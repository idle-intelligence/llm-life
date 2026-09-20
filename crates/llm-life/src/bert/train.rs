//! Training loop for BERT of Life, the MLP baseline, and the lookup-table
//! floor (CONCEPT.md §11 "BERT of Life" backlog item).
//!
//! Trains on the exhaustive 512 cases (it is a finite function — CONCEPT.md
//! §12) with a held-out split for the generalisation row, then evaluates the
//! IoU rollout on real 16x16 grids exactly the way variant A/B's `score`
//! module scores the LLM.

use anyhow::Result;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Int, Tensor, TensorData};
use life::Rule;
use std::path::PathBuf;
use std::time::Instant;

use super::data::{grid_cases, split_512};
use super::model::{BertConfig, BertOfLife, LookupTable, MlpConfig, MlpOfLife};
use crate::score::score;

type AB = Autodiff<Wgpu>;
/// Inference-only backend: `.valid()` on an `AutodiffModule` returns a
/// module over this backend, so eval tensors are built on it directly rather
/// than converted after the fact.
type IB = Wgpu;

fn tokens_of<B: Backend>(cases: &[([u8; 9], u8)], device: &B::Device) -> (Tensor<B, 2, Int>, Vec<u8>) {
    let n = cases.len();
    let mut toks = Vec::with_capacity(n * 9);
    let mut labels = Vec::with_capacity(n);
    for (c, y) in cases {
        toks.extend(c.iter().map(|&b| b as i64));
        labels.push(*y);
    }
    let t = Tensor::from_data(TensorData::new(toks, [n, 9]), device);
    (t, labels)
}

fn bits_of<B: Backend>(cases: &[([u8; 9], u8)], device: &B::Device) -> (Tensor<B, 2>, Vec<u8>) {
    let n = cases.len();
    let mut v = Vec::with_capacity(n * 9);
    let mut y = Vec::with_capacity(n);
    for (c, l) in cases {
        v.extend(c.iter().map(|&b| b as f32));
        y.push(*l);
    }
    (Tensor::from_data(TensorData::new(v, [n, 9]), device), y)
}

fn cross_entropy<B: Backend>(logits: Tensor<B, 2>, labels: &[u8], device: &B::Device) -> Tensor<B, 1> {
    let n = labels.len();
    let targets: Tensor<B, 1, Int> = Tensor::from_data(
        TensorData::new(labels.iter().map(|&l| l as i64).collect::<Vec<_>>(), [n]),
        device,
    );
    burn::nn::loss::CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits, targets)
}

fn accuracy<B: Backend>(logits: &Tensor<B, 2>, labels: &[u8]) -> f64 {
    let pred = logits.clone().argmax(1).into_data().into_vec::<i64>().unwrap();
    let ok = pred.iter().zip(labels).filter(|(&p, &l)| p as u8 == l).count();
    ok as f64 / labels.len() as f64
}

/// p(alive) per cell of a 16x16 grid from `[n, 2]` logits, plus the IoU
/// rollout for 5 generations at one seed, teacher-forced (each generation
/// starts from true Life, matching `docs/runs/2026-09-20-a-rollout.md`'s
/// convention).
fn rollout_iou(logits_fn: impl Fn(&[([u8; 9], u8)]) -> Vec<f32>, rule: &Rule, seed: u64) -> Vec<f64> {
    let mut grid = life::Grid::random(16, 16, seed, 0.28);
    let mut out = Vec::with_capacity(5);
    for _gen in 1..=5 {
        let truth = grid.step(rule);
        let cases = grid_cases(&grid, rule);
        let p = logits_fn(&cases);
        let pred_cells: Vec<u8> = p.iter().map(|&v| (v >= 0.5) as u8).collect();
        let pred_grid = life::Grid::from_cells(16, 16, pred_cells);
        let s = score(&truth, &pred_grid, &p, 1);
        out.push(s.iou);
        grid = truth;
    }
    out
}

pub struct BertResult {
    pub params: usize,
    pub steps_to_512: Option<usize>,
    pub held_out_acc: f64,
    /// 15 IoU values: seeds 1..=3, generations 1..=5 each.
    pub iou_per_gen: Vec<f64>,
}

/// Train one BERT-of-Life size to convergence on the 512-case lookup, report
/// steps-to-exact (first step all 512 are correct), held-out accuracy on 64
/// never-trained cases, and the IoU rollout on real 16x16 grids.
pub fn train_bert(
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    steps: usize,
    lr: f64,
    _seed: u64,
    out: &PathBuf,
) -> Result<BertResult> {
    let rule = Rule::life();
    let device = WgpuDevice::default();
    let cfg = BertConfig::small(d_model, n_layers, n_heads);
    let mut model: BertOfLife<AB> = cfg.init(&device);
    let params = model.num_params();

    let (train, held) = split_512(&rule, 64);
    let all = super::data::all_cases(&rule);
    let (train_t, train_y) = tokens_of::<AB>(&train, &device);
    let (held_t, held_y) = tokens_of::<IB>(&held, &device);
    let (all_t, all_y) = tokens_of::<IB>(&all, &device);

    let mut opt = AdamConfig::new().init();
    let mut steps_to_512: Option<usize> = None;
    let start = Instant::now();

    for step in 1..=steps {
        let logits = model.forward(train_t.clone());
        let loss = cross_entropy::<AB>(logits, &train_y, &device);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = opt.step(lr, model, grads);

        if steps_to_512.is_none() && step.is_multiple_of(5) {
            let valid_model = model.valid();
            let logits = valid_model.forward(all_t.clone());
            if accuracy(&logits, &all_y) >= 1.0 {
                steps_to_512 = Some(step);
                println!(
                    "  d={d_model} L={n_layers}: all 512 exact at step {step} ({:.1}s)",
                    start.elapsed().as_secs_f64()
                );
                break;
            }
        }
    }

    let valid_model = model.valid();
    let held_out_acc = accuracy(&valid_model.forward(held_t), &held_y);

    let mut iou_per_gen = Vec::new();
    for seed_n in 1..=3u64 {
        iou_per_gen.extend(rollout_iou(
            |cases| {
                let (t, _) = tokens_of::<IB>(cases, &device);
                let logits = valid_model.forward(t);
                let probs = burn::tensor::activation::softmax(logits, 1);
                probs.slice([0..cases.len(), 1..2]).into_data().into_vec::<f32>().unwrap()
            },
            &rule,
            seed_n,
        ));
    }

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes = recorder.record(model.into_record(), ())?;
    std::fs::write(out, &bytes)?;
    println!("  wrote {} ({} bytes)", out.display(), bytes.len());

    Ok(BertResult {
        params,
        steps_to_512,
        held_out_acc,
        iou_per_gen,
    })
}

/// The one-layer MLP baseline on the raw 9 bits ("the honest floor" — a model
/// with no encoder structure at all). Returns `(params, steps_to_512,
/// held_out_acc, iou_per_gen)`.
pub fn train_mlp(hidden: usize, steps: usize, lr: f64, out: &PathBuf) -> Result<(usize, Option<usize>, f64, Vec<f64>)> {
    let rule = Rule::life();
    let device = WgpuDevice::default();
    let cfg = MlpConfig { hidden };
    let mut model: MlpOfLife<AB> = cfg.init(&device);
    let params = model.num_params();

    let (train, held) = split_512(&rule, 64);
    let all = super::data::all_cases(&rule);
    let (train_t, train_y) = bits_of::<AB>(&train, &device);
    let (held_t, held_y) = bits_of::<IB>(&held, &device);
    let (all_t, all_y) = bits_of::<IB>(&all, &device);

    let mut opt = AdamConfig::new().init();
    let mut steps_to_512 = None;
    for step in 1..=steps {
        let logits = model.forward(train_t.clone());
        let loss = cross_entropy::<AB>(logits, &train_y, &device);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = opt.step(lr, model, grads);
        if steps_to_512.is_none() && step.is_multiple_of(5) {
            let valid = model.valid();
            if accuracy(&valid.forward(all_t.clone()), &all_y) >= 1.0 {
                steps_to_512 = Some(step);
                break;
            }
        }
    }
    let valid = model.valid();
    let held_acc = accuracy(&valid.forward(held_t), &held_y);

    let mut iou_per_gen = Vec::new();
    for seed_n in 1..=3u64 {
        iou_per_gen.extend(rollout_iou(
            |cases| {
                let (t, _) = bits_of::<IB>(cases, &device);
                let logits = valid.forward(t);
                let probs = burn::tensor::activation::softmax(logits, 1);
                probs.slice([0..cases.len(), 1..2]).into_data().into_vec::<f32>().unwrap()
            },
            &rule,
            seed_n,
        ));
    }

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes = recorder.record(model.into_record(), ())?;
    std::fs::write(out, &bytes)?;

    Ok((params, steps_to_512, held_acc, iou_per_gen))
}

/// The zero-parameter lookup floor: exact by construction (it *is* the
/// rule), included so its IoU rollout row exists next to the trained ones.
/// Returns `(params = 0, held_out_acc = 1.0, iou_per_gen)`.
pub fn eval_lookup() -> (usize, f64, Vec<f64>) {
    let rule = Rule::life();
    let table = LookupTable::from_rule(&rule);
    let mut iou_per_gen = Vec::new();
    for seed_n in 1..=3u64 {
        iou_per_gen.extend(rollout_iou(
            |cases| {
                cases
                    .iter()
                    .map(|(c, _)| {
                        let mut k = 0usize;
                        for (b, &bit) in c[..8].iter().enumerate() {
                            if bit != 0 {
                                k |= 1 << b;
                            }
                        }
                        k |= (c[8] as usize) << 8;
                        table.predict(k) as u32 as f32
                    })
                    .collect()
            },
            &rule,
            seed_n,
        ));
    }
    (0, 1.0, iou_per_gen)
}
