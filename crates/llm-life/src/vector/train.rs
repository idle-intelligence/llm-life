//! Training loops for the vector-space variants (CONCEPT.md §12).

use anyhow::Result;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Int, Tensor, TensorData};
use life::Rule;
use std::path::PathBuf;

use super::data::density_sweep_batch;
use super::model::{stencil_mask, AttnConfig, AttnOfLife, StencilConfig, StencilOfLife};
use crate::bert::data::{all_cases, split_512, Case};
use crate::score::{score, GenScore};

type AB = Autodiff<Wgpu>;
type IB = Wgpu;

fn bits_of<B: Backend>(cases: &[Case], device: &B::Device) -> (Tensor<B, 2>, Vec<u8>) {
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
        TensorData::new(labels.iter().map(|&l| l as i32).collect::<Vec<_>>(), [n]),
        device,
    );
    burn::nn::loss::CrossEntropyLossConfig::new().init(device).forward(logits, targets)
}

fn accuracy<B: Backend>(logits: &Tensor<B, 2>, labels: &[u8]) -> f64 {
    let pred = logits.clone().argmax(1).into_data().into_vec::<i32>().unwrap();
    let ok = pred.iter().zip(labels).filter(|(&p, &l)| p as u8 == l).count();
    ok as f64 / labels.len() as f64
}

/// Same convention as `bert::train::rollout_score`: 5 teacher-forced
/// generations from a random `width x height` grid at `seed`.
fn rollout_score(
    logits_fn: impl Fn(&[Case]) -> Vec<f32>,
    rule: &Rule,
    width: usize,
    height: usize,
    seed: u64,
) -> Vec<GenScore> {
    let mut grid = life::Grid::random(width, height, seed, 0.28);
    let mut out = Vec::with_capacity(5);
    for gen in 1..=5 {
        let truth = grid.step(rule);
        let cases = crate::bert::data::grid_cases(&grid, rule);
        let p = logits_fn(&cases);
        let pred_cells: Vec<u8> = p.iter().map(|&v| (v >= 0.5) as u8).collect();
        let pred_grid = life::Grid::from_cells(width, height, pred_cells);
        out.push(score(&truth, &pred_grid, &p, gen));
        grid = truth;
    }
    out
}

pub struct AttnResult {
    pub params: usize,
    pub steps_to_512: Option<usize>,
    pub held_out_acc: f64,
    pub scores: Vec<GenScore>,
}

/// (i) the attention-over-9-numbers model, trained exactly like
/// `bert::train::train_bert` (same 512-case split, same rollout convention)
/// but on float inputs through `AttnOfLife` instead of token ids through
/// `BertOfLife`.
pub fn train_attn(d_model: usize, steps: usize, lr: f64, out: &PathBuf) -> Result<AttnResult> {
    let rule = Rule::life();
    let device = WgpuDevice::default();
    let cfg = AttnConfig::new(d_model);
    let mut model: AttnOfLife<AB> = cfg.init(&device);
    let params = model.num_params();

    let (train, held) = split_512(&rule, 64);
    let all = all_cases(&rule);
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
    let held_out_acc = accuracy(&valid.forward(held_t), &held_y);

    let mut scores = Vec::new();
    for seed_n in 1..=3u64 {
        scores.extend(rollout_score(
            |cases| {
                let (t, _) = bits_of::<IB>(cases, &device);
                let logits = valid.forward(t);
                let probs = burn::tensor::activation::softmax(logits, 1);
                probs.slice([0..cases.len(), 1..2]).into_data().into_vec::<f32>().unwrap()
            },
            &rule,
            16,
            16,
            seed_n,
        ));
    }

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes = recorder.record(model.into_record(), ())?;
    std::fs::write(out, &bytes)?;

    Ok(AttnResult { params, steps_to_512, held_out_acc, scores })
}

pub struct StencilResult {
    pub params: usize,
    pub steps_to_converge: Option<usize>,
    pub train_iou: f64,
    /// IoU rollouts at 16x16 and 32x32 — the generalisation claim
    /// (CONCEPT.md §12: "the model must generalise across grid size since
    /// the mask is local"), same weights, only the mask tensor rebuilt.
    pub scores_16: Vec<GenScore>,
    pub scores_32: Vec<GenScore>,
}

/// (ii) the whole-grid stencil-masked transformer. Trained on a density
/// sweep of random 16x16 grids (BCE against the true next state), evaluated
/// as an IoU rollout at 16x16 (in-distribution) and 32x32 (CONCEPT.md's
/// generalisation test — the mask is rebuilt for the new size, the weights
/// are not retrained).
pub fn train_stencil(
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    steps: usize,
    lr: f64,
    out: &PathBuf,
) -> Result<StencilResult> {
    let rule = Rule::life();
    let device = WgpuDevice::default();
    let cfg = StencilConfig::new(d_model, n_layers, n_heads);
    let mut model: StencilOfLife<AB> = cfg.init(&device);
    let params = model.num_params();

    let densities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
    let (xs, ys, batch) = density_sweep_batch(&rule, 16, 16, &densities, 8, 1);
    let n = 16 * 16;
    let train_x: Tensor<AB, 2> = Tensor::from_data(TensorData::new(xs, [batch, n]), &device);
    let train_y: Tensor<AB, 2, Int> =
        Tensor::from_data(TensorData::new(ys.iter().map(|&b| b as i32).collect::<Vec<_>>(), [batch, n]), &device);
    let mask_train: Tensor<AB, 2> = stencil_mask(16, 16, &device);
    let bce: BinaryCrossEntropyLoss<AB> = BinaryCrossEntropyLossConfig::new().with_logits(true).init(&device);

    let mut opt = AdamConfig::new().init();
    let mut steps_to_converge = None;
    for step in 1..=steps {
        let logits = model.forward(train_x.clone(), mask_train.clone());
        let loss = bce.forward(logits, train_y.clone());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = opt.step(lr, model, grads);

        if steps_to_converge.is_none() && step.is_multiple_of(20) {
            let valid = model.valid();
            let mask_v: Tensor<IB, 2> = stencil_mask(16, 16, &device);
            let logits = valid.forward(train_x.clone().inner(), mask_v);
            let pred = logits.clone().greater_elem(0.0);
            let target = train_y.clone().inner().equal_elem(1);
            let hit = pred.clone().bool_and(target.clone()).float().sum().into_scalar();
            let union = pred.bool_or(target).float().sum().into_scalar();
            let iou = if union == 0.0 { 1.0 } else { hit as f64 / union as f64 };
            if iou >= 0.999 {
                steps_to_converge = Some(step);
                break;
            }
        }
    }

    let valid = model.valid();
    let mask_v: Tensor<IB, 2> = stencil_mask(16, 16, &device);
    let logits = valid.forward(train_x.inner(), mask_v);
    let pred = logits.clone().greater_elem(0.0);
    let target: Tensor<IB, 2, Int> = train_y.inner();
    let target_bool = target.equal_elem(1);
    let hit = pred.clone().bool_and(target_bool.clone()).float().sum().into_scalar();
    let union = pred.bool_or(target_bool).float().sum().into_scalar();
    let train_iou = if union == 0.0 { 1.0 } else { hit as f64 / union as f64 };

    // `cases` come from `crate::bert::data::grid_cases`, one entry per cell
    // in row-major order — exactly the flat layout `StencilOfLife::forward`
    // expects, so the 9-token case is collapsed back to just the self bit
    // (index 8) for the input and the mask does the rest.
    let forward_probs = |width: usize, height: usize, cases: &[Case]| -> Vec<f32> {
        let n = cases.len();
        let xs: Vec<f32> = cases.iter().map(|(c, _)| c[8] as f32).collect();
        let x: Tensor<IB, 2> = Tensor::from_data(TensorData::new(xs, [1, n]), &device);
        let mask: Tensor<IB, 2> = stencil_mask(width, height, &device);
        let logits = valid.forward(x, mask);
        let probs = burn::tensor::activation::sigmoid(logits);
        probs.into_data().into_vec::<f32>().unwrap()
    };

    let mut scores_16 = Vec::new();
    let mut scores_32 = Vec::new();
    for seed_n in 1..=3u64 {
        scores_16.extend(rollout_score(|cases| forward_probs(16, 16, cases), &rule, 16, 16, seed_n));
        scores_32.extend(rollout_score(|cases| forward_probs(32, 32, cases), &rule, 32, 32, seed_n));
    }

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes = recorder.record(model.into_record(), ())?;
    std::fs::write(out, &bytes)?;

    Ok(StencilResult { params, steps_to_converge, train_iou, scores_16, scores_32 })
}
