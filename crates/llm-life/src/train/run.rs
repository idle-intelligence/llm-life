//! The variant-B fine-tuning loop (CONCEPT.md §5).
//!
//! One training example is one whole grid: the rules prefix, one token per
//! cell, the stencil mask and bag positions exactly as inference builds them
//! (`variant_b::pack`), and a cross-entropy on the two answer logits at every
//! cell against true Life's next state. The grid is the batch — 1024 labelled
//! cells per forward at 32x32 — so there is no batch axis anywhere, which is
//! the same choice the inference path makes.
//!
//! What moves: LoRA `a`/`b` on q/k/v/o (and gate/up with `--lora-mlp`).
//! Everything else, including the token embedding and the tied head, is
//! frozen.

use anyhow::{Context, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Bool, Tensor, TensorData};
use life::{Grid, Rule};
use llm_wasm::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::score::{per_case_recall, score, LifeCase};
use crate::variant_b::{argmax_grid, p_alive, pack, rules_prefix};

use super::data::{held_out, sample_grid, targets, Rng};
use super::load::{load_train_model, LoraSpec};
use super::lora_io;
use super::model::{cell_cross_entropy, TrainModel};
use super::optim::AdamW;

type AB = Autodiff<Wgpu>;

pub struct TrainArgs {
    pub gguf: PathBuf,
    pub tokenizer: PathBuf,
    pub size: usize,
    pub steps: usize,
    pub lr: f64,
    pub lora: LoraSpec,
    pub seed: u64,
    pub eval_every: usize,
    pub eval_grids: usize,
    pub out: PathBuf,
    pub run_doc: Option<PathBuf>,
    /// Stop after this many seconds, whatever `steps` says — the GPU is
    /// shared, so a run has a wall-clock budget, not only a step budget.
    pub max_secs: f64,
}

/// Everything derived from the tokenizer that the loop needs.
pub struct Prompt {
    pub prefix: Vec<u32>,
    pub dead: u32,
    pub alive: u32,
}

pub fn prompt(tokenizer: &Path, rule: &Rule) -> Result<Prompt> {
    let tok = Tokenizer::from_json(&std::fs::read(tokenizer).context("read tokenizer.json")?)?;
    let one = |s: &str| -> Result<u32> {
        let ids = tok.encode(s, false)?;
        anyhow::ensure!(ids.len() == 1, "'{s}' is not a single token: {ids:?}");
        Ok(ids[0])
    };
    Ok(Prompt {
        prefix: tok.encode(&rules_prefix(rule), false)?,
        dead: one("0")?,
        alive: one("1")?,
    })
}

fn mask_tensor<B: burn::prelude::Backend>(
    allowed: &[bool],
    t: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let out: Vec<bool> = allowed.iter().map(|&a| !a).collect();
    Tensor::from_data(TensorData::new(out, [t, t]), device)
}

/// p(alive) per cell from a `[1, T, 2]` logits tensor — the training-side
/// twin of `variant_b::p_alive`, which reads the same two columns out of the
/// engine's readback.
fn p_alive_from(logits: Tensor<AB, 3>, grid_start: usize, n: usize) -> Vec<f32> {
    let v = logits.into_data().into_vec::<f32>().unwrap();
    p_alive(&v, grid_start, n)
}

/// Held-out per-case recall — the number that says whether birth and
/// survival have moved off zero, which grid-level accuracy hides.
pub struct EvalReport {
    pub loss: f64,
    pub accuracy: f64,
    pub iou: f64,
    pub cases: Vec<(LifeCase, usize, usize)>,
}

impl EvalReport {
    pub fn case_fracs(&self) -> Vec<(LifeCase, f64)> {
        self.cases
            .iter()
            .map(|&(c, n, ok)| (c, if n == 0 { 1.0 } else { ok as f64 / n as f64 }))
            .collect()
    }
}

pub fn evaluate(
    model: &TrainModel<AB>,
    p: &Prompt,
    grids: &[Grid],
    rule: &Rule,
    device: &WgpuDevice,
) -> Result<EvalReport> {
    let mut totals: Vec<(LifeCase, usize, usize)> =
        LifeCase::ALL.iter().map(|&c| (c, 0, 0)).collect();
    let mut loss_sum = 0.0;
    let mut acc_sum = 0.0;
    let mut iou_sum = 0.0;
    for g in grids {
        let packed = pack(g, &p.prefix, p.dead, p.alive);
        let t = packed.tokens.len();
        let n = g.width() * g.height();
        let mask = mask_tensor::<AB>(&packed.allowed, t, device);
        let logits = model.forward(&packed.tokens, &packed.positions, &mask)?;
        let tgt = targets(g, rule);
        let rows: Vec<u32> = (packed.grid_start as u32..(packed.grid_start + n) as u32).collect();
        let l = cell_cross_entropy(logits.clone(), &rows, &tgt, device);
        loss_sum += l.into_data().into_vec::<f32>().unwrap()[0] as f64;

        let pa = p_alive_from(logits, packed.grid_start, n);
        let model_grid = argmax_grid(&pa, g.width(), g.height());
        let truth = g.step(rule);
        let s = score(&truth, &model_grid, &pa, 1);
        acc_sum += s.accuracy;
        iou_sum += s.iou;
        for (i, (_, count)) in per_case_recall(g, &model_grid, rule).iter().enumerate() {
            totals[i].1 += count.count;
            totals[i].2 += count.correct;
        }
    }
    let k = grids.len() as f64;
    Ok(EvalReport {
        loss: loss_sum / k,
        accuracy: acc_sum / k,
        iou: iou_sum / k,
        cases: totals,
    })
}

fn case_table(r: &EvalReport) -> String {
    let mut s = String::from("| case | n | correct | frac |\n|---|---|---|---|\n");
    for (c, n, ok) in &r.cases {
        let f = if *n == 0 { 1.0 } else { *ok as f64 / *n as f64 };
        s.push_str(&format!("| {} | {n} | {ok} | {f:.4} |\n", c.label()));
    }
    s
}

pub fn run(args: TrainArgs) -> Result<()> {
    let rule = Rule::life();
    let device = WgpuDevice::default();
    let p = prompt(&args.tokenizer, &rule)?;

    println!(
        "prefix {} tokens, answer '0'={} '1'={}, grid {}x{} -> T={}",
        p.prefix.len(),
        p.dead,
        p.alive,
        args.size,
        args.size,
        p.prefix.len() + args.size * args.size
    );

    let load_start = Instant::now();
    let mut model: TrainModel<AB> = load_train_model(
        &args.gguf,
        &[p.dead, p.alive],
        Some(args.lora),
        &device,
    )?;
    println!(
        "loaded {} layers, hidden {}, in {:.1}s",
        model.config.num_layers,
        model.config.hidden_size,
        load_start.elapsed().as_secs_f64()
    );

    let mut params: Vec<_> = model
        .lora_params()
        .into_iter()
        .map(|t| t.require_grad())
        .collect();
    model.set_lora_params(params.clone());
    let n_trainable: usize = params.iter().map(|t| t.dims()[0] * t.dims()[1]).sum();
    println!("{} LoRA matrices, {n_trainable} trainable parameters", params.len());

    let eval_grids = held_out(args.size, args.eval_grids);
    let before = evaluate(&model, &p, &eval_grids, &rule, &device)?;
    println!(
        "step 0 (base): held-out loss {:.4} acc {:.4} iou {:.4}",
        before.loss, before.accuracy, before.iou
    );
    for (c, f) in before.case_fracs() {
        println!("  {:<16} {f:.4}", c.label());
    }

    let mut opt = AdamW::<AB>::new(&params, args.lr);
    let mut rng = Rng::new(args.seed);
    let mut log: Vec<(usize, f64, f64)> = Vec::new();
    let start = Instant::now();
    let mut step = 0usize;
    let mut last_eval: Option<EvalReport> = None;

    while step < args.steps {
        if start.elapsed().as_secs_f64() > args.max_secs {
            println!("wall-clock budget reached at step {step}");
            break;
        }
        step += 1;
        let grid = sample_grid(&mut rng, args.size);
        let packed = pack(&grid, &p.prefix, p.dead, p.alive);
        let t = packed.tokens.len();
        let n = args.size * args.size;
        let mask = mask_tensor::<AB>(&packed.allowed, t, &device);
        let logits = model.forward(&packed.tokens, &packed.positions, &mask)?;
        let tgt = targets(&grid, &rule);
        let rows: Vec<u32> = (packed.grid_start as u32..(packed.grid_start + n) as u32).collect();
        let loss = cell_cross_entropy(logits, &rows, &tgt, &device);
        let loss_v = loss.clone().into_data().into_vec::<f32>().unwrap()[0] as f64;

        let grads = loss.backward();
        params = opt.step(params, &grads);
        model.set_lora_params(params.clone());

        let secs = start.elapsed().as_secs_f64();
        log.push((step, loss_v, secs));
        println!("step {step:4} loss {loss_v:.4}  {:.1}s", secs);

        if args.eval_every > 0 && step.is_multiple_of(args.eval_every) {
            let e = evaluate(&model, &p, &eval_grids, &rule, &device)?;
            println!("  held-out loss {:.4} acc {:.4} iou {:.4}", e.loss, e.accuracy, e.iou);
            for (c, f) in e.case_fracs() {
                println!("    {:<16} {f:.4}", c.label());
            }
            last_eval = Some(e);
        }
    }

    let after = match last_eval {
        Some(e) if args.eval_every > 0 && step.is_multiple_of(args.eval_every) => e,
        _ => evaluate(&model, &p, &eval_grids, &rule, &device)?,
    };

    lora_io::save(&args.out, &params)?;
    println!("wrote {}", args.out.display());

    if let Some(doc) = &args.run_doc {
        let mut s = format!(
            "# Fine-tune run\n\n\
             machine: {}\n\
             model: {}\n\
             grid: {}x{} (torus), rule {}\n\
             LoRA: rank {}, alpha {}, q/k/v/o{}\n\
             steps: {step} (budget {}), lr {}, seed {}\n\
             trainable parameters: {n_trainable}\n\
             wall clock: {:.1}s\n\n\
             Timings provisional: the Metal GPU is shared with another repo's training job.\n\n",
            machine(),
            args.gguf.display(),
            args.size,
            args.size,
            rule.to_rulestring(),
            args.lora.rank,
            args.lora.alpha,
            if args.lora.mlp { " + gate/up" } else { "" },
            args.steps,
            args.lr,
            args.seed,
            start.elapsed().as_secs_f64(),
        );
        s.push_str("## Loss\n\n| step | loss | s |\n|---|---|---|\n");
        for (i, l, secs) in &log {
            s.push_str(&format!("| {i} | {l:.4} | {secs:.1} |\n"));
        }
        s.push_str(&format!(
            "\n## Held-out per-case recall ({} grids, never drawn from the training RNG)\n\n             ### before (base model, LoRA at zero)\n\n             loss {:.4}, accuracy {:.4}, IoU {:.4}\n\n{}\n             ### after ({step} steps)\n\n             loss {:.4}, accuracy {:.4}, IoU {:.4}\n\n{}",
            eval_grids.len(),
            before.loss,
            before.accuracy,
            before.iou,
            case_table(&before),
            after.loss,
            after.accuracy,
            after.iou,
            case_table(&after),
        ));
        if let Some(dir) = doc.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(doc, s)?;
        println!("wrote {}", doc.display());
    }
    Ok(())
}

fn machine() -> String {
    let out = std::process::Command::new("uname").arg("-srm").output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "unknown".into(),
    }
}
