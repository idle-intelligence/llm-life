//! Fine-tuning variant A (CONCEPT.md §11): a per-cell prompt, LoRA on the
//! same `TrainModel` the variant-B fine-tune uses, cross-entropy on the
//! single answer token. Unlike B, the "grid" here is not spatial — it is the
//! 512-entry (8 neighbours, self) lookup table in full, so the labelled data
//! is exhaustive rather than sampled: every one of the 512 cases is seen,
//! cycled in a shuffled order, `--batch` at a time per step.
//!
//! Two adapters share this loop and differ only in the prefix text
//! (`variant_a::rules_prefix` vs `variant_a::norules_prefix`), which is the
//! `norules` flag on `RunAArgs`.

use anyhow::{Context, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Bool, Tensor, TensorData};
use life::{Grid, Rule};
use llm_wasm::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::score::{classify, score, LifeCase};
use crate::variant_a::{self, argmax_grid_a, p_alive_grid_a, Chunk};

use super::data::Rng;
use super::load::{load_train_model, LoraSpec};
use super::lora_io;
use super::model::{cell_cross_entropy, TrainModel};
use super::optim::AdamW;

type AB = Autodiff<Wgpu>;

pub struct RunAArgs {
    pub gguf: PathBuf,
    pub tokenizer: PathBuf,
    pub norules: bool,
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub lora: LoraSpec,
    pub seed: u64,
    pub eval_every: usize,
    pub out: PathBuf,
    pub run_doc: Option<PathBuf>,
    pub max_secs: f64,
}

pub struct PromptA {
    pub prefix: Vec<u32>,
    pub dead: u32,
    pub alive: u32,
}

pub fn prompt_a(tokenizer: &Path, rule: &Rule, norules: bool) -> Result<PromptA> {
    let tok = Tokenizer::from_json(&std::fs::read(tokenizer).context("read tokenizer.json")?)?;
    let one = |s: &str| -> Result<u32> {
        let ids = tok.encode(s, false)?;
        anyhow::ensure!(ids.len() == 1, "'{s}' is not a single token: {ids:?}");
        Ok(ids[0])
    };
    let text = if norules {
        variant_a::norules_prefix()
    } else {
        variant_a::rules_prefix(rule)
    };
    Ok(PromptA {
        prefix: tok.encode(&text, false)?,
        dead: one("0")?,
        alive: one("1")?,
    })
}

/// All 512 `(neighbors, self)` cases, in table order (`k = neighbors_bits |
/// self << 8`), with the true Life target for each.
pub fn all_cases(rule: &Rule) -> Vec<([u8; 8], u8, u8)> {
    (0u32..512)
        .map(|k| {
            let nb: [u8; 8] = std::array::from_fn(|b| ((k >> b) & 1) as u8);
            let self_state = ((k >> 8) & 1) as u8;
            let n = nb.iter().filter(|&&b| b != 0).count();
            let next = rule.next(self_state != 0, n) as u8;
            (nb, self_state, next)
        })
        .collect()
}

/// Tokenize one case's prompt with a leading newline, matching `web.rs`'s
/// per-cell prompt set (the model was never shown a prompt starting at
/// column 0 of a fresh line during pretraining, but it also never saw one
/// mid-prefix without a separator, so this keeps the prefix boundary clean).
fn tokenize_case(tok: &Tokenizer, nb: &[u8; 8], self_state: u8) -> Result<Vec<u32>> {
    let text = format!("\n{}", variant_a::cell_prompt(nb, self_state));
    tok.encode(&text, false)
}

/// Pack `prefix` and a chunk's suffix into one square-masked sequence:
/// `variant_a::pack_chunk`'s mask already has this exact shape for its
/// suffix rows (`[t, prefix_len + t]`, columns 0..prefix_len the resident
/// prefix in full); the only piece missing for a from-scratch forward
/// (training has no persistent KV cache) is the causal block for the
/// prefix's own rows.
fn full_sequence(prefix: &[u32], chunk: &Chunk, device: &WgpuDevice) -> (Vec<u32>, Vec<u32>, Tensor<AB, 2, Bool>) {
    let p = prefix.len();
    let t = chunk.tokens.len();
    let total = p + t;

    let mut tokens = Vec::with_capacity(total);
    tokens.extend_from_slice(prefix);
    tokens.extend_from_slice(&chunk.tokens);
    let mut positions: Vec<u32> = (0..p as u32).collect();
    positions.extend_from_slice(&chunk.positions);

    let mut masked_out = vec![true; total * total];
    for i in 0..p {
        for j in 0..=i {
            masked_out[i * total + j] = false;
        }
    }
    for i in 0..t {
        let row = (p + i) * total;
        let src = i * total; // chunk.allowed is [t, p+t] = [t, total]
        for j in 0..total {
            masked_out[row + j] = !chunk.allowed[src + j];
        }
    }
    let mask = Tensor::from_data(TensorData::new(masked_out, [total, total]), device);
    (tokens, positions, mask)
}

pub struct EvalReportA {
    pub loss: f64,
    pub accuracy: f64,
    pub cases: Vec<(LifeCase, usize, usize)>,
    pub iou: f64,
}

/// Evaluate every one of the 512 cases (chunked to keep the mask a sane
/// size) plus IoU on `real_grids` (never drawn from the training RNG).
pub fn evaluate_a(
    model: &TrainModel<AB>,
    p: &PromptA,
    tok: &Tokenizer,
    cases: &[([u8; 8], u8, u8)],
    real_grids: &[Grid],
    rule: &Rule,
    device: &WgpuDevice,
) -> Result<EvalReportA> {
    // Small enough that the autodiff graph over T = prefix + CHUNK*~20
    // tokens (every intermediate activation of all 24 layers, kept live for
    // a backward this eval never runs) does not blow the M2's shared
    // memory. Variant B's per-step forward at a comparable T (~335 tokens,
    // 16x16 grid) is the size this was picked to stay near.
    const CHUNK: usize = 32;
    let mut loss_sum = 0.0;
    let mut correct = 0usize;
    let mut case_totals: Vec<(LifeCase, usize, usize)> = LifeCase::ALL.iter().map(|&c| (c, 0, 0)).collect();

    for slice in cases.chunks(CHUNK) {
        let ids: Vec<usize> = (0..slice.len()).collect();
        let chunk = variant_a::pack_chunk(
            &ids,
            |i| tokenize_case(tok, &slice[i].0, slice[i].1).unwrap(),
            p.prefix.len(),
        );
        let (tokens, positions, mask) = full_sequence(&p.prefix, &chunk, device);
        let logits = model.forward(&tokens, &positions, &mask)?;
        let rows: Vec<u32> = chunk
            .answer_rows()
            .iter()
            .map(|&r| (r + p.prefix.len()) as u32)
            .collect();
        let targets: Vec<u8> = slice.iter().map(|&(_, _, t)| t).collect();
        let l = cell_cross_entropy(logits.clone(), &rows, &targets, device);
        loss_sum += l.clone().into_data().into_vec::<f32>().unwrap()[0] as f64 * slice.len() as f64;
        // Burn's autodiff backend retains every forward's graph nodes until
        // a `backward()` walks and consumes them (`train_oracle.rs` never
        // hit this: it forwards once). An eval loop of dozens of forwards
        // with no backward between them accumulates gigabytes of dead graph
        // per call and jetsam-kills the process; the eval-only gradients
        // this produces are never read (`AdamW::step` takes its own
        // `Gradients` object per real step), so dropping them is free.
        drop(l.backward());

        let data = logits.into_data().into_vec::<f32>().unwrap();
        let pa = variant_a::p_alive_chunk(&data[p.prefix.len() * 2..], &chunk);
        for (i, &(cell, prob)) in pa.iter().enumerate() {
            debug_assert_eq!(cell, i);
            let pred = (prob > 0.5) as u8;
            let (nb, self_state, target) = slice[i];
            if pred == target {
                correct += 1;
            }
            let n = nb.iter().filter(|&&b| b != 0).count();
            let case = classify(self_state != 0, n);
            let idx = LifeCase::ALL.iter().position(|&c| c == case).unwrap();
            case_totals[idx].1 += 1;
            if pred == target {
                case_totals[idx].2 += 1;
            }
        }
    }

    let mut iou_sum = 0.0;
    for g in real_grids {
        let n = g.width() * g.height();
        let mut pa = vec![0.0f32; n];
        let all_ids: Vec<usize> = (0..n).collect();
        for cell_slice in all_ids.chunks(CHUNK) {
            let ids: Vec<usize> = (0..cell_slice.len()).collect();
            let chunk = variant_a::pack_chunk(
                &ids,
                |i| {
                    let c = cell_slice[i];
                    let nb: Vec<u8> = g.neighbor_indices(c).iter().map(|&j| g.cells()[j]).collect();
                    let arr: [u8; 8] = nb.try_into().unwrap();
                    tokenize_case(tok, &arr, g.cells()[c]).unwrap()
                },
                p.prefix.len(),
            );
            let (tokens, positions, mask) = full_sequence(&p.prefix, &chunk, device);
            let logits = model.forward(&tokens, &positions, &mask)?;
            let data = logits.clone().into_data().into_vec::<f32>().unwrap();
            drop(logits.sum().backward()); // flush the autodiff tape, see above
            for (i, prob) in p_alive_grid_a(&data[p.prefix.len() * 2..], &chunk, cell_slice.len())
                .into_iter()
                .enumerate()
            {
                pa[cell_slice[i]] = prob;
            }
        }
        let model_grid = argmax_grid_a(&pa, g.width(), g.height());
        let truth = g.step(rule);
        let s = score(&truth, &model_grid, &pa, 1);
        iou_sum += s.iou;
    }

    Ok(EvalReportA {
        loss: loss_sum / cases.len() as f64,
        accuracy: correct as f64 / cases.len() as f64,
        cases: case_totals,
        iou: iou_sum / real_grids.len().max(1) as f64,
    })
}

fn case_table(r: &EvalReportA) -> String {
    let mut s = String::from("| case | n | correct | frac |\n|---|---|---|---|\n");
    for (c, n, ok) in &r.cases {
        let f = if *n == 0 { 1.0 } else { *ok as f64 / *n as f64 };
        s.push_str(&format!("| {} | {n} | {ok} | {f:.4} |\n", c.label()));
    }
    s
}

/// A fixed, held-out real-grid set for the IoU column: never drawn from the
/// training RNG (which never sees a real grid in the first place — variant
/// A's training set is the exhaustive 512, not sampled grids).
fn real_grids(size: usize) -> Vec<Grid> {
    let mut glider = Grid::new(size, size);
    glider.place_glider(size / 2, size / 2);
    let mut blinker = Grid::new(size, size);
    blinker.set(size / 2 - 1, size / 2, 1);
    blinker.set(size / 2, size / 2, 1);
    blinker.set(size / 2 + 1, size / 2, 1);
    let random = Grid::random(size, size, 999_999, 0.28);
    vec![glider, blinker, random]
}

pub fn run(args: RunAArgs) -> Result<()> {
    let rule = Rule::life();
    let device = WgpuDevice::default();
    let p = prompt_a(&args.tokenizer, &rule, args.norules)?;
    let tok = Tokenizer::from_json(&std::fs::read(&args.tokenizer)?)?;

    println!(
        "variant A {} adapter: prefix {} tokens, answer '0'={} '1'={}",
        if args.norules { "a-norules" } else { "a-rules" },
        p.prefix.len(),
        p.dead,
        p.alive
    );

    let load_start = Instant::now();
    let mut model: TrainModel<AB> = load_train_model(&args.gguf, &[p.dead, p.alive], Some(args.lora), &device)?;
    println!(
        "loaded {} layers, hidden {}, in {:.1}s",
        model.config.num_layers,
        model.config.hidden_size,
        load_start.elapsed().as_secs_f64()
    );

    let mut params: Vec<_> = model.lora_params().into_iter().map(|t| t.require_grad()).collect();
    model.set_lora_params(params.clone());
    let n_trainable: usize = params.iter().map(|t| t.dims()[0] * t.dims()[1]).sum();
    println!("{} LoRA matrices, {n_trainable} trainable parameters", params.len());

    let cases = all_cases(&rule);
    let grids = real_grids(16);

    let before = evaluate_a(&model, &p, &tok, &cases, &grids, &rule, &device)?;
    println!(
        "step 0 (base): loss {:.4} acc {:.4} ({}/{}) iou {:.4}",
        before.loss,
        before.accuracy,
        (before.accuracy * cases.len() as f64).round() as usize,
        cases.len(),
        before.iou
    );

    let mut opt = AdamW::<AB>::new(&params, args.lr);
    let mut rng = Rng::new(args.seed);
    let mut order: Vec<usize> = (0..cases.len()).collect();
    let mut cursor = cases.len(); // forces a shuffle before step 1
    let mut log: Vec<(usize, f64, f64)> = Vec::new();
    let mut evals: Vec<(usize, EvalReportA)> = Vec::new();
    let start = Instant::now();
    let mut step = 0usize;
    let mut best: Option<(f64, Vec<Tensor<AB, 2>>)> = None;
    let mut stopped_full = false;

    while step < args.steps {
        if start.elapsed().as_secs_f64() > args.max_secs {
            println!("wall-clock budget reached at step {step}");
            break;
        }
        step += 1;

        if cursor + args.batch > order.len() {
            // Fisher-Yates over the xorshift RNG already in scope.
            for i in (1..order.len()).rev() {
                let j = (rng.unit() * (i + 1) as f64) as usize;
                order.swap(i, j.min(i));
            }
            cursor = 0;
        }
        let batch_idx = &order[cursor..cursor + args.batch];
        cursor += args.batch;
        let batch: Vec<([u8; 8], u8, u8)> = batch_idx.iter().map(|&i| cases[i]).collect();

        let ids: Vec<usize> = (0..batch.len()).collect();
        let chunk = variant_a::pack_chunk(
            &ids,
            |i| tokenize_case(&tok, &batch[i].0, batch[i].1).unwrap(),
            p.prefix.len(),
        );
        let (tokens, positions, mask) = full_sequence(&p.prefix, &chunk, &device);
        let logits = model.forward(&tokens, &positions, &mask)?;
        let rows: Vec<u32> = chunk
            .answer_rows()
            .iter()
            .map(|&r| (r + p.prefix.len()) as u32)
            .collect();
        let targets: Vec<u8> = batch.iter().map(|&(_, _, t)| t).collect();
        let loss = cell_cross_entropy(logits, &rows, &targets, &device);
        let loss_v = loss.clone().into_data().into_vec::<f32>().unwrap()[0] as f64;

        let grads = loss.backward();
        params = opt.step(params, &grads);
        model.set_lora_params(params.clone());

        let secs = start.elapsed().as_secs_f64();
        log.push((step, loss_v, secs));
        println!("step {step:4} loss {loss_v:.4}  {:.1}s", secs);

        if args.eval_every > 0 && step.is_multiple_of(args.eval_every) {
            let e = evaluate_a(&model, &p, &tok, &cases, &grids, &rule, &device)?;
            println!(
                "  held-out: loss {:.4} acc {:.4} ({}/{}) iou {:.4}",
                e.loss,
                e.accuracy,
                (e.accuracy * cases.len() as f64).round() as usize,
                cases.len(),
                e.iou
            );
            let is_best = best.as_ref().map(|(a, _)| e.accuracy > *a).unwrap_or(true);
            if is_best {
                best = Some((e.accuracy, params.clone()));
            }
            let full = e.accuracy >= 1.0;
            evals.push((step, e));
            if full {
                println!("512/512 correct at step {step}, stopping");
                stopped_full = true;
                break;
            }
        }
    }

    let (best_acc, best_params) = best.unwrap_or_else(|| (before.accuracy, params.clone()));
    lora_io::save(&args.out, &args.lora, &best_params)?;
    println!("wrote {} (best-by-accuracy {:.4})", args.out.display(), best_acc);

    if let Some(doc) = &args.run_doc {
        model.set_lora_params(best_params);
        let best_eval = evaluate_a(&model, &p, &tok, &cases, &grids, &rule, &device)?;
        let mut s = format!(
            "# Variant A LoRA fine-tune — {}\n\n\
             machine: {}\n\
             model: {}\n\
             adapter: {}\n\
             LoRA: rank {}, alpha {}, q/k/v/o{}\n\
             steps: {step}{}, batch {}, lr {}, seed {}\n\
             trainable parameters: {n_trainable}\n\
             wall clock: {:.1}s\n\n",
            if args.norules { "a-norules" } else { "a-rules" },
            machine(),
            args.gguf.display(),
            if args.norules { "a-norules (no rule text in the prefix)" } else { "a-rules (rules in the prefix)" },
            args.lora.rank,
            args.lora.alpha,
            if args.lora.mlp { " + gate/up" } else { "" },
            if stopped_full { " (512/512, early stop)" } else { "" },
            args.batch,
            args.lr,
            args.seed,
            start.elapsed().as_secs_f64(),
        );
        s.push_str(&format!(
            "## Base (step 0)\n\nloss {:.4}, accuracy {:.4} ({}/512)\n\n{}\n",
            before.loss,
            before.accuracy,
            (before.accuracy * 512.0).round() as usize,
            case_table(&before)
        ));
        s.push_str("## Loss\n\n| step | loss | s |\n|---|---|---|\n");
        for (i, l, secs) in &log {
            s.push_str(&format!("| {i} | {l:.4} | {secs:.1} |\n"));
        }
        s.push_str("\n## Held-out (every eval-every steps)\n\n");
        for (i, e) in &evals {
            s.push_str(&format!(
                "### step {i}\n\nloss {:.4}, accuracy {:.4} ({}/512), IoU (3 real grids) {:.4}\n\n{}\n",
                e.loss,
                e.accuracy,
                (e.accuracy * 512.0).round() as usize,
                e.iou,
                case_table(e)
            ));
        }
        s.push_str(&format!(
            "## Best-by-accuracy checkpoint (saved to `{}`)\n\naccuracy {:.4} ({}/512), IoU {:.4}\n\n{}\n",
            args.out.display(),
            best_eval.accuracy,
            (best_eval.accuracy * 512.0).round() as usize,
            best_eval.iou,
            case_table(&best_eval)
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
