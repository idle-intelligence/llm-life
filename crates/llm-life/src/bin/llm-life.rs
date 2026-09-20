//! Native driver: run variant A or B against a real GGUF and write the pictures.
//!
//! Backend features are selected here, in the consumer crate (CLAUDE.md
//! "Code"). The GPU is shared with another repo's training job — every run
//! here is one short inference per generation, and every timing it prints is
//! provisional.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use life::{Grid, Rule};
use llm_life::pgm::{read_pgm, write_binary_pgm, write_pgm};
use llm_life::score::{median_threshold_grid, otsu_threshold_grid, per_case_recall, score, zscore_threshold_grid};
use llm_life::variant_a;
use llm_life::variant_b::{
    argmax_grid, fewshot_rules_prefix, p_alive, pack_sparse, rules_prefix, MAX_STENCIL_KEYS,
};
use llm_life::train::load::{load_train_model, LoraSpec};
use llm_life::train::lora_io;
use llm_life::train::model::TrainModel;
use llm_life::train::run::{TrainArgs};
use llm_life::train::run_a::{self, RunAArgs};
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::kv::KvCache;
use llm_wasm::model::{ForwardSpec, LlmModel, SparseMask};
use llm_wasm::tokenizer::Tokenizer;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;

#[derive(Parser)]
#[command(name = "llm-life", about = "A language model as the update rule of a cellular automaton")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// B/S rulestring for the classical truth and, where the subcommand
    /// builds a prefix, the rules-in-prompt variant (CONCEPT.md §11 backlog:
    /// rule families). `norules` prefixes are unaffected by design — that's
    /// the point of the ablation.
    #[arg(long, global = true, default_value = "B3/S23")]
    rule: String,
}

#[derive(Subcommand)]
enum Command {
    /// Run variant B for N generations and write pictures + a text summary.
    Picture {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "64")]
        size: usize,
        #[arg(long, default_value = "10")]
        generations: usize,
        /// Seeds to run; `glider` is also accepted as a seed name.
        #[arg(long, value_delimiter = ',', default_value = "glider,1,2")]
        seeds: Vec<String>,
        #[arg(long, default_value = "0.28")]
        density: f64,
        #[arg(long, default_value = "docs/pictures")]
        out: PathBuf,
        /// Feed the model its own previous output instead of true Life's
        /// (the honest "let it run" mode). Off = teacher forcing: every
        /// generation starts from the true grid, which isolates one-step
        /// rule accuracy from error accumulation.
        #[arg(long)]
        freerun: bool,
        /// Insert variant A's six worked examples into the prefix before
        /// `Grid:`.
        #[arg(long)]
        fewshot: bool,
        /// Filename prefix for this run's pictures and summary.
        #[arg(long, default_value = "b")]
        tag: String,
        /// Run the forward through the f32 training model instead of the
        /// Q4 engine, applying this LoRA file. Slower and heavier; it is
        /// how a fine-tune is scored with the same pipeline as the base.
        #[arg(long)]
        lora: Option<PathBuf>,
        /// The same f32 forward with no adapter — the control that says how
        /// much of a fine-tuned row is the LoRA and how much is f32 weights.
        #[arg(long)]
        f32_forward: bool,
    },
    /// Run variant A (packed per-cell prompts) for N generations and write
    /// pictures + a text summary.
    PictureA {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "64")]
        size: usize,
        #[arg(long, default_value = "10")]
        generations: usize,
        #[arg(long, value_delimiter = ',', default_value = "glider,1,2")]
        seeds: Vec<String>,
        #[arg(long, default_value = "0.28")]
        density: f64,
        #[arg(long, default_value = "docs/pictures")]
        out: PathBuf,
        #[arg(long)]
        freerun: bool,
        /// Cells per chunk. The chunk's dense mask and attention scores are
        /// quadratic in `chunk_cells * tokens_per_cell`, so this is the one
        /// knob that decides whether a generation fits in memory.
        #[arg(long, default_value = "128")]
        chunk_cells: usize,
        /// Prepend six worked neighborhood->next examples to the prefix.
        #[arg(long)]
        fewshot: bool,
        /// Filename prefix for this attempt's pictures and summary.
        #[arg(long, default_value = "a")]
        tag: String,
        /// Run the f32 training model with this LoRA adapter applied,
        /// instead of the Q4 engine with no adapter. This is how
        /// `artifacts/lora-a-{rules,norules}-300.bin` get scored on real
        /// grids: the same forward the fine-tune was evaluated with
        /// (`train::run_a::evaluate_a`), not the base model.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// The adapter's prefix had no rule text (`lora-a-norules-*.bin`).
        /// Must match how the adapter file was trained or its LoRA weights
        /// are being applied under the wrong prompt.
        #[arg(long)]
        norules: bool,
    },
    /// Time one full forward pass at several sizes, to see how it scales.
    ///
    /// Variant B: the whole grid in one pass (prefill + sliced-head
    /// readback) at each square grid size. Variant A: one chunk's forward
    /// against the resident prefix, at each chunk size. `reps` timed
    /// repetitions after one warm-up; the table reports the median.
    BenchForward {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        /// Square grid edge lengths for variant B.
        #[arg(long, value_delimiter = ',', default_value = "16,32,45,64,128")]
        sizes: Vec<usize>,
        /// Cells per chunk for variant A.
        #[arg(long, value_delimiter = ',', default_value = "16,32,64,128")]
        chunks: Vec<usize>,
        #[arg(long, default_value = "3")]
        reps: usize,
        #[arg(long, default_value = "0.28")]
        density: f64,
        #[arg(long)]
        fewshot: bool,
        /// After each variant-B row, print a per-component breakdown of one
        /// extra forward (`LLM_PROFILE` in llm-wasm). Every component is
        /// bracketed by a GPU sync, so the rows do not sum to the timed
        /// wall clock above them — they say where the work is.
        #[arg(long)]
        profile: bool,
    },
    /// Fine-tune variant B with LoRA on true Life (CONCEPT.md §5).
    Train {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "32")]
        size: usize,
        #[arg(long, default_value = "200")]
        steps: usize,
        #[arg(long, default_value = "1e-4")]
        lr: f64,
        #[arg(long, default_value = "8")]
        rank: usize,
        #[arg(long, default_value = "16")]
        alpha: f32,
        /// Adapt gate/up as well as q/k/v/o.
        #[arg(long)]
        lora_mlp: bool,
        #[arg(long, default_value = "1")]
        seed: u64,
        #[arg(long, default_value = "25")]
        eval_every: usize,
        #[arg(long, default_value = "5")]
        eval_grids: usize,
        #[arg(long, default_value = "artifacts/lora-b.bin")]
        out: PathBuf,
        /// Research log for this run (machine, commit, command, tables).
        #[arg(long)]
        run_doc: Option<PathBuf>,
        /// Wall-clock budget in seconds — the GPU is shared.
        #[arg(long, default_value = "1500")]
        max_secs: f64,
        /// Load this pretrained LoRA file's own rank/alpha/params instead of
        /// starting from a fresh zero-initialized adapter — combined with
        /// `--steps 0`, scores an existing checkpoint (from this or another
        /// variant) through variant B's whole-grid forward with no training.
        #[arg(long)]
        adapter: Option<PathBuf>,
    },
    /// Fine-tune variant A with LoRA on the exhaustive 512-case lookup
    /// (CONCEPT.md §11).
    TrainA {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        /// `a-norules`: the prefix states only the answer format, no rule text.
        #[arg(long)]
        norules: bool,
        #[arg(long, default_value = "400")]
        steps: usize,
        #[arg(long, default_value = "64")]
        batch: usize,
        #[arg(long, default_value = "3e-5")]
        lr: f64,
        #[arg(long, default_value = "8")]
        rank: usize,
        #[arg(long, default_value = "16")]
        alpha: f32,
        #[arg(long)]
        lora_mlp: bool,
        #[arg(long, default_value = "1")]
        seed: u64,
        #[arg(long, default_value = "20")]
        eval_every: usize,
        /// Number of the 512 cases used for the before-training and
        /// periodic evals (the final eval always uses all 512).
        #[arg(long, default_value = "128")]
        eval_cases: usize,
        /// Number of real grids (of 3) used for the before-training and
        /// periodic evals (the final eval always uses all 3).
        #[arg(long, default_value = "1")]
        eval_grids: usize,
        #[arg(long, default_value = "artifacts/lora-a.bin")]
        out: PathBuf,
        #[arg(long)]
        run_doc: Option<PathBuf>,
        #[arg(long, default_value = "900")]
        max_secs: f64,
    },
    /// Train BERT of Life: a small transformer encoder from scratch on the
    /// 512 exhaustive neighbourhood cases (CONCEPT.md §11 backlog item).
    /// Sweeps several sizes plus the MLP baseline and the lookup floor in
    /// one run; writes a checkpoint per size under `artifacts/bert/` and one
    /// run doc with the whole table.
    TrainBert {
        #[arg(long, default_value = "artifacts/bert")]
        out: PathBuf,
        #[arg(long, default_value = "docs/runs/2026-09-20-bert.md")]
        run_doc: PathBuf,
        #[arg(long, default_value = "400")]
        steps: usize,
        #[arg(long, default_value = "1e-2")]
        lr: f64,
        #[arg(long, default_value = "1")]
        seed: u64,
    },
    /// s/generation for a BERT-of-Life checkpoint through the native wgpu
    /// backend (the 3080's own GPU forward, not the CPU-ndarray path).
    BenchBert {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value = "64")]
        d_model: usize,
        #[arg(long, default_value = "2")]
        n_layers: usize,
        #[arg(long, default_value = "2")]
        n_heads: usize,
        #[arg(long, value_delimiter = ',', default_value = "16,32,64")]
        sizes: Vec<usize>,
        #[arg(long, default_value = "5")]
        reps: usize,
    },
    /// ns/cell for a BERT-of-Life checkpoint on the CPU (burn-ndarray, not
    /// wgpu — safe to run on the laptop while the GPU belongs to another
    /// worker). Requires `--features cpu`.
    #[cfg(feature = "cpu")]
    BenchBertCpu {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value = "64")]
        d_model: usize,
        #[arg(long, default_value = "2")]
        n_layers: usize,
        #[arg(long, default_value = "2")]
        n_heads: usize,
        #[arg(long, value_delimiter = ',', default_value = "16,32,64")]
        sizes: Vec<usize>,
        #[arg(long, default_value = "5")]
        reps: usize,
    },
    /// Train the vector-space variants (CONCEPT.md §12): (i) attention over
    /// the 9 neighbourhood numbers (plus the MLP baseline, reused from
    /// `bert::model::MlpOfLife`) and (ii) the whole-grid stencil-masked
    /// transformer. Writes checkpoints under `--out` and one run doc with
    /// both tables.
    TrainVec {
        #[arg(long, default_value = "artifacts/vector")]
        out: PathBuf,
        #[arg(long, default_value = "docs/runs/2026-09-20-vector.md")]
        run_doc: PathBuf,
        #[arg(long, default_value = "600")]
        steps: usize,
        #[arg(long, default_value = "1e-2")]
        lr: f64,
        #[arg(long, default_value = "1200")]
        stencil_steps: usize,
        #[arg(long, default_value = "3e-3")]
        stencil_lr: f64,
        #[arg(long, default_value = "1")]
        seed: u64,
    },
    /// Rerun (i) only, with a corrected learning-rate schedule (cosine decay
    /// instead of the fixed lr=1e-2/600-steps that collapsed the first
    /// sweep — the same failure mode the BERT d=32/64 runs hit, fixed there
    /// by lr 2e-3/800 steps). MLP h ∈ {8,16,32}, a 2-layer MLP body
    /// (9->32->32->2), and Attn d ∈ {8,16,32}. Appends a corrected section
    /// to `--run-doc` rather than overwriting the first sweep.
    TrainVecILrFix {
        #[arg(long, default_value = "artifacts/vector")]
        out: PathBuf,
        #[arg(long, default_value = "docs/runs/2026-09-20-vector.md")]
        run_doc: PathBuf,
        #[arg(long, default_value = "2000")]
        steps: usize,
        #[arg(long, default_value = "2e-3")]
        lr: f64,
    },
    /// s/generation for a vector-space checkpoint through the native wgpu
    /// backend, at 16x16/32x32/64x64 (stencil) — the same convention as
    /// `BenchBert`.
    BenchVec {
        #[arg(long)]
        checkpoint: PathBuf,
        /// `attn` (9-number attention model, always 16x16 per-cell) or
        /// `stencil` (whole-grid model, generalises across grid size).
        #[arg(long)]
        kind: String,
        #[arg(long, default_value = "16")]
        d_model: usize,
        #[arg(long, default_value = "1")]
        n_layers: usize,
        #[arg(long, default_value = "1")]
        n_heads: usize,
        #[arg(long, value_delimiter = ',', default_value = "16,32,64")]
        sizes: Vec<usize>,
        #[arg(long, default_value = "5")]
        reps: usize,
    },
    /// s/generation for every rung of `docs/LADDER.md`'s compute ladder, on
    /// this machine, at the same board (density/seed) for every rung. "CPU
    /// loop" and "512-entry lookup" are the plain-Rust equivalents of the
    /// tab's live JS measurement; "LLM per pixel"/"LLM batched (adapter)"
    /// are the same forward (`TrainRunnerA`, `a-norules`) narrated twice per
    /// LADDER.md's own convention, measured only at `--llm-sizes` (each
    /// cell is a real forward — never projected here); BERT of Life, "9
    /// numbers -> centre" and stencil are cheap whole-batch/whole-grid
    /// native forwards, measured at every `--sizes` entry. Prints one JSON
    /// object and a human table; `--out-json`/`--out-md` also write them to
    /// files.
    BenchLadder {
        #[arg(long, value_delimiter = ',', default_value = "16,32,64")]
        sizes: Vec<usize>,
        /// Sizes to actually run the two LLM rungs at (one real forward per
        /// size, no projection). Sizes requested via `--sizes` but not here
        /// are reported as `null` for those two rows.
        #[arg(long, value_delimiter = ',', default_value = "16")]
        llm_sizes: Vec<usize>,
        #[arg(long, default_value = "models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf")]
        gguf: PathBuf,
        #[arg(long, default_value = "models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "artifacts/lora-a-norules-300.bin")]
        adapter: PathBuf,
        #[arg(long, default_value = "artifacts/bert/bert-d16-L1.bin")]
        bert_checkpoint: PathBuf,
        #[arg(long, default_value = "artifacts/vector/mlp2-32-lrfix.bin")]
        mlp2_checkpoint: PathBuf,
        #[arg(long, default_value = "artifacts/vector/stencil-d16-L1.bin")]
        stencil_checkpoint: PathBuf,
        /// The stencil forward is O(n) with the current `stencil_neighbors`
        /// gather table, but is skipped above this size as a safety cap
        /// while the crate's stencil implementation is still being worked
        /// on elsewhere.
        #[arg(long, default_value = "64")]
        stencil_max_size: usize,
        #[arg(long, default_value = "0.28")]
        density: f64,
        #[arg(long, default_value = "1")]
        seed: u64,
        #[arg(long, default_value = "5")]
        reps: usize,
        #[arg(long)]
        out_json: Option<PathBuf>,
        #[arg(long)]
        out_md: Option<PathBuf>,
    },
    /// Re-score an already-written `<tag>-<seed>-gen1-{palive,true}.pgm`
    /// pair at the Otsu and z-score (k=2) thresholds, label-free, no rerun.
    Rescore {
        #[arg(long, default_value = "docs/pictures")]
        dir: PathBuf,
        #[arg(long)]
        tag: String,
        #[arg(long)]
        seed: String,
    },
}

/// Everything variant B needs that does not change between generations.
struct Runner {
    model: LlmModel,
    prefix: Vec<u32>,
    dead: u32,
    alive: u32,
    head: burn::tensor::Tensor<burn::backend::Wgpu, 2>,
    fewshot: bool,
}

impl Runner {
    fn new(
        gguf: &PathBuf,
        tokenizer: &PathBuf,
        rule: &Rule,
        fewshot: bool,
        device: &WgpuDevice,
    ) -> Result<Self> {
        let tok = Tokenizer::from_json(&std::fs::read(tokenizer).context("read tokenizer.json")?)?;
        let prefix_text = if fewshot {
            fewshot_rules_prefix(rule)
        } else {
            rules_prefix(rule)
        };
        let prefix = tok.encode(&prefix_text, false)?;

        // The two answer tokens must each be exactly one token, or "read the
        // logits at this position" means something else than intended.
        let dead = single_token(&tok, "0")?;
        let alive = single_token(&tok, "1")?;

        let reader = std::io::BufReader::new(std::fs::File::open(gguf).context("open gguf")?);
        let mut loader = Q4ModelLoader::new(reader)?;
        let parts = loader.load_deferred(device)?;
        drop(loader);
        let model = parts.finalize(device)?;

        let cfg = model.config();
        println!(
            "model: {} layers, hidden {}, {} heads / {} kv, intermediate {}, vocab {}, ctx {}",
            cfg.num_layers,
            cfg.hidden_size,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.intermediate_size,
            cfg.vocab_size,
            cfg.max_seq_len
        );
        println!("prefix: {} tokens, answer tokens: '0'={dead} '1'={alive}", prefix.len());

        let head = model.head_slice(&[dead, alive])?;
        Ok(Runner {
            model,
            prefix,
            dead,
            alive,
            head,
            fewshot,
        })
    }

    /// One generation: pack the grid, one forward pass, read p(alive) at
    /// every cell.
    fn step(&self, grid: &Grid) -> Result<(Vec<f32>, f64)> {
        let packed = pack_sparse(grid, &self.prefix, self.dead, self.alive);
        let t = packed.tokens.len();
        let spec = ForwardSpec::default()
            .with_positions(packed.positions.clone())
            .with_sparse(SparseMask::new(
                &packed.prefix_len,
                &packed.n_keys,
                &packed.keys,
                MAX_STENCIL_KEYS,
                t,
                self.model.device(),
            ));

        let start = Instant::now();
        let mut cache = self.model.new_cache(t);
        let hidden = self.model.forward_hidden_spec(&packed.tokens, &mut cache, &spec)?;
        let dev = self.model.device().clone();
        let logits = llm_wasm::profile::scope("lm_head_sliced", &dev, || {
            self.model.lm_head_sliced(hidden, &self.head)
        });
        let logits = llm_wasm::profile::scope("logits_readback", &dev, || {
            llm_wasm::model::logits_to_vec(logits)
        })?;
        let secs = start.elapsed().as_secs_f64();

        let n = grid.width() * grid.height();
        Ok((p_alive(&logits, packed.grid_start, n), secs))
    }
}

/// Variant B through the f32 training forward (`TrainModel`) instead of the
/// Q4 engine, optionally with a LoRA applied.
///
/// This is how a fine-tune is scored with the same pipeline as the base
/// model: merging a LoRA back into Q4_0 weights would have to re-quantize
/// them, and a rank-8 delta after a short run is comfortably smaller than
/// Q4_0's step size — the merge would erase most of what was learned. So the
/// eval runs the adapter where it was trained, in f32, and `--f32-forward`
/// with no `--lora` is the control that separates "the LoRA did something"
/// from "f32 weights did something".
struct TrainRunner {
    model: TrainModel<burn::backend::Wgpu>,
    prefix: Vec<u32>,
    dead: u32,
    alive: u32,
    lora: Option<PathBuf>,
}

impl TrainRunner {
    fn new(
        gguf: &Path,
        tokenizer: &Path,
        rule: &Rule,
        lora: Option<&Path>,
        device: &WgpuDevice,
    ) -> Result<Self> {
        let p = llm_life::train::run::prompt(tokenizer, rule)?;
        // The file carries the `LoraSpec` it was trained with, so the
        // adapter set is rebuilt exactly, not guessed.
        let loaded = lora
            .map(|path| lora_io::load::<burn::backend::Wgpu>(path, device))
            .transpose()?;
        let spec: Option<LoraSpec> = loaded.as_ref().map(|(s, _)| *s);
        let mut model: TrainModel<burn::backend::Wgpu> =
            load_train_model(gguf, &[p.dead, p.alive], spec, device)?;
        if let Some((_, params)) = loaded {
            anyhow::ensure!(
                params.len() == model.lora_params().len(),
                "LoRA file has {} matrices, this model wants {}",
                params.len(),
                model.lora_params().len()
            );
            model.set_lora_params(params);
        }
        println!(
            "f32 forward: {} layers, hidden {}, lora {}",
            model.config.num_layers,
            model.config.hidden_size,
            lora.map(|p| p.display().to_string()).unwrap_or_else(|| "none".into())
        );
        Ok(TrainRunner {
            model,
            prefix: p.prefix,
            dead: p.dead,
            alive: p.alive,
            lora: lora.map(|p| p.to_path_buf()),
        })
    }
}

impl Stepper for TrainRunner {
    fn step(&self, grid: &Grid) -> Result<(Vec<f32>, f64)> {
        use burn::tensor::{Bool, TensorData};
        let packed = llm_life::variant_b::pack(grid, &self.prefix, self.dead, self.alive);
        let t = packed.tokens.len();
        let masked_out: Vec<bool> = packed.allowed.iter().map(|&a| !a).collect();
        let mask: burn::tensor::Tensor<burn::backend::Wgpu, 2, Bool> =
            burn::tensor::Tensor::from_data(TensorData::new(masked_out, [t, t]), self.model.device());
        let start = Instant::now();
        let logits = self.model.forward(&packed.tokens, &packed.positions, &mask)?;
        let v = logits.into_data().into_vec::<f32>().unwrap();
        let secs = start.elapsed().as_secs_f64();
        let n = grid.width() * grid.height();
        Ok((p_alive(&v, packed.grid_start, n), secs))
    }

    fn header(&self) -> String {
        format!(
            "positions: bag (all grid tokens share one position id)\n\
             mask: prefix (causal) + self + 8 neighbors (dense)\n\
             head: sliced to the two answer tokens\n\
             forward: f32 training model (pure Burn ops), lora {}\n\
             prefix: {} tokens (rules only)\n",
            self.lora.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none".into()),
            self.prefix.len(),
        )
    }
}

/// Variant A's f32-forward twin of `TrainRunner`: packed per-cell prompts
/// (`variant_a::pack_chunk`), forwarded from scratch each chunk against
/// `TrainModel<Wgpu>` with an optional LoRA adapter applied — the exact
/// forward `train::run_a::evaluate_a` uses to score a fine-tune, run here on
/// real 16x16/32x32 grids instead of the training eval's own grid set.
struct TrainRunnerA {
    model: TrainModel<burn::backend::Wgpu>,
    prefix: Vec<u32>,
    tok: Tokenizer,
    chunk_cells: usize,
    lora: Option<PathBuf>,
    norules: bool,
}

impl TrainRunnerA {
    fn new(
        gguf: &Path,
        tokenizer: &Path,
        rule: &Rule,
        lora: Option<&Path>,
        norules: bool,
        chunk_cells: usize,
        device: &WgpuDevice,
    ) -> Result<Self> {
        let tok = Tokenizer::from_json(&std::fs::read(tokenizer).context("read tokenizer.json")?)?;
        let p = run_a::prompt_a(tokenizer, rule, norules)?;
        let loaded = lora
            .map(|path| lora_io::load::<burn::backend::Wgpu>(path, device))
            .transpose()?;
        let spec: Option<LoraSpec> = loaded.as_ref().map(|(s, _)| *s);
        let mut model: TrainModel<burn::backend::Wgpu> =
            load_train_model(gguf, &[p.dead, p.alive], spec, device)?;
        if let Some((_, params)) = loaded {
            anyhow::ensure!(
                params.len() == model.lora_params().len(),
                "LoRA file has {} matrices, this model wants {}",
                params.len(),
                model.lora_params().len()
            );
            model.set_lora_params(params);
        }
        println!(
            "f32 forward (variant A): {} layers, hidden {}, adapter {}, prefix {}",
            model.config.num_layers,
            model.config.hidden_size,
            lora.map(|p| p.display().to_string()).unwrap_or_else(|| "none".into()),
            if norules { "norules" } else { "rules" },
        );
        Ok(TrainRunnerA {
            model,
            prefix: p.prefix,
            tok,
            chunk_cells,
            lora: lora.map(|p| p.to_path_buf()),
            norules,
        })
    }
}

impl Stepper for TrainRunnerA {
    fn step(&self, grid: &Grid) -> Result<(Vec<f32>, f64)> {
        let n = grid.cells().len();
        let mut p = vec![0.0f32; n];
        let mut secs = 0.0;
        for first in (0..n).step_by(self.chunk_cells) {
            let cells: Vec<usize> = (first..(first + self.chunk_cells).min(n)).collect();
            let chunk = variant_a::pack_chunk(
                &cells,
                |c| {
                    let nb: Vec<u8> = grid
                        .neighbor_indices(c)
                        .iter()
                        .map(|&j| grid.cells()[j])
                        .collect();
                    let arr: [u8; 8] = nb.try_into().unwrap();
                    run_a::tokenize_case(&self.tok, &arr, grid.cells()[c]).unwrap()
                },
                self.prefix.len(),
            );
            let (tokens, positions, mask) =
                run_a::full_sequence::<burn::backend::Wgpu>(&self.prefix, &chunk, self.model.device());
            let start = Instant::now();
            let logits = self.model.forward(&tokens, &positions, &mask)?;
            let data = logits.into_data().into_vec::<f32>().unwrap();
            secs += start.elapsed().as_secs_f64();
            for (cell, v) in variant_a::p_alive_chunk(&data[self.prefix.len() * 2..], &chunk) {
                p[cell] = v;
            }
        }
        Ok((p, secs))
    }

    fn header(&self) -> String {
        format!(
            "positions: restart at the prefix for every cell\n\
             mask: block-diagonal — prefix (causal) + the cell's own prompt\n\
             head: sliced to the two answer tokens\n\
             forward: f32 training model (pure Burn ops), no persistent KV cache — full prefix+chunk forward every chunk\n\
             adapter: {}, prefix: {} tokens ({})\n",
            self.lora.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none".into()),
            self.prefix.len(),
            if self.norules { "norules" } else { "rules" },
        )
    }
}

/// One run of either variant, from the driver's point of view: turn a grid
/// into a p(alive) per cell, and say how long the GPU part took.
trait Stepper {
    fn step(&self, grid: &Grid) -> Result<(Vec<f32>, f64)>;
    /// Extra lines for the summary header, describing this run's setup.
    fn header(&self) -> String;
}

impl Stepper for Runner {
    fn step(&self, grid: &Grid) -> Result<(Vec<f32>, f64)> {
        Runner::step(self, grid)
    }

    fn header(&self) -> String {
        format!(
            "positions: bag (all grid tokens share one position id)\n\
             mask: prefix (causal) + self + 8 neighbors\n\
             head: sliced to the two answer tokens\n\
             prefix: {} tokens ({})\n",
            self.prefix.len(),
            if self.fewshot {
                "rules + variant A's 6 worked examples"
            } else {
                "rules only"
            },
        )
    }
}

/// Everything variant A needs that does not change between generations.
///
/// The rules prefix is prefilled into `cache` once, here, and stays resident:
/// every chunk is forwarded against it and then rewinds the cache to
/// `prefix_len` (`KvCache::snapshot`/`restore`), so the prefix is paid once
/// per process, not once per chunk.
struct RunnerA {
    model: LlmModel,
    cache: RefCell<KvCache>,
    prefix_len: usize,
    /// Tokenized per-cell prompt for each of the 512 (neighborhood, self)
    /// cases — the template has only that many distinct strings, so the
    /// tokenizer runs 512 times per process instead of 4096 times per
    /// generation.
    prompts: Vec<Vec<u32>>,
    chunk_cells: usize,
    head: burn::tensor::Tensor<burn::backend::Wgpu, 2>,
    fewshot: bool,
}

/// Index into `RunnerA::prompts`: the 8 neighbor bits, then self.
fn case_index(neighbors: &[u8], self_state: u8) -> usize {
    let mut k = 0usize;
    for (b, &n) in neighbors.iter().enumerate() {
        if n != 0 {
            k |= 1 << b;
        }
    }
    k | ((self_state != 0) as usize) << 8
}

impl RunnerA {
    fn new(
        gguf: &PathBuf,
        tokenizer: &PathBuf,
        rule: &Rule,
        chunk_cells: usize,
        fewshot: bool,
        device: &WgpuDevice,
    ) -> Result<Self> {
        let tok = Tokenizer::from_json(&std::fs::read(tokenizer).context("read tokenizer.json")?)?;
        let mut prefix_text = variant_a::rules_prefix(rule);
        if fewshot {
            prefix_text.push_str(&variant_a::fewshot_examples());
        }
        let prefix = tok.encode(&prefix_text, false)?;


        let mut prompts = vec![Vec::new(); 512];
        for (k, slot) in prompts.iter_mut().enumerate() {
            let nb: Vec<u8> = (0..8).map(|b| ((k >> b) & 1) as u8).collect();
            let text = format!("\n{}", variant_a::cell_prompt(&nb, ((k >> 8) & 1) as u8));
            *slot = tok.encode(&text, false)?;
        }
        let max_cell_tokens = prompts.iter().map(|p| p.len()).max().unwrap();
        // The prompt ends on its trailing space, so the read-out position
        // takes the bare digit tokens.
        let dead = single_token(&tok, "0")?;
        let alive = single_token(&tok, "1")?;

        let reader = std::io::BufReader::new(std::fs::File::open(gguf).context("open gguf")?);
        let mut loader = Q4ModelLoader::new(reader)?;
        let parts = loader.load_deferred(device)?;
        drop(loader);
        let model = parts.finalize(device)?;

        let cfg = model.config();
        println!(
            "model: {} layers, hidden {}, {} heads / {} kv, intermediate {}, vocab {}, ctx {}",
            cfg.num_layers,
            cfg.hidden_size,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.intermediate_size,
            cfg.vocab_size,
            cfg.max_seq_len
        );
        println!(
            "prefix: {} tokens ({}), per-cell prompt: {} tokens, chunk: {chunk_cells} cells = {} tokens",
            prefix.len(),
            if fewshot { "few-shot" } else { "rules only" },
            max_cell_tokens,
            chunk_cells * max_cell_tokens
        );
        println!("answer tokens: '0'={dead} '1'={alive}");

        let head = model.head_slice(&[dead, alive])?;
        let mut cache = model.new_cache(prefix.len() + chunk_cells * max_cell_tokens);
        model.forward_hidden(&prefix, &mut cache)?;

        Ok(RunnerA {
            model,
            cache: RefCell::new(cache),
            prefix_len: prefix.len(),
            prompts,
            chunk_cells,
            head,
            fewshot,
        })
    }

    fn chunks(&self, n_cells: usize) -> usize {
        n_cells.div_ceil(self.chunk_cells)
    }

    fn step(&self, grid: &Grid) -> Result<(Vec<f32>, f64)> {
        let n = grid.cells().len();
        let mut p = vec![0.0f32; n];
        let mut secs = 0.0;
        let mut cache = self.cache.borrow_mut();

        for first in (0..n).step_by(self.chunk_cells) {
            let cells: Vec<usize> = (first..(first + self.chunk_cells).min(n)).collect();
            let chunk = variant_a::pack_chunk(
                &cells,
                |c| {
                    let nb: Vec<u8> = grid
                        .neighbor_indices(c)
                        .iter()
                        .map(|&j| grid.cells()[j])
                        .collect();
                    self.prompts[case_index(&nb, grid.cells()[c])].clone()
                },
                self.prefix_len,
            );
            let t = chunk.len();
            let spec = ForwardSpec::default()
                .with_positions(chunk.positions.clone())
                .with_allowed(&chunk.allowed, t, self.prefix_len + t, self.model.device());

            let start = Instant::now();
            let resident = cache.snapshot();
            let hidden = self
                .model
                .forward_hidden_spec(&chunk.tokens, &mut cache, &spec)?;
            let logits = self.model.lm_head_sliced(hidden, &self.head);
            let logits = llm_wasm::model::logits_to_vec(logits)?;
            cache.restore(resident);
            secs += start.elapsed().as_secs_f64();

            for (cell, v) in variant_a::p_alive_chunk(&logits, &chunk) {
                p[cell] = v;
            }
        }
        Ok((p, secs))
    }

    /// One chunk's forward against the resident prefix, timed. Returns
    /// `(tokens, seconds)` where `tokens` counts the resident prefix plus
    /// the chunk's own tokens — the KV length the pass attends over.
    fn bench_chunk(&self, grid: &Grid, chunk_cells: usize) -> Result<(usize, f64)> {
        let cells: Vec<usize> = (0..chunk_cells).collect();
        let chunk = variant_a::pack_chunk(
            &cells,
            |c| {
                let nb: Vec<u8> = grid
                    .neighbor_indices(c)
                    .iter()
                    .map(|&j| grid.cells()[j])
                    .collect();
                self.prompts[case_index(&nb, grid.cells()[c])].clone()
            },
            self.prefix_len,
        );
        let t = chunk.len();
        let spec = ForwardSpec::default()
            .with_positions(chunk.positions.clone())
            .with_allowed(&chunk.allowed, t, self.prefix_len + t, self.model.device());

        let mut cache = self.cache.borrow_mut();
        let start = Instant::now();
        let resident = cache.snapshot();
        let hidden = self
            .model
            .forward_hidden_spec(&chunk.tokens, &mut cache, &spec)?;
        let logits = self.model.lm_head_sliced(hidden, &self.head);
        let _ = llm_wasm::model::logits_to_vec(logits)?;
        cache.restore(resident);
        Ok((self.prefix_len + t, start.elapsed().as_secs_f64()))
    }
}

impl Stepper for RunnerA {
    fn step(&self, grid: &Grid) -> Result<(Vec<f32>, f64)> {
        RunnerA::step(self, grid)
    }

    fn header(&self) -> String {
        format!(
            "positions: restart at the prefix for every cell\n\
             mask: block-diagonal — resident prefix + the cell's own prompt\n\
             head: sliced to the two answer tokens (the prompt ends on a space)\n\
             prefix: {} tokens ({}), resident in the KV cache\n\
             chunk: {} cells\n",
            self.prefix_len,
            if self.fewshot { "rules + 6 worked examples" } else { "rules only" },
            self.chunk_cells,
        )
    }
}

/// The reporting loop both variants share: N generations per seed, a markdown
/// table of per-generation scores, and four PGMs at generation 1.
#[allow(clippy::too_many_arguments)]
fn run_pictures(
    stepper: &dyn Stepper,
    title: &str,
    tag: &str,
    gguf: &Path,
    rule: &Rule,
    size: usize,
    generations: usize,
    seeds: &[String],
    density: f64,
    out: &Path,
    freerun: bool,
) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut summary = format!(
        "# {title}\n\n\
         model: {}\nrule: {}\ngrid: {size}x{size} (torus)\nmode: {}\n{}\
         Scored at two thresholds: p(alive) >= 0.5, and the grid median of\n\
         p(alive) (equivalently, the logit difference `1`-`0` calibrated to the\n\
         grid — the median hands the model the live fraction, so its live recall\n\
         is an upper bound, not an accuracy claim).\n\n\
         Timings provisional: the Metal GPU is shared with another repo's training job.\n\n",
        gguf.display(),
        rule.to_rulestring(),
        if freerun {
            "free-running (model eats its own output)"
        } else {
            "teacher-forced (each generation starts from true Life)"
        },
        stepper.header(),
    );

    for seed_name in seeds {
        println!("\n=== seed {seed_name} ===");
        summary.push_str(&format!(
            "## seed {seed_name}\n\n\
             | gen | accuracy | IoU | alive recall | dead recall | alive precision | TP | FP | FN | TN | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |\n\
             |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        ));
        let mut input = seed_grid(seed_name, size, density)?;
        for gen in 1..=generations {
            let truth = input.step(rule);
            let (p, secs) = stepper.step(&input)?;
            let model_grid = argmax_grid(&p, size, size);
            let med_grid = median_threshold_grid(&p, size, size);
            let s = score(&truth, &model_grid, &p, gen);
            let m = score(&truth, &med_grid, &p, gen);
            println!(
                "gen {gen:2}: acc={:.4} iou={:.4} alive_recall={:.4} dead_recall={:.4} alive_precision={:.4} tp={} fp={} fn={} tn={} | median acc={:.4} live_recall={:.4} | gap={:+.4} true_live={} model_live={} {:.2}s",
                s.accuracy, s.iou, s.alive_recall, s.dead_recall, s.alive_precision, s.tp, s.fp, s.fn_, s.tn,
                m.accuracy, m.live_recall, s.confidence_gap, s.true_live, s.model_live, secs
            );
            summary.push_str(&format!(
                "| {gen} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} | {} | {} | {} | {:.4} | {:.4} | {:+.4} | {} | {} | {:.2} |\n",
                s.accuracy, s.iou, s.alive_recall, s.dead_recall, s.alive_precision, s.tp, s.fp, s.fn_, s.tn,
                m.accuracy, m.live_recall, s.confidence_gap, s.true_live, s.model_live, secs
            ));

            if gen == 1 {
                let name = format!("{tag}-{seed_name}-gen{gen}");
                write_pgm(&out.join(format!("{name}-palive.pgm")), &p, size, size, 6)?;
                write_binary_pgm(&out.join(format!("{name}-argmax.pgm")), model_grid.cells(), size, size, 6)?;
                write_binary_pgm(&out.join(format!("{name}-true.pgm")), truth.cells(), size, size, 6)?;
                write_binary_pgm(&out.join(format!("{name}-diff.pgm")), &truth.diff(&model_grid), size, size, 6)?;
            }

            input = if freerun { model_grid } else { truth };
        }
        summary.push('\n');
    }

    let path = out.join(format!("{tag}-picture.md"));
    std::fs::write(&path, summary)?;
    println!("\nwrote {}", path.display());
    Ok(())
}

/// Print llm-wasm's accumulated per-component profile as a markdown table.
fn print_profile(label: &str, t: usize) {
    let mut rows = llm_wasm::profile::take();
    let total: f64 = rows.iter().map(|(_, s, _)| s).sum();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nProfile, {label} (T = {t}), one forward, GPU-synced per component:\n");
    println!("| component | calls | total s | % | ms/call |");
    println!("|---|---|---|---|---|");
    for (label, secs, calls) in &rows {
        println!(
            "| {label} | {calls} | {secs:.3} | {:.1} | {:.2} |",
            100.0 * secs / total,
            1000.0 * secs / *calls as f64
        );
    }
    println!("| **sum** | | **{total:.3}** | | |\n");
}

/// Median of a small sample — the reported statistic for every bench row.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time one forward pass at several sizes, for both variants, and print the
/// two markdown tables that go into `docs/runs/`.
///
/// The two models are loaded in sequence, not together: each is ~430MB of Q4
/// weights on the GPU and there is no reason to hold both.
#[allow(clippy::too_many_arguments)]
fn bench_forward(
    gguf: &PathBuf,
    tokenizer: &PathBuf,
    rule: &Rule,
    sizes: &[usize],
    chunks: &[usize],
    reps: usize,
    density: f64,
    fewshot: bool,
    profile: bool,
    device: &WgpuDevice,
) -> Result<()> {
    if profile {
        std::env::set_var("LLM_PROFILE", "1");
        anyhow::ensure!(
            llm_wasm::profile::enabled(),
            "--profile set but llm-wasm's profiler is off"
        );
    }
    {
        let runner = Runner::new(gguf, tokenizer, rule, fewshot, device)?;
        println!("\n## Variant B — one full forward (prefill + sliced-head readback)\n");
        println!("| grid | cells | tokens T | median s | s/token |");
        println!("|---|---|---|---|---|");
        for &size in sizes {
            let grid = Grid::random(size, size, 1, density);
            runner.step(&grid)?;
            let mut ts = Vec::with_capacity(reps);
            for _ in 0..reps {
                ts.push(runner.step(&grid)?.1);
            }
            let m = median(ts);
            let t = runner.prefix.len() + size * size;
            println!(
                "| {size}x{size} | {} | {t} | {m:.3} | {:.6} |",
                size * size,
                m / t as f64
            );
            if profile {
                llm_wasm::profile::reset();
                runner.step(&grid)?;
                print_profile(&format!("{size}x{size}"), t);
            }
        }
    }

    {
        let max_chunk = *chunks.iter().max().unwrap();
        let runner = RunnerA::new(gguf, tokenizer, rule, max_chunk, fewshot, device)?;
        // Any grid at least `max_chunk` cells wide will do: the chunk is the
        // unit being timed, not the grid.
        let grid = Grid::random(64, 64, 1, density);
        println!("\n## Variant A — one chunk's forward against the resident prefix\n");
        println!("| chunk cells | tokens T | median s | s/token |");
        println!("|---|---|---|---|");
        for &c in chunks {
            runner.bench_chunk(&grid, c)?;
            let mut ts = Vec::with_capacity(reps);
            let mut t = 0;
            for _ in 0..reps {
                let (tok, secs) = runner.bench_chunk(&grid, c)?;
                t = tok;
                ts.push(secs);
            }
            let m = median(ts);
            println!("| {c} | {t} | {m:.3} | {:.6} |", m / t as f64);
        }
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

fn single_token(tok: &Tokenizer, s: &str) -> Result<u32> {
    let ids = tok.encode(s, false)?;
    anyhow::ensure!(ids.len() == 1, "'{s}' is not a single token: {ids:?}");
    Ok(ids[0])
}

fn seed_grid(name: &str, size: usize, density: f64) -> Result<Grid> {
    if name == "glider" {
        let mut g = Grid::new(size, size);
        g.place_glider(size / 2, size / 2);
        return Ok(g);
    }
    let seed: u64 = name.parse().context("seed must be a number or 'glider'")?;
    Ok(Grid::random(size, size, seed, density))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let rule = Rule::parse(&cli.rule).with_context(|| format!("invalid --rule '{}'", cli.rule))?;
    let device = WgpuDevice::default();
    match cli.command {
        Command::Picture {
            gguf,
            tokenizer,
            size,
            generations,
            seeds,
            density,
            out,
            freerun,
            fewshot,
            tag,
            lora,
            f32_forward,
        } => {
            let title = "Variant B — stencil mask, native";
            if lora.is_some() || f32_forward {
                anyhow::ensure!(!fewshot, "--fewshot is not wired into the f32 forward");
                let runner = TrainRunner::new(&gguf, &tokenizer, &rule, lora.as_deref(), &device)?;
                run_pictures(
                    &runner, title, &tag, &gguf, &rule, size, generations, &seeds, density, &out,
                    freerun,
                )
            } else {
                let runner = Runner::new(&gguf, &tokenizer, &rule, fewshot, &device)?;
                run_pictures(
                    &runner, title, &tag, &gguf, &rule, size, generations, &seeds, density, &out,
                    freerun,
                )
            }
        }
        Command::PictureA {
            gguf,
            tokenizer,
            size,
            generations,
            seeds,
            density,
            out,
            freerun,
            chunk_cells,
            fewshot,
            tag,
            adapter,
            norules,
        } => {
            let title = "Variant A — packed per-cell prompts, native";
            if adapter.is_some() {
                anyhow::ensure!(!fewshot, "--fewshot is not wired into the f32 forward");
                let runner =
                    TrainRunnerA::new(&gguf, &tokenizer, &rule, adapter.as_deref(), norules, chunk_cells, &device)?;
                run_pictures(
                    &runner, title, &tag, &gguf, &rule, size, generations, &seeds, density, &out,
                    freerun,
                )
            } else {
                let runner = RunnerA::new(&gguf, &tokenizer, &rule, chunk_cells, fewshot, &device)?;
                println!("{} chunks per generation", runner.chunks(size * size));
                run_pictures(
                    &runner, title, &tag, &gguf, &rule, size, generations, &seeds, density, &out,
                    freerun,
                )
            }
        }
        Command::BenchForward {
            gguf,
            tokenizer,
            sizes,
            chunks,
            reps,
            density,
            fewshot,
            profile,
        } => bench_forward(
            &gguf, &tokenizer, &rule, &sizes, &chunks, reps, density, fewshot, profile, &device,
        ),
        Command::Train {
            gguf,
            tokenizer,
            size,
            steps,
            lr,
            rank,
            alpha,
            lora_mlp,
            seed,
            eval_every,
            eval_grids,
            out,
            run_doc,
            max_secs,
            adapter,
        } => llm_life::train::run::run(TrainArgs {
            gguf,
            tokenizer,
            size,
            steps,
            lr,
            lora: LoraSpec {
                rank,
                alpha,
                mlp: lora_mlp,
            },
            seed,
            eval_every,
            eval_grids,
            out,
            run_doc,
            adapter,
            max_secs,
        }),
        Command::TrainA {
            gguf,
            tokenizer,
            norules,
            steps,
            batch,
            lr,
            rank,
            alpha,
            lora_mlp,
            seed,
            eval_every,
            eval_cases,
            eval_grids,
            out,
            run_doc,
            max_secs,
        } => llm_life::train::run_a::run(RunAArgs {
            gguf,
            tokenizer,
            rule,
            norules,
            steps,
            batch,
            lr,
            lora: LoraSpec {
                rank,
                alpha,
                mlp: lora_mlp,
            },
            seed,
            eval_every,
            eval_cases,
            eval_grids,
            out,
            run_doc,
            max_secs,
        }),
        Command::TrainBert {
            out,
            run_doc,
            steps,
            lr,
            seed,
        } => train_bert_sweep(&out, &run_doc, steps, lr, seed),
        Command::BenchBert {
            checkpoint,
            d_model,
            n_layers,
            n_heads,
            sizes,
            reps,
        } => bench_bert_native(&checkpoint, d_model, n_layers, n_heads, &sizes, reps, &device),
        #[cfg(feature = "cpu")]
        Command::BenchBertCpu {
            checkpoint,
            d_model,
            n_layers,
            n_heads,
            sizes,
            reps,
        } => bench_bert_cpu(&checkpoint, d_model, n_layers, n_heads, &sizes, reps),
        Command::TrainVec {
            out,
            run_doc,
            steps,
            lr,
            stencil_steps,
            stencil_lr,
            seed,
        } => train_vec_sweep(&out, &run_doc, steps, lr, stencil_steps, stencil_lr, seed),
        Command::TrainVecILrFix { out, run_doc, steps, lr } => train_vec_i_lr_fix(&out, &run_doc, steps, lr),
        Command::BenchVec {
            checkpoint,
            kind,
            d_model,
            n_layers,
            n_heads,
            sizes,
            reps,
        } => bench_vec_native(&checkpoint, &kind, d_model, n_layers, n_heads, &sizes, reps, &device),
        Command::BenchLadder {
            sizes,
            llm_sizes,
            gguf,
            tokenizer,
            adapter,
            bert_checkpoint,
            mlp2_checkpoint,
            stencil_checkpoint,
            stencil_max_size,
            density,
            seed,
            reps,
            out_json,
            out_md,
        } => bench_ladder(
            &sizes,
            &llm_sizes,
            &gguf,
            &tokenizer,
            &adapter,
            &bert_checkpoint,
            &mlp2_checkpoint,
            &stencil_checkpoint,
            stencil_max_size,
            density,
            seed,
            reps,
            &rule,
            out_json.as_deref(),
            out_md.as_deref(),
            &device,
        ),
        Command::Rescore { dir, tag, seed } => rescore(&dir, &tag, &seed),
    }
}

fn bench_bert_native(
    checkpoint: &Path,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    sizes: &[usize],
    reps: usize,
    device: &WgpuDevice,
) -> Result<()> {
    use burn::backend::Wgpu;
    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::{Int, Tensor, TensorData};
    use llm_life::bert::data::grid_cases;
    use llm_life::bert::model::BertConfig;

    let cfg = BertConfig::small(d_model, n_layers, n_heads);
    let model: llm_life::bert::model::BertOfLife<Wgpu> = cfg.init(device);
    let bytes = std::fs::read(checkpoint).context("read checkpoint")?;
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(bytes, device)?;
    let model = model.load_record(record);

    let rule = Rule::life();
    println!("| grid | cells | median s/gen | ms/cell |");
    println!("|---|---|---|---|");
    for &size in sizes {
        let grid = Grid::random(size, size, 1, 0.28);
        let cases = grid_cases(&grid, &rule);
        let n = cases.len();
        let mut toks = Vec::with_capacity(n * 9);
        for (c, _) in &cases {
            toks.extend(c.iter().map(|&b| b as i32));
        }
        let t: Tensor<Wgpu, 2, Int> = Tensor::from_data(TensorData::new(toks, [n, 9]), device);

        let _ = model.forward(t.clone()).into_data(); // warm-up
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let logits = model.forward(t.clone());
            let _ = logits.into_data();
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = times[times.len() / 2];
        println!("| {size}x{size} | {n} | {m:.6} | {:.4} |", 1000.0 * m / n as f64);
    }
    Ok(())
}

#[cfg(feature = "cpu")]
fn bench_bert_cpu(
    checkpoint: &Path,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    sizes: &[usize],
    reps: usize,
) -> Result<()> {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::{Int, Tensor, TensorData};
    use llm_life::bert::data::grid_cases;
    use llm_life::bert::model::BertConfig;

    let device = <NdArray as burn::tensor::backend::Backend>::Device::default();
    let cfg = BertConfig::small(d_model, n_layers, n_heads);
    let model: llm_life::bert::model::BertOfLife<NdArray> = cfg.init(&device);
    let bytes = std::fs::read(checkpoint).context("read checkpoint")?;
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(bytes, &device)?;
    let model = model.load_record(record);

    let rule = Rule::life();
    println!("| grid | cells | median s/gen | ns/cell |");
    println!("|---|---|---|---|");
    for &size in sizes {
        let grid = Grid::random(size, size, 1, 0.28);
        let cases = grid_cases(&grid, &rule);
        let n = cases.len();
        let mut toks = Vec::with_capacity(n * 9);
        for (c, _) in &cases {
            toks.extend(c.iter().map(|&b| b as i64));
        }
        let t: Tensor<NdArray, 2, Int> = Tensor::from_data(TensorData::new(toks, [n, 9]), &device);

        let _ = model.forward(t.clone()); // warm-up
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let logits = model.forward(t.clone());
            let _ = logits.into_data();
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = times[times.len() / 2];
        println!("| {size}x{size} | {n} | {m:.6} | {:.1} |", 1e9 * m / n as f64);
    }
    Ok(())
}

/// Sweep BERT of Life sizes + baselines, writing checkpoints and one run doc
/// with the whole table (CONCEPT.md §11 backlog item, task step 3).
fn train_bert_sweep(out: &Path, run_doc: &Path, steps: usize, lr: f64, seed: u64) -> Result<()> {
    use llm_life::bert::train::{eval_lookup, train_bert, train_mlp};

    let start = Instant::now();
    let mut table = String::from(
        "| model | params | steps to 512/512 | held-out acc (64 cases) | IoU 16\u{b2} gen 1..5 (seed 1) | IoU 16\u{b2} gen 1..5 (seed 2) | IoU 16\u{b2} gen 1..5 (seed 3) | alive recall gen 1..5 (seed 1/2/3) | dead recall gen 1..5 (seed 1/2/3) |\n\
         |---|---|---|---|---|---|---|---|---|\n",
    );

    let fmt_iou = |v: &[f64]| -> String {
        v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(",")
    };
    let fmt_field = |s: &[llm_life::score::GenScore], f: fn(&llm_life::score::GenScore) -> f64| -> String {
        [&s[0..5], &s[5..10], &s[10..15]]
            .iter()
            .map(|seed_scores| seed_scores.iter().map(|g| format!("{:.3}", f(g))).collect::<Vec<_>>().join(","))
            .collect::<Vec<_>>()
            .join(" | ")
    };

    println!("=== lookup (zero-parameter floor) ===");
    let (lp, lacc, lscores) = eval_lookup();
    let liou: Vec<f64> = lscores.iter().map(|s| s.iou).collect();
    table.push_str(&format!(
        "| lookup (512-entry table) | {lp} | 0 (exact by construction) | {lacc:.4} | {} | {} | {} | {} | {} |\n",
        fmt_iou(&liou[0..5]), fmt_iou(&liou[5..10]), fmt_iou(&liou[10..15]),
        fmt_field(&lscores, |g| g.alive_recall), fmt_field(&lscores, |g| g.dead_recall)
    ));

    println!("=== mlp baseline ===");
    let (mp, msteps, macc, mscores) = train_mlp(16, steps, lr, &out.join("mlp-16.bin"))?;
    let miou: Vec<f64> = mscores.iter().map(|s| s.iou).collect();
    table.push_str(&format!(
        "| MLP (9->16->2) | {mp} | {} | {macc:.4} | {} | {} | {} | {} | {} |\n",
        msteps.map(|s| s.to_string()).unwrap_or_else(|| format!(">{steps}")),
        fmt_iou(&miou[0..5]), fmt_iou(&miou[5..10]), fmt_iou(&miou[10..15]),
        fmt_field(&mscores, |g| g.alive_recall), fmt_field(&mscores, |g| g.dead_recall)
    ));

    for (d, l, h) in [(16usize, 1usize, 1usize), (32, 2, 2), (64, 2, 2)] {
        println!("=== bert d={d} L={l} H={h} ===");
        let r = train_bert(d, l, h, steps, lr, seed, &out.join(format!("bert-d{d}-L{l}.bin")))?;
        let riou: Vec<f64> = r.scores.iter().map(|s| s.iou).collect();
        table.push_str(&format!(
            "| BERT d={d} L={l} H={h} | {} | {} | {:.4} | {} | {} | {} | {} | {} |\n",
            r.params,
            r.steps_to_512.map(|s| s.to_string()).unwrap_or_else(|| format!(">{steps}")),
            r.held_out_acc,
            fmt_iou(&riou[0..5]), fmt_iou(&riou[5..10]), fmt_iou(&riou[10..15]),
            fmt_field(&r.scores, |g| g.alive_recall), fmt_field(&r.scores, |g| g.dead_recall)
        ));
    }

    let doc = format!(
        "# BERT of Life sweep\n\n\
         machine: {}\n\
         data: exhaustive 512 neighbourhood cases (train on all 512 minus a\n\
         64-case held-out split; a finite function, CONCEPT.md §12), IoU rollout\n\
         on real 16x16 grids, 5 generations, seeds 1-3, teacher-forced\n\
         (each generation starts from true Life, same convention as\n\
         `docs/runs/2026-09-20-a-rollout.md`).\n\
         steps budget: {steps}, lr {lr}, seed {seed}\n\
         wall clock: {:.1}s\n\n\
         ## Table\n\n{table}\n",
        machine(),
        start.elapsed().as_secs_f64(),
    );
    if let Some(dir) = run_doc.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(run_doc, doc)?;
    println!("\nwrote {}", run_doc.display());
    Ok(())
}

/// s/generation for a vector-space checkpoint through native wgpu.
/// `attn` always runs at 16x16 (one forward per cell, batched as one
/// tensor); `stencil` runs at every size in `sizes` (one forward for the
/// whole grid), which is the generalisation number CONCEPT.md §12 asks for.
#[allow(clippy::too_many_arguments)]
fn bench_vec_native(
    checkpoint: &Path,
    kind: &str,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    sizes: &[usize],
    reps: usize,
    device: &WgpuDevice,
) -> Result<()> {
    use burn::backend::Wgpu;
    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::{Tensor, TensorData};
    use llm_life::bert::data::grid_cases;
    use llm_life::vector::model::{stencil_neighbors, AttnConfig, StencilConfig};

    let rule = Rule::life();
    let bytes = std::fs::read(checkpoint).context("read checkpoint")?;
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();

    match kind {
        "attn" => {
            let cfg = AttnConfig::new(d_model);
            let model = cfg.init::<Wgpu>(device);
            let record = recorder.load(bytes, device)?;
            let model = model.load_record(record);

            let grid = Grid::random(16, 16, 1, 0.28);
            let cases = grid_cases(&grid, &rule);
            let n = cases.len();
            let mut xs = Vec::with_capacity(n * 9);
            for (c, _) in &cases {
                xs.extend(c.iter().map(|&b| b as f32));
            }
            let t: Tensor<Wgpu, 2> = Tensor::from_data(TensorData::new(xs, [n, 9]), device);
            let _ = model.forward(t.clone()).into_data();
            println!("| grid | cells | median s/gen | ms/cell |");
            println!("|---|---|---|---|");
            let mut times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                let logits = model.forward(t.clone());
                let _ = logits.into_data();
                times.push(start.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = times[times.len() / 2];
            println!("| 16x16 | {n} | {m:.6} | {:.4} |", 1000.0 * m / n as f64);
        }
        "stencil" => {
            let cfg = StencilConfig::new(d_model, n_layers, n_heads);
            let model = cfg.init::<Wgpu>(device);
            let record = recorder.load(bytes, device)?;
            let model = model.load_record(record);

            println!("| grid | cells | median s/gen | ms/cell |");
            println!("|---|---|---|---|");
            for &size in sizes {
                let n = size * size;
                let grid = Grid::random(size, size, 1, 0.28);
                let cells: Vec<f32> = grid.cells().iter().map(|&c| c as f32).collect();
                let x: Tensor<Wgpu, 2> = Tensor::from_data(TensorData::new(cells, [1, n]), device);
                let neighbors: Tensor<Wgpu, 1, burn::tensor::Int> = stencil_neighbors(size, size, device);
                let _ = model.forward(x.clone(), neighbors.clone()).into_data();
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let start = Instant::now();
                    let logits = model.forward(x.clone(), neighbors.clone());
                    let _ = logits.into_data();
                    times.push(start.elapsed().as_secs_f64());
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let m = times[times.len() / 2];
                println!("| {size}x{size} | {n} | {m:.6} | {:.4} |", 1000.0 * m / n as f64);
            }
        }
        other => anyhow::bail!("unknown --kind {other} (expected attn or stencil)"),
    }
    Ok(())
}

/// Sweep the vector-space variants (CONCEPT.md §12) and write one run doc
/// with both tables: (i) MLP + attention on the 9-number neighbourhood
/// (reusing `bert::train::train_mlp` and `bert::data::split_512`'s 512-case
/// convention) and (ii) the whole-grid stencil transformer, generalising
/// from a 16x16 training grid to a 32x32 rollout with the same weights.
fn train_vec_sweep(
    out: &Path,
    run_doc: &Path,
    steps: usize,
    lr: f64,
    stencil_steps: usize,
    stencil_lr: f64,
    _seed: u64,
) -> Result<()> {
    use llm_life::bert::train::train_mlp;
    use llm_life::vector::train::{train_attn, train_stencil};

    let start = Instant::now();

    let fmt_iou = |v: &[f64]| -> String {
        v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(",")
    };
    let fmt_field = |s: &[llm_life::score::GenScore], f: fn(&llm_life::score::GenScore) -> f64| -> String {
        [&s[0..5], &s[5..10], &s[10..15]]
            .iter()
            .map(|seed_scores| seed_scores.iter().map(|g| format!("{:.3}", f(g))).collect::<Vec<_>>().join(","))
            .collect::<Vec<_>>()
            .join(" | ")
    };

    let mut table_i = String::from(
        "| model | params | steps to 512/512 | held-out acc (64 cases) | IoU 16\u{b2} gen 1..5 (seed 1) | IoU 16\u{b2} gen 1..5 (seed 2) | IoU 16\u{b2} gen 1..5 (seed 3) | alive recall gen 1..5 (seed 1/2/3) | dead recall gen 1..5 (seed 1/2/3) |\n\
         |---|---|---|---|---|---|---|---|---|\n",
    );
    for h in [8usize, 16, 32] {
        println!("=== (i) MLP h={h} ===");
        let (p, s, acc, scores) = train_mlp(h, steps, lr, &out.join(format!("mlp-{h}.bin")))?;
        let iou: Vec<f64> = scores.iter().map(|s| s.iou).collect();
        table_i.push_str(&format!(
            "| MLP (9->{h}->2) | {p} | {} | {acc:.4} | {} | {} | {} | {} | {} |\n",
            s.map(|s| s.to_string()).unwrap_or_else(|| format!(">{steps}")),
            fmt_iou(&iou[0..5]), fmt_iou(&iou[5..10]), fmt_iou(&iou[10..15]),
            fmt_field(&scores, |g| g.alive_recall), fmt_field(&scores, |g| g.dead_recall)
        ));
    }
    for d in [8usize, 16, 32] {
        println!("=== (i) attention d={d} ===");
        let r = train_attn(d, steps, lr, &out.join(format!("attn-{d}.bin")))?;
        let iou: Vec<f64> = r.scores.iter().map(|s| s.iou).collect();
        table_i.push_str(&format!(
            "| Attn (9 numbers, d={d}) | {} | {} | {:.4} | {} | {} | {} | {} | {} |\n",
            r.params,
            r.steps_to_512.map(|s| s.to_string()).unwrap_or_else(|| format!(">{steps}")),
            r.held_out_acc,
            fmt_iou(&iou[0..5]), fmt_iou(&iou[5..10]), fmt_iou(&iou[10..15]),
            fmt_field(&r.scores, |g| g.alive_recall), fmt_field(&r.scores, |g| g.dead_recall)
        ));
    }

    let mut table_ii = String::from(
        "| model | params | steps to train IoU>=0.999 | train IoU | IoU 16\u{b2} gen 1..5 (seed 1/2/3) | IoU 32\u{b2} gen 1..5 (seed 1/2/3) | alive recall 32\u{b2} (seed 1/2/3) | dead recall 32\u{b2} (seed 1/2/3) |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for (d, l, h) in [(16usize, 1usize, 1usize), (32, 2, 2)] {
        println!("=== (ii) stencil d={d} L={l} H={h} ===");
        let r = train_stencil(d, l, h, stencil_steps, stencil_lr, &out.join(format!("stencil-d{d}-L{l}.bin")))?;
        let iou16: Vec<f64> = r.scores_16.iter().map(|s| s.iou).collect();
        let iou32: Vec<f64> = r.scores_32.iter().map(|s| s.iou).collect();
        let iou16_s = format!("{} | {} | {}", fmt_iou(&iou16[0..5]), fmt_iou(&iou16[5..10]), fmt_iou(&iou16[10..15]));
        let iou32_s = format!("{} | {} | {}", fmt_iou(&iou32[0..5]), fmt_iou(&iou32[5..10]), fmt_iou(&iou32[10..15]));
        table_ii.push_str(&format!(
            "| Stencil d={d} L={l} H={h} | {} | {} | {:.4} | {iou16_s} | {iou32_s} | {} | {} |\n",
            r.params,
            r.steps_to_converge.map(|s| s.to_string()).unwrap_or_else(|| format!(">{stencil_steps}")),
            r.train_iou,
            fmt_field(&r.scores_32, |g| g.alive_recall), fmt_field(&r.scores_32, |g| g.dead_recall)
        ));
    }

    let named_cases = [
        ("lonely cell dies (0 neighbours)", [0u8, 0, 0, 0, 0, 0, 0, 0], 1u8, 0u8),
        ("underpopulation (1 neighbour)", [1, 0, 0, 0, 0, 0, 0, 0], 1, 0),
        ("survival on 2", [1, 1, 0, 0, 0, 0, 0, 0], 1, 1),
        ("survival on 3", [1, 1, 1, 0, 0, 0, 0, 0], 1, 1),
        ("birth on 3", [1, 1, 1, 0, 0, 0, 0, 0], 0, 1),
        ("overcrowding (4 neighbours)", [1, 1, 1, 1, 0, 0, 0, 0], 1, 0),
    ];
    let mut cases_table = String::from("| case (000/0X0/000 style, B3/S23) | neighbours (NW,N,NE,W,E,SW,S,SE) | self | -> next |\n|---|---|---|---|\n");
    for (label, nb, self_state, next) in named_cases {
        cases_table.push_str(&format!(
            "| {label} | {} | {self_state} | {next} |\n",
            nb.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(" ")
        ));
    }

    let doc = format!(
        "# Vector-space variants (CONCEPT.md \u{a7}12/\u{a7}13/\u{a7}14)\n\n\
         machine: {}\n\
         wall clock: {:.1}s\n\n\
         (i) \"3x3 -> centre, as numbers\": no tokens, the 9 neighbourhood values\n\
         (8 neighbours + self, `bert::data`'s order) enter as floats. MLP is\n\
         `bert::model::MlpOfLife` (9->h->2), reused as-is (see\n\
         `vector::model` doc comment for why this is already \"an MLP\n\
         (9->h->1)\" up to the 2-class head every other model here shares).\n\
         Attention is `vector::model::AttnOfLife`: 9 scalar tokens through a\n\
         `Linear(1,d)`, learned absolute positions (fixed length, safe here),\n\
         one self-attention block, centre-token readout. Trained on the same\n\
         512-case split as BERT of Life (`bert::data::split_512`, 64 held out),\n\
         same IoU rollout convention (16x16, 5 generations, seeds 1-3,\n\
         teacher-forced).\n\
         steps budget: {steps}, lr {lr}\n\n\
         ## (i) table\n\n{table_i}\n\
         ### named cases (B3/S23)\n\n{cases_table}\n\
         (ii) \"grid -> grid, one channel\": jacobi2000's stencil-masked\n\
         attention (`jacobi2000::model::Block`, `jacobi2000::mask::Mask::Stencil`,\n\
         read-only reference, idea copied not the crate) with one scalar\n\
         channel instead of jacobi2000's multi-channel PDE state — this is\n\
         jacobi2000 with one channel, said explicitly. `vector::model::StencilOfLife`:\n\
         cells enter as floats through a `Linear(1,d)`, **no** positional\n\
         embedding (a learned absolute position would break generalisation\n\
         the moment the grid size changes), 1-2 stencil-masked attention\n\
         blocks (`vector::model::stencil_neighbors` — a flat [n*9] gather\n\
         table, self+8 `life::Grid::neighbor_indices` neighbours, O(9n) not\n\
         O(n^2), same toroidal boundary `life::Grid::step` uses),\n\
         BCE-with-logits per cell.\n\
         Trained on a density sweep (0.10-0.50, 9 densities x 8 seeds) of\n\
         random 16x16 grids against the true next state; evaluated as an IoU\n\
         rollout at 16x16 (in-distribution) **and 32x32** with the identical\n\
         trained weights (only the mask tensor rebuilt for the new size) —\n\
         the CONCEPT.md \u{a7}12 generalisation claim.\n\
         steps budget: {stencil_steps}, lr {stencil_lr}\n\n\
         ## (ii) table\n\n{table_ii}\n\
         ## (iii) 3D Life (note, no code)\n\n\
         Extending this to 3D Life changes exactly one thing structurally:\n\
         the neighbourhood grows from the 8 Moore neighbours of a 3x3 patch to\n\
         the 26 neighbours of a 3x3x3 cube (27 cells including self). Every\n\
         piece of this rung's machinery carries over unchanged in kind: (i)'s\n\
         attention model becomes a 27-token sequence through the same\n\
         `Linear(1,d)` + learned positions + one attention block; (ii)'s\n\
         stencil mask becomes a 3D-Moore mask (`life::Grid::neighbor_indices`\n\
         generalised to a `width*height*depth` index space, still exactly 26\n\
         allowed keys per query, still toroidal, still no positional\n\
         embedding so it still generalises across box size). What does *not*\n\
         carry over is the zero-parameter floor: a 9-bit index (2^9=512\n\
         entries) into a lookup table is free in 2D, but 3D Life's neighbourhood\n\
         is 27 bits, and 2^27 = 134,217,728 entries — the table itself becomes\n\
         a ~134M-row array (order gigabytes at even one byte per entry,\n\
         before any real B/S ruleset like B5766 needs more than 1 bit of\n\
         state per cell). The \"brute force lookup\" rung of the compute ladder\n\
         (docs/LADDER.md) simply disappears in 3D: it is no longer a\n\
         reasonable thing to build, memory-bound or not. This is exactly the\n\
         point CONCEPT.md \u{a7}12 makes about this project's rungs: the learned\n\
         models ((i)/(ii)) generalise to this case with no architectural\n\
         change and a parameter count in the hundreds to low thousands, while\n\
         \"the classical answer\" (an explicit table) stops being available at\n\
         all. 3D Life becomes jacobi2000's first discrete dataset precisely\n\
         because jacobi2000's stencil machinery (a local mask over an\n\
         N-dimensional grid) does not care whether N is 2 or 3.\n",
        machine(),
        start.elapsed().as_secs_f64(),
    );
    if let Some(dir) = run_doc.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(run_doc, doc)?;
    println!("\nwrote {}", run_doc.display());
    Ok(())
}

/// Rerun (i) only with a corrected lr schedule (see `Command::TrainVecILrFix`
/// doc comment for why). Appends a "corrected" section to `run_doc`,
/// keeping whatever the file already has (the first, lr=1e-2 sweep) as a
/// documented lr-sensitivity result rather than overwriting it.
fn train_vec_i_lr_fix(out: &Path, run_doc: &Path, steps: usize, lr: f64) -> Result<()> {
    use llm_life::vector::train::{train_attn_decay, train_mlp2_decay, train_mlp_decay};

    let start = Instant::now();
    let fmt_iou = |v: &[f64]| -> String {
        v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(",")
    };

    let mut table = String::from(
        "| model | params | steps to 512/512 | held-out acc (64 cases) | IoU 16\u{b2} gen 1..5 (seed 1) | IoU 16\u{b2} gen 1..5 (seed 2) | IoU 16\u{b2} gen 1..5 (seed 3) |\n\
         |---|---|---|---|---|---|---|\n",
    );
    let mut row = |name: &str, params: usize, steps_to_512: Option<usize>, held_out_acc: f64, scores: &[llm_life::score::GenScore]| {
        let iou: Vec<f64> = scores.iter().map(|s| s.iou).collect();
        table.push_str(&format!(
            "| {name} | {params} | {} | {held_out_acc:.4} | {} | {} | {} |\n",
            steps_to_512.map(|s| s.to_string()).unwrap_or_else(|| format!(">{steps}")),
            fmt_iou(&iou[0..5]), fmt_iou(&iou[5..10]), fmt_iou(&iou[10..15]),
        ));
    };

    for h in [8usize, 16, 32] {
        println!("=== (i) lr-fix MLP h={h} ===");
        let r = train_mlp_decay(h, steps, lr, &out.join(format!("mlp-{h}-lrfix.bin")))?;
        row(&format!("MLP (9->{h}->2), lr-fix"), r.params, r.steps_to_512, r.held_out_acc, &r.scores);
    }
    {
        println!("=== (i) lr-fix MLP2 h=32 (9->32->32->2) ===");
        let r = train_mlp2_decay(32, steps, lr, &out.join("mlp2-32-lrfix.bin"))?;
        row("MLP2 (9->32->32->2), lr-fix", r.params, r.steps_to_512, r.held_out_acc, &r.scores);
    }
    for d in [8usize, 16, 32] {
        println!("=== (i) lr-fix Attn d={d} ===");
        let r = train_attn_decay(d, steps, lr, &out.join(format!("attn-{d}-lrfix.bin")))?;
        row(&format!("Attn (9 numbers, d={d}), lr-fix"), r.params, r.steps_to_512, r.held_out_acc, &r.scores);
    }

    let existing = std::fs::read_to_string(run_doc).unwrap_or_default();
    let section = format!(
        "\n## (i) corrected: cosine lr decay {lr}->{:.1e} over {steps} steps\n\n\
         The first (i) sweep above used a fixed lr=1e-2 for 600 steps — the\n\
         same setting that collapsed the BERT d=32/64 runs, fixed there by\n\
         lr 2e-3/800 steps (docs/runs/2026-09-20-bert.md). This section\n\
         reruns MLP h\u{2208}{{8,16,32}}, a 2-layer MLP body (9->32->32->2), and\n\
         Attn d\u{2208}{{8,16,32}} with cosine decay from {lr} down to {:.1e} instead,\n\
         same 512-case split, same 16\u{b2} rollout convention. Kept alongside the\n\
         first sweep as a documented lr-sensitivity result, not a\n\
         replacement of it.\n\
         wall clock: {:.1}s\n\n\
         {table}\n",
        lr / 10.0, lr / 10.0, start.elapsed().as_secs_f64(),
    );
    std::fs::write(run_doc, existing + &section)?;
    println!("\nappended corrected section to {}", run_doc.display());
    Ok(())
}

/// Re-score a picture already on disk (no GPU, no rerun): read its
/// `palive`/`true`/`argmax` PGMs and print accuracy, IoU, Hamming, F1 and the
/// per-case recall table, plus (for comparison) live recall at the Otsu and
/// z-score (k=2) label-free thresholds and the median.
///
/// The seed grid is not on disk — only its output is (`*-true.pgm`) — so it
/// is regenerated with `seed_grid` exactly as `run_pictures` built it
/// (density 0.28, the CLI default, since no picture run used a different
/// one), and asserted to reproduce the true PGM: if that assertion fails the
/// picture was made with different parameters and every downstream number
/// here would be wrong.
fn rescore(dir: &Path, tag: &str, seed: &str) -> Result<()> {
    let rule = Rule::life();
    let (p, w, h) = read_pgm(&dir.join(format!("{tag}-{seed}-gen1-palive.pgm")), 6)?;
    let (true_cells, tw, th) = read_pgm(&dir.join(format!("{tag}-{seed}-gen1-true.pgm")), 6)?;
    let (argmax_cells, aw, ah) = read_pgm(&dir.join(format!("{tag}-{seed}-gen1-argmax.pgm")), 6)?;
    anyhow::ensure!((w, h) == (tw, th) && (w, h) == (aw, ah), "pgm dimension mismatch");
    let truth = Grid::from_cells(w, h, true_cells.iter().map(|&v| (v > 0.5) as u8).collect());
    let model = Grid::from_cells(w, h, argmax_cells.iter().map(|&v| (v > 0.5) as u8).collect());

    let input = seed_grid(seed, w, 0.28)?;
    anyhow::ensure!(
        input.step(&rule) == truth,
        "{tag} {seed}: regenerated seed's Life step does not match the true PGM on disk \
         (picture run used different parameters — rescore's assumptions are wrong)"
    );

    let s = score(&truth, &model, &p, 1);
    println!(
        "{tag} {seed}: acc={:.4} iou={:.4} hamming={} f1={:.4} precision={:.4} recall={:.4} true_live={} model_live={}",
        s.accuracy, s.iou, s.wrong_cells, s.f1, s.precision, s.recall, s.true_live, s.model_live,
    );
    println!(
        "  alive_recall={:.4} dead_recall={:.4} alive_precision={:.4} | tp={} fp={} fn={} tn={}",
        s.alive_recall, s.dead_recall, s.alive_precision, s.tp, s.fp, s.fn_, s.tn,
    );

    println!("  per-case recall (from the input grid, B3/S23):");
    for (case, c) in per_case_recall(&input, &model, &rule) {
        println!("    {:<20} n={:<5} correct={:<5} frac={:.4}", case.label(), c.count, c.correct, c.fraction());
    }

    let otsu = otsu_threshold_grid(&p, w, h);
    let z2 = zscore_threshold_grid(&p, w, h, 2.0);
    let med = median_threshold_grid(&p, w, h);
    let s_otsu = score(&truth, &otsu, &p, 1);
    let s_z2 = score(&truth, &z2, &p, 1);
    let s_med = score(&truth, &med, &p, 1);
    println!(
        "  otsu acc={:.4} recall={:.4} live={} | z2 acc={:.4} recall={:.4} live={} | median acc={:.4} recall={:.4} live={}",
        s_otsu.accuracy, s_otsu.live_recall, s_otsu.model_live,
        s_z2.accuracy, s_z2.live_recall, s_z2.model_live,
        s_med.accuracy, s_med.live_recall, s_med.model_live,
    );
    Ok(())
}

/// One row of `docs/LADDER.md`'s table: a rung name and its s/generation at
/// each requested size (`None` = not measured at that size, never
/// projected).
struct LadderRow {
    rung: String,
    sizes: Vec<(usize, Option<f64>)>,
    note: Option<String>,
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn ladder_json(device: &str, rows: &[LadderRow]) -> String {
    let mut out = String::from("{\n");
    out.push_str(&format!("  \"device\": \"{}\",\n", json_escape(device)));
    out.push_str("  \"backend\": \"wgpu\",\n");
    out.push_str("  \"rows\": [\n");
    for (i, row) in rows.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"rung\": \"{}\",\n", json_escape(&row.rung)));
        let fields: Vec<String> = row
            .sizes
            .iter()
            .map(|(sz, v)| match v {
                Some(x) => format!("\"{sz}\": {x:.6}"),
                None => format!("\"{sz}\": null"),
            })
            .collect();
        out.push_str(&format!("      \"s_per_gen\": {{{}}}", fields.join(", ")));
        match &row.note {
            Some(note) => out.push_str(&format!(",\n      \"note\": \"{}\"\n", json_escape(note))),
            None => out.push('\n'),
        }
        out.push_str("    }");
        if i + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    out
}

fn ladder_table(rows: &[LadderRow], columns: &[usize]) -> String {
    let mut out = String::from("| rung |");
    for c in columns {
        out.push_str(&format!(" {c}\u{b2} s/gen |"));
    }
    out.push_str(" note |\n|---|");
    for _ in columns {
        out.push_str("---|");
    }
    out.push_str("---|\n");
    for row in rows {
        out.push_str(&format!("| {} |", row.rung));
        for c in columns {
            let v = row.sizes.iter().find(|(sz, _)| sz == c).and_then(|(_, v)| *v);
            match v {
                Some(x) => out.push_str(&format!(" {x:.6} |")),
                None => out.push_str(" — |"),
            }
        }
        out.push_str(&format!(" {} |\n", row.note.as_deref().unwrap_or("")));
    }
    out
}

/// One generation, timed, for every `docs/LADDER.md` rung, on this machine.
/// Same board (density/seed) for every rung. Never projects: a rung not
/// measured at a size is `null`, not `ms_per_cell * n^2`.
#[allow(clippy::too_many_arguments)]
fn bench_ladder(
    sizes: &[usize],
    llm_sizes: &[usize],
    gguf: &Path,
    tokenizer: &Path,
    adapter: &Path,
    bert_checkpoint: &Path,
    mlp2_checkpoint: &Path,
    stencil_checkpoint: &Path,
    stencil_max_size: usize,
    density: f64,
    seed: u64,
    reps: usize,
    rule: &Rule,
    out_json: Option<&Path>,
    out_md: Option<&Path>,
    device: &WgpuDevice,
) -> Result<()> {
    use burn::backend::Wgpu;
    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::{Int, Tensor, TensorData};
    use llm_life::bert::data::grid_cases;
    use llm_life::bert::model::BertConfig;
    use llm_life::vector::model::{stencil_neighbors, Mlp2Config, StencilConfig};

    let mut rows: Vec<LadderRow> = Vec::new();

    // "CPU loop": life::Grid::step is exactly the plain nested loop over 8
    // neighbours the tab's JS measures live.
    {
        let mut cells = Vec::with_capacity(sizes.len());
        for &size in sizes {
            let grid = Grid::random(size, size, seed, density);
            let _ = grid.step(rule);
            let mut times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                let _ = grid.step(rule);
                times.push(start.elapsed().as_secs_f64());
            }
            cells.push((size, Some(median(times))));
        }
        rows.push(LadderRow { rung: "CPU loop".into(), sizes: cells, note: None });
    }

    // "512-entry lookup": precompute the 9-bit-index table once (outside
    // the timed loop), then a per-cell gather + table read.
    {
        let mut table = [false; 512];
        for self_state in 0u8..2 {
            for mask in 0usize..256 {
                let neighbors: Vec<u8> = (0..8).map(|b| ((mask >> b) & 1) as u8).collect();
                let idx = case_index(&neighbors, self_state);
                let count = neighbors.iter().filter(|&&x| x != 0).count();
                table[idx] = rule.next(self_state != 0, count);
            }
        }
        let lookup_step = |grid: &Grid| -> Grid {
            let n = grid.width() * grid.height();
            let mut out = vec![0u8; n];
            for (i, o) in out.iter_mut().enumerate() {
                let neighbors: Vec<u8> = grid.neighbor_indices(i).iter().map(|&j| grid.cells()[j]).collect();
                let idx = case_index(&neighbors, grid.cells()[i]);
                *o = table[idx] as u8;
            }
            Grid::from_cells(grid.width(), grid.height(), out)
        };
        let mut cells = Vec::with_capacity(sizes.len());
        for &size in sizes {
            let grid = Grid::random(size, size, seed, density);
            let _ = lookup_step(&grid);
            let mut times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                let _ = lookup_step(&grid);
                times.push(start.elapsed().as_secs_f64());
            }
            cells.push((size, Some(median(times))));
        }
        rows.push(LadderRow { rung: "512-entry lookup".into(), sizes: cells, note: None });
    }

    // "LLM per pixel (adapter)" / "LLM batched (adapter)": the same forward
    // (`TrainRunnerA`, `a-norules`), per LADDER.md's own note that the two
    // narrations share one set of numbers. Measured only at `llm_sizes` —
    // each cell is a real forward, never projected.
    {
        let have_model = gguf.exists() && tokenizer.exists() && adapter.exists();
        if have_model {
            let chunk_cells = 64;
            let runner = TrainRunnerA::new(gguf, tokenizer, rule, Some(adapter), true, chunk_cells, device)?;
            let mut measured = Vec::with_capacity(llm_sizes.len());
            for &size in llm_sizes {
                let grid = Grid::random(size, size, seed, density);
                let _ = Stepper::step(&runner, &grid)?;
                let (_, secs) = Stepper::step(&runner, &grid)?;
                measured.push((size, secs));
            }
            let cells: Vec<(usize, Option<f64>)> = sizes
                .iter()
                .map(|&size| (size, measured.iter().find(|(s, _)| *s == size).map(|(_, v)| *v)))
                .collect();
            rows.push(LadderRow { rung: "LLM per pixel (adapter)".into(), sizes: cells.clone(), note: None });
            rows.push(LadderRow {
                rung: "LLM batched (adapter)".into(),
                sizes: cells,
                note: Some("identical forward to per-pixel (LADDER.md)".into()),
            });
        } else {
            let note = format!(
                "skipped: model files not local ({}, {}, {})",
                gguf.display(),
                tokenizer.display(),
                adapter.display()
            );
            let cells: Vec<(usize, Option<f64>)> = sizes.iter().map(|&s| (s, None)).collect();
            rows.push(LadderRow { rung: "LLM per pixel (adapter)".into(), sizes: cells.clone(), note: Some(note.clone()) });
            rows.push(LadderRow { rung: "LLM batched (adapter)".into(), sizes: cells, note: Some(note) });
        }
    }

    // "BERT of Life": whole-batch native forward (not one LLM call per
    // cell), cheap at every requested size.
    {
        if bert_checkpoint.exists() {
            let cfg = BertConfig::small(16, 1, 1);
            let model: llm_life::bert::model::BertOfLife<Wgpu> = cfg.init(device);
            let bytes = std::fs::read(bert_checkpoint).context("read bert checkpoint")?;
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
            let record = recorder.load(bytes, device)?;
            let model = model.load_record(record);

            let mut cells = Vec::with_capacity(sizes.len());
            for &size in sizes {
                let grid = Grid::random(size, size, seed, density);
                let cases = grid_cases(&grid, rule);
                let n = cases.len();
                let mut toks = Vec::with_capacity(n * 9);
                for (c, _) in &cases {
                    toks.extend(c.iter().map(|&b| b as i32));
                }
                let t: Tensor<Wgpu, 2, Int> = Tensor::from_data(TensorData::new(toks, [n, 9]), device);
                let _ = model.forward(t.clone()).into_data();
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let start = Instant::now();
                    let logits = model.forward(t.clone());
                    let _ = logits.into_data();
                    times.push(start.elapsed().as_secs_f64());
                }
                cells.push((size, Some(median(times))));
            }
            rows.push(LadderRow { rung: "BERT of Life".into(), sizes: cells, note: None });
        } else {
            let note = format!("skipped: checkpoint not local ({})", bert_checkpoint.display());
            let cells: Vec<(usize, Option<f64>)> = sizes.iter().map(|&s| (s, None)).collect();
            rows.push(LadderRow { rung: "BERT of Life".into(), sizes: cells, note: Some(note) });
        }
    }

    // "9 numbers -> centre": MLP2 (9->32->32->2), the exact checkpoint.
    {
        if mlp2_checkpoint.exists() {
            let cfg = Mlp2Config::new(32);
            let model = cfg.init::<Wgpu>(device);
            let bytes = std::fs::read(mlp2_checkpoint).context("read mlp2 checkpoint")?;
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
            let record = recorder.load(bytes, device)?;
            let model = model.load_record(record);

            let mut cells = Vec::with_capacity(sizes.len());
            for &size in sizes {
                let grid = Grid::random(size, size, seed, density);
                let cases = grid_cases(&grid, rule);
                let n = cases.len();
                let mut xs = Vec::with_capacity(n * 9);
                for (c, _) in &cases {
                    xs.extend(c.iter().map(|&b| b as f32));
                }
                let t: Tensor<Wgpu, 2> = Tensor::from_data(TensorData::new(xs, [n, 9]), device);
                let _ = model.forward(t.clone()).into_data();
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let start = Instant::now();
                    let logits = model.forward(t.clone());
                    let _ = logits.into_data();
                    times.push(start.elapsed().as_secs_f64());
                }
                cells.push((size, Some(median(times))));
            }
            rows.push(LadderRow { rung: "9 numbers -> centre".into(), sizes: cells, note: None });
        } else {
            let note = format!("skipped: checkpoint not local ({})", mlp2_checkpoint.display());
            let cells: Vec<(usize, Option<f64>)> = sizes.iter().map(|&s| (s, None)).collect();
            rows.push(LadderRow { rung: "9 numbers -> centre".into(), sizes: cells, note: Some(note) });
        }
    }

    // "stencil (grid -> grid)": whole grid in one forward pass. Capped at
    // `stencil_max_size` — the current O(9n) `stencil_neighbors` gather is
    // fine at 16/32/64, but until the crate's stencil implementation is
    // settled elsewhere, larger sizes are not attempted here rather than
    // risking a GPU OOM mid-ladder.
    {
        if stencil_checkpoint.exists() {
            let cfg = StencilConfig::new(16, 1, 1);
            let model = cfg.init::<Wgpu>(device);
            let bytes = std::fs::read(stencil_checkpoint).context("read stencil checkpoint")?;
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
            let record = recorder.load(bytes, device)?;
            let model = model.load_record(record);

            let mut cells = Vec::with_capacity(sizes.len());
            let mut skipped = Vec::new();
            for &size in sizes {
                if size > stencil_max_size {
                    cells.push((size, None));
                    skipped.push(size);
                    continue;
                }
                let n = size * size;
                let grid = Grid::random(size, size, seed, density);
                let x_cells: Vec<f32> = grid.cells().iter().map(|&c| c as f32).collect();
                let x: Tensor<Wgpu, 2> = Tensor::from_data(TensorData::new(x_cells, [1, n]), device);
                let neighbors: Tensor<Wgpu, 1, Int> = stencil_neighbors(size, size, device);
                let _ = model.forward(x.clone(), neighbors.clone()).into_data();
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let start = Instant::now();
                    let logits = model.forward(x.clone(), neighbors.clone());
                    let _ = logits.into_data();
                    times.push(start.elapsed().as_secs_f64());
                }
                cells.push((size, Some(median(times))));
            }
            let note = if skipped.is_empty() {
                None
            } else {
                Some(format!(
                    "{skipped:?} skipped above --stencil-max-size={stencil_max_size} (pending stencil fix)"
                ))
            };
            rows.push(LadderRow { rung: "stencil (grid -> grid)".into(), sizes: cells, note });
        } else {
            let note = format!("skipped: checkpoint not local ({})", stencil_checkpoint.display());
            let cells: Vec<(usize, Option<f64>)> = sizes.iter().map(|&s| (s, None)).collect();
            rows.push(LadderRow { rung: "stencil (grid -> grid)".into(), sizes: cells, note: Some(note) });
        }
    }

    let mut columns: Vec<usize> = sizes.to_vec();
    for &s in llm_sizes {
        if !columns.contains(&s) {
            columns.push(s);
        }
    }
    columns.sort_unstable();

    let device_str = machine();
    let json = ladder_json(&device_str, &rows);
    let table = ladder_table(&rows, &columns);

    println!("{json}");
    println!("{table}");

    if let Some(path) = out_json {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, &json)?;
        println!("wrote {}", path.display());
    }
    if let Some(path) = out_md {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let doc = format!(
            "# Ladder bench\n\n\
             machine: {device_str}\n\
             backend: wgpu\n\
             board: density {density}, seed {seed}\n\
             reps: {reps} (median), LLM rungs: 1\n\
             sizes: {sizes:?}, llm-sizes: {llm_sizes:?}\n\n\
             ## Table\n\n{table}\n",
        );
        std::fs::write(path, doc)?;
        println!("wrote {}", path.display());
    }

    Ok(())
}
