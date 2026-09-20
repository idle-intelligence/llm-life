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
             | gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |\n\
             |---|---|---|---|---|---|---|---|---|---|\n"
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
                "gen {gen:2}: acc={:.4} iou={:.4} live_recall={:.4} | median acc={:.4} live_recall={:.4} | gap={:+.4} true_live={} model_live={} {:.2}s",
                s.accuracy, s.iou, s.live_recall, m.accuracy, m.live_recall, s.confidence_gap, s.true_live, s.model_live, secs
            );
            summary.push_str(&format!(
                "| {gen} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:+.4} | {} | {} | {:.2} |\n",
                s.accuracy, s.iou, s.live_recall, m.accuracy, m.live_recall, s.confidence_gap, s.true_live, s.model_live, secs
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
    let rule = Rule::life();
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
        Command::Rescore { dir, tag, seed } => rescore(&dir, &tag, &seed),
    }
}

/// Sweep BERT of Life sizes + baselines, writing checkpoints and one run doc
/// with the whole table (CONCEPT.md §11 backlog item, task step 3).
fn train_bert_sweep(out: &Path, run_doc: &Path, steps: usize, lr: f64, seed: u64) -> Result<()> {
    use llm_life::bert::train::{eval_lookup, train_bert, train_mlp};

    let start = Instant::now();
    let mut table = String::from(
        "| model | params | steps to 512/512 | held-out acc (64 cases) | IoU 16\u{b2} gen 1..5 (seed 1) | IoU 16\u{b2} gen 1..5 (seed 2) | IoU 16\u{b2} gen 1..5 (seed 3) |\n\
         |---|---|---|---|---|---|---|\n",
    );

    let fmt_iou = |v: &[f64]| -> String {
        v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(",")
    };

    println!("=== lookup (zero-parameter floor) ===");
    let (lp, lacc, liou) = eval_lookup();
    table.push_str(&format!(
        "| lookup (512-entry table) | {lp} | 0 (exact by construction) | {lacc:.4} | {} | {} | {} |\n",
        fmt_iou(&liou[0..5]), fmt_iou(&liou[5..10]), fmt_iou(&liou[10..15])
    ));

    println!("=== mlp baseline ===");
    let (mp, msteps, macc, miou) = train_mlp(16, steps, lr, &out.join("mlp-16.bin"))?;
    table.push_str(&format!(
        "| MLP (9->16->2) | {mp} | {} | {macc:.4} | {} | {} | {} |\n",
        msteps.map(|s| s.to_string()).unwrap_or_else(|| format!(">{steps}")),
        fmt_iou(&miou[0..5]), fmt_iou(&miou[5..10]), fmt_iou(&miou[10..15])
    ));

    for (d, l, h) in [(16usize, 1usize, 1usize), (32, 2, 2), (64, 2, 2)] {
        println!("=== bert d={d} L={l} H={h} ===");
        let r = train_bert(d, l, h, steps, lr, seed, &out.join(format!("bert-d{d}-L{l}.bin")))?;
        table.push_str(&format!(
            "| BERT d={d} L={l} H={h} | {} | {} | {:.4} | {} | {} | {} |\n",
            r.params,
            r.steps_to_512.map(|s| s.to_string()).unwrap_or_else(|| format!(">{steps}")),
            r.held_out_acc,
            fmt_iou(&r.iou_per_gen[0..5]), fmt_iou(&r.iou_per_gen[5..10]), fmt_iou(&r.iou_per_gen[10..15])
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
