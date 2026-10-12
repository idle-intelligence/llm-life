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
    /// s/generation for each way of computing a Life step, on this machine,
    /// at the same board (density/seed) for every row. "CPU loop" and
    /// "512-entry lookup" are the plain-Rust equivalents of the tab's live JS
    /// measurement; BERT of Life, "9 numbers -> centre" and stencil are cheap
    /// whole-batch/whole-grid native forwards, measured at every `--sizes`
    /// entry. Prints one JSON object and a human table; `--out-json` and
    /// `--out-md` also write them to files.
    BenchLadder {
        #[arg(long, value_delimiter = ',', default_value = "16,32,64")]
        sizes: Vec<usize>,
        #[arg(long, default_value = "artifacts/bert-d16-L1.bin")]
        bert_checkpoint: PathBuf,
        #[arg(long, default_value = "artifacts/mlp2-32-lrfix.bin")]
        mlp2_checkpoint: PathBuf,
        #[arg(long, default_value = "artifacts/stencil-d16-L1.bin")]
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
        } => {
            let title = "Variant B — stencil mask, native";
            let runner = Runner::new(&gguf, &tokenizer, &rule, fewshot, &device)?;
            run_pictures(
                &runner, title, &tag, &gguf, &rule, size, generations, &seeds, density, &out,
                freerun,
            )
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
        } => {
            let title = "Variant A — packed per-cell prompts, native";
            let runner = RunnerA::new(&gguf, &tokenizer, &rule, chunk_cells, fewshot, &device)?;
            println!("{} chunks per generation", runner.chunks(size * size));
            run_pictures(
                &runner, title, &tag, &gguf, &rule, size, generations, &seeds, density, &out,
                freerun,
            )
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
             reps: {reps} (median)\n\
             sizes: {sizes:?}\n\n\
             ## Table\n\n{table}\n",
        );
        std::fs::write(path, doc)?;
        println!("wrote {}", path.display());
    }

    Ok(())
}
