//! Native driver: run variant A or B against a real GGUF and write the pictures.
//!
//! Backend features are selected here, in the consumer crate (CLAUDE.md
//! "Code"). The GPU is shared with another repo's training job — every run
//! here is one short inference per generation, and every timing it prints is
//! provisional.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use life::{Grid, Rule};
use llm_life::pgm::{write_binary_pgm, write_pgm};
use llm_life::score::{median_threshold_grid, score};
use llm_life::variant_a;
use llm_life::variant_b::{argmax_grid, p_alive, pack, rules_prefix};
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::kv::KvCache;
use llm_wasm::model::{ForwardSpec, LlmModel};
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
        /// Threshold p(alive) at the grid median instead of 0.5.
        #[arg(long)]
        median: bool,
        /// Filename prefix for this attempt's pictures and summary.
        #[arg(long, default_value = "a")]
        tag: String,
    },
}

/// Everything variant B needs that does not change between generations.
struct Runner {
    model: LlmModel,
    prefix: Vec<u32>,
    dead: u32,
    alive: u32,
    head: burn::tensor::Tensor<burn::backend::Wgpu, 2>,
}

impl Runner {
    fn new(gguf: &PathBuf, tokenizer: &PathBuf, rule: &Rule, device: &WgpuDevice) -> Result<Self> {
        let tok = Tokenizer::from_json(&std::fs::read(tokenizer).context("read tokenizer.json")?)?;
        let prefix = tok.encode(&rules_prefix(rule), false)?;

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
        })
    }

    /// One generation: pack the grid, one forward pass, read p(alive) at
    /// every cell.
    fn step(&self, grid: &Grid) -> Result<(Vec<f32>, f64)> {
        let packed = pack(grid, &self.prefix, self.dead, self.alive);
        let t = packed.len();
        let spec = ForwardSpec::default()
            .with_positions(packed.positions.clone())
            .with_allowed(&packed.allowed, t, t, self.model.device());

        let start = Instant::now();
        let mut cache = self.model.new_cache(t);
        let hidden = self.model.forward_hidden_spec(&packed.tokens, &mut cache, &spec)?;
        let logits = self.model.lm_head_sliced(hidden, &self.head);
        let logits = llm_wasm::model::logits_to_vec(logits)?;
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
        "positions: bag (all grid tokens share one position id)\n\
         mask: prefix (causal) + self + 8 neighbors\n\
         head: sliced to the two answer tokens\n"
            .to_string()
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
    median: bool,
) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut summary = format!(
        "# {title}\n\n\
         model: {}\nrule: {}\ngrid: {size}x{size} (torus)\nmode: {}\n{}\
         threshold: {}\n\n\
         Timings provisional: the Metal GPU is shared with another repo's training job.\n\n",
        gguf.display(),
        rule.to_rulestring(),
        if freerun {
            "free-running (model eats its own output)"
        } else {
            "teacher-forced (each generation starts from true Life)"
        },
        stepper.header(),
        if median { "grid median of p(alive)" } else { "p(alive) >= 0.5" },
    );

    for seed_name in seeds {
        println!("\n=== seed {seed_name} ===");
        summary.push_str(&format!(
            "## seed {seed_name}\n\n\
             | gen | accuracy | live recall | confidence gap | wrong cells | true live | model live | s/gen |\n\
             |---|---|---|---|---|---|---|---|\n"
        ));
        let mut input = seed_grid(seed_name, size, density)?;
        for gen in 1..=generations {
            let truth = input.step(rule);
            let (p, secs) = stepper.step(&input)?;
            let model_grid = if median {
                median_threshold_grid(&p, size, size)
            } else {
                argmax_grid(&p, size, size)
            };
            let s = score(&truth, &model_grid, &p, gen);
            println!(
                "gen {gen:2}: acc={:.4} live_recall={:.4} gap={:+.4} wrong={} true_live={} model_live={} {:.2}s",
                s.accuracy, s.live_recall, s.confidence_gap, s.wrong_cells, s.true_live, s.model_live, secs
            );
            summary.push_str(&format!(
                "| {gen} | {:.4} | {:.4} | {:+.4} | {} | {} | {} | {:.2} |\n",
                s.accuracy, s.live_recall, s.confidence_gap, s.wrong_cells, s.true_live, s.model_live, secs
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
        } => {
            let runner = Runner::new(&gguf, &tokenizer, &rule, &device)?;
            run_pictures(
                &runner,
                "First picture — variant B, native",
                "b",
                &gguf,
                &rule,
                size,
                generations,
                &seeds,
                density,
                &out,
                freerun,
                false,
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
            median,
            tag,
        } => {
            let runner = RunnerA::new(&gguf, &tokenizer, &rule, chunk_cells, fewshot, &device)?;
            println!("{} chunks per generation", runner.chunks(size * size));
            run_pictures(
                &runner,
                "Variant A — packed per-cell prompts, native",
                &tag,
                &gguf,
                &rule,
                size,
                generations,
                &seeds,
                density,
                &out,
                freerun,
                median,
            )
        }
    }
}
