//! Native driver: run variant B against a real GGUF and write the pictures.
//!
//! Backend features are selected here, in the consumer crate (CLAUDE.md
//! "Code"). The GPU is shared with another repo's training job — every run
//! here is one short inference per generation, and every timing it prints is
//! provisional.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use life::{Grid, Rule};
use llm_life::pgm::{write_binary_pgm, write_pgm};
use llm_life::score::score;
use llm_life::variant_b::{argmax_grid, p_alive, pack, rules_prefix};
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::model::{ForwardSpec, LlmModel};
use llm_wasm::tokenizer::Tokenizer;
use std::path::PathBuf;
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
            let rule = Rule::life();
            let device = WgpuDevice::default();
            let runner = Runner::new(&gguf, &tokenizer, &rule, &device)?;
            std::fs::create_dir_all(&out)?;

            let mut summary = String::new();
            summary.push_str(&format!(
                "# First picture — variant B, native\n\n\
                 model: {}\nrule: {}\ngrid: {size}x{size} (torus)\nmode: {}\n\
                 positions: bag (all grid tokens share one position id)\n\
                 mask: prefix (causal) + self + 8 neighbors\n\
                 head: sliced to the two answer tokens\n\n\
                 Timings provisional: the Metal GPU is shared with another repo's training job.\n\n",
                gguf.display(),
                rule.to_rulestring(),
                if freerun { "free-running (model eats its own output)" } else { "teacher-forced (each generation starts from true Life)" },
            ));

            for seed_name in &seeds {
                println!("\n=== seed {seed_name} ===");
                summary.push_str(&format!(
                    "## seed {seed_name}\n\n\
                     | gen | accuracy | live recall | confidence gap | wrong cells | true live | model live | s/gen |\n\
                     |---|---|---|---|---|---|---|---|\n"
                ));
                let mut input = seed_grid(seed_name, size, density)?;
                for gen in 1..=generations {
                    let truth = input.step(&rule);
                    let (p, secs) = runner.step(&input)?;
                    let model_grid = argmax_grid(&p, size, size);
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
                        let tag = format!("b-{seed_name}-gen{gen}");
                        write_pgm(&out.join(format!("{tag}-palive.pgm")), &p, size, size, 6)?;
                        write_binary_pgm(&out.join(format!("{tag}-argmax.pgm")), model_grid.cells(), size, size, 6)?;
                        write_binary_pgm(&out.join(format!("{tag}-true.pgm")), truth.cells(), size, size, 6)?;
                        write_binary_pgm(&out.join(format!("{tag}-diff.pgm")), &truth.diff(&model_grid), size, size, 6)?;
                    }

                    input = if freerun { model_grid } else { truth };
                }
                summary.push('\n');
            }

            let path = out.join("first-picture.md");
            std::fs::write(&path, summary)?;
            println!("\nwrote {}", path.display());
        }
    }
    Ok(())
}
