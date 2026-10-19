//! Native driver: run variant A or B against a real GGUF and write the pictures.
//!
//! Runs on lean like the page: WebGPU when the machine has an adapter,
//! otherwise lean's CPU backend. Every timing it prints is one run on a
//! machine that may be busy with other work.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use life::{Grid, Rule};
use llm_life::pgm::{read_pgm, write_binary_pgm, write_pgm};
use llm_life::score::{
    median_threshold_grid, otsu_threshold_grid, per_case_recall, score, zscore_threshold_grid,
};
use llm_life::variant_a;
use llm_life::variant_b::{argmax_grid, fewshot_rules_prefix, rules_prefix};
use llm_life_lean::lean::engine::Engine;
use llm_life_lean::{case_index, p_alive, LifeLean};
use pollster::block_on;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "llm-life",
    about = "A language model as the update rule of a cellular automaton"
)]
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

/// Which variant a run asks, and how it was set up.
enum Variant {
    /// Whole grid in one forward (stencil mask).
    B,
    /// Packed per-cell prompts against the resident prefix.
    A { chunk_cells: usize },
}

/// The loaded model and one variant's setup: turn a grid into a p(alive)
/// per cell, and say how long the forward part took.
struct Runner {
    life: LifeLean,
    variant: Variant,
    fewshot: bool,
}

impl Runner {
    /// Load the GGUF on WebGPU (or the CPU without an adapter) and set the
    /// variant's prefix: rules (+ the six worked examples with `fewshot`).
    fn new(
        gguf: &Path,
        tokenizer: &Path,
        rulestring: &str,
        variant: Variant,
        fewshot: bool,
    ) -> Result<Self> {
        let tok = std::fs::read(tokenizer).context("read tokenizer.json")?;
        let reader = || -> Result<_> {
            Ok(std::io::BufReader::new(
                std::fs::File::open(gguf).context("open gguf")?,
            ))
        };
        let mut life = match Engine::new() {
            Ok(engine) => block_on(LifeLean::load(engine, reader()?, &tok, rulestring, 0, 0))?,
            Err(e) => {
                println!("no GPU adapter ({e}), using the CPU");
                block_on(LifeLean::load_cpu(reader()?, &tok, rulestring, 0, 0))?
            }
        };
        let rule = Rule::parse(rulestring).context("bad rulestring")?;
        let cfg = life.config();
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
        println!("backend: {}", life.backend_name());
        let [dead, alive] = life.head();
        match variant {
            Variant::B => {
                life.set_prefix_b(&if fewshot {
                    fewshot_rules_prefix(&rule)
                } else {
                    rules_prefix(&rule)
                })?;
                println!(
                    "prefix: {} tokens, answer tokens: '0'={dead} '1'={alive}",
                    life.prefix_b_ids().len()
                );
            }
            Variant::A { chunk_cells } => {
                let mut prefix = variant_a::rules_prefix(&rule);
                if fewshot {
                    prefix.push_str(&variant_a::fewshot_examples());
                }
                block_on(life.set_prefix_a(&prefix))?;
                block_on(life.set_chunk_cells(chunk_cells))?;
                println!(
                    "prefix: {} tokens ({}), per-cell prompt: {} tokens, chunk: {chunk_cells} cells = {} tokens",
                    life.prefix_tokens_a(),
                    if fewshot { "few-shot" } else { "rules only" },
                    life.cell_tokens(),
                    chunk_cells * life.cell_tokens()
                );
                println!("answer tokens: '0'={dead} '1'={alive}");
            }
        }
        Ok(Runner {
            life,
            variant,
            fewshot,
        })
    }

    /// One generation: p(alive) at every cell, and the seconds spent in the
    /// forwards (every chunk's, for variant A).
    fn step(&mut self, grid: &Grid) -> Result<(Vec<f32>, f64)> {
        let n = grid.cells().len();
        match self.variant {
            Variant::B => {
                self.life.set_grid(grid.width(), grid.height());
                let start = Instant::now();
                let logits = block_on(self.life.logits_b(grid.cells().to_vec()))?;
                Ok((
                    logits.into_iter().map(p_alive).collect(),
                    start.elapsed().as_secs_f64(),
                ))
            }
            Variant::A { chunk_cells } => {
                let mut p = Vec::with_capacity(n);
                let mut secs = 0.0;
                for first in (0..n).step_by(chunk_cells) {
                    let cases: Vec<usize> = (first..(first + chunk_cells).min(n))
                        .map(|c| case_index(grid, c))
                        .collect();
                    let start = Instant::now();
                    let logits = block_on(self.life.logits_cases_a(&cases))?;
                    secs += start.elapsed().as_secs_f64();
                    p.extend(logits.into_iter().map(p_alive));
                }
                Ok((p, secs))
            }
        }
    }

    /// One chunk's forward (the grid's first `chunk_cells` cells) against
    /// the resident prefix, timed. Returns `(tokens, seconds)` where
    /// `tokens` counts the resident prefix plus the chunk's own tokens, the
    /// KV length the pass attends over.
    fn bench_chunk(&mut self, grid: &Grid, chunk_cells: usize) -> Result<(usize, f64)> {
        let cases: Vec<usize> = (0..chunk_cells).map(|c| case_index(grid, c)).collect();
        let t: usize = cases.iter().map(|&k| self.life.prompt_ids(k).len()).sum();
        let start = Instant::now();
        block_on(self.life.logits_cases_a(&cases))?;
        Ok((
            self.life.prefix_tokens_a() + t,
            start.elapsed().as_secs_f64(),
        ))
    }

    /// Extra lines for the summary header, describing this run's setup.
    fn header(&self) -> String {
        match self.variant {
            Variant::B => format!(
                "positions: bag (all grid tokens share one position id)\n\
                 mask: prefix (causal) + self + 8 neighbors\n\
                 head: sliced to the two answer tokens\n\
                 prefix: {} tokens ({})\n",
                self.life.prefix_b_ids().len(),
                if self.fewshot {
                    "rules + variant A's 6 worked examples"
                } else {
                    "rules only"
                },
            ),
            Variant::A { chunk_cells } => format!(
                "positions: restart at the prefix for every cell\n\
                 mask: block-diagonal — resident prefix + the cell's own prompt\n\
                 head: sliced to the two answer tokens (the prompt ends on a space)\n\
                 prefix: {} tokens ({}), resident in the KV cache\n\
                 chunk: {} cells\n",
                self.life.prefix_tokens_a(),
                if self.fewshot {
                    "rules + 6 worked examples"
                } else {
                    "rules only"
                },
                chunk_cells,
            ),
        }
    }
}

/// The reporting loop both variants share: N generations per seed, a markdown
/// table of per-generation scores, and four PGMs at generation 1.
#[allow(clippy::too_many_arguments)]
fn run_pictures(
    runner: &mut Runner,
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
        runner.header(),
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
            let (p, secs) = runner.step(&input)?;
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
                write_binary_pgm(
                    &out.join(format!("{name}-argmax.pgm")),
                    model_grid.cells(),
                    size,
                    size,
                    6,
                )?;
                write_binary_pgm(
                    &out.join(format!("{name}-true.pgm")),
                    truth.cells(),
                    size,
                    size,
                    6,
                )?;
                write_binary_pgm(
                    &out.join(format!("{name}-diff.pgm")),
                    &truth.diff(&model_grid),
                    size,
                    size,
                    6,
                )?;
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

/// Median of a small sample — the reported statistic for every bench row.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time one forward pass at several sizes, for both variants, and print the
/// two markdown tables that go into `docs/runs/`.
#[allow(clippy::too_many_arguments)]
fn bench_forward(
    gguf: &Path,
    tokenizer: &Path,
    rulestring: &str,
    sizes: &[usize],
    chunks: &[usize],
    reps: usize,
    density: f64,
    fewshot: bool,
) -> Result<()> {
    let mut runner = Runner::new(gguf, tokenizer, rulestring, Variant::B, fewshot)?;
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
        let t = runner.life.prefix_b_ids().len() + size * size;
        println!(
            "| {size}x{size} | {} | {t} | {m:.3} | {:.6} |",
            size * size,
            m / t as f64
        );
    }

    // Same model, now set up for variant A.
    let max_chunk = *chunks.iter().max().unwrap();
    let mut runner = Runner::new(
        gguf,
        tokenizer,
        rulestring,
        Variant::A {
            chunk_cells: max_chunk,
        },
        fewshot,
    )?;
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
    Ok(())
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
            let mut runner = Runner::new(&gguf, &tokenizer, &cli.rule, Variant::B, fewshot)?;
            run_pictures(
                &mut runner,
                title,
                &tag,
                &gguf,
                &rule,
                size,
                generations,
                &seeds,
                density,
                &out,
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
            let mut runner = Runner::new(
                &gguf,
                &tokenizer,
                &cli.rule,
                Variant::A { chunk_cells },
                fewshot,
            )?;
            println!(
                "{} chunks per generation",
                (size * size).div_ceil(chunk_cells)
            );
            run_pictures(
                &mut runner,
                title,
                &tag,
                &gguf,
                &rule,
                size,
                generations,
                &seeds,
                density,
                &out,
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
        } => bench_forward(
            &gguf, &tokenizer, &cli.rule, &sizes, &chunks, reps, density, fewshot,
        ),
        Command::Rescore { dir, tag, seed } => rescore(&dir, &tag, &seed),
    }
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
    anyhow::ensure!(
        (w, h) == (tw, th) && (w, h) == (aw, ah),
        "pgm dimension mismatch"
    );
    let truth = Grid::from_cells(w, h, true_cells.iter().map(|&v| (v > 0.5) as u8).collect());
    let model = Grid::from_cells(
        w,
        h,
        argmax_cells.iter().map(|&v| (v > 0.5) as u8).collect(),
    );

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
        println!(
            "    {:<20} n={:<5} correct={:<5} frac={:.4}",
            case.label(),
            c.count,
            c.correct,
            c.fraction()
        );
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
