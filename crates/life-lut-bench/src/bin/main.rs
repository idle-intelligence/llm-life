use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use life::grid::Grid;
use life::rule::Rule;
use life_lut_bench::{cpu, gpu, lut, pack};
use serde::Serialize;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Bit-exact check of every implementation against `life::Grid::step`.
    Correctness,
    /// Timed benchmark across grid sizes.
    Bench {
        #[arg(long, value_delimiter = ',', default_value = "16,64,256,1024,4096,16384")]
        sizes: Vec<usize>,
        /// Extra large square grid to also measure for the LUT encoding (u32/cell, GPU; u8/cell, CPU).
        #[arg(long)]
        max_lut: Option<usize>,
        /// Extra large square grid to also measure for the bit-packed encoding.
        #[arg(long)]
        max_bitpack: Option<usize>,
        #[arg(long)]
        json: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        skip_cpu: bool,
        #[arg(long, default_value_t = false)]
        skip_gpu: bool,
    },
}

fn gens_for(size: usize) -> (u32, u32, u32) {
    // (warmup, timed generations, flush_every) — chosen to keep each cell
    // under ~1-2 minutes of wall time while giving a steady-state average;
    // not a per-device tuning, the same table is used on every machine.
    let timed = match size {
        s if s <= 64 => 20_000,
        s if s <= 256 => 10_000,
        s if s <= 1024 => 2_000,
        s if s <= 4096 => 200,
        s if s <= 16_384 => 20,
        _ => 5,
    };
    let warmup = (timed / 10).clamp(2, 200);
    let flush_every = flush_every_for(timed);
    (warmup, timed, flush_every)
}

fn flush_every_for(timed: u32) -> u32 {
    timed.clamp(1, 50)
}

#[derive(Serialize)]
struct Row {
    size: usize,
    encoding: &'static str,
    implementation: &'static str,
    generations: u32,
    seconds: f64,
    gens_per_sec: f64,
    cells_per_sec: f64,
    gb_per_sec: f64,
}

fn record(rows: &mut Vec<Row>, size: usize, encoding: &'static str, implementation: &'static str, generations: u32, elapsed: Duration, bytes_per_cell_roundtrip: f64) {
    let seconds = elapsed.as_secs_f64();
    let cells = (size * size) as f64;
    let gens_per_sec = generations as f64 / seconds;
    let cells_per_sec = cells * gens_per_sec;
    let gb_per_sec = cells_per_sec * bytes_per_cell_roundtrip / 1e9;
    println!(
        "{size:>7}x{size:<7} {encoding:<10} {implementation:<14} gens={generations:<6} {seconds:>10.4}s  {gens_per_sec:>12.2} gen/s  {cells_per_sec:>14.4e} cells/s  {gb_per_sec:>8.2} GB/s"
    );
    rows.push(Row {
        size,
        encoding,
        implementation,
        generations,
        seconds,
        gens_per_sec,
        cells_per_sec,
        gb_per_sec,
    });
}

fn correctness() -> anyhow::Result<()> {
    let rule = Rule::life();
    let table = lut::build_lut(&rule);
    let (birth_mask, survive_mask) = lut::birth_survive_masks(&rule);
    let engine = gpu::Engine::new()?;

    let mut all_ok = true;

    // Aligned size: every implementation (byte-LUT and both bit-packed
    // widths) runs here.
    for (w, h, seed, density, generations) in [(128usize, 64usize, 1u64, 0.35, 30usize), (128, 64, 2, 0.5, 30), (128, 64, 3, 0.15, 30)] {
        let mut reference = Grid::random(w, h, seed, density);
        let initial = reference.cells().to_vec();
        for _ in 0..generations {
            reference = reference.step(&rule);
        }
        let want = reference.cells().to_vec();

        let mut a = initial.clone();
        let mut b = vec![0u8; w * h];
        for _ in 0..generations {
            cpu::lut_step_scalar(&a, &mut b, w, h, &table);
            std::mem::swap(&mut a, &mut b);
        }
        all_ok &= check("cpu lut scalar", w, h, seed, &a, &want);

        let mut a = initial.clone();
        let mut b = vec![0u8; w * h];
        for _ in 0..generations {
            cpu::lut_step_rayon(&a, &mut b, w, h, &table);
            std::mem::swap(&mut a, &mut b);
        }
        all_ok &= check("cpu lut rayon", w, h, seed, &a, &want);

        let wpr64 = w / 64;
        let mut a = pack::pack_u64(&initial, w, h);
        let mut b = vec![0u64; wpr64 * h];
        for _ in 0..generations {
            cpu::bitpack_step_scalar(&a, &mut b, wpr64, h, birth_mask, survive_mask);
            std::mem::swap(&mut a, &mut b);
        }
        all_ok &= check("cpu bitpack64 scalar", w, h, seed, &pack::unpack_u64(&a, w, h), &want);

        let mut a = pack::pack_u64(&initial, w, h);
        let mut b = vec![0u64; wpr64 * h];
        for _ in 0..generations {
            cpu::bitpack_step_rayon(&a, &mut b, wpr64, h, birth_mask, survive_mask);
            std::mem::swap(&mut a, &mut b);
        }
        all_ok &= check("cpu bitpack64 rayon", w, h, seed, &pack::unpack_u64(&a, w, h), &want);

        let initial_u32: Vec<u32> = initial.iter().map(|&c| c as u32).collect();
        let (out, _) = engine.run_lut_timed(w as u32, h as u32, &table, &initial_u32, 0, generations as u32, 1000);
        let got: Vec<u8> = out.iter().map(|&c| c as u8).collect();
        all_ok &= check("gpu lut", w, h, seed, &got, &want);

        let wpr32 = w / 32;
        let packed = pack::pack_u32(&initial, w, h);
        let (out, _) = engine.run_bitpack_timed(wpr32 as u32, h as u32, birth_mask, survive_mask, &packed, 0, generations as u32, 1000);
        let got = pack::unpack_u32(&out, w, h);
        all_ok &= check("gpu bitpack32", w, h, seed, &got, &want);
    }

    // Non-word-aligned width: byte-LUT paths only (CPU + GPU), the
    // bit-packed kernels require width % 32/64 == 0 and are not exercised
    // here.
    for (w, h, seed, density, generations) in [(100usize, 50usize, 7u64, 0.4, 15usize)] {
        let mut reference = Grid::random(w, h, seed, density);
        let initial = reference.cells().to_vec();
        for _ in 0..generations {
            reference = reference.step(&rule);
        }
        let want = reference.cells().to_vec();

        let mut a = initial.clone();
        let mut b = vec![0u8; w * h];
        for _ in 0..generations {
            cpu::lut_step_scalar(&a, &mut b, w, h, &table);
            std::mem::swap(&mut a, &mut b);
        }
        all_ok &= check("cpu lut scalar (non-aligned)", w, h, seed, &a, &want);

        let initial_u32: Vec<u32> = initial.iter().map(|&c| c as u32).collect();
        let (out, _) = engine.run_lut_timed(w as u32, h as u32, &table, &initial_u32, 0, generations as u32, 1000);
        let got: Vec<u8> = out.iter().map(|&c| c as u8).collect();
        all_ok &= check("gpu lut (non-aligned)", w, h, seed, &got, &want);
    }

    // A non-Conway rule, to prove the bit-packed rule masks are not
    // hardcoded to B3/S23: HighLife, B36/S23.
    {
        let rule2 = Rule::parse("B36/S23").expect("literal rulestring");
        let table2 = lut::build_lut(&rule2);
        let (bm2, sm2) = lut::birth_survive_masks(&rule2);
        let (w, h, seed, density, generations) = (128usize, 64usize, 9u64, 0.3, 25usize);
        let mut reference = Grid::random(w, h, seed, density);
        let initial = reference.cells().to_vec();
        for _ in 0..generations {
            reference = reference.step(&rule2);
        }
        let want = reference.cells().to_vec();

        let wpr64 = w / 64;
        let mut a = pack::pack_u64(&initial, w, h);
        let mut b = vec![0u64; wpr64 * h];
        for _ in 0..generations {
            cpu::bitpack_step_scalar(&a, &mut b, wpr64, h, bm2, sm2);
            std::mem::swap(&mut a, &mut b);
        }
        all_ok &= check("cpu bitpack64 (B36/S23)", w, h, seed, &pack::unpack_u64(&a, w, h), &want);

        let wpr32 = w / 32;
        let packed = pack::pack_u32(&initial, w, h);
        let (out, _) = engine.run_bitpack_timed(wpr32 as u32, h as u32, bm2, sm2, &packed, 0, generations as u32, 1000);
        let got = pack::unpack_u32(&out, w, h);
        all_ok &= check("gpu bitpack32 (B36/S23)", w, h, seed, &got, &want);

        let initial_u32: Vec<u32> = initial.iter().map(|&c| c as u32).collect();
        let (out, _) = engine.run_lut_timed(w as u32, h as u32, &table2, &initial_u32, 0, generations as u32, 1000);
        let got: Vec<u8> = out.iter().map(|&c| c as u8).collect();
        all_ok &= check("gpu lut (B36/S23)", w, h, seed, &got, &want);
    }

    if all_ok {
        println!("\nALL CORRECTNESS CHECKS PASSED");
        Ok(())
    } else {
        anyhow::bail!("one or more correctness checks FAILED");
    }
}

fn check(name: &str, w: usize, h: usize, seed: u64, got: &[u8], want: &[u8]) -> bool {
    if got == want {
        println!("PASS  {name:<28} {w}x{h} seed={seed}");
        true
    } else {
        let diffs = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        println!("FAIL  {name:<28} {w}x{h} seed={seed}  {diffs} of {} cells differ", got.len());
        false
    }
}

fn bench(sizes: Vec<usize>, max_lut: Option<usize>, max_bitpack: Option<usize>, json: Option<PathBuf>, skip_cpu: bool, skip_gpu: bool) -> anyhow::Result<()> {
    let rule = Rule::life();
    let table = lut::build_lut(&rule);
    let (birth_mask, survive_mask) = lut::birth_survive_masks(&rule);

    let engine = if skip_gpu { None } else { Some(gpu::Engine::new()?) };
    if let Some(e) = &engine {
        println!("GPU limits: max_storage_buffer_binding_size = {} bytes", e.max_storage_buffer_binding_size());
    }

    let mut rows = Vec::new();

    let mut lut_sizes = sizes.clone();
    if let Some(m) = max_lut {
        lut_sizes.push(m);
    }
    for &size in &lut_sizes {
        let grid = Grid::random(size, size, 42, 0.35);
        let initial: Vec<u32> = grid.cells().iter().map(|&c| c as u32).collect();
        let initial_u8 = grid.cells().to_vec();
        let (warmup, gens, flush) = gens_for(size);

        if !skip_cpu {
            let mut a = initial_u8.clone();
            let mut b = vec![0u8; size * size];
            for _ in 0..warmup {
                cpu::lut_step_scalar(&a, &mut b, size, size, &table);
                std::mem::swap(&mut a, &mut b);
            }
            let start = Instant::now();
            for _ in 0..gens {
                cpu::lut_step_scalar(&a, &mut b, size, size, &table);
                std::mem::swap(&mut a, &mut b);
            }
            record(&mut rows, size, "lut", "cpu-1thread", gens, start.elapsed(), 2.0);

            let mut a = initial_u8.clone();
            let mut b = vec![0u8; size * size];
            for _ in 0..warmup {
                cpu::lut_step_rayon(&a, &mut b, size, size, &table);
                std::mem::swap(&mut a, &mut b);
            }
            let start = Instant::now();
            for _ in 0..gens {
                cpu::lut_step_rayon(&a, &mut b, size, size, &table);
                std::mem::swap(&mut a, &mut b);
            }
            record(&mut rows, size, "lut", "cpu-rayon", gens, start.elapsed(), 2.0);
        }

        if let Some(e) = &engine {
            let (_, elapsed) = e.run_lut_timed(size as u32, size as u32, &table, &initial, warmup, gens, flush);
            record(&mut rows, size, "lut", "gpu", gens, elapsed, 8.0);
        }
    }

    let mut bitpack_sizes: Vec<usize> = sizes.iter().copied().filter(|&s| s % 32 == 0).collect();
    if let Some(m) = max_bitpack {
        bitpack_sizes.push(m);
    }
    for &size in &bitpack_sizes {
        let grid = Grid::random(size, size, 42, 0.35);
        let initial_u8 = grid.cells().to_vec();
        let (warmup, gens, flush) = gens_for(size);

        if !skip_cpu && size % 64 == 0 {
            let wpr = size / 64;
            let mut a = pack::pack_u64(&initial_u8, size, size);
            let mut b = vec![0u64; wpr * size];
            for _ in 0..warmup {
                cpu::bitpack_step_scalar(&a, &mut b, wpr, size, birth_mask, survive_mask);
                std::mem::swap(&mut a, &mut b);
            }
            let start = Instant::now();
            for _ in 0..gens {
                cpu::bitpack_step_scalar(&a, &mut b, wpr, size, birth_mask, survive_mask);
                std::mem::swap(&mut a, &mut b);
            }
            record(&mut rows, size, "bitpack", "cpu-1thread", gens, start.elapsed(), 0.25);

            let mut a = pack::pack_u64(&initial_u8, size, size);
            let mut b = vec![0u64; wpr * size];
            for _ in 0..warmup {
                cpu::bitpack_step_rayon(&a, &mut b, wpr, size, birth_mask, survive_mask);
                std::mem::swap(&mut a, &mut b);
            }
            let start = Instant::now();
            for _ in 0..gens {
                cpu::bitpack_step_rayon(&a, &mut b, wpr, size, birth_mask, survive_mask);
                std::mem::swap(&mut a, &mut b);
            }
            record(&mut rows, size, "bitpack", "cpu-rayon", gens, start.elapsed(), 0.25);
        } else if !skip_cpu {
            println!("{size:>7}x{size:<7} bitpack    cpu-*          N/A (width % 64 != 0)");
        }

        if let Some(e) = &engine {
            let wpr = size / 32;
            let packed = pack::pack_u32(&initial_u8, size, size);
            let (_, elapsed) = e.run_bitpack_timed(wpr as u32, size as u32, birth_mask, survive_mask, &packed, warmup, gens, flush);
            record(&mut rows, size, "bitpack", "gpu", gens, elapsed, 0.25);
        }
    }

    if let Some(path) = json {
        std::fs::write(&path, serde_json::to_string_pretty(&rows)?)?;
        println!("\nwrote {}", path.display());
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Correctness => correctness(),
        Cmd::Bench { sizes, max_lut, max_bitpack, json, skip_cpu, skip_gpu } => bench(sizes, max_lut, max_bitpack, json, skip_cpu, skip_gpu),
    }
}
