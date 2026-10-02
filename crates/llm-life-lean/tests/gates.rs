//! Gates for the lean port of the language-model methods, natively:
//!
//! (i)  per cell, the 512 (8 neighbours, self) cases: base model with the
//!      rules prefix 215/512, the `a-norules-300` adapter 512/512 (the
//!      numbers of docs/runs/2026-09-20-eval-a-base-rules.md and
//!      2026-09-20-runtime-lora.md), all 512 at 64 cells per packed forward
//!      (the batched row) and every 8th case at one cell per forward (the
//!      per-cell rows; the full 512 at one cell per forward is ~3 min per
//!      configuration natively, so it is sampled to keep a GPU run short);
//! (ii) token-exact against HF transformers + PEFT
//!      (`tests/fixtures/hf_peft_reference.json`, written by
//!      tools/parity/hf_peft_reference.py): the same answer token at every
//!      answer position, greedy, for the four per-cell configurations and
//!      the whole-grid grids (base and adapter, 16x16 and 32x32).
//!
//! ```sh
//! LLM_LIFE_GGUF=.../qwen2.5-0.5b-instruct-q4_0.gguf \
//! LLM_LIFE_TOKENIZER=.../Qwen2.5-0.5B-Instruct/tokenizer.json \
//! LLM_LIFE_ADAPTERS=<dir with lora-*.bin> \
//! cargo test -p llm-life-lean --release --test gates -- --ignored --nocapture --test-threads=1 [name]
//! ```

use std::io::BufReader;

use llm_life_lean::lean::engine::Engine;
use llm_life_lean::{case_cells, variant_a, LifeLean, CHUNK_CELLS};
use life::Rule;
use serde::Deserialize;

#[derive(Deserialize)]
struct PerCell {
    name: String,
    adapter: Option<String>,
    prefix_ids: Vec<u32>,
    prompt_ids: Vec<Vec<u32>>,
    logits: Vec<[f32; 2]>,
}

#[derive(Deserialize)]
struct WholeGrid {
    name: String,
    adapter: Option<String>,
    width: usize,
    height: usize,
    cells: Vec<u8>,
    prefix_ids: Vec<u32>,
    logits: Vec<[f32; 2]>,
}

#[derive(Deserialize)]
struct Fixture {
    per_cell: Vec<PerCell>,
    whole_grid: Vec<WholeGrid>,
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("set {k} to run this test"))
}

fn load() -> LifeLean {
    let gguf = std::fs::File::open(env("LLM_LIFE_GGUF")).expect("open gguf");
    let tok = std::fs::read(env("LLM_LIFE_TOKENIZER")).expect("read tokenizer.json");
    pollster::block_on(LifeLean::load(Engine::new().expect("wgpu device"), BufReader::new(gguf), &tok, "B3/S23", 16, 16)).expect("load")
}

fn adapter(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/{name}", env("LLM_LIFE_ADAPTERS"))).expect("read adapter")
}

fn truth(k: usize) -> bool {
    let (nb, me) = case_cells(k);
    Rule::life().next(me != 0, nb.iter().filter(|&&b| b != 0).count())
}

/// Compare engine logits to the reference: (answers that differ, with the
/// reference's own margin; max |logit diff|).
fn compare(got: &[[f32; 2]], want: &[[f32; 2]]) -> (Vec<(usize, f32)>, f32) {
    assert_eq!(got.len(), want.len());
    let mut diffs = Vec::new();
    let mut max = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if (g[1] > g[0]) != (w[1] > w[0]) {
            diffs.push((i, (w[1] - w[0]).abs()));
        }
        max = max.max((g[0] - w[0]).abs()).max((g[1] - w[1]).abs());
    }
    (diffs, max)
}

fn fixture() -> Fixture {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/hf_peft_reference.json")).unwrap()).unwrap()
}

/// One per-cell configuration of the fixture, gates (i) and (ii).
fn per_cell(name: &str) {
    let fixture = fixture();
    let cfg = fixture.per_cell.iter().find(|c| c.name == name).expect("configuration in fixture");
    let mut life = load();
    match (cfg.name.as_str(), cfg.adapter.as_deref()) {
        ("base-fewshot", None) => {}
        ("base-rules", None) => pollster::block_on(life.set_prefix_a(&variant_a::rules_prefix(&Rule::life()))).unwrap(),
        (_, Some(a)) => pollster::block_on(life.load_adapter(&adapter(a), a.contains("norules"))).unwrap(),
        (n, None) => panic!("unknown base configuration {n}"),
    }
    assert_eq!(life.prefix_a_ids(), &cfg.prefix_ids[..], "{}: prefix tokens differ from the reference", cfg.name);
    for k in 0..512 {
        assert_eq!(life.prompt_ids(k), &cfg.prompt_ids[k][..], "{}: case {k} tokens differ", cfg.name);
    }

    let cases: Vec<usize> = (0..512).collect();
    let mut batched = Vec::new();
    for chunk in cases.chunks(CHUNK_CELLS) {
        batched.extend(pollster::block_on(life.logits_cases_a(chunk)).unwrap());
    }
    let sampled: Vec<usize> = (0..512).step_by(8).collect();
    let per_call: Vec<[f32; 2]> = sampled.iter().map(|&k| pollster::block_on(life.logits_cases_a(&[k])).unwrap()[0]).collect();
    let want_sampled: Vec<[f32; 2]> = sampled.iter().map(|&k| cfg.logits[k]).collect();

    let correct = batched.iter().enumerate().filter(|(k, l)| (l[1] > l[0]) == truth(*k)).count();
    let mut failed = false;
    for (mode, got, want, ids) in [
        ("64 cells per forward", &batched, &cfg.logits, &cases),
        ("one cell per forward", &per_call, &want_sampled, &sampled),
    ] {
        let (diffs, max) = compare(got, want);
        println!(
            "[gate] per-cell {:<14} {:<21} {:>3} cases, answers vs HF+PEFT: {} differ, max |logit diff| {max:.3e}",
            cfg.name,
            mode,
            got.len(),
            diffs.len()
        );
        for (i, margin) in &diffs {
            println!("[gate]   case {}: HF margin {margin:.4}", ids[*i]);
        }
        failed |= !diffs.is_empty();
    }
    println!("[gate] per-cell {:<14} {correct}/512 correct", cfg.name);
    match cfg.name.as_str() {
        "base-rules" => assert_eq!(correct, 215, "gate (i): base model, rules prefix"),
        "a-norules-300" => assert_eq!(correct, 512, "gate (i): a-norules-300 adapter"),
        _ => {}
    }
    assert!(!failed, "gate (ii): {} answers differ from HF+PEFT", cfg.name);
}

#[test]
#[ignore = "needs the GGUF, tokenizer and adapters on disk"]
fn per_cell_base_fewshot() {
    per_cell("base-fewshot");
}

#[test]
#[ignore = "needs the GGUF, tokenizer and adapters on disk"]
fn per_cell_base_rules() {
    per_cell("base-rules");
}

#[test]
#[ignore = "needs the GGUF, tokenizer and adapters on disk"]
fn per_cell_a_norules() {
    per_cell("a-norules-300");
}

#[test]
#[ignore = "needs the GGUF, tokenizer and adapters on disk"]
fn per_cell_a_rules() {
    per_cell("a-rules-300");
}

#[test]
#[ignore = "needs the GGUF, tokenizer and adapters on disk"]
fn whole_grid() {
    let fixture = fixture();
    let mut life = load();
    let mut failures = Vec::new();
    for g in &fixture.whole_grid {
        match g.adapter.as_deref() {
            None => life.clear_adapter(),
            Some(a) => pollster::block_on(life.load_adapter(&adapter(a), false)).unwrap(),
        }
        life.set_grid(g.width, g.height);
        assert_eq!(life.prefix_b_ids(), &g.prefix_ids[..], "{}: prefix tokens differ", g.name);
        let got = pollster::block_on(life.logits_b(g.cells.clone())).unwrap();
        let grid = life::Grid::from_cells(g.width, g.height, g.cells.clone());
        let next = grid.step(&Rule::life());
        let correct = got.iter().zip(next.cells()).filter(|(l, &t)| (l[1] > l[0]) == (t != 0)).count();
        let (diffs, max) = compare(&got, &g.logits);
        println!(
            "[gate] whole grid {:<28} {correct}/{} correct, answers vs HF+PEFT: {} differ, max |logit diff| {max:.3e}",
            g.name,
            g.width * g.height,
            diffs.len()
        );
        for (c, margin) in &diffs {
            println!("[gate]   cell {c}: HF margin {margin:.4}");
        }
        if !diffs.is_empty() {
            failures.push(g.name.clone());
        }
    }
    assert!(failures.is_empty(), "gate (ii): answers differ from HF+PEFT in {failures:?}");
}
