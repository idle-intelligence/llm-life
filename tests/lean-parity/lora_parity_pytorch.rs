//! Same check as `lora_parity.rs`'s `lora_and_sliced_head_match_reference`,
//! against `reference/fixture_lora_pytorch.json` instead — a LoRA adapter
//! trained from scratch in PyTorch (llm-life's pytorch-training branch),
//! converted PEFT-safetensors -> LLMLIFE2, rather than the natively
//! Burn-trained published adapter the other fixture uses. Not committed:
//! ad hoc verification for the PyTorch migration, run once.

use lean::engine::Engine;
use lean::model::{build_rope_tables, forward_chunk_spec, ForwardSpec, GpuModel, KvCache};
use serde::Deserialize;

#[derive(Deserialize)]
struct Logits {
    dead: f32,
    alive: f32,
}

#[derive(Deserialize)]
struct Fixture {
    input_ids: Vec<u32>,
    dead_token_id: u32,
    alive_token_id: u32,
    base_logits: Logits,
    lora_logits: Logits,
}

const TOL: f32 = 2e-2;

#[test]
#[ignore = "needs LEAN_GGUF and a real LLMLIFE2 .bin on disk; never committed"]
fn pytorch_trained_lora_matches_reference() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let lora_path = std::env::var("LEAN_LORA_BIN").expect("set LEAN_LORA_BIN to run this test");

    let fixture_json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_lora_pytorch.json"))
        .expect("reading fixture_lora_pytorch.json (run gen_fixture_lora.py against the pytorch-trained adapter first)");
    let fixture: Fixture = serde_json::from_str(&fixture_json).expect("parsing fixture_lora_pytorch.json");

    let engine = Engine::new().expect("wgpu engine init");
    let mut model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let max_ctx = fixture.input_ids.len() as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let t = fixture.input_ids.len() as u32;
    let ids = [fixture.dead_token_id, fixture.alive_token_id];

    model.pool.reset();
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let hidden = pollster::block_on(forward_chunk_spec(&engine, &model, &mut cache, &fixture.input_ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
    let logits = pollster::block_on(model.lm_head_sliced(&engine, &hidden, t, &ids));
    let last = &logits[((t - 1) * 2) as usize..];
    eprintln!("[lora_parity_pytorch] base: dead={:.6} alive={:.6} (fixture dead={:.6} alive={:.6})", last[0], last[1], fixture.base_logits.dead, fixture.base_logits.alive);
    assert!((last[0] - fixture.base_logits.dead).abs() < TOL, "base dead logit mismatch: got {} want {}", last[0], fixture.base_logits.dead);
    assert!((last[1] - fixture.base_logits.alive).abs() < TOL, "base alive logit mismatch: got {} want {}", last[1], fixture.base_logits.alive);

    let lora_bytes = std::fs::read(&lora_path).expect("reading LEAN_LORA_BIN");
    model.apply_lora(&engine, &lora_bytes).expect("apply_lora");
    assert!(model.has_lora());

    model.pool.reset();
    let mut cache2 = KvCache::new(&engine, &model.config, max_ctx);
    let hidden2 = pollster::block_on(forward_chunk_spec(&engine, &model, &mut cache2, &fixture.input_ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
    let logits2 = pollster::block_on(model.lm_head_sliced(&engine, &hidden2, t, &ids));
    let last2 = &logits2[((t - 1) * 2) as usize..];
    eprintln!("[lora_parity_pytorch] +lora: dead={:.6} alive={:.6} (fixture dead={:.6} alive={:.6})", last2[0], last2[1], fixture.lora_logits.dead, fixture.lora_logits.alive);
    assert!((last2[0] - fixture.lora_logits.dead).abs() < TOL, "lora dead logit mismatch: got {} want {}", last2[0], fixture.lora_logits.dead);
    assert!((last2[1] - fixture.lora_logits.alive).abs() < TOL, "lora alive logit mismatch: got {} want {}", last2[1], fixture.lora_logits.alive);
}
