//! One-shot bridge for the Burn -> PyTorch training migration.
//!
//! Dumps the published checkpoints to safetensors (tiny models: raw weight
//! names; LoRA adapters: PEFT layout) plus fixed-input oracle logits, so the
//! PyTorch port can be checked against the exact weights and math the Rust
//! trainer produced, not just against the README's headline numbers.
//!
//! Not part of the ongoing build. Usage:
//!
//! ```text
//! cargo run --release -p llm-life --bin export --no-default-features \
//!   --features native,cpu,export -- \
//!   --stencil-life <dir with bert-d16-L1.bin etc.> \
//!   --lora <dir with lora-*.bin> \
//!   --gguf <qwen2.5-0.5b-instruct-q4_0.gguf> \
//!   --tokenizer <tokenizer.json> \
//!   --out <output dir>
//! ```

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Int, Tensor, TensorData};
use life::{Grid, Rule};
use llm_life::bert::model::{BertConfig, BertOfLife, MlpConfig, MlpOfLife};
use llm_life::train::run_a::{all_cases, full_sequence, prompt_a, tokenize_case};
use llm_life::train::load_train_model;
use llm_life::variant_a::pack_chunk;
use llm_life::variant_b::{pack, p_alive, rules_prefix};
use llm_life::vector::model::{stencil_neighbors, Mlp2Config, Mlp2OfLife, StencilConfig, StencilOfLife};
use llm_wasm::tokenizer::Tokenizer;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

type CB = NdArray;

fn arg(name: &str) -> Option<String> {
    let flag = format!("--{name}");
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == &flag).and_then(|i| args.get(i + 1)).cloned()
}

/// A named f32 tensor pending safetensors serialization. Data must outlive
/// the `TensorView`s built from it, hence the two-pass collect-then-write.
struct Entry {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn t2(t: Tensor<CB, 2>) -> (Vec<usize>, Vec<f32>) {
    let dims = t.dims().to_vec();
    (dims, t.into_data().into_vec::<f32>().unwrap())
}
fn t1(t: Tensor<CB, 1>) -> (Vec<usize>, Vec<f32>) {
    let dims = t.dims().to_vec();
    (dims, t.into_data().into_vec::<f32>().unwrap())
}

fn write_safetensors(path: &Path, entries: Vec<Entry>) -> Result<()> {
    let views: Vec<(String, TensorView)> = entries
        .iter()
        .map(|e| {
            let view = TensorView::new(safetensors::Dtype::F32, e.shape.clone(), bytemuck(&e.data)).unwrap();
            (e.name.clone(), view)
        })
        .collect();
    let map: HashMap<String, TensorView> = views.into_iter().collect();
    safetensors::serialize_to_file(&map, &None, path)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn bytemuck(v: &[f32]) -> &[u8] {
    // f32 -> bytes, little-endian native (safetensors always reads LE).
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn export_bert(dir: &Path, out: &Path) -> Result<()> {
    let path = dir.join("bert-d16-L1.bin");
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let device = Default::default();
    let cfg = BertConfig::small(16, 1, 1);
    let model: BertOfLife<CB> = cfg.init(&device);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(bytes, &device)?;
    let model = model.load_record(record);

    let mut entries = Vec::new();
    let (s, d) = t2(model.tok_emb.weight.val());
    entries.push(Entry { name: "tok_emb.weight".into(), shape: s, data: d });
    let (s, d) = t2(model.pos_emb.weight.val());
    entries.push(Entry { name: "pos_emb.weight".into(), shape: s, data: d });
    let layer = &model.encoder.layers[0];
    for (name, lin) in [
        ("mha.query", &layer.mha.query),
        ("mha.key", &layer.mha.key),
        ("mha.value", &layer.mha.value),
        ("mha.output", &layer.mha.output),
        ("pwff.linear_inner", &layer.pwff.linear_inner),
        ("pwff.linear_outer", &layer.pwff.linear_outer),
    ] {
        let (s, d) = t2(lin.weight.val());
        entries.push(Entry { name: format!("{name}.weight"), shape: s, data: d });
        if let Some(b) = &lin.bias {
            let (s, d) = t1(b.val());
            entries.push(Entry { name: format!("{name}.bias"), shape: s, data: d });
        }
    }
    for (name, ln) in [("norm_1", &layer.norm_1), ("norm_2", &layer.norm_2)] {
        let (s, d) = t1(ln.gamma.val());
        entries.push(Entry { name: format!("{name}.gamma"), shape: s, data: d });
        if let Some(b) = &ln.beta {
            let (s, d) = t1(b.val());
            entries.push(Entry { name: format!("{name}.beta"), shape: s, data: d });
        }
    }
    let (s, d) = t2(model.head.weight.val());
    entries.push(Entry { name: "head.weight".into(), shape: s, data: d });
    if let Some(b) = &model.head.bias {
        let (s, d) = t1(b.val());
        entries.push(Entry { name: "head.bias".into(), shape: s, data: d });
    }

    // Oracle: logits for all 512 exhaustive neighbourhood cases.
    let rule = Rule::life();
    let cases = llm_life::bert::data::all_cases(&rule);
    let n = cases.len();
    let mut toks = Vec::with_capacity(n * 9);
    for (c, _) in &cases {
        toks.extend(c.iter().map(|&b| b as i32));
    }
    let t: Tensor<CB, 2, Int> = Tensor::from_data(TensorData::new(toks, [n, 9]), &device);
    let logits = model.forward(t);
    let (ls, ld) = t2(logits);
    write_oracle(&out.join("bert-oracle.json"), &ls, &ld, &cases.iter().map(|(_, y)| *y).collect::<Vec<_>>())?;

    write_safetensors(&out.join("bert-d16-L1.safetensors"), entries)
}

fn export_mlp(dir: &Path, out: &Path) -> Result<()> {
    // MLP baseline shipped alongside BERT (1 hidden layer, README's older
    // "9 numbers to centre" row before the 2-layer lr-fix rerun). Not
    // referenced by stencil-life's config.json (mlp2 is), included only if
    // the .bin is present.
    let path = dir.join("mlp-32.bin");
    if !path.exists() {
        return Ok(());
    }
    let bytes = std::fs::read(&path)?;
    let device = Default::default();
    let cfg = MlpConfig { hidden: 32 };
    let model: MlpOfLife<CB> = cfg.init(&device);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(bytes, &device)?;
    let model = model.load_record(record);
    let mut entries = Vec::new();
    for (name, lin) in [("fc1", &model.fc1), ("fc2", &model.fc2)] {
        let (s, d) = t2(lin.weight.val());
        entries.push(Entry { name: format!("{name}.weight"), shape: s, data: d });
        if let Some(b) = &lin.bias {
            let (s, d) = t1(b.val());
            entries.push(Entry { name: format!("{name}.bias"), shape: s, data: d });
        }
    }
    write_safetensors(&out.join("mlp-32.safetensors"), entries)
}

fn export_mlp2(dir: &Path, out: &Path) -> Result<()> {
    let path = dir.join("mlp2-32-lrfix.bin");
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let device = Default::default();
    let cfg = Mlp2Config { hidden: 32 };
    let model: Mlp2OfLife<CB> = cfg.init(&device);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(bytes, &device)?;
    let model = model.load_record(record);
    let mut entries = Vec::new();
    for (name, lin) in [("fc1", &model.fc1), ("fc2", &model.fc2), ("fc3", &model.fc3)] {
        let (s, d) = t2(lin.weight.val());
        entries.push(Entry { name: format!("{name}.weight"), shape: s, data: d });
        if let Some(b) = &lin.bias {
            let (s, d) = t1(b.val());
            entries.push(Entry { name: format!("{name}.bias"), shape: s, data: d });
        }
    }

    let rule = Rule::life();
    let cases = llm_life::bert::data::all_cases(&rule);
    let n = cases.len();
    let mut bits = Vec::with_capacity(n * 9);
    for (c, _) in &cases {
        bits.extend(c.iter().map(|&b| b as f32));
    }
    let t: Tensor<CB, 2> = Tensor::from_data(TensorData::new(bits, [n, 9]), &device);
    let logits = model.forward(t);
    let (ls, ld) = t2(logits);
    write_oracle(&out.join("mlp2-oracle.json"), &ls, &ld, &cases.iter().map(|(_, y)| *y).collect::<Vec<_>>())?;

    write_safetensors(&out.join("mlp2-32-lrfix.safetensors"), entries)
}

fn export_stencil(dir: &Path, out: &Path) -> Result<()> {
    let path = dir.join("stencil-d16-L1.bin");
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let device = Default::default();
    let cfg = StencilConfig::new(16, 1, 1);
    let model: StencilOfLife<CB> = cfg.init(&device);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(bytes, &device)?;
    let model = model.load_record(record);

    let mut entries = Vec::new();
    let (s, d) = t2(model.embed.weight.val());
    entries.push(Entry { name: "embed.weight".into(), shape: s, data: d });
    if let Some(b) = &model.embed.bias {
        let (s, d) = t1(b.val());
        entries.push(Entry { name: "embed.bias".into(), shape: s, data: d });
    }
    for (i, blk) in model.blocks.iter().enumerate() {
        for (name, lin) in [
            ("q", &blk.q),
            ("k", &blk.k),
            ("v", &blk.v),
            ("proj", &blk.proj),
            ("ff1", &blk.ff1),
            ("ff2", &blk.ff2),
        ] {
            let (s, d) = t2(lin.weight.val());
            entries.push(Entry { name: format!("blocks.{i}.{name}.weight"), shape: s, data: d });
            if let Some(b) = &lin.bias {
                let (s, d) = t1(b.val());
                entries.push(Entry { name: format!("blocks.{i}.{name}.bias"), shape: s, data: d });
            }
        }
        for (name, ln) in [("norm1", &blk.norm1), ("norm2", &blk.norm2)] {
            let (s, d) = t1(ln.gamma.val());
            entries.push(Entry { name: format!("blocks.{i}.{name}.gamma"), shape: s, data: d });
            if let Some(b) = &ln.beta {
                let (s, d) = t1(b.val());
                entries.push(Entry { name: format!("blocks.{i}.{name}.beta"), shape: s, data: d });
            }
        }
    }
    let (s, d) = t2(model.head.weight.val());
    entries.push(Entry { name: "head.weight".into(), shape: s, data: d });
    if let Some(b) = &model.head.bias {
        let (s, d) = t1(b.val());
        entries.push(Entry { name: "head.bias".into(), shape: s, data: d });
    }

    // Oracle: a fixed 16x16 grid (seed 1_000_000, density 0.28 — the exact
    // held-out convention `train::data::held_out` uses), whole-grid logits.
    let rule = Rule::life();
    let g = Grid::random(16, 16, 1_000_000, 0.28);
    let n = g.width() * g.height();
    let cells: Vec<f32> = g.cells().iter().map(|&c| c as f32).collect();
    let cells_t: Tensor<CB, 2> = Tensor::from_data(TensorData::new(cells, [1, n]), &device);
    let neighbors = stencil_neighbors::<CB>(16, 16, &device);
    let logits = model.forward(cells_t, neighbors);
    let data = logits.into_data().into_vec::<f32>().unwrap();
    let truth = g.step(&rule).cells().to_vec();
    write_oracle_1d(&out.join("stencil-oracle.json"), &data, &truth)?;

    write_safetensors(&out.join("stencil-d16-L1.safetensors"), entries)
}

fn write_oracle(path: &Path, shape: &[usize], logits: &[f32], targets: &[u8]) -> Result<()> {
    let json = serde_json::json!({ "shape": shape, "logits": logits, "targets": targets });
    std::fs::write(path, serde_json::to_string(&json)?)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn write_oracle_1d(path: &Path, logits: &[f32], targets: &[u8]) -> Result<()> {
    let json = serde_json::json!({ "logits": logits, "targets": targets });
    std::fs::write(path, serde_json::to_string(&json)?)?;
    println!("wrote {}", path.display());
    Ok(())
}

/// One LoRA `.bin` file (`LLMLIFE2`) -> PEFT-layout safetensors: `a` is
/// stored `[in, rank]` here and PEFT's `lora_A.weight` is `[rank, in]`
/// (transpose), `b` is `[rank, out]` here vs PEFT's `lora_B.weight`
/// `[out, rank]` (transpose). The delta is the same either way:
/// `(x @ a) @ b * scale` here equals `x @ lora_A^T @ lora_B^T * scale` there.
fn export_lora(dir: &Path, name: &str, out: &Path) -> Result<()> {
    let path = dir.join(name);
    if !path.exists() {
        println!("skip {} (not found)", path.display());
        return Ok(());
    }
    let device = Default::default();
    let (spec, params) = llm_life::train::lora_io::load::<CB>(&path, &device)?;
    anyhow::ensure!(!spec.mlp, "gate/up adapters not handled by this exporter");
    let per_layer = 8; // q.a,q.b,k.a,k.b,v.a,v.b,o.a,o.b
    anyhow::ensure!(params.len() % per_layer == 0, "unexpected LoRA matrix count {}", params.len());
    let n_layers = params.len() / per_layer;

    let mut entries = Vec::new();
    let names = ["q_proj", "k_proj", "v_proj", "o_proj"];
    for layer in 0..n_layers {
        for (m, proj) in names.iter().enumerate() {
            let a = params[layer * per_layer + m * 2].clone();
            let b = params[layer * per_layer + m * 2 + 1].clone();
            // transpose a: [in, rank] -> [rank, in]
            let a_t = a.transpose();
            let (s, d) = t2(a_t);
            entries.push(Entry {
                name: format!("base_model.model.model.layers.{layer}.self_attn.{proj}.lora_A.weight"),
                shape: s,
                data: d,
            });
            // transpose b: [rank, out] -> [out, rank]
            let b_t = b.transpose();
            let (s, d) = t2(b_t);
            entries.push(Entry {
                name: format!("base_model.model.model.layers.{layer}.self_attn.{proj}.lora_B.weight"),
                shape: s,
                data: d,
            });
        }
    }

    let stem = name.trim_end_matches(".bin");
    write_safetensors(&out.join(format!("{stem}.safetensors")), entries)?;

    let cfg = serde_json::json!({
        "r": spec.rank,
        "lora_alpha": spec.alpha,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    });
    std::fs::write(out.join(format!("{stem}.adapter_config.json")), serde_json::to_string_pretty(&cfg)?)?;
    Ok(())
}

/// Oracle for one LoRA adapter's variant-B whole-grid forward: the exact
/// `TrainModel` math (RMSNorm, RoPE, GQA, sliced head) this repo's Burn
/// trainer runs, on CPU (`NdArray`), on the published Q4_0 GGUF dequantized
/// the same way training dequantized it — so a PyTorch replica loading the
/// same GGUF through `transformers`' GGUF loader has something exact to
/// match, not just an accuracy claim.
fn export_lora_forward_oracle(
    dir: &Path,
    name: &str,
    gguf: &Path,
    tokenizer: &Path,
    grid_size: usize,
    out: &Path,
) -> Result<()> {
    let path = dir.join(name);
    if !path.exists() {
        println!("skip forward oracle for {} (not found)", path.display());
        return Ok(());
    }
    let device = Default::default();
    let (spec, params) = llm_life::train::lora_io::load::<CB>(&path, &device)?;

    let tok = Tokenizer::from_json(&std::fs::read(tokenizer)?)?;
    let rule = Rule::life();
    let prefix = tok.encode(&rules_prefix(&rule), false)?;
    let dead = tok.encode("0", false)?[0];
    let alive = tok.encode("1", false)?[0];

    let mut model = load_train_model::<CB>(gguf, &[dead, alive], Some(spec), &device)?;
    anyhow::ensure!(model.lora_params().len() == params.len(), "LoRA/model shape mismatch");
    model.set_lora_params(params);

    let g = Grid::random(grid_size, grid_size, 1_000_000, 0.28);
    let packed = pack(&g, &prefix, dead, alive);
    let t = packed.tokens.len();
    let mask_data: Vec<bool> = packed.allowed.iter().map(|&a| !a).collect();
    let mask: Tensor<CB, 2, burn::tensor::Bool> = Tensor::from_data(TensorData::new(mask_data, [t, t]), &device);
    let logits = model.forward(&packed.tokens, &packed.positions, &mask)?;
    let data = logits.into_data().into_vec::<f32>().unwrap();
    let pa = p_alive(&data, packed.grid_start, grid_size * grid_size);
    let truth = g.step(&rule).cells().to_vec();

    let stem = name.trim_end_matches(".bin");
    let json = serde_json::json!({
        "grid_size": grid_size,
        "seed": 1_000_000,
        "density": 0.28,
        "prefix_len": prefix.len(),
        "dead_token": dead,
        "alive_token": alive,
        "p_alive": pa,
        "truth": truth,
        "logits_all": data,
        "grid_start": packed.grid_start,
    });
    let out_path = out.join(format!("{stem}-forward-oracle.json"));
    std::fs::write(&out_path, serde_json::to_string(&json)?)?;
    println!("wrote {}", out_path.display());
    Ok(())
}

/// Oracle for one variant-A LoRA adapter's per-cell forward: the same
/// `TrainModel` math as `export_lora_forward_oracle`, but through the
/// block-diagonal causal packing `run_a::full_sequence` builds (one chunk of
/// 64 cells, the default `--eval-cases`/`--batch` chunk size), so the
/// PyTorch port's per-cell packer (distinct code from variant B's stencil
/// packer) has its own exact target.
fn export_variant_a_oracle(
    dir: &Path,
    name: &str,
    norules: bool,
    gguf: &Path,
    tokenizer: &Path,
    out: &Path,
) -> Result<()> {
    let path = dir.join(name);
    if !path.exists() {
        println!("skip variant-A oracle for {} (not found)", path.display());
        return Ok(());
    }
    let device = Default::default();
    let (spec, params) = llm_life::train::lora_io::load::<CB>(&path, &device)?;

    let rule = Rule::life();
    let tok = Tokenizer::from_json(&std::fs::read(tokenizer)?)?;
    let p = prompt_a(tokenizer, &rule, norules)?;

    let mut model = load_train_model::<CB>(gguf, &[p.dead, p.alive], Some(spec), &device)?;
    anyhow::ensure!(model.lora_params().len() == params.len(), "LoRA/model shape mismatch");
    model.set_lora_params(params);

    let cases = all_cases(&rule);
    let chunk_cases = &cases[..64];
    let ids: Vec<usize> = (0..chunk_cases.len()).collect();
    let chunk = pack_chunk(
        &ids,
        |i| tokenize_case(&tok, &chunk_cases[i].0, chunk_cases[i].1).unwrap(),
        p.prefix.len(),
    );
    let (tokens, positions, mask) = full_sequence::<CB>(&p.prefix, &chunk, &device);
    let logits = model.forward(&tokens, &positions, &mask)?;
    let data = logits.into_data().into_vec::<f32>().unwrap();
    let answer_rows: Vec<usize> = chunk.answer_rows();
    let targets: Vec<u8> = chunk_cases.iter().map(|&(_, _, t)| t).collect();

    let stem = name.trim_end_matches(".bin");
    let json = serde_json::json!({
        "prefix_len": p.prefix.len(),
        "dead_token": p.dead,
        "alive_token": p.alive,
        "cases": chunk_cases.iter().map(|(nb, s, t)| serde_json::json!({"neighbors": nb, "self": s, "target": t})).collect::<Vec<_>>(),
        "answer_rows": answer_rows,
        "targets": targets,
        "logits_all": data,
        "total_tokens": tokens.len(),
    });
    let out_path = out.join(format!("{stem}-variant-a-oracle.json"));
    std::fs::write(&out_path, serde_json::to_string(&json)?)?;
    println!("wrote {}", out_path.display());
    Ok(())
}

/// Reverse of `export_lora`: a PEFT-layout safetensors adapter (this port's
/// training output) back to `LLMLIFE2`, so it can be handed to `lean`'s
/// `apply_lora` (`crates/lean/src/lora.rs` in llm-web) the same way a
/// natively Burn-trained adapter is. `lora_A.weight` `[r, in]` -> `a`
/// `[in, r]`, `lora_B.weight` `[out, r]` -> `b` `[r, out]` (transpose back).
fn import_lora(path: &Path, rank: u32, alpha: f32, out: &Path) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;

    let mut n_layers = 0usize;
    for (name, _) in st.tensors() {
        if let Some(rest) = name.strip_prefix("base_model.model.model.layers.") {
            let idx: usize = rest.split('.').next().unwrap().parse()?;
            n_layers = n_layers.max(idx + 1);
        }
    }
    anyhow::ensure!(n_layers > 0, "no layer tensors found in {}", path.display());

    let device = Default::default();
    let read2 = |name: &str, transpose: bool| -> Result<Tensor<CB, 2>> {
        let view = st.tensor(name)?;
        anyhow::ensure!(view.dtype() == safetensors::Dtype::F32, "{name} is not f32");
        let shape = view.shape().to_vec();
        let data: Vec<f32> = view.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let t: Tensor<CB, 2> = Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), &device);
        Ok(if transpose { t.transpose() } else { t })
    };

    let mut params = Vec::with_capacity(n_layers * 8);
    for layer in 0..n_layers {
        for proj in ["q", "k", "v", "o"] {
            let a = read2(&format!("base_model.model.model.layers.{layer}.self_attn.{proj}_proj.lora_A.weight"), true)?;
            let b = read2(&format!("base_model.model.model.layers.{layer}.self_attn.{proj}_proj.lora_B.weight"), true)?;
            params.push(a);
            params.push(b);
        }
    }

    let spec = llm_life::train::LoraSpec { rank: rank as usize, alpha, mlp: false };
    llm_life::train::lora_io::save::<CB>(out, &spec, &params)?;
    println!("wrote {} ({} layers, rank {rank}, alpha {alpha})", out.display(), n_layers);
    Ok(())
}

fn main() -> Result<()> {
    if let Some(path) = arg("import-lora") {
        let rank: u32 = arg("rank").unwrap_or_else(|| "8".into()).parse()?;
        let alpha: f32 = arg("alpha").unwrap_or_else(|| "16".into()).parse()?;
        let out = PathBuf::from(arg("out").expect("--out required"));
        return import_lora(&PathBuf::from(path), rank, alpha, &out);
    }
    let stencil_dir = PathBuf::from(arg("stencil-life").unwrap_or_else(|| ".".into()));
    let lora_dir = PathBuf::from(arg("lora").unwrap_or_else(|| ".".into()));
    let out = PathBuf::from(arg("out").unwrap_or_else(|| "export-out".into()));
    std::fs::create_dir_all(&out)?;

    export_bert(&stencil_dir, &out)?;
    export_mlp(&stencil_dir, &out)?;
    export_mlp2(&stencil_dir, &out)?;
    export_stencil(&stencil_dir, &out)?;

    for f in ["lora-a-rules-300.bin", "lora-a-norules-300.bin", "lora-b-16-s3-300.bin", "lora-b-32.bin"] {
        export_lora(&lora_dir, f, &out)?;
    }

    if let (Some(gguf), Some(tokenizer)) = (arg("gguf"), arg("tokenizer")) {
        let gguf = PathBuf::from(gguf);
        let tokenizer = PathBuf::from(tokenizer);
        export_lora_forward_oracle(&lora_dir, "lora-b-16-s3-300.bin", &gguf, &tokenizer, 16, &out)?;
        export_variant_a_oracle(&lora_dir, "lora-a-rules-300.bin", false, &gguf, &tokenizer, &out)?;
        export_variant_a_oracle(&lora_dir, "lora-a-norules-300.bin", true, &gguf, &tokenizer, &out)?;
    } else {
        println!("no --gguf/--tokenizer given, skipping the LoRA forward oracle");
    }

    Ok(())
}
