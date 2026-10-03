//! The language-model methods of llm-life (per cell and whole grid, base
//! model and LoRA adapters) on lean, the project's raw wgpu + WGSL engine
//! (crates/lean in idle-intelligence/llm-web), native and wasm from one
//! source. The Burn engine (`crates/llm-life`) keeps training, the native
//! drivers and the small from-scratch models.
//!
//! The prompt text and the packing are not duplicated: `variant_a.rs`,
//! `variant_b.rs` and `score.rs` are compiled here straight from
//! `crates/llm-life/src` (they depend on nothing but the `life` crate), so
//! the two engines can never drift apart on what they ask the model.
//!
//! Two backends, picked by the caller: WebGPU (`LifeLean::load`) or the
//! CPU (`LifeLean::load_cpu`: lean's `CpuModel`, WASM SIMD128 in the
//! browser, threaded when built with lean's `threads` feature). The prompts,
//! packing, positions, masks and head are the same on both.
//!
//! Head: the `[dead, alive]` logits are read through the token embedding
//! rows (`GpuModel::embed_head_sliced`), the head the adapters were trained
//! against, not the GGUF's separately quantized `output.weight`.

#[path = "../../llm-life/src/score.rs"]
pub mod score;
#[path = "../../llm-life/src/variant_a.rs"]
pub mod variant_a;
#[path = "../../llm-life/src/variant_b.rs"]
pub mod variant_b;

#[cfg(feature = "web")]
pub mod web;

use anyhow::{anyhow, ensure, Context, Result};
use lean::cpu::{CpuKvCache, CpuModel};
use lean::engine::Engine;
use lean::model::{build_rope_tables, forward_chunk_spec, ForwardSpec, GpuModel, KvCache};
use life::{Grid, Rule};
use tokenizers::Tokenizer;
use variant_b::{pack_sparse, PackedSparse, MAX_STENCIL_KEYS};

pub use lean;

/// wasm-bindgen-rayon's pool bootstrap: the page calls `await
/// initThreadPool(navigator.hardwareConcurrency)` once after `init()`, before
/// `LifeEngine.createCpu`, in the threaded build only.
#[cfg(all(feature = "web-mt", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Cells per per-cell ("variant A") chunk: the same 64 the Burn engine
/// packs, so the batched row forwards the same blocks.
pub const CHUNK_CELLS: usize = 64;

/// RoPE table length. Per-cell positions stop at the resident prefix plus
/// one cell prompt, whole-grid positions at the prefix plus one (every cell
/// token shares the position after the prefix), both far below this.
const MAX_POS: usize = 2048;

/// The 512 per-cell cases in the order the training eval and the gates use:
/// bit `b` (0..8) of `k` is neighbour `b`, bit 8 is the cell itself.
pub fn case_cells(k: usize) -> ([u8; 8], u8) {
    let mut nb = [0u8; 8];
    for (b, v) in nb.iter_mut().enumerate() {
        *v = ((k >> b) & 1) as u8;
    }
    (nb, ((k >> 8) & 1) as u8)
}

/// Index into the 512 cases of cell `c`'s neighbourhood in `grid`.
pub fn case_index(grid: &Grid, c: usize) -> usize {
    let mut k = 0usize;
    for (b, &j) in grid.neighbor_indices(c).iter().enumerate() {
        if grid.cells()[j] != 0 {
            k |= 1 << b;
        }
    }
    k | (((grid.cells()[c] != 0) as usize) << 8)
}

/// The whole-grid stencil as lean's dense bitset (`[t, t]`, bit set where
/// query row `i` may attend key `j`), built straight from the sparse form:
/// prefix rows causal, cell rows the prefix plus self and 8 neighbours. Same
/// topology as `variant_b::pack`'s bool mask, without the `t * t` bools.
pub fn stencil_bits(p: &PackedSparse) -> Vec<u32> {
    let t = p.tokens.len();
    let mut bits = vec![0u32; (t * t).div_ceil(32)];
    let mut set = |idx: usize| bits[idx / 32] |= 1u32 << (idx % 32);
    for row in 0..t {
        for j in 0..p.prefix_len[row] as usize {
            set(row * t + j);
        }
        for k in 0..p.n_keys[row] as usize {
            set(row * t + p.keys[row * MAX_STENCIL_KEYS + k] as usize);
        }
    }
    bits
}

/// Softmax over the two answer logits: the same reading as
/// `variant_a::p_alive_chunk` / `variant_b::p_alive`.
pub fn p_alive([d, a]: [f32; 2]) -> f32 {
    let m = d.max(a);
    let (ed, ea) = ((d - m).exp(), (a - m).exp());
    ea / (ed + ea)
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec())
}

/// Which KV cache the model's pool last bound. lean's pool caches bind
/// groups by call site and only invalidates them when one of its *own*
/// buffers is reallocated; a KV cache is allocated on the engine, so moving
/// between the per-cell and whole-grid caches (or to a fresh cache) must
/// reset the pool, or attention keeps reading the previous cache.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    A(u64),
    B(u64),
    Full(u64),
}

/// The three KV caches: `A` per cell (resident prefix at `[0, P)`, each
/// chunk written at `[P, P + t)` and rewound by resetting `kv_len` to `P`,
/// no copy), `B` whole grid (written from position 0 each generation),
/// `Full` per cell without prefix reuse (prefix and prompt from position 0,
/// so the resident prefix in `A` is never overwritten).
#[derive(Clone, Copy)]
enum Slot {
    A,
    B,
    Full,
}

/// KV cache layout is lean's on both backends (`[kv_head][position][head_dim]`
/// per layer, head-major, contiguous, `max_ctx` positions allocated up
/// front).
struct Gpu {
    engine: Engine,
    model: GpuModel,
    cos: wgpu::Buffer,
    sin: wgpu::Buffer,
    caches: [Option<(KvCache, u64)>; 3],
    next_id: u64,
    bound: Option<Bound>,
}

struct Cpu {
    model: CpuModel,
    caches: [Option<CpuKvCache>; 3],
}

/// Where the forwards run: WebGPU (`GpuModel`) or the CPU (`CpuModel`,
/// WASM SIMD128, threaded when built with lean's `threads`). Same packing,
/// positions, masks and head on both.
enum Backend {
    Gpu(Box<Gpu>),
    Cpu(Box<Cpu>),
}

impl Backend {
    fn config(&self) -> &lean::config::Qwen2Config {
        match self {
            Backend::Gpu(g) => &g.model.config,
            Backend::Cpu(c) => &c.model.config,
        }
    }

    /// Forward `tokens` into cache `slot` starting at slot position
    /// `kv_len`, and return the `[dead, alive]` logits at every row
    /// (`[t, 2]`). The cache is (re)allocated with `max_ctx` positions when
    /// `fresh`, absent, or smaller than `kv_len + t`.
    #[allow(clippy::too_many_arguments)]
    async fn forward(&mut self, slot: Slot, fresh: bool, max_ctx: usize, kv_len: usize, tokens: &[u32], spec: &ForwardSpec, head: [u32; 2]) -> Vec<f32> {
        let t = tokens.len();
        let i = slot as usize;
        match self {
            Backend::Gpu(g) => {
                if fresh || g.caches[i].as_ref().is_none_or(|(c, _)| (c.max_ctx as usize) < kv_len + t) {
                    g.caches[i] = Some((KvCache::new(&g.engine, &g.model.config, max_ctx.max(kv_len + t) as u32), g.next_id));
                    g.next_id += 1;
                }
                let (cache, id) = g.caches[i].as_mut().unwrap();
                let b = match slot {
                    Slot::A => Bound::A(*id),
                    Slot::B => Bound::B(*id),
                    Slot::Full => Bound::Full(*id),
                };
                if g.bound != Some(b) {
                    g.model.pool.reset();
                    g.bound = Some(b);
                }
                cache.kv_len = kv_len as u32;
                let hidden = forward_chunk_spec(&g.engine, &g.model, cache, tokens, &g.cos, &g.sin, spec).await;
                g.model.embed_head_sliced(&g.engine, &hidden, t as u32, &head).await
            }
            Backend::Cpu(c) => {
                if fresh || c.caches[i].as_ref().is_none_or(|cache| cache.max_ctx() < kv_len + t) {
                    c.caches[i] = Some(CpuKvCache::new(&c.model.config, max_ctx.max(kv_len + t)));
                }
                let cache = c.caches[i].as_mut().unwrap();
                cache.kv_len = kv_len;
                let hidden = lean::cpu::forward_chunk_spec(&c.model, cache, tokens, spec);
                c.model.embed_head_sliced(&hidden, t, &head)
            }
        }
    }

    fn apply_lora(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            Backend::Gpu(g) => {
                g.model.apply_lora(&g.engine, bytes)?;
                g.bound = None; // apply_lora reset the pool
            }
            Backend::Cpu(c) => c.model.apply_lora(bytes)?,
        }
        Ok(())
    }

    fn clear_lora(&mut self) {
        match self {
            Backend::Gpu(g) => {
                g.model.clear_lora();
                g.bound = None;
            }
            Backend::Cpu(c) => c.model.clear_lora(),
        }
    }
}

pub struct LifeLean {
    backend: Backend,
    tokenizer: Tokenizer,
    rule: Rule,
    dead: u32,
    alive: u32,
    width: usize,
    height: usize,
    /// Whole-grid rules prefix (`variant_b::rules_prefix`).
    prefix_b: Vec<u32>,
    /// Per-cell resident prefix: rules + six worked examples for the base
    /// model, or the adapter's own training-time prefix after
    /// `load_adapter`.
    prefix_a: Vec<u32>,
    /// The 512 per-cell prompts, tokenized once (`case_cells` order).
    prompts: Vec<Vec<u32>>,
    cell_tokens: usize,
}

impl LifeLean {
    /// WebGPU backend. Parse the GGUF (two-phase: each tensor's bytes are
    /// dropped right after their GPU upload inside
    /// `GpuModel::load_from_reader`), tokenize the prefixes and the 512
    /// per-cell prompts, prefill the base model's per-cell prefix.
    pub async fn load<R: std::io::Read + std::io::Seek>(
        engine: Engine,
        gguf: R,
        tokenizer_json: &[u8],
        rulestring: &str,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        let model = GpuModel::load_from_reader(&engine, gguf, true).context("load gguf")?;
        let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, MAX_POS);
        let cos = engine.buf_f32(&cos, "rope_cos");
        let sin = engine.buf_f32(&sin, "rope_sin");
        let backend = Backend::Gpu(Box::new(Gpu { engine, model, cos, sin, caches: [None, None, None], next_id: 1, bound: None }));
        Self::finish_load(backend, tokenizer_json, rulestring, width, height).await
    }

    /// CPU backend: the same model, held as the GGUF's quantized bytes
    /// (`CpuModel`), the same prompts, packing and head.
    pub async fn load_cpu<R: std::io::Read + std::io::Seek>(gguf: R, tokenizer_json: &[u8], rulestring: &str, width: usize, height: usize) -> Result<Self> {
        let model = CpuModel::load_from_reader(gguf).context("load gguf")?;
        Self::finish_load(Backend::Cpu(Box::new(Cpu { model, caches: [None, None, None] })), tokenizer_json, rulestring, width, height).await
    }

    async fn finish_load(backend: Backend, tokenizer_json: &[u8], rulestring: &str, width: usize, height: usize) -> Result<Self> {
        let rule = Rule::parse(rulestring).ok_or_else(|| anyhow!("bad rulestring {rulestring:?}"))?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let single = |s: &str| -> Result<u32> {
            let ids = encode(&tokenizer, s)?;
            ensure!(ids.len() == 1, "{s:?} is not a single token");
            Ok(ids[0])
        };
        let (dead, alive) = (single("0")?, single("1")?);
        let prefix_b = encode(&tokenizer, &variant_b::rules_prefix(&rule))?;

        let mut prefix_a_text = variant_a::rules_prefix(&rule);
        prefix_a_text.push_str(&variant_a::fewshot_examples());
        let prefix_a = encode(&tokenizer, &prefix_a_text)?;
        let mut prompts = Vec::with_capacity(512);
        for k in 0..512 {
            let (nb, me) = case_cells(k);
            prompts.push(encode(&tokenizer, &format!("\n{}", variant_a::cell_prompt(&nb, me)))?);
        }
        let cell_tokens = prompts.iter().map(|p| p.len()).max().unwrap();

        let mut me = LifeLean { backend, tokenizer, rule, dead, alive, width, height, prefix_b, prefix_a, prompts, cell_tokens };
        me.prefill_a().await;
        Ok(me)
    }

    /// "webgpu" or "cpu".
    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            Backend::Gpu(_) => "webgpu",
            Backend::Cpu(_) => "cpu",
        }
    }

    fn head(&self) -> [u32; 2] {
        [self.dead, self.alive]
    }

    /// Fresh per-cell cache sized for the current prefix plus one chunk,
    /// with the prefix prefilled (plain causal, whatever adapter is applied).
    async fn prefill_a(&mut self) {
        let p = self.prefix_a.len();
        let head = self.head();
        let _ = self.backend.forward(Slot::A, true, p + CHUNK_CELLS * self.cell_tokens, 0, &self.prefix_a, &ForwardSpec::default(), head).await;
    }

    /// Apply a runtime LoRA adapter (LLMLIFE2, q/k/v/o), replacing any
    /// earlier one, and re-prefill the per-cell prefix the adapter was
    /// trained with (`norules`: no rule text; otherwise the rules, no worked
    /// examples either way) with the adapter applied. The Burn tab prefilled
    /// that prefix *before* applying the adapter; training (and the HF+PEFT
    /// reference) run the prefix through the adapter too, as here.
    pub async fn load_adapter(&mut self, bytes: &[u8], norules: bool) -> Result<()> {
        self.backend.apply_lora(bytes).context("apply adapter")?;
        let text = if norules { variant_a::norules_prefix() } else { variant_a::rules_prefix(&self.rule) };
        self.set_prefix_a(&text).await
    }

    /// Back to the base model (no adapter). The per-cell prefix is left as
    /// is; `set_prefix_a` picks the one to run against.
    pub fn clear_adapter(&mut self) {
        self.backend.clear_lora();
    }

    /// Replace the per-cell resident prefix and prefill it.
    pub async fn set_prefix_a(&mut self, text: &str) -> Result<()> {
        self.prefix_a = encode(&self.tokenizer, text)?;
        self.prefill_a().await;
        Ok(())
    }

    pub fn set_grid(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
    }

    /// `[dead, alive]` logits at the answer position of each per-cell case
    /// (indices into the 512 prompts), packed block-diagonally against the
    /// resident prefix in one forward.
    pub async fn logits_cases_a(&mut self, cases: &[usize]) -> Result<Vec<[f32; 2]>> {
        ensure!(cases.len() <= CHUNK_CELLS, "at most {CHUNK_CELLS} cells per chunk");
        let p = self.prefix_a.len();
        let prompts = &self.prompts;
        let chunk = variant_a::pack_chunk(cases, |k| prompts[k].clone(), p);
        let t = chunk.len();
        ensure!(p + self.cell_tokens < MAX_POS, "prefix too long for the RoPE table");
        let spec = ForwardSpec::default()
            .with_positions(chunk.positions.clone())
            .with_allowed(&chunk.allowed, t, p + t);
        let head = self.head();
        let logits = self.backend.forward(Slot::A, false, p + CHUNK_CELLS * self.cell_tokens, p, &chunk.tokens, &spec, head).await;
        Ok(chunk.answer_rows().into_iter().map(|r| [logits[r * 2], logits[r * 2 + 1]]).collect())
    }

    /// One per-cell case with no prefix reuse: the resident prefix and case
    /// `k`'s prompt in one plain causal forward from position 0, as a caller
    /// that never kept a prefix would run it. Same answer position and head
    /// as `logits_cases_a`; used to measure what reusing the prefix saves.
    pub async fn logits_case_a_full(&mut self, k: usize) -> Result<[f32; 2]> {
        let mut tokens = self.prefix_a.clone();
        tokens.extend_from_slice(&self.prompts[k]);
        let t = tokens.len();
        ensure!(t < MAX_POS, "prompt too long for the RoPE table");
        let head = self.head();
        let max_ctx = self.prefix_a.len() + self.cell_tokens;
        let logits = self.backend.forward(Slot::Full, false, max_ctx, 0, &tokens, &ForwardSpec::default(), head).await;
        let r = t - 1;
        Ok([logits[r * 2], logits[r * 2 + 1]])
    }

    /// p(alive) for a list of per-cell cases (see `logits_cases_a`).
    pub async fn step_cases_a(&mut self, cases: &[usize]) -> Result<Vec<f32>> {
        Ok(self.logits_cases_a(cases).await?.into_iter().map(p_alive).collect())
    }

    pub fn grid(&self, cells: Vec<u8>) -> Result<Grid> {
        ensure!(cells.len() == self.width * self.height, "cells length does not match the grid");
        Ok(Grid::from_cells(self.width, self.height, cells))
    }

    /// One per-cell chunk: cells `chunk * CHUNK_CELLS ..` of the grid.
    pub async fn step_chunk_a(&mut self, cells: Vec<u8>, chunk: usize) -> Result<Vec<f32>> {
        let grid = self.grid(cells)?;
        let n = grid.cells().len();
        let first = chunk * CHUNK_CELLS;
        ensure!(first < n, "chunk index past the end of the grid");
        let cases: Vec<usize> = (first..(first + CHUNK_CELLS).min(n)).map(|c| case_index(&grid, c)).collect();
        self.step_cases_a(&cases).await
    }

    /// One cell, one forward (a chunk of one against the same prefix).
    pub async fn step_cell_a(&mut self, cells: Vec<u8>, index: usize) -> Result<f32> {
        let grid = self.grid(cells)?;
        ensure!(index < grid.cells().len(), "cell index past the end of the grid");
        Ok(self.step_cases_a(&[case_index(&grid, index)]).await?[0])
    }

    /// Whole grid: the rules prefix and one token per cell in one forward,
    /// stencil mask, every cell token at the same position. Returns the
    /// `[dead, alive]` logits of every cell, in cell order.
    pub async fn logits_b(&mut self, cells: Vec<u8>) -> Result<Vec<[f32; 2]>> {
        let grid = self.grid(cells)?;
        let packed = pack_sparse(&grid, &self.prefix_b, self.dead, self.alive);
        let t = packed.tokens.len();
        ensure!(self.prefix_b.len() < MAX_POS, "prefix too long for the RoPE table");
        let spec = ForwardSpec::default()
            .with_positions(packed.positions.clone())
            .with_allowed_bits(stencil_bits(&packed));
        let head = self.head();
        let logits = self.backend.forward(Slot::B, false, t, 0, &packed.tokens, &spec, head).await;
        let g = packed.grid_start;
        Ok((0..grid.cells().len()).map(|c| [logits[(g + c) * 2], logits[(g + c) * 2 + 1]]).collect())
    }

    /// p(alive) per cell for the whole grid (see `logits_b`).
    pub async fn step_b(&mut self, cells: Vec<u8>) -> Result<Vec<f32>> {
        Ok(self.logits_b(cells).await?.into_iter().map(p_alive).collect())
    }

    /// Token ids of the current per-cell prefix and of case `k`'s prompt,
    /// for checking against a reference's own tokenization.
    pub fn prefix_a_ids(&self) -> &[u32] {
        &self.prefix_a
    }

    pub fn prompt_ids(&self, k: usize) -> &[u32] {
        &self.prompts[k]
    }

    pub fn prefix_b_ids(&self) -> &[u32] {
        &self.prefix_b
    }

    pub fn chunk_count(&self) -> usize {
        (self.width * self.height).div_ceil(CHUNK_CELLS)
    }

    pub fn tokens_per_generation_a(&self) -> usize {
        self.chunk_count() * self.prefix_a.len() + self.width * self.height * self.cell_tokens
    }

    pub fn packed_tokens(&self) -> usize {
        self.prefix_b.len() + self.width * self.height
    }

    pub fn cell_tokens(&self) -> usize {
        self.cell_tokens
    }

    pub fn prefix_tokens_a(&self) -> usize {
        self.prefix_a.len()
    }

    pub fn num_layers(&self) -> usize {
        self.backend.config().num_layers
    }
}
