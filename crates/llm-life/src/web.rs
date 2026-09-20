//! wasm-bindgen surface for LLM mode in the tab.
//!
//! Mirrors llm-web's `LlmEngine` conventions: JS fetches the bytes and pushes
//! them in with `appendModelShard` (keeping the >2GB-ArrayBuffer problem on
//! the JS side where it can be chunked), then `load`, then one `step` per
//! generation. All readback is `into_data_async().await` — a sync readback
//! deadlocks the browser.

use crate::score::{otsu_threshold, zscore_threshold};
use crate::variant_a;
use crate::variant_b::{p_alive, pack_sparse, rules_prefix, MAX_STENCIL_KEYS};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use life::{Grid, Rule};
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::kv::KvCache;
use llm_wasm::model::{ForwardSpec, LlmModel, SparseMask};
use llm_wasm::tokenizer::Tokenizer;
use wasm_bindgen::prelude::*;

fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

/// Label-free binarization (`score::otsu_threshold`) — same code path as the
/// native `rescore` tool, so the tab and `docs/pictures/README.md` agree.
#[wasm_bindgen(js_name = otsuThreshold)]
pub fn otsu_threshold_js(p_alive: Vec<f32>) -> f32 {
    otsu_threshold(&p_alive)
}

/// Label-free binarization, mean + k*std (`score::zscore_threshold`).
#[wasm_bindgen(js_name = zscoreThreshold)]
pub fn zscore_threshold_js(p_alive: Vec<f32>, k: f64) -> f32 {
    zscore_threshold(&p_alive, k)
}

// `initWgpuDevice` and the panic hook are NOT redefined here: llm-wasm's
// `web` module already exports both, and wasm-bindgen collects exports from
// every linked crate into one module — a second `#[wasm_bindgen(start)]` or a
// second `initWgpuDevice` would collide. JS imports them from this same pkg.

/// Cells per variant A chunk. The chunk's dense mask is quadratic in
/// `CHUNK_CELLS * tokens_per_cell`, and this is the size the native driver's
/// scaling run (docs/runs/2026-09-18-forward-scaling.md) shows is still on
/// the flat part of the s/token curve.
const CHUNK_CELLS: usize = 64;

#[wasm_bindgen]
pub struct LifeEngine {
    device: WgpuDevice,
    shards: Vec<Vec<u8>>,
    model: Option<LlmModel>,
    head: Option<Tensor<Wgpu, 2>>,
    prefix: Vec<u32>,
    dead: u32,
    alive: u32,
    width: usize,
    height: usize,
    // Variant A, sharing the one model above. Its prefix is a different text
    // from variant B's (it states the answer format and carries the six
    // worked examples), and unlike B's it is *resident* in `cache_a`:
    // prefilled once at load, and every chunk forward rewinds the cache back
    // to it with `snapshot`/`restore`.
    prefix_a: Vec<u32>,
    cache_a: Option<KvCache>,
    /// The 512 distinct (8 neighbors, self) per-cell prompts, tokenized once.
    prompts: Vec<Vec<u32>>,
    cell_tokens: usize,
}

#[wasm_bindgen]
impl LifeEngine {
    /// Call **after** `initWgpuDevice()` has been awaited: the engine picks
    /// up the device that call created. A `WgpuDevice::default()` here would
    /// spin up a second, uninitialized runtime whose readback panics in the
    /// browser ("Failed to read tensor data synchronously").
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> LifeEngine {
        LifeEngine {
            device: llm_wasm::web::wgpu_device()
                .expect("initWgpuDevice() must be awaited before constructing LifeEngine"),
            shards: Vec::new(),
            model: None,
            head: None,
            prefix: Vec::new(),
            dead: 0,
            alive: 0,
            width,
            height,
            prefix_a: Vec::new(),
            cache_a: None,
            prompts: Vec::new(),
            cell_tokens: 0,
        }
    }

    /// Change the grid the engine packs for. Variant B runs at 64x64 and
    /// variant A at 16x16, so the page sets this when the mode changes
    /// rather than rebuilding the engine (which would reload the weights).
    #[wasm_bindgen(js_name = setGrid)]
    pub fn set_grid(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
    }

    /// Append one GGUF chunk. Call before `load`; several chunks are fine
    /// (they are read through a sharded cursor, so no single ArrayBuffer has
    /// to hold the whole file).
    #[wasm_bindgen(js_name = appendModelShard)]
    pub fn append_model_shard(&mut self, shard: &[u8]) {
        self.shards.push(shard.to_vec());
    }

    /// Parse the GGUF, upload to GPU, tokenize the rules prefix.
    #[wasm_bindgen(js_name = load)]
    pub async fn load(&mut self, tokenizer_json: String, rulestring: String) -> Result<(), JsError> {
        if self.shards.is_empty() {
            return Err(JsError::new("no shards appended"));
        }
        let rule = Rule::parse(&rulestring).ok_or_else(|| JsError::new("bad rulestring"))?;

        let tokenizer = Tokenizer::from_json(tokenizer_json.as_bytes())
            .map_err(|e| JsError::new(&format!("tokenizer: {e}")))?;
        let encode_one = |s: &str| -> Result<u32, JsError> {
            let ids = tokenizer
                .encode(s, false)
                .map_err(|e| JsError::new(&format!("encode: {e}")))?;
            if ids.len() != 1 {
                return Err(JsError::new(&format!("'{s}' is not a single token")));
            }
            Ok(ids[0])
        };
        self.dead = encode_one("0")?;
        self.alive = encode_one("1")?;
        self.prefix = tokenizer
            .encode(&rules_prefix(&rule), false)
            .map_err(|e| JsError::new(&format!("encode prefix: {e}")))?;

        // Two-phase: the GGUF reader (and its shard bytes) is dropped before
        // the GPU tensors are finalized, so the 4GB wasm address space never
        // has to hold both.
        let parts = {
            let shards = std::mem::take(&mut self.shards);
            let mut loader = Q4ModelLoader::from_shards(shards)
                .map_err(|e| JsError::new(&format!("open gguf: {e}")))?;
            loader
                .load_deferred(&self.device)
                .map_err(|e| JsError::new(&format!("load: {e}")))?
        };
        let model = parts
            .finalize(&self.device)
            .map_err(|e| JsError::new(&format!("finalize: {e}")))?;

        self.head = Some(
            model
                .head_slice(&[self.dead, self.alive])
                .map_err(|e| JsError::new(&format!("head slice: {e}")))?,
        );

        // Variant A's own prompt set and resident prefix, off the same model
        // and tokenizer. Few-shot is on and not optional: without the six
        // worked examples the model answers `0` almost everywhere
        // (docs/OVERNIGHT-REPORT.md), which is not worth putting in the tab.
        let mut prefix_a_text = variant_a::rules_prefix(&rule);
        prefix_a_text.push_str(&variant_a::fewshot_examples());
        self.prefix_a = tokenizer
            .encode(&prefix_a_text, false)
            .map_err(|e| JsError::new(&format!("encode variant A prefix: {e}")))?;
        // Only 512 distinct per-cell prompts exist, so the tokenizer runs 512
        // times per load instead of once per cell per generation.
        self.prompts = Vec::with_capacity(512);
        for k in 0..512usize {
            let nb: Vec<u8> = (0..8).map(|b| ((k >> b) & 1) as u8).collect();
            let text = format!("\n{}", variant_a::cell_prompt(&nb, ((k >> 8) & 1) as u8));
            self.prompts.push(
                tokenizer
                    .encode(&text, false)
                    .map_err(|e| JsError::new(&format!("encode cell prompt: {e}")))?,
            );
        }
        self.cell_tokens = self.prompts.iter().map(|p| p.len()).max().unwrap();
        let mut cache = model.new_cache(self.prefix_a.len() + CHUNK_CELLS * self.cell_tokens);
        model
            .forward_hidden(&self.prefix_a, &mut cache)
            .map_err(|e| JsError::new(&format!("prefill variant A prefix: {e}")))?;
        self.cache_a = Some(cache);

        log(&format!(
            "[llm-life] loaded: {} layers, hidden {}, prefix {} tokens, grid {}x{}",
            model.config().num_layers,
            model.config().hidden_size,
            self.prefix.len(),
            self.width,
            self.height
        ));
        self.model = Some(model);
        Ok(())
    }

    /// Load a runtime LoRA adapter (llm-wasm's `lora` module, LLMLIFE2
    /// format — see llm-life's `tools/merge/lora_io.py` / `train/lora_io.rs`
    /// for the on-disk layout) onto the already-`load()`-ed model. Applies
    /// q/k/v/o deltas on every subsequent `step`/`stepChunkA` forward;
    /// replaces any adapter loaded earlier, does not stack.
    #[wasm_bindgen(js_name = loadAdapter)]
    pub fn load_adapter(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        let model = self.model.as_mut().ok_or_else(|| JsError::new("not loaded"))?;
        let adapter = llm_wasm::lora::LoraAdapter::from_bytes(bytes, model.config().num_layers, &self.device)
            .map_err(|e| JsError::new(&format!("parse adapter: {e}")))?;
        model
            .apply_lora(adapter)
            .map_err(|e| JsError::new(&format!("apply adapter: {e}")))?;
        log(&format!("[llm-life] LoRA adapter applied ({} bytes)", bytes.len()));
        Ok(())
    }

    /// One generation. `cells` is the current grid (row-major, 0/1); returns
    /// p(alive) per cell.
    #[wasm_bindgen(js_name = step)]
    pub async fn step(&self, cells: Vec<u8>) -> Result<Vec<f32>, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("not loaded"))?;
        let head = self.head.as_ref().unwrap();
        if cells.len() != self.width * self.height {
            return Err(JsError::new("cells length does not match the grid"));
        }
        let grid = Grid::from_cells(self.width, self.height, cells);
        let packed = pack_sparse(&grid, &self.prefix, self.dead, self.alive);
        let t = packed.tokens.len();
        let sparse = SparseMask::new(
            &packed.prefix_len,
            &packed.n_keys,
            &packed.keys,
            MAX_STENCIL_KEYS,
            t,
            &self.device,
        );
        let spec = ForwardSpec::default()
            .with_positions(packed.positions.clone())
            .with_sparse(sparse);

        let mut cache = model.new_cache(t);
        let hidden = model
            .forward_hidden_spec(&packed.tokens, &mut cache, &spec)
            .map_err(|e| JsError::new(&format!("forward: {e}")))?;
        let logits = model.lm_head_sliced(hidden, head);
        let logits = llm_wasm::model::logits_to_vec_async(logits)
            .await
            .map_err(|e| JsError::new(&format!("readback: {e}")))?;
        Ok(p_alive(&logits, packed.grid_start, self.width * self.height))
    }

    /// Number of variant A chunks one generation of the current grid takes.
    #[wasm_bindgen(js_name = chunkCount)]
    pub fn chunk_count(&self) -> usize {
        (self.width * self.height).div_ceil(CHUNK_CELLS)
    }

    /// Tokens variant A forwards per generation: every chunk re-attends the
    /// resident prefix, plus one per-cell prompt per cell.
    #[wasm_bindgen(js_name = tokensPerGenerationA)]
    pub fn tokens_per_generation_a(&self) -> usize {
        self.chunk_count() * self.prefix_a.len() + self.width * self.height * self.cell_tokens
    }

    /// One variant A chunk: the cells `chunk * CHUNK_CELLS ..` packed
    /// block-diagonally against the resident prefix, one forward, and
    /// p(alive) for those cells in order.
    ///
    /// The cache is rewound to the prefix afterwards (`snapshot`/`restore`,
    /// both O(1) — they move the fill length only), so the next chunk
    /// overwrites this one's rows and the prefix is prefilled once per load.
    #[wasm_bindgen(js_name = stepChunkA)]
    pub async fn step_chunk_a(&mut self, cells: Vec<u8>, chunk: usize) -> Result<Vec<f32>, JsError> {
        if self.model.is_none() {
            return Err(JsError::new("not loaded"));
        }
        if cells.len() != self.width * self.height {
            return Err(JsError::new("cells length does not match the grid"));
        }
        let n = self.width * self.height;
        let first = chunk * CHUNK_CELLS;
        if first >= n {
            return Err(JsError::new("chunk index past the end of the grid"));
        }
        let grid = Grid::from_cells(self.width, self.height, cells);
        let cell_ids: Vec<usize> = (first..(first + CHUNK_CELLS).min(n)).collect();

        let prompts = &self.prompts;
        let packed = variant_a::pack_chunk(
            &cell_ids,
            |c| {
                let mut k = 0usize;
                for (b, &j) in grid.neighbor_indices(c).iter().enumerate() {
                    if grid.cells()[j] != 0 {
                        k |= 1 << b;
                    }
                }
                k |= ((grid.cells()[c] != 0) as usize) << 8;
                prompts[k].clone()
            },
            self.prefix_a.len(),
        );

        let model = self.model.as_ref().unwrap();
        let head = self.head.as_ref().unwrap();
        let prefix_len = self.prefix_a.len();
        let t = packed.len();
        let spec = ForwardSpec::default()
            .with_positions(packed.positions.clone())
            .with_allowed(&packed.allowed, t, prefix_len + t, &self.device);

        let cache = self.cache_a.as_mut().unwrap();
        let resident = cache.snapshot();
        let hidden = model
            .forward_hidden_spec(&packed.tokens, cache, &spec)
            .map_err(|e| JsError::new(&format!("forward: {e}")))?;
        let logits = model.lm_head_sliced(hidden, head);
        let logits = llm_wasm::model::logits_to_vec_async(logits)
            .await
            .map_err(|e| JsError::new(&format!("readback: {e}")))?;
        self.cache_a.as_mut().unwrap().restore(resident);

        Ok(variant_a::p_alive_chunk(&logits, &packed)
            .into_iter()
            .map(|(_, v)| v)
            .collect())
    }

    /// Token count of one packed forward pass — what the PERFORMANCE panel
    /// reports alongside seconds per generation.
    #[wasm_bindgen(js_name = packedTokens)]
    pub fn packed_tokens(&self) -> usize {
        self.prefix.len() + self.width * self.height
    }
}
