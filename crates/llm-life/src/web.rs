//! wasm-bindgen surface for LLM mode in the tab.
//!
//! Mirrors llm-web's `LlmEngine` conventions: JS fetches the bytes and pushes
//! them in with `appendModelShard` (keeping the >2GB-ArrayBuffer problem on
//! the JS side where it can be chunked), then `load`, then one `step` per
//! generation. All readback is `into_data_async().await` — a sync readback
//! deadlocks the browser.

use crate::variant_b::{p_alive, pack, rules_prefix};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use life::{Grid, Rule};
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::model::{ForwardSpec, LlmModel};
use llm_wasm::tokenizer::Tokenizer;
use wasm_bindgen::prelude::*;

fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

// `initWgpuDevice` and the panic hook are NOT redefined here: llm-wasm's
// `web` module already exports both, and wasm-bindgen collects exports from
// every linked crate into one module — a second `#[wasm_bindgen(start)]` or a
// second `initWgpuDevice` would collide. JS imports them from this same pkg.

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
}

#[wasm_bindgen]
impl LifeEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> LifeEngine {
        LifeEngine {
            device: WgpuDevice::default(),
            shards: Vec::new(),
            model: None,
            head: None,
            prefix: Vec::new(),
            dead: 0,
            alive: 0,
            width,
            height,
        }
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
        let packed = pack(&grid, &self.prefix, self.dead, self.alive);
        let t = packed.len();
        let spec = ForwardSpec::default()
            .with_positions(packed.positions.clone())
            .with_allowed(&packed.allowed, t, t, &self.device);

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

    /// Token count of one packed forward pass — what the PERFORMANCE panel
    /// reports alongside seconds per generation.
    #[wasm_bindgen(js_name = packedTokens)]
    pub fn packed_tokens(&self) -> usize {
        self.prefix.len() + self.width * self.height
    }
}
