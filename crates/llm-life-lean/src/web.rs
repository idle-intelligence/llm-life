//! wasm-bindgen surface: the same `LifeEngine` calls the tab made on the
//! Burn engine (`crates/llm-life/src/web.rs`), so web/worker.js changes only
//! where the engine comes from. Two differences: the engine is built with
//! `await LifeEngine.create(w, h)` (lean requests its own WebGPU device,
//! which is async), and `loadAdapter` is async (it re-prefills the per-cell
//! prefix with the adapter applied). All readback is async.

use std::io::{Read, Seek, SeekFrom};

use crate::score::{otsu_threshold, zscore_threshold};
use crate::LifeLean;
use lean::engine::Engine;
use wasm_bindgen::prelude::*;

fn js_err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

#[wasm_bindgen(js_name = otsuThreshold)]
pub fn otsu_threshold_js(p_alive: Vec<f32>) -> f32 {
    otsu_threshold(&p_alive)
}

#[wasm_bindgen(js_name = zscoreThreshold)]
pub fn zscore_threshold_js(p_alive: Vec<f32>, k: f64) -> f32 {
    zscore_threshold(&p_alive, k)
}

/// `Read + Seek` over the GGUF as JS-side chunks: the bytes stay in JS
/// memory and are copied into wasm one read at a time, so neither a single
/// 2GB ArrayBuffer nor a second full copy in wasm memory is ever needed.
struct JsChunksReader {
    chunks: Vec<js_sys::Uint8Array>,
    starts: Vec<u64>,
    len: u64,
    pos: u64,
}

impl JsChunksReader {
    fn new(chunks: Vec<js_sys::Uint8Array>) -> Self {
        let mut starts = Vec::with_capacity(chunks.len());
        let mut len = 0u64;
        for c in &chunks {
            starts.push(len);
            len += c.length() as u64;
        }
        JsChunksReader { chunks, starts, len, pos: 0 }
    }
}

impl Read for JsChunksReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let i = self.starts.partition_point(|&s| s <= self.pos) - 1;
        let off = (self.pos - self.starts[i]) as u32;
        let chunk = &self.chunks[i];
        let n = (chunk.length() - off).min(buf.len() as u32);
        chunk.subarray(off, off + n).copy_to(&mut buf[..n as usize]);
        self.pos += n as u64;
        Ok(n as usize)
    }
}

impl Seek for JsChunksReader {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let pos = match to {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(d) => self.len as i64 + d,
            SeekFrom::Current(d) => self.pos as i64 + d,
        };
        if pos < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before start"));
        }
        self.pos = pos as u64;
        Ok(self.pos)
    }
}

#[wasm_bindgen]
pub struct LifeEngine {
    width: usize,
    height: usize,
    engine: Option<Engine>,
    shards: Vec<js_sys::Uint8Array>,
    inner: Option<LifeLean>,
}

#[wasm_bindgen]
impl LifeEngine {
    /// Requests the WebGPU device and builds lean's pipelines.
    #[wasm_bindgen(js_name = create)]
    pub async fn create(width: usize, height: usize) -> Result<LifeEngine, JsError> {
        console_error_panic_hook::set_once();
        let engine = Engine::new_async().await.map_err(js_err)?;
        Ok(LifeEngine { width, height, engine: Some(engine), shards: Vec::new(), inner: None })
    }

    #[wasm_bindgen(js_name = setGrid)]
    pub fn set_grid(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        if let Some(l) = self.inner.as_mut() {
            l.set_grid(width, height);
        }
    }

    /// Append one GGUF chunk (kept on the JS side, see `JsChunksReader`).
    #[wasm_bindgen(js_name = appendModelShard)]
    pub fn append_model_shard(&mut self, shard: js_sys::Uint8Array) {
        self.shards.push(shard);
    }

    #[wasm_bindgen(js_name = load)]
    pub async fn load(&mut self, tokenizer_json: String, rulestring: String) -> Result<(), JsError> {
        if self.shards.is_empty() {
            return Err(JsError::new("no shards appended"));
        }
        let engine = self.engine.take().ok_or_else(|| JsError::new("already loaded"))?;
        let reader = JsChunksReader::new(std::mem::take(&mut self.shards));
        let inner = LifeLean::load(engine, reader, tokenizer_json.as_bytes(), &rulestring, self.width, self.height)
            .await
            .map_err(js_err)?;
        web_sys::console::log_1(
            &format!(
                "[llm-life] lean engine loaded: {} layers, whole-grid prefix {} tokens, grid {}x{}",
                inner.num_layers(),
                inner.packed_tokens() - self.width * self.height,
                self.width,
                self.height
            )
            .into(),
        );
        self.inner = Some(inner);
        Ok(())
    }

    fn inner(&mut self) -> Result<&mut LifeLean, JsError> {
        self.inner.as_mut().ok_or_else(|| JsError::new("not loaded"))
    }

    #[wasm_bindgen(js_name = loadAdapter)]
    pub async fn load_adapter(&mut self, bytes: Vec<u8>, norules: bool) -> Result<(), JsError> {
        self.inner()?.load_adapter(&bytes, norules).await.map_err(js_err)?;
        web_sys::console::log_1(
            &format!("[llm-life] LoRA adapter applied ({} bytes, {})", bytes.len(), if norules { "norules prefix" } else { "rules prefix" }).into(),
        );
        Ok(())
    }

    /// Whole grid, one forward: p(alive) per cell.
    #[wasm_bindgen(js_name = step)]
    pub async fn step(&mut self, cells: Vec<u8>) -> Result<Vec<f32>, JsError> {
        self.inner()?.step_b(cells).await.map_err(js_err)
    }

    #[wasm_bindgen(js_name = stepChunkA)]
    pub async fn step_chunk_a(&mut self, cells: Vec<u8>, chunk: usize) -> Result<Vec<f32>, JsError> {
        self.inner()?.step_chunk_a(cells, chunk).await.map_err(js_err)
    }

    #[wasm_bindgen(js_name = stepCellA)]
    pub async fn step_cell_a(&mut self, cells: Vec<u8>, index: usize) -> Result<f32, JsError> {
        self.inner()?.step_cell_a(cells, index).await.map_err(js_err)
    }

    #[wasm_bindgen(js_name = chunkCount)]
    pub fn chunk_count(&mut self) -> Result<usize, JsError> {
        Ok(self.inner()?.chunk_count())
    }

    #[wasm_bindgen(js_name = tokensPerGenerationA)]
    pub fn tokens_per_generation_a(&mut self) -> Result<usize, JsError> {
        Ok(self.inner()?.tokens_per_generation_a())
    }

    #[wasm_bindgen(js_name = packedTokens)]
    pub fn packed_tokens(&mut self) -> Result<usize, JsError> {
        Ok(self.inner()?.packed_tokens())
    }

    #[wasm_bindgen(js_name = cellTokens)]
    pub fn cell_tokens(&mut self) -> Result<usize, JsError> {
        Ok(self.inner()?.cell_tokens())
    }

    #[wasm_bindgen(js_name = prefixTokensA)]
    pub fn prefix_tokens_a(&mut self) -> Result<usize, JsError> {
        Ok(self.inner()?.prefix_tokens_a())
    }
}
