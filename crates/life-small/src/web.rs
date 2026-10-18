//! wasm-bindgen surface for the page's worker: one `SmallModel` for each of
//! the three models, created with `await SmallModel.create(...)`.

use crate::Small;
use wasm_bindgen::prelude::*;

fn js_err(e: life_models::Error) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct SmallModel {
    inner: Small,
}

#[wasm_bindgen]
impl SmallModel {
    /// `use_gpu` is the page's capability check (an adapter was granted);
    /// the CPU is taken when the device then fails to open.
    pub async fn create(safetensors: Vec<u8>, config_json: String, use_gpu: bool) -> Result<SmallModel, JsError> {
        Ok(SmallModel { inner: Small::new(&safetensors, &config_json, use_gpu).await.map_err(js_err)? })
    }

    /// `webgpu` or `single`.
    pub fn backend(&self) -> String {
        match self.inner.backend() {
            crate::Backend::Gpu(_) => "webgpu".to_string(),
            crate::Backend::Cpu => "single".to_string(),
        }
    }

    pub fn label(&self) -> String {
        self.inner.label()
    }

    pub fn kind(&self) -> String {
        self.inner.kind().id().to_string()
    }

    /// p(alive) for every cell of a `width * height` torus in one pass.
    #[wasm_bindgen(js_name = stepGrid)]
    pub async fn step_grid(&self, cells: Vec<u8>, width: usize, height: usize) -> Result<Vec<f32>, JsError> {
        self.inner.p_alive(&cells, width, height).await.map_err(js_err)
    }

    /// p(alive) for one cell, from its own 3x3 neighbourhood.
    #[wasm_bindgen(js_name = stepCell)]
    pub async fn step_cell(&self, cells: Vec<u8>, width: usize, height: usize, index: usize) -> Result<f32, JsError> {
        self.inner.p_alive_cell(&cells, width, height, index).await.map_err(js_err)
    }
}
