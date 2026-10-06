//! CPU-side dequantization of a GGUF's quantized weights into f32 tensors.
//!
//! The inference engine keeps every projection as Q4_0 bytes on the GPU and
//! dequantizes inside its WGSL kernels; autodiff cannot differentiate through
//! those. Training therefore reads the same GGUF and materializes f32 weights
//! — 0.5B x 4 B = ~2 GB, which is fine natively and is why this path is
//! native-only.
//!
//! Layout convention: GGUF (and `Q4Linear`) store a projection as
//! `[out_features, in_features]` and compute `x @ W^T`. Everything here
//! transposes at load into `[in_features, out_features]` so the training
//! forward is a plain `x.matmul(w)`.

use anyhow::{bail, Context, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use llm_wasm::gguf::{f16_to_f32, GgmlDtype, Q4ModelLoader};
use llm_wasm::LlmConfig;
use std::io::{Read, Seek};

/// Dequantize a Q4_0 blob (18-byte blocks: f16 scale + 16 paired nibbles)
/// into f32. Mirrors `EmbeddingStore::embed_id_add_cpu`'s block reader — the
/// pairing is element `j` (low nibble) with element `j + 16` (high nibble).
pub fn dequant_q4_0(bytes: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let blocks = n_elements / 32;
    if bytes.len() != blocks * 18 {
        bail!("Q4_0 blob is {} bytes, expected {}", bytes.len(), blocks * 18);
    }
    let mut out = vec![0.0f32; n_elements];
    for blk in 0..blocks {
        let bo = blk * 18;
        let d = f16_to_f32(u16::from_le_bytes([bytes[bo], bytes[bo + 1]]));
        let base = blk * 32;
        for j in 0..16 {
            let byte = bytes[bo + 2 + j];
            out[base + j] = ((byte & 0x0F) as f32 - 8.0) * d;
            out[base + j + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
        }
    }
    Ok(out)
}

/// Dequantize a Q8_0 blob (34-byte blocks: f16 scale + 32 i8 quants) into f32.
pub fn dequant_q8_0(bytes: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let blocks = n_elements / 32;
    if bytes.len() != blocks * 34 {
        bail!("Q8_0 blob is {} bytes, expected {}", bytes.len(), blocks * 34);
    }
    let mut out = vec![0.0f32; n_elements];
    for blk in 0..blocks {
        let bo = blk * 34;
        let d = f16_to_f32(u16::from_le_bytes([bytes[bo], bytes[bo + 1]]));
        for j in 0..32 {
            out[blk * 32 + j] = (bytes[bo + 2 + j] as i8) as f32 * d;
        }
    }
    Ok(out)
}

/// Transpose a row-major `[rows, cols]` slice into `[cols, rows]`.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// One 2D weight, already transposed to `[in, out]`.
pub struct LoadedWeight {
    pub data: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl LoadedWeight {
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(self.data.clone(), [self.in_features, self.out_features]),
            device,
        )
    }
}

/// Reads one GGUF tensor and hands back f32, whatever block format it is in.
pub struct GgufF32Reader<R: Read + Seek> {
    loader: Q4ModelLoader<R>,
}

impl<R: Read + Seek> GgufF32Reader<R> {
    pub fn new(loader: Q4ModelLoader<R>) -> Self {
        Self { loader }
    }

    pub fn config(&self) -> Result<LlmConfig> {
        llm_wasm::gguf::config_from_gguf(self.loader.reader())
    }

    /// GGUF stores dimensions fastest-varying first, so `ne = [in, out]` for
    /// a projection this crate thinks of as `[out, in]`.
    pub fn dims(&self, name: &str) -> Result<Vec<usize>> {
        let info = self
            .loader
            .reader()
            .tensor_info(name)
            .with_context(|| format!("tensor '{name}' not found"))?;
        Ok(info.shape().iter().rev().map(|&d| d as usize).collect())
    }

    pub fn dtype_of(&self, name: &str) -> Result<GgmlDtype> {
        Ok(self
            .loader
            .reader()
            .tensor_info(name)
            .with_context(|| format!("tensor '{name}' not found"))?
            .dtype())
    }

    pub fn f32_vec(&mut self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .loader
            .reader()
            .tensor_info(name)
            .with_context(|| format!("tensor '{name}' not found"))?
            .clone();
        let n = info.num_elements()? as usize;
        let bytes = self.loader.tensor_bytes(name)?;
        match info.dtype() {
            GgmlDtype::F32 => Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()),
            GgmlDtype::F16 => Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                .collect()),
            GgmlDtype::Q4_0 => dequant_q4_0(&bytes, n),
            GgmlDtype::Q8_0 => dequant_q8_0(&bytes, n),
        }
    }

    /// A 2D projection, transposed to `[in, out]`.
    pub fn weight(&mut self, name: &str) -> Result<LoadedWeight> {
        let dims = self.dims(name)?;
        if dims.len() != 2 {
            bail!("tensor '{name}' is not 2D: {dims:?}");
        }
        let (out_features, in_features) = (dims[0], dims[1]);
        let data = self.f32_vec(name)?;
        Ok(LoadedWeight {
            data: transpose(&data, out_features, in_features),
            in_features,
            out_features,
        })
    }

    pub fn into_inner(self) -> Q4ModelLoader<R> {
        self.loader
    }
}
