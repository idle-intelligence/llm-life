//! On-disk format for a set of LoRA matrices.
//!
//! Deliberately not a Burn recorder: the file is `[a, b]` per adapted linear
//! in `TrainModel::lora_params`' fixed order, and nothing else needs to read
//! it. Magic + a shape table so a mismatched rank or layer count fails loudly
//! instead of loading garbage.

use anyhow::{bail, ensure, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"LLMLIFE1";

pub fn save<B: Backend>(path: &Path, params: &[Tensor<B, 2>]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(MAGIC)?;
    f.write_all(&(params.len() as u32).to_le_bytes())?;
    for t in params {
        let [r, c] = t.dims();
        f.write_all(&(r as u32).to_le_bytes())?;
        f.write_all(&(c as u32).to_le_bytes())?;
        let data = t.clone().into_data().into_vec::<f32>().unwrap();
        for v in data {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn load<B: Backend>(path: &Path, device: &B::Device) -> Result<Vec<Tensor<B, 2>>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    ensure!(bytes.len() >= 12 && &bytes[..8] == MAGIC, "not a LoRA file: {}", path.display());
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut off = 12;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if off + 8 > bytes.len() {
            bail!("LoRA file truncated at tensor {i}");
        }
        let r = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        let c = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        off += 8;
        let len = r * c;
        ensure!(off + len * 4 <= bytes.len(), "LoRA file truncated in tensor {i}");
        let data: Vec<f32> = bytes[off..off + len * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        off += len * 4;
        out.push(Tensor::from_data(TensorData::new(data, [r, c]), device));
    }
    Ok(out)
}
