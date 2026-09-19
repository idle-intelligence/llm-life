//! Build a [`TrainModel`] from a GGUF on disk.
//!
//! Two-pass, like `Q4ModelLoader::load_deferred`: every projection is read,
//! dequantized, transposed and uploaded one tensor at a time so only one
//! weight is in CPU memory at once; the token embedding keeps its quantized
//! bytes (it is frozen and the f32 form would be 544 MB at this vocab).

use anyhow::{Context, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use llm_wasm::gguf::{EmbeddingStore, Q4ModelLoader};
use std::io::{Read, Seek};
use std::path::Path;

use super::model::{Lora, TrainLayer, TrainLinear, TrainModel};
use super::weights::GgufF32Reader;

/// Which projections get a LoRA adapter, and how big.
#[derive(Clone, Copy, Debug)]
pub struct LoraSpec {
    pub rank: usize,
    pub alpha: f32,
    /// gate/up as well as q/k/v/o. `o` and the MLP are where most of the
    /// capacity is; q/k/v/o alone is the cheap configuration.
    pub mlp: bool,
}

impl Default for LoraSpec {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            mlp: false,
        }
    }
}

pub fn load_train_model<B: Backend>(
    gguf: &Path,
    answer_tokens: &[u32],
    lora: Option<LoraSpec>,
    device: &B::Device,
) -> Result<TrainModel<B>> {
    let file = std::io::BufReader::new(std::fs::File::open(gguf).context("open gguf")?);
    let loader = Q4ModelLoader::new(file)?;
    build(GgufF32Reader::new(loader), answer_tokens, lora, device)
}

fn build<B: Backend, R: Read + Seek>(
    mut r: GgufF32Reader<R>,
    answer_tokens: &[u32],
    lora: Option<LoraSpec>,
    device: &B::Device,
) -> Result<TrainModel<B>> {
    let config = r.config()?;
    let hidden = config.hidden_size;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let p = format!("blk.{i}");
        let mk = |r: &mut GgufF32Reader<R>, name: &str, bias: Option<&str>, adapt: bool| -> Result<TrainLinear<B>> {
            let w = r.weight(name)?;
            let bias = match bias {
                Some(b) => Some(vec1::<B>(&r.f32_vec(b)?, device)),
                None => None,
            };
            let lora = match (adapt, lora) {
                (true, Some(s)) => Some(Lora::new(w.in_features, w.out_features, s.rank, s.alpha, device)),
                _ => None,
            };
            Ok(TrainLinear {
                w: w.to_tensor(device),
                bias,
                lora,
            })
        };
        let mlp_adapt = lora.map(|s| s.mlp).unwrap_or(false);
        layers.push(TrainLayer {
            attn_norm: vec1::<B>(&r.f32_vec(&format!("{p}.attn_norm.weight"))?, device),
            q: mk(&mut r, &format!("{p}.attn_q.weight"), Some(&format!("{p}.attn_q.bias")), true)?,
            k: mk(&mut r, &format!("{p}.attn_k.weight"), Some(&format!("{p}.attn_k.bias")), true)?,
            v: mk(&mut r, &format!("{p}.attn_v.weight"), Some(&format!("{p}.attn_v.bias")), true)?,
            o: mk(&mut r, &format!("{p}.attn_output.weight"), None, true)?,
            ffn_norm: vec1::<B>(&r.f32_vec(&format!("{p}.ffn_norm.weight"))?, device),
            gate: mk(&mut r, &format!("{p}.ffn_gate.weight"), None, mlp_adapt)?,
            up: mk(&mut r, &format!("{p}.ffn_up.weight"), None, mlp_adapt)?,
            down: mk(&mut r, &format!("{p}.ffn_down.weight"), None, false)?,
        });
    }

    let out_norm = vec1::<B>(&r.f32_vec("output_norm.weight")?, device);

    let embd_dims = r.dims("token_embd.weight")?;
    let (vocab, dim) = (embd_dims[0], embd_dims[1]);
    let embd_dtype = r.dtype_of("token_embd.weight")?;
    let mut loader = r.into_inner();
    let embd_bytes = loader.tensor_bytes("token_embd.weight")?;
    drop(loader);
    let embed = EmbeddingStore::new_with_dtype(embd_bytes, embd_dtype, vocab, dim);

    let mut head_data = vec![0.0f32; answer_tokens.len() * dim];
    for (i, &id) in answer_tokens.iter().enumerate() {
        embed.embed_id_add_cpu(id, &mut head_data[i * dim..(i + 1) * dim])?;
    }
    let head: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(head_data, [answer_tokens.len(), dim]),
        device,
    );

    debug_assert_eq!(hidden, dim);
    Ok(TrainModel::new(embed, layers, out_norm, head, config, device.clone()))
}

fn vec1<B: Backend>(data: &[f32], device: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
}
