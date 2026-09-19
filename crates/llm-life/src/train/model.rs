//! `TrainModel` — Qwen2's forward pass in pure Burn tensor ops, so autodiff
//! can run through it, with LoRA adapters as the only trainable parameters.
//!
//! This is a **reimplementation of `llm_wasm::model::LlmModel`'s math**, not a
//! wrapper around it: that model's projections are Q4_0 blobs read by custom
//! WGSL kernels, and there is no gradient through a hand-written kernel. The
//! two must agree numerically, which is what `tests/train_oracle.rs` pins:
//! with the LoRA `b` matrices at zero, `TrainModel`'s logits equal
//! `LlmModel::lm_head_sliced(forward_hidden_spec(..))` on the same weights.
//!
//! Matched piece by piece against `llm_wasm::model`:
//!   * RMSNorm — `burn::nn::RmsNorm::forward`'s exact expression
//!     (`x / sqrt(mean(x^2) + eps) * gamma`), which `gguf::rmsnorm_fused`
//!     reproduces.
//!   * RoPE — rotate-half (HF Llama/Qwen2), `cos`/`sin` tables built as
//!     `cat([freqs, freqs])` and indexed by caller-supplied positions, which
//!     is `model::rope_positions`.
//!   * Attention — GQA with q/k/v bias and no o bias, `repeat_kv`, a
//!     caller-supplied "masked out" bool mask, scale `head_dim^-0.5`,
//!     queries chunked exactly like `model::attention_with_mask`.
//!   * MLP — 3-matrix SwiGLU, `down(silu(gate(x)) * up(x))`.
//!   * Head — the tied embedding sliced to the answer tokens, applied by
//!     broadcast-multiply-and-sum, which is `LlmModel::lm_head_sliced`.
//!
//! Weight layout differs on purpose: `Q4Linear` holds `[out, in]` and does
//! `x @ W^T`; here every weight is transposed once at load into `[in, out]`
//! and the forward is `x.matmul(w)`.

use anyhow::{ensure, Result};
use burn::prelude::Backend;
use burn::tensor::activation::{log_softmax, silu, softmax};
use burn::tensor::{Bool, Int, Tensor, TensorData};
use llm_wasm::gguf::EmbeddingStore;
use llm_wasm::LlmConfig;

/// Largest query chunk for the attention score/softmax/PV passes. Same value
/// and same reason as `llm_wasm::model::ATTN_QUERY_CHUNK`: it keeps each
/// individual matmul's shape inside what this backend's kernel handles.
const ATTN_QUERY_CHUNK: usize = 256;

/// A LoRA adapter: `delta(x) = (x @ a) @ b * (alpha / rank)`.
///
/// `b` is zero-initialized, so a freshly built adapter is the identity and
/// the model is exactly the base model — that is what makes the oracle test
/// a test of the forward pass rather than of the adapter.
pub struct Lora<B: Backend> {
    pub a: Tensor<B, 2>,
    pub b: Tensor<B, 2>,
    pub scale: f32,
}

impl<B: Backend> Lora<B> {
    /// `a` ~ N(0, 1/rank) (so the product starts at a sane magnitude once `b`
    /// leaves zero), `b` = 0.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, device: &B::Device) -> Self {
        let a = Tensor::random(
            [in_features, rank],
            burn::tensor::Distribution::Normal(0.0, (1.0 / rank as f64).sqrt()),
            device,
        );
        let b = Tensor::zeros([rank, out_features], device);
        Self {
            a,
            b,
            scale: alpha / rank as f32,
        }
    }

    fn delta(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let r = self.a.dims()[1];
        let n = self.b.dims()[1];
        let h = x
            .matmul(self.a.clone().unsqueeze::<3>())
            .reshape([b, t, r]);
        h.matmul(self.b.clone().unsqueeze::<3>())
            .reshape([b, t, n])
            .mul_scalar(self.scale)
    }
}

/// `x @ w + bias`, optionally plus a LoRA delta. `w` is `[in, out]`.
pub struct TrainLinear<B: Backend> {
    pub w: Tensor<B, 2>,
    pub bias: Option<Tensor<B, 1>>,
    pub lora: Option<Lora<B>>,
}

impl<B: Backend> TrainLinear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let base = x.clone().matmul(self.w.clone().unsqueeze::<3>());
        let base = match &self.bias {
            Some(b) => base + b.clone().unsqueeze::<3>(),
            None => base,
        };
        match &self.lora {
            Some(l) => base + l.delta(x),
            None => base,
        }
    }
}

pub struct TrainLayer<B: Backend> {
    pub attn_norm: Tensor<B, 1>,
    pub q: TrainLinear<B>,
    pub k: TrainLinear<B>,
    pub v: TrainLinear<B>,
    pub o: TrainLinear<B>,
    pub ffn_norm: Tensor<B, 1>,
    pub gate: TrainLinear<B>,
    pub up: TrainLinear<B>,
    pub down: TrainLinear<B>,
}

impl<B: Backend> TrainLayer<B> {
    fn linears_mut(&mut self) -> [&mut TrainLinear<B>; 6] {
        [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.o,
            &mut self.gate,
            &mut self.up,
        ]
    }

    fn linears(&self) -> [&TrainLinear<B>; 6] {
        [&self.q, &self.k, &self.v, &self.o, &self.gate, &self.up]
    }
}

/// The training-side model. `embed` stays quantized on the CPU (the tied
/// embedding is 136M values at this model's vocab — materializing it as f32
/// would cost more than every other weight put together, and it is frozen
/// anyway).
pub struct TrainModel<B: Backend> {
    pub embed: EmbeddingStore,
    pub layers: Vec<TrainLayer<B>>,
    pub out_norm: Tensor<B, 1>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
    /// `[K, hidden]` — the tied embedding sliced to the answer tokens.
    head: Tensor<B, 2>,
    pub config: LlmConfig,
    device: B::Device,
}

impl<B: Backend> TrainModel<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed: EmbeddingStore,
        layers: Vec<TrainLayer<B>>,
        out_norm: Tensor<B, 1>,
        head: Tensor<B, 2>,
        config: LlmConfig,
        device: B::Device,
    ) -> Self {
        let head_dim = config.hidden_size / config.num_heads;
        let (cos, sin) = rope_tables(head_dim, config.max_seq_len, config.rope_theta, &device);
        Self {
            embed,
            layers,
            out_norm,
            cos,
            sin,
            head,
            config,
            device,
        }
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Every LoRA matrix, in a fixed order: per layer, `q.a, q.b, k.a, k.b,
    /// v.a, v.b, o.a, o.b, gate.a, gate.b, up.a, up.b`, skipping any linear
    /// without an adapter. The order is the contract between
    /// `lora_params`, `set_lora_params` and the on-disk file.
    pub fn lora_params(&self) -> Vec<Tensor<B, 2>> {
        let mut out = Vec::new();
        for layer in &self.layers {
            for lin in layer.linears() {
                if let Some(l) = &lin.lora {
                    out.push(l.a.clone());
                    out.push(l.b.clone());
                }
            }
        }
        out
    }

    pub fn set_lora_params(&mut self, params: Vec<Tensor<B, 2>>) {
        let mut it = params.into_iter();
        for layer in &mut self.layers {
            for lin in layer.linears_mut() {
                if let Some(l) = &mut lin.lora {
                    l.a = it.next().expect("too few LoRA params");
                    l.b = it.next().expect("too few LoRA params");
                }
            }
        }
        assert!(it.next().is_none(), "too many LoRA params");
    }

    /// Embed on the CPU (frozen, quantized) and upload as `[1, T, hidden]`.
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor<B, 3>> {
        let hidden = self.config.hidden_size;
        let mut data = vec![0.0f32; token_ids.len() * hidden];
        for (i, &id) in token_ids.iter().enumerate() {
            self.embed
                .embed_id_add_cpu(id, &mut data[i * hidden..(i + 1) * hidden])?;
        }
        Ok(Tensor::from_data(
            TensorData::new(data, [1, token_ids.len(), hidden]),
            &self.device,
        ))
    }

    /// `mask_out` is `[T, T]`, `true` where the score must be masked out —
    /// the same convention as `llm_wasm::model::ForwardSpec::mask_out`.
    /// Returns `[1, T, K]` logits over the answer tokens.
    pub fn forward(
        &self,
        token_ids: &[u32],
        positions: &[u32],
        mask_out: &Tensor<B, 2, Bool>,
    ) -> Result<Tensor<B, 3>> {
        let t = token_ids.len();
        ensure!(positions.len() == t, "one position per token");
        ensure!(mask_out.dims() == [t, t], "mask must be [T, T]");

        let head_dim = self.config.hidden_size / self.config.num_heads;
        let (cos, sin) = self.rope_rows(positions, head_dim);

        let mut x = self.embed_tokens(token_ids)?;
        for layer in &self.layers {
            let normed = rms_norm(x.clone(), &layer.attn_norm, self.config.rms_norm_eps);
            let attn = self.attention(layer, normed, &cos, &sin, mask_out, head_dim);
            x = x + attn;
            let normed = rms_norm(x.clone(), &layer.ffn_norm, self.config.rms_norm_eps);
            let gate = silu(layer.gate.forward(normed.clone()));
            let ffn = layer.down.forward(gate * layer.up.forward(normed));
            x = x + ffn;
        }
        let x = rms_norm(x, &self.out_norm, self.config.rms_norm_eps);
        Ok(self.head_sliced(x))
    }

    fn rope_rows(&self, positions: &[u32], head_dim: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let t = positions.len();
        let idx: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [t]), &self.device);
        let cos = self
            .cos
            .clone()
            .select(0, idx.clone())
            .reshape([1, t, 1, head_dim]);
        let sin = self.sin.clone().select(0, idx).reshape([1, t, 1, head_dim]);
        (cos, sin)
    }

    fn attention(
        &self,
        layer: &TrainLayer<B>,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        mask_out: &Tensor<B, 2, Bool>,
        head_dim: usize,
    ) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let n_heads = self.config.num_heads;
        let n_kv = self.config.num_kv_heads;

        let q = layer.q.forward(x.clone()).reshape([b, t, n_heads, head_dim]);
        let k = layer.k.forward(x.clone()).reshape([b, t, n_kv, head_dim]);
        let v = layer
            .v
            .forward(x)
            .reshape([b, t, n_kv, head_dim])
            .permute([0, 2, 1, 3]);

        let q = apply_rope(q, cos, sin).permute([0, 2, 1, 3]);
        let k = apply_rope(k, cos, sin).permute([0, 2, 1, 3]);

        let n_rep = n_heads / n_kv;
        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);

        let scale = (head_dim as f32).powf(-0.5);
        let kt = k.swap_dims(2, 3);
        let mut chunks = Vec::with_capacity(t.div_ceil(ATTN_QUERY_CHUNK));
        let mut start = 0usize;
        while start < t {
            let len = ATTN_QUERY_CHUNK.min(t - start);
            let scores = q.clone().narrow(2, start, len).matmul(kt.clone()) * scale;
            let m = mask_out.clone().narrow(0, start, len).unsqueeze::<4>();
            let probs = softmax(scores.mask_fill(m, f32::NEG_INFINITY), 3);
            chunks.push(probs.matmul(v.clone()));
            start += len;
        }
        let out = if chunks.len() == 1 {
            chunks.pop().unwrap()
        } else {
            Tensor::cat(chunks, 2)
        };
        let out = out.permute([0, 2, 1, 3]).reshape([b, t, n_heads * head_dim]);
        layer.o.forward(out)
    }

    /// `LlmModel::lm_head_sliced`: broadcast-multiply-and-sum, not `matmul`
    /// — at K=2 the output width is tiny while the contraction is `hidden`,
    /// the shape regime where this backend's matmul kernel was found to be
    /// silently wrong (llm-web `model.rs`, `PV_KV_CHUNK`).
    fn head_sliced(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, t, d] = hidden.dims();
        let k = self.head.dims()[0];
        let h = hidden.reshape([1, t, 1, d]);
        let w = self.head.clone().reshape([1, 1, k, d]);
        (h * w).sum_dim(3).reshape([1, t, k])
    }
}

/// `x / sqrt(mean(x^2) + eps) * gamma` — `burn::nn::RmsNorm::forward`.
pub fn rms_norm<B: Backend>(x: Tensor<B, 3>, gamma: &Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let rms = (x.clone().powi_scalar(2).mean_dim(2) + eps).sqrt();
    (x / rms) * gamma.clone().unsqueeze()
}

/// cos/sin tables `[max_seq_len, head_dim]`, HF `emb = cat([freqs, freqs])`.
fn rope_tables<B: Backend>(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let mut emb = vec![0.0f32; max_seq_len * head_dim];
    for pos in 0..max_seq_len {
        for j in 0..half {
            let v = pos as f32 * inv_freq[j];
            emb[pos * head_dim + j] = v;
            emb[pos * head_dim + half + j] = v;
        }
    }
    let emb: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(emb, [max_seq_len, head_dim]), device);
    (emb.clone().cos(), emb.sin())
}

/// `x * cos + rotate_half(x) * sin` over `[1, T, H, Dh]`.
fn apply_rope<B: Backend>(x: Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
    let half = x.dims()[3] / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let rotated = Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3);
    x * cos.clone() + rotated * sin.clone()
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let n_kv = x.dims()[1];
    let mut heads = Vec::with_capacity(n_kv * n_rep);
    for h in 0..n_kv {
        let slice = x.clone().narrow(1, h, 1);
        for _ in 0..n_rep {
            heads.push(slice.clone());
        }
    }
    Tensor::cat(heads, 1)
}

/// Mean cross-entropy of `logits` `[1, T, K]` against `targets` (one class
/// index per **selected** row), restricted to rows `rows`.
///
/// Variant B asks the same question at every cell, so the loss is the mean
/// over cells of `-log p(true next state)` — there is no sequence to sum
/// over, only a grid.
pub fn cell_cross_entropy<B: Backend>(
    logits: Tensor<B, 3>,
    rows: &[u32],
    targets: &[u8],
    device: &B::Device,
) -> Tensor<B, 1> {
    assert_eq!(rows.len(), targets.len());
    let [_, _, k] = logits.dims();
    let n = rows.len();
    let idx: Vec<i64> = rows.iter().map(|&r| r as i64).collect();
    let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [n]), device);
    let t = logits.dims()[1];
    let cell_logits = logits.reshape([t, k]).select(0, idx);
    let logp = log_softmax(cell_logits, 1);

    // One-hot the targets and pick with a sum — `gather` over a 2-wide axis
    // is the same arithmetic and this keeps the graph to elementwise ops.
    let mut onehot = vec![0.0f32; n * k];
    for (i, &t) in targets.iter().enumerate() {
        onehot[i * k + t as usize] = 1.0;
    }
    let onehot: Tensor<B, 2> = Tensor::from_data(TensorData::new(onehot, [n, k]), device);
    -(logp * onehot).sum().div_scalar(n as f32)
}
