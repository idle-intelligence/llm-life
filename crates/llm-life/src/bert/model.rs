//! BERT of Life: a small transformer encoder over the 9 neighbourhood cells
//! as tokens (CONCEPT.md §11 "brute force with the right data" pole,
//! "BERT of Life" backlog item).
//!
//! Input is 9 tokens — the 8 neighbours (NW,N,NE,W,E,SW,S,SE, the same
//! `Grid::neighbor_indices` order variant A and B use) then the cell's own
//! state last, each a `0`/`1` id into a 2-row embedding — plus a learned
//! position embedding over the 9 slots. No causal mask: every token attends
//! every token, since the whole neighbourhood is known at once (this is an
//! encoder, not an autoregressive decoder). The read-out takes the encoded
//! centre token (index 8, the cell's own state) and projects to 2 classes
//! (dead/alive next generation) — the same `[dead, alive]` convention variant
//! A/B's sliced LM head uses, so `score::score` and friends need no changes.
//!
//! Reference: burn-nn 0.20's own `TransformerEncoder`
//! (`burn-nn-0.20.1/src/modules/transformer/encoder.rs`), used as a library
//! component, not reimplemented — this rung is about model *size*, not about
//! writing attention from scratch again (variant B already did that).

use burn::config::Config;
use burn::module::Module;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

/// Neighbourhood tokens per cell: 8 neighbours + the cell's own state.
pub const SEQ_LEN: usize = 9;

#[derive(Config, Debug)]
pub struct BertConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Feed-forward width. Defaults to `4*d_model`, the usual transformer
    /// ratio, small enough here that it barely matters.
    pub d_ff: usize,
}

impl BertConfig {
    pub fn small(d_model: usize, n_layers: usize, n_heads: usize) -> Self {
        BertConfig::new(d_model, n_layers, n_heads, d_model * 4)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BertOfLife<B> {
        BertOfLife {
            tok_emb: EmbeddingConfig::new(2, self.d_model).init(device),
            pos_emb: EmbeddingConfig::new(SEQ_LEN, self.d_model).init(device),
            encoder: TransformerEncoderConfig::new(self.d_model, self.d_ff, self.n_heads, self.n_layers)
                .with_dropout(0.0)
                .init(device),
            head: LinearConfig::new(self.d_model, 2).init(device),
            d_model: self.d_model,
        }
    }
}

#[derive(Module, Debug)]
pub struct BertOfLife<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    encoder: TransformerEncoder<B>,
    head: Linear<B>,
    d_model: usize,
}

impl<B: Backend> BertOfLife<B> {
    /// `tokens`: `[batch, 9]`, values 0/1 (see module docs for token order).
    /// Returns `[batch, 2]` logits, `[dead, alive]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [b, t] = tokens.dims();
        let device = tokens.device();
        let x = self.tok_emb.forward(tokens);
        let pos_ids: Tensor<B, 1, Int> = Tensor::arange(0..t as i64, &device);
        let pos_ids = pos_ids.unsqueeze::<2>().repeat_dim(0, b);
        let p = self.pos_emb.forward(pos_ids);
        let x = x + p;
        let out = self.encoder.forward(TransformerEncoderInput::new(x));
        let center = out.slice([0..b, SEQ_LEN - 1..SEQ_LEN]).reshape([b, self.d_model]);
        self.head.forward(center)
    }

    pub fn num_params(&self) -> usize {
        self.tok_emb.weight.dims().iter().product::<usize>()
            + self.pos_emb.weight.dims().iter().product::<usize>()
            + self.encoder_params()
            + self.head.weight.dims().iter().product::<usize>()
            + self.head.bias.as_ref().map(|b| b.dims().iter().product::<usize>()).unwrap_or(0)
    }

    fn encoder_params(&self) -> usize {
        // Standard per-layer count: 4 * d^2 (q,k,v,o) + 2 * d * d_ff (pwff) +
        // biases + 2 layer norms (2*d each). Computed rather than walked
        // through the module tree, since `TransformerEncoderLayer`'s fields
        // are private to burn-nn.
        let d = self.d_model as f64;
        let d_ff = self.encoder.d_ff as f64;
        let n_layers = self.encoder.n_layers as f64;
        let attn = 4.0 * d * d + 4.0 * d;
        let pwff = 2.0 * d * d_ff + d + d_ff;
        let ln = 4.0 * d; // two LayerNorms, weight+bias each
        ((attn + pwff + ln) * n_layers) as usize
    }
}

/// One-layer MLP baseline on the raw 9 bits — the "honest floor" of a model
/// that is not shaped like an encoder at all (CONCEPT.md task: "a one-layer
/// MLP baseline on the 9 bits").
#[derive(Module, Debug)]
pub struct MlpOfLife<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    pub hidden: usize,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MlpOfLife<B> {
        MlpOfLife {
            fc1: LinearConfig::new(SEQ_LEN, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, 2).init(device),
        }
    }
}

impl<B: Backend> MlpOfLife<B> {
    /// `bits`: `[batch, 9]` f32, already 0.0/1.0 (no embedding — the MLP eats
    /// the raw bits directly).
    pub fn forward(&self, bits: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::relu(self.fc1.forward(bits));
        self.fc2.forward(h)
    }

    pub fn num_params(&self) -> usize {
        self.fc1.weight.dims().iter().product::<usize>()
            + self.fc1.bias.as_ref().map(|b| b.dims().iter().product::<usize>()).unwrap_or(0)
            + self.fc2.weight.dims().iter().product::<usize>()
            + self.fc2.bias.as_ref().map(|b| b.dims().iter().product::<usize>()).unwrap_or(0)
    }
}

/// The zero-parameter floor: a direct 512-entry lookup table, indexed
/// exactly as `case_index` in `bert::data` builds it (8 neighbour bits, then
/// the cell's own state as bit 8). This is not a model — it's the classical
/// rule restated as a table, included so the ladder's "no learning at all"
/// row is on the same axis as the trained ones.
pub struct LookupTable {
    pub alive: [bool; 512],
}

impl LookupTable {
    pub fn from_rule(rule: &life::Rule) -> Self {
        let mut alive = [false; 512];
        for (k, out) in alive.iter_mut().enumerate() {
            let self_state = (k >> 8) & 1 != 0;
            let n = (0..8).filter(|b| (k >> b) & 1 != 0).count();
            *out = rule.next(self_state, n);
        }
        LookupTable { alive }
    }

    pub fn predict(&self, case: usize) -> bool {
        self.alive[case]
    }
}
