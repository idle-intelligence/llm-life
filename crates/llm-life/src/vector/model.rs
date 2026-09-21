//! Vector-space models (CONCEPT.md §12): no tokens, no embedding table —
//! cell states enter as floats through a `Linear`.
//!
//! `AttnOfLife` is variant (i) — CONCEPT.md's "1-layer attention block over
//! 9 scalar tokens with learned positions". The MLP baseline for (i) is
//! `bert::model::MlpOfLife` (9 raw bits -> hidden -> 2 classes), reused as-is
//! rather than rewritten with a 1-wide head: it is already exactly "an MLP
//! (9->h->1)" up to the label being one-hot over 2 classes instead of a
//! single sigmoid unit, and the rest of this crate's scoring/training
//! plumbing (`cross_entropy`, `score::score`'s `[dead, alive]` convention)
//! already expects that shape.
//!
//! `StencilOfLife` is variant (ii) — jacobi2000's stencil-masked attention
//! (`jacobi2000::model::Block`, `jacobi2000::mask::Mask::Stencil`, read-only
//! reference, the idea copied not the crate) with one scalar channel: cells
//! enter as `[batch, cells]` floats, a `Linear(1, d)` embeds each one, no
//! positional embedding at all (CONCEPT.md's requirement that the model
//! "must generalise across grid size since the mask is local" — a learned
//! absolute position would break that the moment the grid size changes), a
//! few stencil-masked attention blocks mix each cell with its 8
//! `life::Grid::neighbor_indices` neighbours (toroidal, same boundary as
//! `life::Grid::step`), and a `Linear(d, 1)` reads one BCE-with-logits value
//! per cell.

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::activation::softmax;
use burn::tensor::{Int, Tensor};
use life::Grid;

fn linear_params<B: Backend>(l: &Linear<B>) -> usize {
    l.weight.dims().iter().product::<usize>()
        + l.bias.as_ref().map(|b| b.dims().iter().product::<usize>()).unwrap_or(0)
}

/// (i) a 2-layer MLP body (9->hidden->hidden->2), for the lr-fix rerun: the
/// 1-layer `bert::model::MlpOfLife` at lr=1e-2/600 steps did not reach exact
/// 512/512 (docs/runs/2026-09-20-vector.md's first sweep), and the extra
/// depth is the other knob CONCEPT.md's "smallest model that fits" search
/// asks about, alongside the corrected learning rate.
#[derive(Config, Debug)]
pub struct Mlp2Config {
    pub hidden: usize,
}

impl Mlp2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp2OfLife<B> {
        Mlp2OfLife {
            fc1: LinearConfig::new(9, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            fc3: LinearConfig::new(self.hidden, 2).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Mlp2OfLife<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> Mlp2OfLife<B> {
    pub fn forward(&self, bits: Tensor<B, 2>) -> Tensor<B, 2> {
        use burn::tensor::activation::relu;
        let h = relu(self.fc1.forward(bits));
        let h = relu(self.fc2.forward(h));
        self.fc3.forward(h)
    }

    pub fn num_params(&self) -> usize {
        linear_params(&self.fc1) + linear_params(&self.fc2) + linear_params(&self.fc3)
    }
}

/// (i) attention over the 9 neighbourhood cells as scalar float tokens.
/// Fixed sequence length (9), so — unlike `StencilOfLife` — a learned
/// absolute position embedding is fine here: this model is never asked to
/// generalise to a different neighbourhood size.
#[derive(Config, Debug)]
pub struct AttnConfig {
    pub d_model: usize,
}

impl AttnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AttnOfLife<B> {
        let d = self.d_model;
        AttnOfLife {
            embed: LinearConfig::new(1, d).init(device),
            pos: Param::from_tensor(Tensor::zeros([1, 9, d], device)),
            q: LinearConfig::new(d, d).init(device),
            k: LinearConfig::new(d, d).init(device),
            v: LinearConfig::new(d, d).init(device),
            proj: LinearConfig::new(d, d).init(device),
            head: LinearConfig::new(d, 2).init(device),
            d_model: d,
        }
    }
}

#[derive(Module, Debug)]
pub struct AttnOfLife<B: Backend> {
    embed: Linear<B>,
    pos: Param<Tensor<B, 3>>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    proj: Linear<B>,
    head: Linear<B>,
    d_model: usize,
}

impl<B: Backend> AttnOfLife<B> {
    /// `bits`: `[batch, 9]` f32, the 9 neighbourhood values (0.0/1.0) in
    /// `bert::data`'s order — 8 neighbours then self last. Returns
    /// `[batch, 2]` logits, `[dead, alive]`.
    pub fn forward(&self, bits: Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, t] = bits.dims();
        let x = self.embed.forward(bits.reshape([b, t, 1])) + self.pos.val();
        let q = self.q.forward(x.clone());
        let k = self.k.forward(x.clone());
        let v = self.v.forward(x);
        let scores = q.matmul(k.swap_dims(1, 2)) / (self.d_model as f32).sqrt();
        let w = softmax(scores, 2);
        let out = self.proj.forward(w.matmul(v));
        let center = out.slice([0..b, t - 1..t]).reshape([b, self.d_model]);
        self.head.forward(center)
    }

    pub fn num_params(&self) -> usize {
        linear_params(&self.embed)
            + self.pos.val().dims().iter().product::<usize>()
            + linear_params(&self.q)
            + linear_params(&self.k)
            + linear_params(&self.v)
            + linear_params(&self.proj)
            + linear_params(&self.head)
    }
}

/// (ii) one stencil-masked attention block: pre-norm self-attention with an
/// additive mask, pre-norm MLP, both residual — the same shape as
/// jacobi2000's `Block` (`jacobi2000::model::Block::forward`/`attention`).
#[derive(Module, Debug)]
struct StencilBlock<B: Backend> {
    norm1: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    proj: Linear<B>,
    norm2: LayerNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
    act: Gelu,
}

impl<B: Backend> StencilBlock<B> {
    fn new(d: usize, d_ff: usize, device: &B::Device) -> Self {
        StencilBlock {
            norm1: LayerNormConfig::new(d).init(device),
            q: LinearConfig::new(d, d).init(device),
            k: LinearConfig::new(d, d).init(device),
            v: LinearConfig::new(d, d).init(device),
            proj: LinearConfig::new(d, d).init(device),
            norm2: LayerNormConfig::new(d).init(device),
            ff1: LinearConfig::new(d, d_ff).init(device),
            ff2: LinearConfig::new(d_ff, d).init(device),
            act: Gelu::new(),
        }
    }

    /// `neighbors`: `[n * 9]` int indices into the sequence dim (flattened
    /// `stencil_neighbors` — row `i`'s 9 taps at `neighbors[i*9..i*9+9]`),
    /// the same for every batch element and every head.
    fn forward(&self, x: Tensor<B, 3>, neighbors: Tensor<B, 1, Int>, heads: usize) -> Tensor<B, 3> {
        let h = self.norm1.forward(x.clone());
        let a = self.attention(h, neighbors, heads);
        let x = x + self.proj.forward(a);
        let h = self.norm2.forward(x.clone());
        x + self.ff2.forward(self.act.forward(self.ff1.forward(h)))
    }

    /// O(9n), not O(n^2): every query only ever attends to its 9
    /// `life::Grid::neighbor_indices` taps (self + 8 neighbours, all always
    /// in-bounds on the toroidal grid — no masking needed, just a gather),
    /// so there is never a dense `[n, n]` score tensor to materialise.
    fn attention(&self, x: Tensor<B, 3>, neighbors: Tensor<B, 1, Int>, heads: usize) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();
        let hd = d / heads;
        let n = t;
        let split = |y: Tensor<B, 3>| y.reshape([b, t, heads, hd]).permute([0, 2, 1, 3]);
        let q = split(self.q.forward(x.clone()));
        let k = split(self.k.forward(x.clone()));
        let v = split(self.v.forward(x));

        // gather each query's 9 taps out of k/v along the sequence dim
        let gather = |y: Tensor<B, 4>| -> Tensor<B, 5> {
            y.select(2, neighbors.clone()).reshape([b, heads, n, 9, hd])
        };
        let k_nb = gather(k);
        let v_nb = gather(v);
        let q = q.reshape([b, heads, n, 1, hd]);

        let scores = (q * k_nb).sum_dim(4).reshape([b, heads, n, 9]) / (hd as f32).sqrt();
        let w = softmax(scores, 3).reshape([b, heads, n, 9, 1]);
        (w * v_nb).sum_dim(3).reshape([b, heads, n, hd]).permute([0, 2, 1, 3]).reshape([b, t, d])
    }

    fn num_params(&self) -> usize {
        let norm_params = |n: &LayerNorm<B>| {
            n.gamma.dims().iter().product::<usize>()
                + n.beta.as_ref().map(|b| b.dims().iter().product::<usize>()).unwrap_or(0)
        };
        linear_params(&self.q)
            + linear_params(&self.k)
            + linear_params(&self.v)
            + linear_params(&self.proj)
            + linear_params(&self.ff1)
            + linear_params(&self.ff2)
            + norm_params(&self.norm1)
            + norm_params(&self.norm2)
    }
}

#[derive(Config, Debug)]
pub struct StencilConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

impl StencilConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StencilOfLife<B> {
        let d = self.d_model;
        StencilOfLife {
            embed: LinearConfig::new(1, d).init(device),
            blocks: (0..self.n_layers).map(|_| StencilBlock::new(d, d * 4, device)).collect(),
            head: LinearConfig::new(d, 1).init(device),
            n_heads: self.n_heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct StencilOfLife<B: Backend> {
    embed: Linear<B>,
    blocks: Vec<StencilBlock<B>>,
    head: Linear<B>,
    n_heads: usize,
}

impl<B: Backend> StencilOfLife<B> {
    /// `cells`: `[batch, width*height]` f32, 0.0/1.0, row-major
    /// (`life::Grid::cells` order). `neighbors`: `[n * 9]` int indices from
    /// `stencil_neighbors`, built for the same `width, height` as `cells`.
    /// Returns `[batch, width*height]` logits (BCE-with-logits target).
    pub fn forward(&self, cells: Tensor<B, 2>, neighbors: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let [b, n] = cells.dims();
        let mut x = self.embed.forward(cells.reshape([b, n, 1]));
        for blk in &self.blocks {
            x = blk.forward(x, neighbors.clone(), self.n_heads);
        }
        self.head.forward(x).reshape([b, n])
    }

    pub fn num_params(&self) -> usize {
        linear_params(&self.embed) + linear_params(&self.head) + self.blocks.iter().map(|b| b.num_params()).sum::<usize>()
    }
}

/// The stencil neighbour table (CONCEPT.md §12 / jacobi2000's
/// `Mask::Stencil { radius: 1, dilation: 1 }`, specialised to the exact 8
/// `life::Grid::neighbor_indices` neighbours instead of a Chebyshev ball, so
/// it is precisely the neighbourhood `life::Grid::step` reads). Flat
/// `[n * 9]` int indices — row `i`'s 9 taps (its 8 neighbours, then itself
/// last) at `[i*9..i*9+9]` — never a dense `[n, n]` mask: every query has
/// exactly 9 valid keys (the grid is toroidal, so all 9 are always
/// in-bounds), so this is O(9n) memory instead of O(n^2). Depends only on
/// `width, height`, not on any cell values, so the same table serves every
/// grid of that size and a model trained at one size needs a freshly built
/// table (not retrained weights) to run at another.
pub fn stencil_neighbors<B: Backend>(width: usize, height: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    let g = Grid::new(width, height);
    let n = width * height;
    let mut data = Vec::with_capacity(n * 9);
    for i in 0..n {
        data.extend(g.neighbor_indices(i).iter().map(|&j| j as i32));
        data.push(i as i32);
    }
    Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(data, [n * 9]), device)
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    //! Equivalence test for the O(n^2)-mask -> O(9n)-gather rewrite
    //! (docs/runs/2026-09-20-stencil-any-size.md): the dense `[n, n]`
    //! additive-mask attention this module used to run is reproduced here
    //! byte-for-byte (same q/k/v/proj/norm/ff weights, same softmax), and
    //! its output is compared against `StencilBlock::forward`'s new
    //! neighbour-gather path on random grids. NdArray, not wgpu: this is a
    //! numerics check, not a benchmark.
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray;

    fn dense_mask(width: usize, height: usize, device: &<B as Backend>::Device) -> Tensor<B, 2> {
        let g = Grid::new(width, height);
        let n = width * height;
        let mut data = vec![f32::NEG_INFINITY; n * n];
        for i in 0..n {
            data[i * n + i] = 0.0;
            for j in g.neighbor_indices(i) {
                data[i * n + j] = 0.0;
            }
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([n, n])
    }

    /// The removed `StencilBlock::attention`/`forward`, verbatim, taking a
    /// dense additive mask instead of the gather table.
    fn dense_block_forward(blk: &StencilBlock<B>, x: Tensor<B, 3>, mask: Tensor<B, 4>, heads: usize) -> Tensor<B, 3> {
        let h = blk.norm1.forward(x.clone());
        let [b, t, d] = h.dims();
        let hd = d / heads;
        let split = |y: Tensor<B, 3>| y.reshape([b, t, heads, hd]).permute([0, 2, 1, 3]);
        let q = split(blk.q.forward(h.clone()));
        let k = split(blk.k.forward(h.clone()));
        let v = split(blk.v.forward(h));
        let scores = q.matmul(k.permute([0, 1, 3, 2])) / (hd as f32).sqrt() + mask;
        let w = softmax(scores, 3);
        let a = w.matmul(v).permute([0, 2, 1, 3]).reshape([b, t, d]);
        let x = x + blk.proj.forward(a);
        let h = blk.norm2.forward(x.clone());
        x + blk.ff2.forward(blk.act.forward(blk.ff1.forward(h)))
    }

    fn dense_forward(model: &StencilOfLife<B>, cells: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, n] = cells.dims();
        let mut x = model.embed.forward(cells.reshape([b, n, 1]));
        let m = mask.reshape([1, 1, n, n]);
        for blk in &model.blocks {
            x = dense_block_forward(blk, x, m.clone(), model.n_heads);
        }
        model.head.forward(x).reshape([b, n])
    }

    fn check_equivalence(width: usize, height: usize) {
        let device = Default::default();
        let cfg = StencilConfig::new(8, 2, 2);
        let model: StencilOfLife<B> = cfg.init(&device);
        let n = width * height;
        let cells: Tensor<B, 2> = Tensor::random([1, n], Distribution::Bernoulli(0.5), &device);

        let neighbors = stencil_neighbors::<B>(width, height, &device);
        let got = model.forward(cells.clone(), neighbors);

        let mask = dense_mask(width, height, &device);
        let want = dense_forward(&model, cells, mask);

        let diff = (got - want).abs().max().into_scalar();
        assert!(diff < 1e-6, "{width}x{height}: max abs diff {diff} >= 1e-6");
    }

    #[test]
    fn gather_matches_dense_mask_16x16() {
        check_equivalence(16, 16);
    }

    #[test]
    fn gather_matches_dense_mask_32x32() {
        check_equivalence(32, 32);
    }
}
