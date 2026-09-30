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

/// Rows of queries gathered per attention chunk (see `StencilBlock::attention`).
/// In tests this is set tiny so the 16x16/32x32 boards already exercised by
/// `tests::gather_matches_dense_mask_*` and `nonsquare_wgpu_tests` cross
/// several chunk boundaries, proving the chunk/concat path is bit-identical
/// to the unchunked one at a size cheap enough to actually run.
#[cfg(not(test))]
const ATTN_GATHER_CHUNK: usize = 1 << 16;
#[cfg(test)]
const ATTN_GATHER_CHUNK: usize = 7;

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
    ///
    /// The gather itself still blows a query up by 9x, into a
    /// `[b, heads, chunk, 9, hd]` tensor -- at the full grid (`chunk == n`)
    /// that one buffer scales with *total cell count*, not with
    /// width/height individually: at 3840x2160 (8,294,400 cells) and the
    /// deployed `d_model=16, n_heads=1` (`VEC_STENCIL_MODEL` in
    /// web/compare/index.html) it is `8_294_400 * 9 * 16 * 4` bytes =
    /// 4,777,574,400 -- confirmed by a native release run panicking with
    /// exactly that `can't allocate buffer of size` from cubecl-wgpu.
    /// Every square board this was checked against before (16..2048,
    /// <=4,194,304 cells) stayed under that threshold, so it read as a
    /// "square works, non-square doesn't" bug when it was really "small
    /// enough works, big enough doesn't" and no square board this size had
    /// been tried. Chunking the *query* dimension of the gather (k/v for
    /// the whole sequence are cheap -- `[b, heads, n, hd]`, no 9x blow-up --
    /// so only the gather output needs bounding) keeps that buffer's size
    /// independent of total grid size.
    fn attention(&self, x: Tensor<B, 3>, neighbors: Tensor<B, 1, Int>, heads: usize) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();
        let hd = d / heads;
        let n = t;
        let split = |y: Tensor<B, 3>| y.reshape([b, t, heads, hd]).permute([0, 2, 1, 3]);
        let q = split(self.q.forward(x.clone()));
        let k = split(self.k.forward(x.clone()));
        let v = split(self.v.forward(x));

        let mut chunks = Vec::with_capacity(n.div_ceil(ATTN_GATHER_CHUNK));
        for start in (0..n).step_by(ATTN_GATHER_CHUNK) {
            let end = (start + ATTN_GATHER_CHUNK).min(n);
            let rows = end - start;
            let nb_chunk = neighbors.clone().slice(start * 9..end * 9);
            // gather this chunk's queries' 9 taps out of k/v along the sequence dim
            let gather = |y: Tensor<B, 4>| -> Tensor<B, 5> {
                y.select(2, nb_chunk.clone()).reshape([b, heads, rows, 9, hd])
            };
            let k_nb = gather(k.clone());
            let v_nb = gather(v.clone());
            let q_chunk = q.clone().slice([0..b, 0..heads, start..end, 0..hd]).reshape([b, heads, rows, 1, hd]);

            let scores = (q_chunk * k_nb).sum_dim(4).reshape([b, heads, rows, 9]) / (hd as f32).sqrt();
            let w = softmax(scores, 3).reshape([b, heads, rows, 9, 1]);
            chunks.push((w * v_nb).sum_dim(3).reshape([b, heads, rows, hd]));
        }
        let out = if chunks.len() == 1 { chunks.pop().unwrap() } else { Tensor::cat(chunks, 2) };
        out.permute([0, 2, 1, 3]).reshape([b, t, d])
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

/// Repro for the 3840x2160 "6,244,398/8,294,400 correct after 3
/// generations" report: cross-backend equivalence (NdArray reference vs.
/// the wgpu backend the browser actually runs) at small non-square boards,
/// same weights (loaded via `into_record`/`load_record` so both backends
/// run byte-identical parameters), same input cells. NdArray-vs-NdArray
/// (the module above) can't see a wgpu-only kernel bug; this can.
#[cfg(all(test, feature = "cpu"))]
mod nonsquare_wgpu_tests {
    use super::*;
    use burn::backend::{NdArray, Wgpu};
    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

    fn check_wgpu_matches_reference(width: usize, height: usize) {
        let cpu_device = Default::default();
        let cfg = StencilConfig::new(8, 2, 2);
        let cpu_model: StencilOfLife<NdArray> = cfg.init(&cpu_device);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        let bytes = recorder.record(cpu_model.clone().into_record(), ()).expect("serialize record");

        let gpu_device: <Wgpu as Backend>::Device = Default::default();
        let record = recorder.load(bytes, &gpu_device).expect("deserialize record");
        let gpu_model: StencilOfLife<Wgpu> = cfg.init(&gpu_device).load_record(record);

        let n = width * height;
        // deterministic 0/1 pattern, not RNG, so it's identical on both
        // backends without relying on matching `Distribution` samplers.
        let cells_data: Vec<f32> = (0..n).map(|i| ((i * 2654435761u64 as usize) % 7 < 3) as u8 as f32).collect();

        let cpu_cells: Tensor<NdArray, 2> = Tensor::<NdArray, 1>::from_floats(cells_data.as_slice(), &cpu_device).reshape([1, n]);
        let gpu_cells: Tensor<Wgpu, 2> = Tensor::<Wgpu, 1>::from_floats(cells_data.as_slice(), &gpu_device).reshape([1, n]);

        let cpu_neighbors = stencil_neighbors::<NdArray>(width, height, &cpu_device);
        let gpu_neighbors = stencil_neighbors::<Wgpu>(width, height, &gpu_device);

        let cpu_out = cpu_model.forward(cpu_cells, cpu_neighbors);
        let gpu_out = gpu_model.forward(gpu_cells, gpu_neighbors);

        let cpu_vec = cpu_out.into_data().to_vec::<f32>().unwrap();
        let gpu_vec = gpu_out.into_data().to_vec::<f32>().unwrap();

        let mismatches: Vec<(usize, f32, f32)> = cpu_vec
            .iter()
            .zip(gpu_vec.iter())
            .enumerate()
            .filter(|(_, (a, b))| (**a - **b).abs() > 1e-3)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        assert!(
            mismatches.is_empty(),
            "{width}x{height}: {}/{n} mismatched, e.g. {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(10)]
        );
    }

    #[test]
    fn wgpu_matches_reference_32x16() {
        check_wgpu_matches_reference(32, 16);
    }

    #[test]
    fn wgpu_matches_reference_16x32() {
        check_wgpu_matches_reference(16, 32);
    }

    #[test]
    fn wgpu_matches_reference_48x16() {
        check_wgpu_matches_reference(48, 16);
    }

    #[test]
    fn wgpu_matches_reference_64x96() {
        check_wgpu_matches_reference(64, 96);
    }

    #[test]
    fn wgpu_matches_reference_square_32x32() {
        check_wgpu_matches_reference(32, 32);
    }

    // Bisection toward the reported 3840x2160 (8,294,400 cells) failure:
    // is it non-square shape, or total cell count? 2880x2880 has the exact
    // same total (2880*2880 == 3840*2160 == 8,294,400) but is square.
    #[test]
    #[ignore]
    fn wgpu_matches_reference_3840x2160() {
        check_wgpu_matches_reference(3840, 2160);
    }

    #[test]
    #[ignore]
    fn wgpu_matches_reference_square_2880x2880() {
        check_wgpu_matches_reference(2880, 2880);
    }

    #[test]
    #[ignore]
    fn wgpu_matches_reference_1920x1080() {
        check_wgpu_matches_reference(1920, 1080);
    }

}
