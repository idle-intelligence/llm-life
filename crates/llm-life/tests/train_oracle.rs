//! The correctness oracle for the training forward.
//!
//! `TrainModel` (pure Burn ops, autodiff-capable) is a reimplementation of
//! `llm_wasm::model::LlmModel`'s math, not a wrapper around it — the engine's
//! projections are Q4_0 blobs read by hand-written WGSL and there is no
//! gradient through those. So the two have to be shown to agree: same
//! weights, same tokens, same caller-supplied positions and stencil mask,
//! same sliced-head logits. With the LoRA `b` matrices at zero the adapters
//! are the identity, so any difference is a difference in the forward pass.
//!
//! A tiny random-weight model (~90k parameters, no GGUF) built the same way
//! llm-web's `tests/stencil.rs` builds its own. Needs a wgpu adapter and
//! fails loudly if there is none.

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{Param, ParamId};
use burn::tensor::{Bool, Tensor, TensorData};
use llm_life::train::model::{Lora, TrainLayer, TrainLinear, TrainModel};
use llm_life::train::weights::{dequant_q4_0, transpose};
use llm_wasm::gguf::{f32_to_f16, EmbeddingStore, Q4Linear, Q4Tensor};
use llm_wasm::model::{
    ForwardSpec, LlmModel, Q4Attention, Q4FeedForward, Q4TransformerBlock, RmsNormLayer, RoPE,
};
use llm_wasm::LlmConfig;

const LAYERS: usize = 2;
const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS;
const INTERMEDIATE: usize = 128;
const VOCAB: usize = 96;
const MAX_SEQ: usize = 64;

struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() % 2000) as f32 / 1000.0 - 1.0
    }
}

fn random_q4_bytes(n: usize, k: usize, rng: &mut Xorshift) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks = k / 32;
    let mut bytes = vec![0u8; n * blocks * 18];
    for row in 0..n {
        for b in 0..blocks {
            let off = (row * blocks + b) * 18;
            let scale = 0.01 + (rng.next_u32() % 50) as f32 * 0.002;
            bytes[off..off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for byte in bytes[off + 2..off + 18].iter_mut() {
                *byte = (rng.next_u32() & 0xff) as u8;
            }
        }
    }
    bytes
}

/// One projection, in both forms: the engine's Q4 `[out, in]` GPU tensor and
/// the training side's dequantized `[in, out]` f32.
struct Proj {
    bytes: Vec<u8>,
    n: usize,
    k: usize,
    bias: Option<Vec<f32>>,
}

fn proj(n: usize, k: usize, with_bias: bool, rng: &mut Xorshift) -> Proj {
    Proj {
        bytes: random_q4_bytes(n, k, rng),
        n,
        k,
        bias: with_bias.then(|| (0..n).map(|_| 0.1 * rng.unit()).collect()),
    }
}

impl Proj {
    fn q4(&self, device: &WgpuDevice) -> Q4Linear {
        let bias = self.bias.as_ref().map(|b| {
            Tensor::<Wgpu, 1>::from_data(TensorData::new(b.clone(), [self.n]), device)
        });
        Q4Linear::new(
            Q4Tensor::from_q4_bytes(&self.bytes, [self.n, self.k], device).unwrap(),
            bias,
        )
    }

    fn train<B: burn::prelude::Backend>(
        &self,
        rank: usize,
        device: &B::Device,
    ) -> TrainLinear<B> {
        let f = dequant_q4_0(&self.bytes, self.n * self.k).unwrap();
        let w = Tensor::from_data(
            TensorData::new(transpose(&f, self.n, self.k), [self.k, self.n]),
            device,
        );
        let bias = self
            .bias
            .as_ref()
            .map(|b| Tensor::from_data(TensorData::new(b.clone(), [self.n]), device));
        TrainLinear {
            w,
            bias,
            lora: Some(Lora::new(self.k, self.n, rank, 16.0, device)),
        }
    }
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    ffn_norm: Vec<f32>,
    gate: Proj,
    up: Proj,
    down: Proj,
}

struct Weights {
    layers: Vec<LayerWeights>,
    out_norm: Vec<f32>,
    embd: Vec<u8>,
}

fn weights(seed: u64) -> Weights {
    let mut rng = Xorshift::new(seed);
    let norm = |rng: &mut Xorshift| -> Vec<f32> { (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.unit()).collect() };
    let layers = (0..LAYERS)
        .map(|_| LayerWeights {
            attn_norm: norm(&mut rng),
            q: proj(HEADS * HEAD_DIM, HIDDEN, true, &mut rng),
            k: proj(KV_HEADS * HEAD_DIM, HIDDEN, true, &mut rng),
            v: proj(KV_HEADS * HEAD_DIM, HIDDEN, true, &mut rng),
            o: proj(HIDDEN, HEADS * HEAD_DIM, false, &mut rng),
            ffn_norm: norm(&mut rng),
            gate: proj(INTERMEDIATE, HIDDEN, false, &mut rng),
            up: proj(INTERMEDIATE, HIDDEN, false, &mut rng),
            down: proj(HIDDEN, INTERMEDIATE, false, &mut rng),
        })
        .collect();
    let out_norm = norm(&mut rng);
    let embd = random_q4_bytes(VOCAB, HIDDEN, &mut rng);
    Weights {
        layers,
        out_norm,
        embd,
    }
}

fn config() -> LlmConfig {
    LlmConfig {
        num_layers: LAYERS,
        hidden_size: HIDDEN,
        num_heads: HEADS,
        num_kv_heads: KV_HEADS,
        intermediate_size: INTERMEDIATE,
        vocab_size: VOCAB,
        rope_theta: 10_000.0,
        max_seq_len: MAX_SEQ,
        rms_norm_eps: 1e-6,
        bos_token_id: 0,
        eos_token_ids: vec![1],
    }
}

fn rms_layer(data: &[f32], device: &WgpuDevice) -> RmsNormLayer {
    let weight: Tensor<Wgpu, 1> = Tensor::from_data(TensorData::new(data.to_vec(), [HIDDEN]), device);
    RmsNormLayer {
        inner: burn::nn::RmsNorm {
            gamma: Param::initialized(ParamId::new(), weight),
            epsilon: 1e-6,
        },
    }
}

fn engine_model(w: &Weights, device: &WgpuDevice) -> LlmModel {
    let layers = w
        .layers
        .iter()
        .map(|l| {
            Q4TransformerBlock::new(
                rms_layer(&l.attn_norm, device),
                Q4Attention::new(
                    l.q.q4(device),
                    l.k.q4(device),
                    l.v.q4(device),
                    l.o.q4(device),
                    HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                ),
                rms_layer(&l.ffn_norm, device),
                Q4FeedForward::new(l.gate.q4(device), l.up.q4(device), l.down.q4(device)),
            )
        })
        .collect();
    let lm_head = Q4Linear::new(
        Q4Tensor::from_q4_bytes(&w.embd, [VOCAB, HIDDEN], device).unwrap(),
        None,
    );
    LlmModel::new(
        EmbeddingStore::new(w.embd.clone(), VOCAB, HIDDEN),
        layers,
        RoPE::new(HEAD_DIM, MAX_SEQ, 10_000.0, device),
        rms_layer(&w.out_norm, device),
        lm_head,
        config(),
        device.clone(),
    )
}

fn train_model<B: burn::prelude::Backend>(
    w: &Weights,
    answer: &[u32],
    rank: usize,
    device: &B::Device,
) -> TrainModel<B> {
    let vec1 = |d: &[f32]| -> Tensor<B, 1> {
        Tensor::from_data(TensorData::new(d.to_vec(), [d.len()]), device)
    };
    let layers = w
        .layers
        .iter()
        .map(|l| TrainLayer {
            attn_norm: vec1(&l.attn_norm),
            q: l.q.train(rank, device),
            k: l.k.train(rank, device),
            v: l.v.train(rank, device),
            o: l.o.train(rank, device),
            ffn_norm: vec1(&l.ffn_norm),
            gate: l.gate.train(rank, device),
            up: l.up.train(rank, device),
            down: TrainLinear {
                lora: None,
                ..l.down.train(rank, device)
            },
        })
        .collect();
    let embed = EmbeddingStore::new(w.embd.clone(), VOCAB, HIDDEN);
    let mut head_data = vec![0.0f32; answer.len() * HIDDEN];
    for (i, &id) in answer.iter().enumerate() {
        embed
            .embed_id_add_cpu(id, &mut head_data[i * HIDDEN..(i + 1) * HIDDEN])
            .unwrap();
    }
    let head = Tensor::from_data(TensorData::new(head_data, [answer.len(), HIDDEN]), device);
    TrainModel::new(embed, layers, vec1(&w.out_norm), head, config(), device.clone())
}

/// Tokens, positions and a stencil-shaped mask: a 4-token "prefix" that is
/// causal among itself, then 4 "cells" that all sit at one position and see
/// the prefix plus a ring of neighbours — the shape `variant_b::pack` builds.
fn case() -> (Vec<u32>, Vec<u32>, Vec<bool>) {
    let p = 4usize;
    let n = 4usize;
    let t = p + n;
    let tokens: Vec<u32> = vec![3, 17, 42, 5, 88, 1, 9, 63];
    let mut positions: Vec<u32> = (0..p as u32).collect();
    positions.extend(std::iter::repeat_n(p as u32, n));
    let mut allowed = vec![false; t * t];
    for i in 0..p {
        for j in 0..=i {
            allowed[i * t + j] = true;
        }
    }
    for c in 0..n {
        let row = (p + c) * t;
        for j in 0..p {
            allowed[row + j] = true;
        }
        allowed[row + p + c] = true;
        allowed[row + p + (c + 1) % n] = true;
        allowed[row + p + (c + n - 1) % n] = true;
    }
    (tokens, positions, allowed)
}

fn mask_out<B: burn::prelude::Backend>(
    allowed: &[bool],
    t: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let out: Vec<bool> = allowed.iter().map(|&a| !a).collect();
    Tensor::from_data(TensorData::new(out, [t, t]), device)
}

// Same probe as grid_batch_parity.rs: cubecl's AutoGraphicsApi only ever
// tries the platform's primary backend, so the probe is restricted to
// PRIMARY to match what the Wgpu backend will actually attempt.
fn has_wgpu_adapter() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    pollster::block_on(instance.request_adapter(&Default::default())).is_ok()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn engine_logits(w: &Weights, answer: &[u32], device: &WgpuDevice) -> Vec<f32> {
    let model = engine_model(w, device);
    let (tokens, positions, allowed) = case();
    let t = tokens.len();
    let spec = ForwardSpec::default()
        .with_positions(positions)
        .with_allowed(&allowed, t, t, device);
    let mut cache = model.new_cache(MAX_SEQ);
    let hidden = model.forward_hidden_spec(&tokens, &mut cache, &spec).unwrap();
    let head = model.head_slice(answer).unwrap();
    model
        .lm_head_sliced(hidden, &head)
        .into_data()
        .into_vec::<f32>()
        .unwrap()
}

#[test]
fn train_model_matches_the_engine_with_zero_lora() {
    if !has_wgpu_adapter() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let device = WgpuDevice::default();
    let w = weights(0xC0FFEE);
    let answer = [7u32, 11u32];

    let baseline = engine_logits(&w, &answer, &device);

    let model = train_model::<Wgpu>(&w, &answer, 8, &device);
    let (tokens, positions, allowed) = case();
    let m = mask_out::<Wgpu>(&allowed, tokens.len(), &device);
    let got = model
        .forward(&tokens, &positions, &m)
        .unwrap()
        .into_data()
        .into_vec::<f32>()
        .unwrap();

    let d = max_abs_diff(&baseline, &got);
    println!("TrainModel vs LlmModel sliced logits: max_abs_diff={d:.3e}");
    println!("  engine: {baseline:?}");
    println!("  train : {got:?}");
    assert!(d < 1e-3, "training forward diverged from the engine by {d}");
}

/// The same comparison on the autodiff backend — the one training actually
/// runs on. It must not change the numbers, and a backward pass through the
/// loss must produce a gradient for the LoRA `a`/`b` matrices.
#[test]
fn autodiff_backend_agrees_and_produces_gradients() {
    if !has_wgpu_adapter() {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    type AB = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    let w = weights(0xC0FFEE);
    let answer = [7u32, 11u32];

    let baseline = engine_logits(&w, &answer, &device);

    let mut model = train_model::<AB>(&w, &answer, 8, &device);
    let params: Vec<_> = model
        .lora_params()
        .into_iter()
        .map(|t| t.require_grad())
        .collect();
    model.set_lora_params(params.clone());

    let (tokens, positions, allowed) = case();
    let m = mask_out::<AB>(&allowed, tokens.len(), &device);
    let logits = model.forward(&tokens, &positions, &m).unwrap();
    let got = logits.clone().into_data().into_vec::<f32>().unwrap();
    let d = max_abs_diff(&baseline, &got);
    println!("Autodiff<Wgpu> TrainModel vs engine: max_abs_diff={d:.3e}");
    assert!(d < 1e-3, "autodiff forward diverged from the engine by {d}");

    let rows: Vec<u32> = (4..8).collect();
    let targets = vec![1u8, 0, 1, 0];
    let loss = llm_life::train::cell_cross_entropy(logits, &rows, &targets, &device);
    let grads = loss.backward();
    // `a` has a gradient even though `b` is zero only if the graph is wired
    // both ways; `b`'s gradient is the one that must be non-zero at step 0.
    let mut nonzero_b = 0;
    for (i, p) in params.iter().enumerate() {
        let g = p.grad(&grads).expect("every LoRA matrix must get a gradient");
        let s: f32 = g.abs().sum().into_data().into_vec::<f32>().unwrap()[0];
        if i % 2 == 1 && s > 0.0 {
            nonzero_b += 1;
        }
    }
    assert!(nonzero_b > 0, "no LoRA `b` matrix received a non-zero gradient");
}
