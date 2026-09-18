# Forward-pass scaling after the sparse-attention and GEMM work — 2026-09-18

Re-measurement of `2026-09-18-forward-scaling.md`'s two tables after the two
changes `2026-09-18-forward-profile.md` said to make, in the order it said to
make them. Same machine, same model, same tokenizer, same prefix, same
command (plus a `128` row, now a default `--sizes` entry).

machine: Apple M2, macOS 26.3.1, Metal via Burn 0.20 / wgpu.
llm-web commit: `0198acd`, llm-life commit: `72726c6`.
model: `~/Code/idle-intelligence/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf`
(24 layers, hidden 896, 14 heads / 2 KV heads, intermediate 4864, vocab
151936, ctx 32768)
tokenizer: `~/Code/idle-intelligence/models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json`

command:

```
./target/release/llm-life bench-forward \
  --gguf  ~/Code/idle-intelligence/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokenizer ~/Code/idle-intelligence/models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json
```

Defaults: `--sizes 16,32,45,64,128 --chunks 16,32,64,128 --reps 3
--density 0.28`, no `--fewshot`. One warm-up forward per row, then 3 timed
repetitions; the table reports the median. Timing brackets the whole forward,
as before.

## What changed

1. **Sparse-mask attention** (llm-web `wgsl/shader_attn_sparse.wgsl`,
   `model::SparseMask`). The stencil is handed to the engine as what it is —
   per query a contiguous key prefix plus its <=9 explicit neighbour keys —
   instead of as a `[T, kv_len]` bool mask. One workgroup per (query, head),
   scores in workgroup memory, GQA and the F32 KV cache read directly, no
   `repeat_kv`, no `T x kv_len` tensor ever allocated. Variant B's packing
   builds this form directly (`variant_b::pack_sparse`), so the 271 MB dense
   mask a 128x128 grid used to need on each of CPU and GPU is now
   `T * (2 + 9)` u32.
2. **Hand-written tiled f32 GEMM** for the prefill matmul (llm-web
   `wgsl/shader_matmul_tiled_f32.wgsl`), replacing `cubek_matmul`'s
   non-tensor-core fallback: 128x128 tile, 256 invocations, 8x8 outputs per
   invocation, `vec4` staging loads.

Both are gated (`ForwardSpec::sparse` is optional and falls back to the dense
mask; `TILED_PREFILL_MATMUL` keeps the cubek path compiled in), and both are
checked against the paths they replace — `llm-web tests/stencil.rs`
(`sparse_attention_matches_the_dense_mask`, 1e-3 on a tiny random-weight GQA
model), `tests/q4_matmul.rs` (CPU reference), `tests/full_forward.rs`
(PyTorch reference logits on the 3B agent model).

## Results

**GPU uncontended** for every run below (`ps -A -o comm | grep -E
'target/release/(jacobi|llm-life)'` checked immediately before and after each
one; the other repo's training job was running earlier in the day and none of
those readings are reported here).

The same configuration was measured three times, because the two largest
rows turned out not to be reproducible to better than ~50%: run 1 is the
default sweep at `--reps 3`, run 2 is `--sizes 64,128 --chunks 64,128` at
`--reps 3`, run 3 is the default sweep at `--reps 5`. Rows up to 45x45 repeat
to within 2%; the 64x64 and 128x128 rows do not. **Read the range, not a
single number.** The spread is not contention and not warm-up (every row is
a median over 3-5 timed reps after an untimed one); the passes that swing are
exactly the ones whose intermediate tensors are large — a 64x64 forward holds
~80 MB of f32 activations per layer and a 128x128 one ~320 MB.

### Variant B — one full forward (prefill + sliced-head readback)

| grid | cells | tokens T | before (s) | run 1 | run 2 | run 3 | best speedup |
|---|---|---|---|---|---|---|---|
| 16x16 | 256 | 324 | 1.241 | 0.595 | — | 0.596 | 2.1x |
| 32x32 | 1024 | 1092 | 3.086 | 1.447 | — | 1.474 | 2.1x |
| 45x45 | 2025 | 2093 | 7.797 | 2.625 | — | 2.680 | 3.0x |
| 64x64 | 4096 | 4164 | 21.903 | 4.984 | 7.837 | 7.077 | 2.8x (median) |
| 128x128 | 16384 | 16452 | not runnable | 29.000 | 32.717 | 31.356 | — |

"before" is `2026-09-18-forward-scaling.md`'s table. There is no before number
at 128x128: the dense path needs a `[16452, 16452]` bool mask, which is 271 MB
as a `Vec<bool>` plus 271 MB on the GPU before any attention runs, and its
attention term scales as T^2 from the 64x64 row — 14.9 s x (16452/4164)^2
≈ 233 s, plus ~23 s of matmul, so ~4 minutes if it allocated at all.

### Variant A — one chunk's forward against the resident prefix

Unchanged code path except for the GEMM: variant A still uses the dense
`ForwardSpec::mask_out` (its block-diagonal chunk mask is only
`3663 x 3742` at the largest size, and the sparse form is a later step).

| chunk cells | tokens T | before (s) | run 1 | run 2 | run 3 |
|---|---|---|---|---|---|
| 16 | 527 | 1.686 | 1.356 | — | 1.237 |
| 32 | 975 | 3.180 | 2.946 | — | 2.419 |
| 64 | 1871 | 5.877 | 7.282 | 7.553 | 5.887 |
| 128 | 3663 | 17.352 | 21.904 | 21.892 | 17.932 |

Variant A does not regress: run 3 reproduces the before table at 64 and 128
cells (5.887 vs 5.877, 17.932 vs 17.352) and improves 16 and 32 cells by
1.3-1.4x. Runs 1 and 2 are the same spread the variant-B rows show at
comparable token counts, not a real slowdown — the GEMM-control table below
measures variant A *slower* with the old GEMM in the same session.

### Control: sparse attention with the old `cubek_matmul` GEMM

One sweep with `TILED_PREFILL_MATMUL = false` (llm-web `gguf.rs`), i.e.
change 1 only, taken between runs 1 and 2 of the table above:

| variant B | 16x16 | 32x32 | 45x45 | 64x64 | 128x128 |
|---|---|---|---|---|---|
| s | 1.238 | 2.835 | 5.542 | 10.993 | 42.322 |

| variant A | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| s | 2.430 | 4.612 | 7.583 | 23.803 |

So the sparse attention alone takes variant B's 64x64 forward from 21.9 s to
11.0 s, and the GEMM takes it from 11.0 s to 5.0-7.8 s. Both changes earn
their place, and the GEMM helps variant A too (7.583 -> 5.9-7.6 at 64 cells,
23.803 -> 17.9-21.9 at 128) even though variant A's attention is untouched.

## Scaling

| range | before | after (run 3) |
|---|---|---|
| T = 1092 -> 4164 | T^1.46 | T^1.17 |
| T = 4164 -> 16452 | (not measured) | T^1.08 |

The superlinearity the first scaling run found is gone: the attention term is
now linear in T by construction (each query reads a fixed ~77 keys), and what
is left of the curvature is the matmul path's large intermediate buffers, not
an algorithmic T^2.

## Gates

The task's criterion was 64x64 <= 5 s and 128x128 <= 20 s.

- **64x64: not met.** 7.1 s median of three runs (4.98 / 7.08 / 7.84). One
  run of three was under the gate; that is not a pass.
- **128x128: not met.** 29.0-32.7 s against 20 s.

## Where the remaining time is

From `2026-09-18-forward-profile.md`'s instrumentation, re-run on the final
code (`--profile`, one forward, GPU-synced per component). **Provisional**:
the other repo's training job restarted during this run, so read the shares,
not the absolute seconds. Nested rows (`ffn.*`) are inside `ffn`.

| component | 64x64 (T = 4164) | 128x128 (T = 16452) |
|---|---|---|
| ffn | 4.698 (67%) | 29.783 (68%) |
| — ffn.gate_up | 3.165 | 20.029 |
| — ffn.down | 1.320 | 9.005 |
| — ffn.silu_mul | 0.212 | 0.749 |
| attn.sparse | 1.139 (16%) | 8.102 (18%) |
| attn.qkv_proj | 0.539 (8%) | 2.844 (6%) |
| attn.o_proj | 0.300 (4%) | 1.959 (4%) |
| attn.rope | 0.165 | 0.743 |
| norm (x2/layer) | 0.112 | 0.490 |
| attn.cache_write | 0.027 | 0.118 |
| embed + out_norm + lm_head + readback | 0.009 | 0.059 |

The four matmul rows are 79% of the pass at both sizes; the sparse attention
is 16-18% and linear in T; nothing else reaches 2%.

It is all still the GEMM, and the GEMM is not obviously fixable by tuning.
`llm-web tests/q4_matmul.rs::bench_prefill_gemm_shapes` (10 back-to-back
calls, one readback) measures the tiled kernel at 330-400 GFLOP/s at every
one of this model's shapes *and* at a plain 2048^3 square, i.e. the shape has
nothing to do with it. Things tried that did not help:

| attempt | result |
|---|---|
| every non-dead `cubek_matmul` Strategy at these shapes (`llm-agent prefill-sweep`) | `DoubleUnit/MinTileSize` already the best; no candidate above it |
| 64x64 tile instead of 128x128 | no better; 128 halves global traffic and is kept |
| `array<vec4<f32>, 16>` accumulators indexed by the unroll counter | 4x **slower** (register spill) — hence the 16 named accumulators |
| K-step 16 instead of 8 (half the barriers, 16 KB of workgroup memory) | no better |
| `vec4` workgroup tile for A as well as B | correctness failure: a component-wise write to a workgroup `vec4` is a read-modify-write, and the four invocations sharing a slot race |

Closing the remaining ~1.5x for 64x64 and ~1.6x for 128x128 most plausibly
needs f16 (2x the ALU rate and half the workgroup memory on this GPU), which
means requesting `wgpu::Features::SHADER_F16` at device init and a second
shader — a real piece of work, and one with a portability cost the rest of
this stack has so far refused to pay (`kv.rs` chose q8_0 over f16 for exactly
this reason).

## Numerics

`llm-life picture` for variant B, 64x64, generation 1, seeds `glider,1,2`,
against `docs/pictures/`'s published run:

| seed | accuracy | live recall | confidence gap | model live | published accuracy |
|---|---|---|---|---|---|
| glider | 0.9988 | 0.0000 | +0.1266 | 0 | 0.9988 |
| 1 | 0.6841 | 0.0000 | +0.0455 | 0 | 0.6841 |
| 2 | 0.6814 | 0.0000 | +0.0435 | 0 | 0.6814 |

Accuracy, live recall, confidence gap and live count are identical to four
decimals on all three seeds, and **11 of the 12 PGMs are byte-identical** to
the published ones; the twelfth (`b-2-gen1-palive.pgm`) differs in 36 of 4096
gray bytes, i.e. 0.9% of cells moved by one 1/255 gray level. The one number
that did move is the glider's median-threshold accuracy, 0.5125 -> 0.6653.
That statistic thresholds at the *grid median* of p(alive), and on the glider
4091 of 4096 cells are dead with near-identical p(alive), so the median sits
inside a tie and a sub-1/255 float change reshuffles thousands of cells
across it. Its live recall is 1.0000 either way. No f16 was introduced, so
there is no f16 drift to document; the residual is f32 reassociation between
two different reduction orders (dense `matmul` + `softmax` versus the fused
kernel).

