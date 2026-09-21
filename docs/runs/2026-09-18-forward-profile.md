# Forward-pass profile by component, native — 2026-09-18

Why: `docs/runs/2026-09-18-forward-scaling.md` measured variant B's forward
growing as T^1.46 and ~3 ms/token where the arithmetic says ~0.7 GFLOP/token
of matmul should cost far less, and concluded "both terms are slow" without
saying which. This run measures it.

machine: Apple M2, macOS 26.3.1, Metal via Burn 0.20 / wgpu. **GPU
uncontended** (`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'`
empty before and after; these timings are not provisional).
llm-web commit: `e212dd6` (branch `llm-life`), llm-life commit: `c97a67a`.
model / tokenizer / prefix: identical to the scaling run.

command:

```
./target/release/llm-life bench-forward \
  --gguf qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokenizer tokenizer.json \
  --sizes 32,64 --chunks 16 --reps 1 --profile
```

`--profile` sets `LLM_PROFILE` in llm-wasm (`crates/llm-wasm/src/profile.rs`),
which brackets each component of one extra forward with a `Backend::sync`.
Burn's wgpu backend dispatches asynchronously, so without the syncs every
component but the last would read as free. The syncs also serialize the
pipeline, so the rows are "where the work is", not a budget that must sum to
the untimed wall clock above them — here they happen to sum to it within 1%,
which says the pass was already sync-bound rather than pipelined.

`attn.dense_masked` is QK^T + `mask_fill` + softmax + PV over the caller's
`[T, kv_len]` bool mask; `attn.kv_read_repeat` is the KV-cache read plus
`repeat_kv`'s GQA head materialization; `ffn` is the three SwiGLU
projections; `attn.qkv_proj` / `attn.o_proj` are the four attention
projections. All four matmul rows go through the same Q4 scratch-dequant +
`cubek_matmul` path.

## T = 1092 (32x32 grid), one forward = 3.191 s

| component | calls | total s | % | ms/call |
|---|---|---|---|---|
| ffn | 24 | 1.742 | 55.2 | 72.60 |
| attn.dense_masked | 24 | 1.054 | 33.4 | 43.91 |
| attn.qkv_proj | 24 | 0.167 | 5.3 | 6.94 |
| attn.o_proj | 24 | 0.116 | 3.7 | 4.83 |
| attn.rope | 24 | 0.026 | 0.8 | 1.09 |
| attn.kv_read_repeat | 24 | 0.021 | 0.7 | 0.86 |
| norm | 48 | 0.019 | 0.6 | 0.39 |
| attn.cache_write | 24 | 0.007 | 0.2 | 0.30 |
| embed | 1 | 0.001 | 0.0 | 1.47 |
| lm_head_sliced | 1 | 0.001 | 0.0 | 0.57 |
| out_norm | 1 | 0.000 | 0.0 | 0.36 |
| logits_readback | 1 | 0.000 | 0.0 | 0.20 |
| sum | | 3.154 | | |

## T = 4164 (64x64 grid), one forward = 22.031 s

| component | calls | total s | % | ms/call |
|---|---|---|---|---|
| attn.dense_masked | 24 | 14.922 | 67.7 | 621.74 |
| ffn | 24 | 5.902 | 26.8 | 245.92 |
| attn.qkv_proj | 24 | 0.641 | 2.9 | 26.71 |
| attn.o_proj | 24 | 0.404 | 1.8 | 16.84 |
| attn.rope | 24 | 0.073 | 0.3 | 3.03 |
| norm | 48 | 0.037 | 0.2 | 0.77 |
| attn.kv_read_repeat | 24 | 0.028 | 0.1 | 1.18 |
| attn.cache_write | 24 | 0.010 | 0.0 | 0.41 |
| embed | 1 | 0.006 | 0.0 | 5.91 |
| lm_head_sliced | 1 | 0.001 | 0.0 | 1.22 |
| out_norm | 1 | 0.001 | 0.0 | 0.73 |
| logits_readback | 1 | 0.000 | 0.0 | 0.22 |
| sum | | 22.025 | | |

## Derived

| quantity | T = 1092 | T = 4164 | growth |
|---|---|---|---|
| tokens | 1092 | 4164 | 3.81x |
| dense masked attention, s | 1.054 | 14.922 | 14.16x |
| all four matmul rows, s | 2.025 | 6.947 | 3.43x |
| everything else, s | 0.075 | 0.156 | 2.08x |
| matmul, GFLOP/s (0.72 GFLOP/token) | 388 | 431 | |
| dense attention, GFLOP/s | 383 | 386 | |

Matmul FLOPs are `2 x 358M non-embedding parameters` per token = 0.72
GFLOP/token (the 0.5B total includes a 136M-parameter embedding table that no
token touches as a matmul). Attention FLOPs are
`2 x 2 x n_heads x T x kv_len x head_dim` per layer over 24 layers.

## Verdict

**The superlinearity is entirely the dense masked attention**, and it grows
as T^2.02 between the two sizes — quadratic, exactly as its `[1, H, T,
kv_len]` score tensor says it must be. It is 33% of the pass at T = 1092 and
68% at T = 4164, which is the whole of the T^1.46 knee the scaling run saw:
the matmul term is linear (3.43x for 3.81x the tokens; the sub-linearity is
fixed per-call overhead amortizing) and everything else is negligible.

**But the matmul term is also slow, independently.** 431 GFLOP/s on an M2
whose f32 peak is ~3.6 TFLOP/s is ~12% of peak, and there is no quadratic
term hiding in it — it is just an inefficient GEMM. Removing the dense
attention entirely would leave a 64x64 forward at ~7 s, still above the 5 s
gate, and a 128x128 forward (T = 16452, 4x the tokens) at ~28 s, above the
20 s gate. So both terms have to be fixed, and the order is: attention first
(it is the larger one and it is the one that gets worse with size), matmul
second.

Note the dense attention is *also* running at ~386 GFLOP/s, i.e. the problem
there is not the kernel's efficiency but that it does ~54x more arithmetic
than the mask asks for: each of the 4096 cell queries attends to 77 keys of
4164, and the dense path computes all 4164 and then throws 98% away.

A third, non-timing cost the profile cannot show: the dense `[T, kv_len]`
bool mask is built on the CPU as a `Vec<bool>` and uploaded. At 64x64 that is
17 MB; at 128x128 it is 271 MB on the CPU plus the same on the GPU, before
any attention runs. Whatever replaces the dense path has to replace the mask
*representation* too, not just the kernel.
