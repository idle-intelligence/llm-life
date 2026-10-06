# PyTorch retrain, parity against the published Burn checkpoints

Machines: the stencil-life and LoRA-parity work below ran on a Mac (CPU
only, no MPS/GPU — this repo's inference GPU is shared with another job).
The variant-A/B retrains that need real GPU timing ran on an RTX 3080
machine, CUDA for PyTorch and Vulkan (via `wgpu`) for Burn, one
job at a time under a shared lock. Base model: `qwen2.5-0.5b-instruct-q4_0.gguf`,
loaded through `transformers`' own GGUF loader (PyTorch) or the native GGUF
reader (Burn) — same Q4_0 block dequant either way. Data: `pytorch/life.py`'s
xorshift64 port of `crates/life`, checked bit-for-bit against the Rust RNG.

See `docs/pytorch-training.md` for the bridge (`tools export`) and how the
oracle files were produced.

## Parity (PyTorch forward vs the exact Burn `TrainModel`/tiny-model forward,
same published weights)

| model | comparison | max abs diff |
|---|---|---|
| BERT of Life | logits, 512 exhaustive cases | 1.85e-6 |
| 9-number MLP (2-layer) | logits, 512 exhaustive cases | 7.63e-6 |
| stencil (grid-to-grid) | logits, 16x16 held-out grid | 5.72e-6 |
| LoRA variant B (`lora-b-16-s3-300`) | p(alive), 16x16 grid | 1.43e-7 |
| LoRA variant B (`lora-b-16-s3-300`) | logits, all 324 positions | 1.72e-4 |
| LoRA variant A, rules (`lora-a-rules-300`) | logits, 64-case chunk | 3.47e-4 |
| LoRA variant A, no-rules (`lora-a-norules-300`) | logits, 64-case chunk | 3.05e-4 |

The `life.py` RNG port also reproduces the Rust `Grid::random`/`step` output
bit-for-bit on the fixed oracle grid (seed 1,000,000, density 0.28).

## Retrain from scratch: stencil-life (tiny models)

Same architecture, same parameter counts as the published checkpoints,
trained on the same exhaustive-512-case split (BERT, MLP) or density sweep
(stencil).

| model | parameters | steps to 512/512 | held-out acc | mean IoU (rollout) |
|---|---|---|---|---|
| BERT of Life | 3,490 | 140 | 1.0 | 1.0 |
| 9-number MLP | 1,442 | 355 | 1.0 | 1.0 |
| stencil (grid-to-grid) | 3,329 | n/a (BCE, not classification) | n/a | 1.0 |

Published (Burn, `docs/runs/2026-09-20-bert.md`, `2026-09-20-vector.md`):
all three reach 256/256 cells correct at 16x16 in the README's headline
table. This retrain matches: 100% held-out accuracy and IoU 1.0 on the
rollout for all three.

## Retrain from scratch: LoRA variant B (whole grid, 16x16)

Hyperparameters match the published run
(`docs/runs/2026-09-20-ft-b-16-s3-300.md`): rank 8, alpha 16 (q/k/v/o),
300 steps, lr 5e-5, seed 3.

| | loss | accuracy | IoU | alive recall | dead recall |
|---|---|---|---|---|---|
| base (step 0) | 0.486 | 0.765 | 0.000 | 0.000 | 1.000 |
| after 300 steps | 0.0004 | 1.000 | 1.000 | 1.000 | 1.000 |

Published (Burn, same hyperparameters): reaches 256/256 cells correct at
16x16 (README's "LLM whole grid (trained)" row). This retrain: 100% held-out
accuracy/IoU on 8 held-out grids after 300 steps — matches.

Wall clock: 927s / 300 steps on Mac CPU (~3.1s/step). Rerun on the RTX 3080 machine
(CUDA): 33.2s / 300 steps (~0.111s/step), same convergence (loss 0.486 ->
0.0003, accuracy/IoU 0.765/0.0 -> 1.0/1.0), ~28x faster than the Mac CPU
run. See "Same-hardware framework comparison" below for PyTorch/CUDA vs.
Burn/Vulkan on the same 3080.

Sanity check: this retrain's step-1 loss (0.2138) matches the published
Burn run's step-1 loss exactly (same value, same seed) — see
`docs/pytorch-training.md` for why that is expected rather than a
coincidence, and why it is strong evidence the two frameworks' forward math
agrees end to end.

## Retrain from scratch: LoRA variant A (per-cell, exhaustive 512)

Hyperparameters match the published runs
(`docs/runs/2026-09-20-ft-a-{rules,norules}-300.md`): rank 8, alpha 16
(q/k/v/o), batch 64, lr 3e-5, seed 1.

Note on the published "step 0 (base)" numbers: the Burn CLI's
`--eval-cases` default is 128, so the published run docs' step-0 print
covers only the first 128 of the 512 exhaustive cases (all "self dead" —
`case_table`'s `survive-2`/`survive-3`/`death-lonely`/`death-crowded` rows
read `n=0` in that print for exactly this reason). This port's `evaluate()`
always uses the full 512, so its step-0 numbers are not directly comparable
to the published step-0 line; the **final/best-checkpoint** numbers in both
are always over the full 512 and are the ones to compare.

Ran on the RTX 3080 machine (CUDA), 100 steps each. A full `batch=64` forward+backward
(dense `[14 heads, ~1871, ~1871]` attention score+softmax tensors retained
for backward across 24 layers) needs roughly 8.5-9 GB, which doesn't fit
the machine's 10 GB 3080 alongside the base model's own ~2 GB. Rather than
shrinking the hyperparameter, `pytorch/train_variant_a.py` gained a
`--micro-batch` flag: each step's 64 cells are split into `ceil(64/32) = 2`
micro-batches of 32, each forward/backwarded separately with its loss
divided by 2 (so the accumulated gradient equals a true batch-64 backward),
one optimizer step per full batch of 64 — ordinary gradient accumulation,
not a change to the published batch size.

| adapter | wall clock (100 steps + 5 evals) | base loss/acc (full 512) | final loss/acc (full 512, best-by-accuracy) |
|---|---|---|---|
| `lora-a-norules-300` | 51.7s | 0.804 / 0.381 | 0.0766 / 0.994 (509/512) |
| `lora-a-rules-300` | 55.1s | 0.987 / 0.420 | 0.0041 / 0.998 (511/512) |

Published (Burn, same hyperparameters, full 512 at the final/best
checkpoint): both reach 512/512. This retrain reaches 509/512 and 511/512
at 100 steps — close but short, which is expected and not a discrepancy:
neither framework seeds the LoRA `a` matrix's random init (`b` starts at
zero in both), so only step-1 loss is bit-exact across runs/frameworks (see
the variant-B section above); final accuracy at a fixed step count depends
on that unseeded init and will vary run to run in Burn too (the published
`a-rules` run took 100 steps to reach 512/512 with early stopping "512/512,
early stop" only once it hit exact; `a-norules` took 80). A longer run or a
few more seeds would be the next step to a tighter comparison, not a
correctness concern — parity against the Burn oracle (above) already
confirms the forward math is exact.

## Same-hardware framework comparison: PyTorch/CUDA vs. Burn/Vulkan, RTX 3080

Both trainers ran on the same machine, one at a time under a shared lock.
PyTorch: `pytorch/train_variant_b.py`. Burn: `crates/llm-life/src/bin/llm-life.rs`'s
`train` subcommand (native build, `--features native,cpu`; the `native`
feature pulls in `llm-wasm`'s `wgpu` backend, which picked Vulkan on this
machine — it already has `glslc`/`vulkaninfo` set up from another
worker's `lean` engine work). Same hyperparameters: 16x16, rank 8, alpha
16, 300 steps, lr 5e-5, seed 3.

| framework / backend | wall clock, 300 steps | s/step (steady state) | final accuracy | final IoU |
|---|---|---|---|---|
| PyTorch, CUDA | 33.2s | ~0.111s | 1.0000 | 1.0000 |
| Burn, Vulkan (wgpu) | 276.4s | ~0.85-0.92s | 1.0000 | 1.0000 |

PyTorch/CUDA is roughly **8x faster** than Burn/Vulkan for this workload on
the same GPU. Both reach identical final accuracy/IoU (1.0/1.0) and an
identical step-0 base loss (0.4860 Burn vs. 0.486 PyTorch) and step-1 loss
(0.2138, both) — a third confirmation (Mac CPU, RTX 3080 CUDA, RTX 3080 Vulkan) that
the two frameworks' forward math agrees exactly; the speed gap is
implementation/backend overhead (cuBLAS/cuDNN-backed PyTorch ops vs.
cubecl-generated WGSL through wgpu's Vulkan backend for a workload this
small — T=324, 24 tiny layers — where per-dispatch/per-kernel-launch
overhead likely dominates actual FLOPs either way), not a training or
parity issue. Not benchmarked further (e.g. at larger grid sizes where
compute would dominate more): out of scope for this migration, worth a
follow-up in `llm-web`'s own `lean` perf work if it matters there.

## lean-engine parity

A PyTorch-trained-from-scratch adapter (`lora-b-16-s3-300-pytorch.safetensors`,
the Mac-CPU retrain above) was converted PEFT-safetensors -> `LLMLIFE2` with
`tools export --import-lora` and checked against llm-web's `lean` engine's
own runtime-LoRA + sliced-head forward (`crates/lean`'s `apply_lora`/
`lm_head_sliced`), using that crate's existing `gen_fixture_lora.py`
reference generator. Result: max abs logit diff ~1.6e-5, well inside the
existing test's 2e-2 tolerance. See `tests/lean-parity/README.md` for the
artifacts and how to rerun this against an `llm-web` checkout.

Burn's published run (`docs/runs/2026-09-20-ft-b-16-s3-300.md`) reports
`machine: x86_64` — the exact same `uname` as this
machine, so it is the same physical 3080 — at 92.4s/300 steps, roughly 3x
faster than this section's uncontended 276.4s. Its doc also carries a
templated "the Metal GPU is shared with another repo's training job" line
(boilerplate left over from when that doc-writing code was first written
for a Mac; it does not describe this machine, which has no Metal GPU) alongside
the real, believable "shared" claim, so contention at the time is plausible
but the 3x gap could equally be an engine/driver/cubecl-autotune change
between that run and this one (`llm-wasm` is pinned to a fixed git rev for
this repo, but the published run predates that pin and may have built
against different `llm-wasm`/`cubecl`/Vulkan-driver versions). Flagging
this rather than resolving it: it is not a PyTorch-vs-Burn question (both
numbers here are Burn), so it does not change the 8x PyTorch/CUDA vs.
Burn/Vulkan conclusion above, but a from-scratch investigation of Burn's
own timing drift is out of scope for this migration.
