# Runtime LoRA on the base Q4 engine (variant A)

Applies `artifacts/lora-a-{norules,rules}-300.bin` at runtime, straight
onto the base Q4_0 `LlmModel` (`llm_wasm::lora`, `Q4Attention`'s q/k/v/o
projections: `y = W_q4·x + scale·B·(A·x)`, both prefill and decode) instead
of the offline GGUF merge (`docs/runs/2026-09-20-merge.md`), which lost
accuracy at Q4_0 (459/512, 423/512) because merging a rank-8 LoRA delta into
an already-4-bit base attenuates it. The runtime path keeps the LoRA delta
in f32 and never re-quantizes anything, so no such attenuation applies.

Code: `crates/llm-wasm/src/lora.rs` (parser/GPU upload), `model.rs`
(`Q4Attention::lora`, `LlmModel::apply_lora`), `web.rs`
(`LlmEngine::loadAdapter`) in the llm-web worktree
(`.claude/worktrees/llm-life`, branch `llm-life`); `crates/llm-life/src/
web.rs`'s `LifeEngine::loadAdapter` and `web/worker.js` in this repo.

## Verification command

`crates/llm-wasm/src/bin/lora_eval.rs` (`lora-eval`, `--features native`) —
a standalone native check duplicating the minimal prompt-format/packing
logic from `variant_a.rs` (llm-wasm can't depend on llm-life). Loads the
base Q4_0 GGUF, applies the adapter via `LoraAdapter::from_bytes` +
`LlmModel::apply_lora`, then scores the exhaustive 512-case
`(neighbors, self)` lookup and one 16x16 generation (glider seed, one step)
by IoU against true Life.

```
./target/release/lora-eval \
  --gguf models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokenizer models/tokenizer.json \
  --adapter artifacts/lora-a-norules-300.bin \
  --norules --chunk-cells 64 --grid-size 16
# (drop --norules for lora-a-rules-300.bin)
```

Machine: RTX 3080, Linux, Vulkan backend. Commit: `930cca1` (llm-web
worktree) / `baaf120`.

## Results

| adapter | 512-case accuracy (runtime) | adapter's own (best checkpoint) | 16x16 IoU |
|---|---|---|---|
| a-norules | 512/512 = 1.0000 | 512/512 = 1.0000 | 1.0000 |
| a-rules | 509/512 = 0.9941 | 509/512 = 0.9941 | 1.0000 |

Both exactly match the adapter's own training-time accuracy
(`docs/runs/2026-09-20-ft-a-norules-300.md` / `-ft-a-rules-300.md`'s
best-by-accuracy checkpoint) — a much stronger result than the offline
Q4_0 merge, which lost 10.3 / 17.4 percentage points to merge-then-quantize
rounding. The runtime path pays for this with the base model's Q4_0
weights unchanged and only ~4.3 MB of extra f32 LoRA matrices resident on
GPU per adapter, applied at forward time instead of baked into the
weights.

## Timings

First run (idle GPU, 0 other compute processes):

| adapter | ms/chunk (64 cells) | chunks | total 512-lookup | 16x16 generation |
|---|---|---|---|---|
| a-norules | 1727.1 | 8 | 13817.2 ms | 1712.6 ms |
| a-rules | 892.8 | 8 | 7142.7 ms | 1766.4 ms |

The ~2x gap between the two runs' ms/chunk (both otherwise identical:
same model, same chunk size, same GPU) is autotune/warm-up noise from
cubecl's first-use kernel benchmarking (no persistent autotune cache
survives between separate process invocations here — see `gguf.rs`'s
`wgpu` feature doc comment on `burn/autotune`), not a property of either
adapter; the two forwards are numerically identical work.

The zero-adapter no-op test (`tests/lora_identity.rs`) and the format
parser's 5 unit tests (`lora::tests`) were run once each on this same box;
one `lora_identity` attempt failed with a Vulkan "Device Lost" error while
another job (`llm-life-b3`) was concurrently using 9875/10240 MiB of GPU
memory at 100% utilization — a resource-contention crash, not a code
defect (the GPU device itself became temporarily unavailable to the
process); a retry immediately after passed cleanly. All further numbers
in this doc other than this document's own timings section were measured
alone on an idle GPU (`nvidia-smi --query-compute-apps` reported 0
processes) before `llm-life-b3` started, except the two unit-test runs
below, run under contention:

- `lora::tests::*` (5/5 pass) — pure format parsing, no GPU device request.
- `lora_identity::zero_adapter_is_a_no_op` (pass on retry) — a zero-valued
  LLMLIFE2 adapter, applied through the exact same `LoraAdapter::from_bytes`
  / `LlmModel::apply_lora` path as the real adapters, reproduces the base
  model's forward output to within 1e-4 max absolute difference (measured
  under GPU contention from `llm-life-b3`; not timed).

## What remains

Headless-browser (Playwright, bundled Chromium) verification of the
`loadAdapter` wiring in `web/worker.js`/`web/index.html` was not run here
per this task's brief (no WebGPU on the M2 laptop that owns this
session) — left to the next worker.
