# The LLM methods move from Burn to lean

2026-10-02. The five language-model rows of the compare page ("LLM per
cell (base)", "(trained)", "(trained, batched)", "LLM whole grid (base)",
"(trained)") and the LLM modes of the demo page now run on lean (crates/lean
in idle-intelligence/llm-web), through a new crate,
`crates/llm-life-lean` (wasm build `web/pkg-lean`). Plan and survey:
docs/plans/lean-migration.md.

## What changed

llm-life (branch `lean-migration`, from `gpu-lookup`):

- `crates/llm-life-lean`: `LifeLean` (native and wasm) and the
  `LifeEngine` wasm class with the calls the tab made on Burn.
  `variant_a.rs`, `variant_b.rs` and `score.rs` are compiled from
  `crates/llm-life/src` (`#[path]`), not copied.
- `web/worker.js`: `load`/`loadAdapter`/`step` for the LLM go to
  `pkg-lean` (imported on first LLM load); BERT of Life and the two vector
  models stay on `pkg-llm` (Burn). Build tags: `ENGINE_BUILD` and both
  pages' `BUILD` are `2026-10-02`.
- `tools/publish-pages.sh` builds and ships `web/pkg-lean` too.
- `tools/parity/hf_peft_reference.py`, `crates/llm-life-lean/tests/gates.rs`
  and `scripts/headless/llm-rows.mjs`: the gates below.

llm-web (branch `lean-lora`, from `lean-stream`, not published):

| commit | change |
|---|---|
| bfba6b1 | `GpuModel::embed_head_sliced`: the `[dead, alive]` head from the token embedding rows (Q4_0), the head the adapters were trained against; lean's `lm_head_sliced` reads `output.weight` (Q8_0) |
| c43815f | `apply_lora`/`clear_lora` reset the buffer pool; before, swapping adapters of the same shape kept running the first adapter |
| 55ac755 | multi-row K/V cache writes as one `kv_scatter` dispatch per buffer per layer instead of one `copy_buffer_to_buffer` per (row, kv head) |

One behaviour differs from the Burn tab on purpose: Burn prefilled the
per-cell prefix *before* applying the adapter, so the trained per-cell rows
read a prefix computed by the base model. Training and the HF+PEFT reference
run the prefix through the adapter; lean does too.

Still on Burn: training (`train`, `train-a`), the native drivers in
`crates/llm-life/src/bin/llm-life.rs`, BERT of Life, the 9-number MLP and
the stencil model (wasm `pkg-llm`, which trucs.ai's LLM-of-Life page loads
for those three). `crates/llm-life/src/web.rs` still contains the Burn
`LifeEngine`; no page calls it any more.

## Build

The `lean` dependency is a git dependency on llm-web with no `rev`: the
`lean-lora` branch is not on GitHub yet. Until it is, builds patch it to a
local checkout:

```
cargo test -p llm-life-lean --config 'patch."https://github.com/idle-intelligence/llm-web".lean.path="<llm-web>/crates/lean"'
LEAN_PATH=<llm-web>/crates/lean tools/publish-pages.sh
```

wasm-pack runs its own `cargo metadata`, which `-- --config` does not reach,
so a wasm-pack build needs the same patch in a `.cargo/config.toml`
(tools/build-lean.sh writes and removes one).

`LEAN_PATH=<llm-web>/crates/lean tools/build-lean.sh` builds both browser
modules: `web/pkg-lean` (WebGPU backend and the single-thread CPU backend,
SIMD128) and `web/pkg-lean-mt` (the threaded CPU backend; nightly with
rust-src, and a cross-origin-isolated page: web/serve.py sends COOP/COEP).
The workers pick WebGPU, then CPU threads, then a single CPU thread by
capability (web/lean-backend.js); `?backend=webgpu|threads|single` forces
one. The CPU backend needs lean-cpu-lora or later (runtime LoRA, ForwardSpec
chunks and the embedding-row head on the CPU).

## Parameters

| | |
|---|---|
| model | Qwen2.5-0.5B-Instruct, `qwen2.5-0.5b-instruct-q4_0.gguf` (428,730,208 bytes) |
| adapters | `lora-a-norules-300.bin`, `lora-a-rules-300.bin`, `lora-b-16-s3-300.bin`, `lora-b-32.bin` (LLMLIFE2, rank 8, alpha 16, q/k/v/o) |
| reference | transformers 5.17.0 (GGUF loader, float32, eager attention, CPU) + PEFT 0.21.0; fixture `crates/llm-life-lean/tests/fixtures/hf_peft_reference.json` |
| native | the M2 laptop, Metal, lean-lora 55ac755 |
| browser | Playwright's Chromium (Chrome for Testing 1243), headless, WebGPU on SwiftShader; servers `web/serve.py` |
| answer | argmax of the `[dead, alive]` logits at each answer position (greedy) |

## Results

### Gate (i): the 512 per-cell cases, native

From the `[gate]` lines of `cargo test -p llm-life-lean --release --test gates`.

| configuration | prefix | adapter | correct / 512 | earlier number |
|---|---|---|---|---|
| base-rules | rules | none | 215 | 215 (2026-09-20-eval-a-base-rules.md) |
| a-norules-300 | no rules | a-norules-300 | 512 | 512 (2026-09-20-runtime-lora.md) |
| a-rules-300 | rules | a-rules-300 | 509 | 509 (2026-09-20-runtime-lora.md) |
| base-fewshot | rules + 6 examples | none | 175 | none recorded |

### Gate (ii): token-exact against HF transformers + PEFT, native

| set | forwards | answers | answers differing | max abs logit diff |
|---|---|---|---|---|
| base-fewshot, 64 cells per forward | 8 | 512 | 0 | 5.722e-5 |
| base-fewshot, one cell per forward (every 8th case) | 64 | 64 | 0 | 4.578e-5 |
| base-rules, 64 cells per forward | 8 | 512 | 0 | 5.913e-5 |
| base-rules, one cell per forward (every 8th case) | 64 | 64 | 0 | 5.913e-5 |
| a-norules-300, 64 cells per forward | 8 | 512 | 0 | 2.537e-4 |
| a-norules-300, one cell per forward (every 8th case) | 64 | 64 | 0 | 1.488e-4 |
| a-rules-300, 64 cells per forward | 8 | 512 | 0 | 2.003e-4 |
| a-rules-300, one cell per forward (every 8th case) | 64 | 64 | 0 | 8.965e-5 |
| whole grid base 16x16 (grid 1) | 1 | 256 | 0 | 6.294e-5 |
| whole grid lora-b-16-s3-300 16x16 (grid 1) | 1 | 256 | 0 | 3.071e-4 |
| whole grid base 16x16 (grid 2) | 1 | 256 | 0 | 4.959e-5 |
| whole grid lora-b-16-s3-300 16x16 (grid 2) | 1 | 256 | 0 | 2.499e-4 |
| whole grid base 32x32 | 1 | 1024 | 0 | 7.629e-5 |
| whole grid lora-b-32 32x32 | 1 | 1024 | 0 | 4.058e-4 |

An earlier run of the same gate, before the K/V scatter change (55ac755),
also ran one cell per forward on all 512 cases of base-fewshot, base-rules
and a-norules-300: 0 answers differing, max abs logit diff 5.341e-5,
7.439e-5 and 2.594e-4.

Whole-grid cells correct in the same runs: grid 1 base 157/256, adapter
256/256; grid 2 base 154/256, adapter 256/256; 32x32 base 677/1024,
adapter 1024/1024.

### Gate (iii): headless browser, 16x16, fixed board

`scripts/headless/llm-rows.mjs` (the compare page's message sequence to
web/worker.js), one fixed board (81 alive of 256). Before: the Burn build
at gpu-lookup (`BUILD 2026-09-23b`). After: this branch (`BUILD
2026-10-02`). The one-cell-per-call rows were stopped after their first 8
cells and the batched row after its first 64-cell chunk (see Observations).

| row | Burn (before) | lean (after) | HF+PEFT on the same board | lean vs HF |
|---|---|---|---|---|
| LLM per cell (base), first 8 cells | 4/8 | 4/8 | 146/256 over the board | max abs p diff 8.09e-6 |
| LLM per cell (trained), first 8 cells | 8/8 | 8/8 | 256/256 over the board | max abs p diff 3.62e-6 |
| LLM per cell (trained, batched), first 64 cells | not run | 64/64 | 256/256 over the board | max abs p diff 4.69e-6 |
| LLM whole grid (base) | 164/256 | 164/256 | 164/256 | 0 cells differ |
| LLM whole grid (trained) | 256/256 | 256/256 | 256/256 | 0 cells differ |

Burn vs lean on the same rows: whole grid, 0 cells differ (base and
trained); per cell, the same 8 answers, max abs p diff 2.7e-4 (base) and
8.49e-2 (trained). Burn vs HF on the trained per-cell cells: max abs p diff
8.49e-2.

The lean batched chunk took 32.8 min under SwiftShader (15:49:25 to
16:22:10 UTC in its log), so it was not repeated on Burn.

Console errors on the compare page: none, in all four browser runs; a
separate load of `index.html` and `compare/` on the lean build: none. The
worker logs one warning per model load, `could not cache: UnknownError:
Failed to execute 'put' on 'Cache'`, in both the Burn and the lean runs.

### Served bytes

| file | bytes | sha256 |
|---|---|---|
| `web/pkg-lean/llm_life_lean_bg.wasm` | 2,572,345 | `0bfadcd57ad7778c5a5eef678369dff4d36e3230e2166a8d214935138667d419` |

`curl 'http://127.0.0.1:8123/pkg-lean/llm_life_lean_bg.wasm?v=2026-10-02' |
shasum -a 256` gave the same hash; the served `compare/` page carries
`const BUILD = '2026-10-02'` and the served `worker.js` `ENGINE_BUILD =
'2026-10-02'`. For comparison the Burn `pkg-llm/llm_life_bg.wasm` is
20,608,886 bytes.

### Native timing (sanity only)

The M2 laptop, Metal, one call after one warm-up, other workers' builds
running (load average 2-5). Not browser numbers.

| forward | before 55ac755 | after 55ac755 |
|---|---|---|
| one cell, base prefix (283 tokens resident) | 382.6 ms | 332.7 ms |
| 64 cells, base prefix | 26,286.9 ms | 17,157.9 ms |
| whole grid 16x16, base | 2,830.8 ms | 1,601.9 ms |

## Observations

- Every answer agrees with HF transformers + PEFT: 0 of 2,048 per-cell
  answers at 64 cells per forward, 0 of 256 sampled one-cell answers, 0 of
  3,072 whole-grid cells. The largest logit difference is 4.058e-4, on the
  32x32 adapter grid; the smallest HF margin between the two answers on the
  base model is 0.0017 (base-fewshot), so none of them was decided by
  noise.
- 215/512 and 512/512 reproduce the Burn-era numbers exactly, and so does
  509/512 for the rules adapter.
- The trained per-cell row is the one place Burn and lean disagree in
  probability (8.49e-2 at most over 8 cells), from the prefix being computed
  before the adapter on Burn; lean sits at 3.62e-6 from HF there. Answers
  are the same on those 8 cells and on the whole 512 natively.
- Under SwiftShader one whole-grid forward took 143.2 s (lean) and 96.5 s
  (Burn); one cell about 12 s on either. The full one-cell-per-call rows
  would take about an hour each per engine, which is why they were cut
  short; the cells they skipped are covered by the native 512-case gates,
  which run the same Rust and WGSL.
- Native lean on Metal is slow at these shapes (17 s for a 64-cell chunk
  of 1,792 tokens). lean's own runs measured native Metal prefill several
  times slower than lean in Chrome on the same machine
  (llm-web docs/runs/2026-09-28-lean-perf.md), so the browser number is
  still to be measured on a quiet machine with a GPU, not inferred from this.

## What was painful (for the next migrations)

1. lean's browser API is chat-shaped (generate, chatGenerate, prefill with
   logit masks). Anything else (custom masks and positions, sliced heads,
   adapters) is only in the Rust API, so each demo needs its own wasm crate
   linking lean as a library. Worth deciding once whether lean grows a
   lower-level wasm surface (forward with spec, sliced logits) or every
   demo keeps its own crate.
2. The pool's bind-group cache does not see buffers it did not allocate. A
   KV cache or adapter allocated on the engine and swapped for another of
   the same size silently keeps the old bindings. The caller must
   `pool.reset()` on every cache switch (`Bound` in lib.rs); lean now does
   it itself for adapters. A cache-identity check inside lean would remove
   the trap for everyone.
3. Head conventions differ between engines: Burn and the trainer read the
   embedding rows, lean read `output.weight`. On this GGUF they are the
   same matrix quantized twice and differ by up to ~0.8 in a logit. Check
   which head a model was trained against before porting.
4. Burn-era behaviour that was never checked against a reference (the
   adapter applied after the prefix) only shows up when there is one. Build
   the HF reference first and compare Burn to it too.
5. Unpublished engine branches: a git dependency with no published rev
   cannot be resolved, and wasm-pack ignores `cargo --config` patches.
   Publish the engine branch first, or plan for the local patch.
6. Per-row `copy_buffer_to_buffer` K/V writes cost nothing at decode and
   dominated every multi-row forward; demos that prefill long or pack many
   prompts hit this first.
7. SwiftShader makes the per-cell rows (256 forwards) an hour each, so the
   browser gate has to sample. A GPU-backed headless browser on the Linux
   test machine (RTX 3080) would let the full rows run.
