# Moving the LLM methods from Burn to lean

2026-10-02. lean is the project's one inference engine (crates/lean in
idle-intelligence/llm-web: raw wgpu + hand-written WGSL, native and wasm).
This moves the five language-model rows of the compare page (and the LLM
modes of the demo page, which share web/worker.js) off the Burn engine.

## How the LLM rows run today (Burn)

- Crate `crates/llm-life`, wasm build `web/pkg-llm` (feature `web`), on the
  Burn engine `llm-wasm` (git dependency on llm-web, pinned at `11362b9`).
  The same wasm also carries BERT of Life and the two vector models.
- wasm API, `LifeEngine` (crates/llm-life/src/web.rs): `new(w, h)` after
  `initWgpuDevice()`, `appendModelShard` x N, `load(tokenizerJson,
  rulestring)`, `loadAdapter(bytes, norules)`, `stepCellA(cells, i)`,
  `stepChunkA(cells, chunk)`, `step(cells)`, plus token-count getters.
- Prompts: `variant_a.rs` (per cell: a resident prefix, then
  `"\nNeighbors: ... / Self: s / Next: "` per cell; base prefix = rules + six
  worked examples, adapter prefix = the adapter's own training prefix) and
  `variant_b.rs` (whole grid: rules prefix + one token per cell, stencil
  mask, every cell token at one position).
- Batching: per cell, 64 cells packed block-diagonally against the resident
  prefix in one forward (`stepChunkA`), or one cell per forward
  (`stepCellA`); whole grid, one forward per generation.
- Adapters: runtime LoRA on q/k/v/o (LLMLIFE2 files from
  idle-intelligence/llm-of-life-lora, 4.3 MB each), applied on top of the
  Q4_0 base without a reload. Head: the token embedding rows of `0`/`1`.

## What lean already has

lean-stream already carries what llm-life needs, built for it: runtime LoRA
on q/k/v/o in the same LLMLIFE2 format (`GpuModel::apply_lora`),
`forward_chunk_spec` with caller-supplied positions and attention bitmask
against a resident KV prefix, and a sliced lm head. Its browser API
(`LeanEngine`: generate, generateStream, chatGenerate, prefill/append with
logit masks) is chat-shaped and does not expose any of that, so llm-life
links lean as a library in its own wasm crate, as it did with Burn.

Two gaps, fixed on llm-web branch `lean-lora` (from lean-stream):

1. lean's sliced head reads `output.weight` (Q8_0). This GGUF stores the
   tied head twice, and the adapters were trained against the Q4_0
   embedding rows; the two differ by up to ~0.8 in a logit. Added
   `GpuModel::embed_head_sliced`.
2. `apply_lora` did not reset the buffer pool, so swapping one adapter for
   another of the same shape kept running the first one. Fixed.

## Adapter choice: (a) runtime LoRA in lean

| option | download for a visitor | engineering delta |
|---|---|---|
| (a) runtime LoRA in lean | 4.3 MB per adapter over the shared 429 MB base | none in kernels: already in lean (two F32 `linear` calls + one add per projection) |
| (b) merge offline | a full 429 MB model per adapter (3 more) | new GGUF exports; and merging into Q4_0 lost accuracy before (459/512 and 423/512, docs/runs/2026-09-20-merge.md) |
| (c) merge at load time in the browser | 4.3 MB per adapter | a requantize path in lean, same Q4_0 attenuation as (b) |

(a) wins on both counts and keeps the outputs exact. Proceeding with it.

## Plan

- New crate `crates/llm-life-lean` (lib + wasm, `web/pkg-lean`): the same
  `LifeEngine` JS surface on lean, with `variant_a.rs`, `variant_b.rs` and
  `score.rs` compiled from `crates/llm-life/src` so the prompts cannot
  drift. The worker imports it only for the LLM.
- Burn stays for training, the native drivers (`llm-life` binary) and the
  small from-scratch models (BERT of Life, the two vector models), which are
  also what trucs.ai loads from the published `pkg-llm`.
- Gates: native first (512 cases, token-exact vs HF transformers + PEFT),
  then wasm in headless Chromium at 16x16.
