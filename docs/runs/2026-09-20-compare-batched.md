# Compare page: batched BERT/MLP forward, LLM one-pass row fix

machine: Darwin 25.3.0 arm64 (M2) — hot/sluggish tonight, cold-machine run pending
backend: wgpu
page: web/compare/index.html, served on 8010
board sizes: 32x32, 64x64
rungs added/changed:
- BERT of Life (cell by cell) — unchanged, `BertEngine::stepCell` per cell
- BERT of Life (batched) — new, `BertEngine::stepGrid`, one `[n, 9]` forward
- 9 numbers -> centre (cell by cell) — unchanged, `VecMlpEngine::stepCell` per cell
- 9 numbers -> centre (batched) — new, `VecMlpEngine::stepGrid`, one `[n, 9]` forward
- LLM per pixel (adapter) — unchanged, variant A, `lora-a-norules-300.bin`
- LLM, whole grid in one pass — replaces the old "LLM batched (adapter)" duplicate row (it ran variant A twice under different narration); now variant B (`engine.step`) with its own adapter, `lora-b-16-s3-300.bin` at 16x16 / `lora-b-32.bin` at 32x32, unavailable at 64x64 (no adapter trained at that size)

## s/gen table

numbers: pending cold-machine run

| method | 32² s/gen | 64² s/gen |
|---|---|---|
| BERT of Life (cell by cell) | — | — |
| BERT of Life (batched) | — | — |
| 9 numbers -> centre (cell by cell) | — | — |
| 9 numbers -> centre (batched) | — | — |
| LLM per pixel (adapter) | — | — |
| LLM, whole grid in one pass | — | (unavailable at 64²) |

## Functional check (headless Chromium, 2026-09-20)

Native parity test (`crates/llm-life/tests/grid_batch_parity.rs`): batched `[n, 9]`
forward vs. `n` per-cell `[1, 9]` forwards, argmax labels compared, 16x16 and
32x32 random boards, `BertOfLife` and `Mlp2OfLife` — both pass.
