# Compare page, all twelve rows in one tab (2026-09-22)

TC's Chrome on an Apple M2 (WebGPU on Metal), web/compare/index.html at main feb0810, grid 16x16, one random board, "run every method". Cells correct is against the true B3/S23 update of the same board. Times are one generation, as the page reports them.

| method | s / generation | parameters | cells correct |
|---|---|---|---|
| Game of Life (rule) | 5.43e-6 | no parameters | 256 / 256 |
| lookup table | 2.98e-6 | 512-entry table | 256 / 256 |
| LLM per cell (base) | 30.92 | 0.5B (no adapter) | 156 / 256 |
| LLM per cell (trained) | 28.46 | 0.5B (+ adapter) | 256 / 256 |
| LLM per cell (trained, batched) | 31.97 | 0.5B (+ adapter) | 256 / 256 |
| LLM whole grid (base) | 0.5740 | 0.5B (no adapter) | 173 / 256 |
| LLM whole grid (trained) | 0.6364 | 0.5B (+ adapter) | 256 / 256 |
| BERT of Life | 1.06 | 3,490 | 256 / 256 |
| BERT of Life (batched) | 0.0195 | 3,490 | 256 / 256 |
| 9 numbers to centre | 0.7794 | 1,442 | 256 / 256 |
| 9 numbers to centre (batched) | 0.0253 | 1,442 | 256 / 256 |
| stencil (grid to grid) | 0.0928 | 3,329 | 256 / 256 |

## Observations

- The base model fails on both prompts: per cell it gets 156 of 256, whole grid it answers dead everywhere (173 is the board's dead count). Every trained row and every from-scratch model is exact.
- Batching the per-cell LLM (64 cells packed in one sequence with a block mask) is not faster than one forward per cell at this size: 31.97 s vs 28.46 s.
- Whole grid in one prompt is about 50x faster than per cell; the from-scratch models batched are 100x to 1000x faster than the LLM rows.
