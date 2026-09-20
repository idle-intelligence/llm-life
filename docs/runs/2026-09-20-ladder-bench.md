# Ladder bench

machine: Darwin 25.3.0 arm64
backend: wgpu
board: density 0.28, seed 1
reps: 5 (median), LLM rungs: 1
sizes: [16, 32, 64], llm-sizes: [16]

## Table

| rung | 16² s/gen | 32² s/gen | 64² s/gen | note |
|---|---|---|---|---|
| CPU loop | 0.000002 | 0.000006 | 0.000024 |  |
| 512-entry lookup | 0.000002 | 0.000006 | 0.000026 |  |
| LLM per pixel (adapter) | 19.019382 | — | — |  |
| LLM batched (adapter) | 19.019382 | — | — | identical forward to per-pixel (LADDER.md) |
| BERT of Life | 0.001527 | 0.005218 | 0.007712 |  |
| 9 numbers -> centre | 0.000501 | 0.000594 | 0.000712 |  |
| stencil (grid -> grid) | 0.001423 | 0.001511 | 0.002687 |  |

