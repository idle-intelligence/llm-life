# Variant B — stencil mask, native

model: /box/models/qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 16x16 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: bag (all grid tokens share one position id)
mask: prefix (causal) + self + 8 neighbors (dense)
head: sliced to the two answer tokens
forward: f32 training model (pure Burn ops), lora artifacts/lora-b-16-s2.bin
prefix: 68 tokens (rules only)
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed 1

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.8281 | 1.0000 | +0.9997 | 83 | 83 | 1.83 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7695 | 1.0000 | +0.9993 | 68 | 68 | 0.12 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7891 | 1.0000 | +0.9995 | 73 | 73 | 0.11 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7734 | 1.0000 | +0.9993 | 69 | 69 | 0.11 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7852 | 1.0000 | +0.9995 | 72 | 72 | 0.11 |

## seed 2

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.7852 | 1.0000 | +0.9998 | 72 | 72 | 0.11 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7344 | 1.0000 | +0.9993 | 59 | 59 | 0.11 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7344 | 1.0000 | +0.9996 | 59 | 59 | 0.11 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7227 | 1.0000 | +0.9996 | 56 | 56 | 0.11 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.6797 | 1.0000 | +0.9997 | 45 | 45 | 0.11 |

## seed 3

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.8125 | 1.0000 | +0.9997 | 79 | 79 | 0.11 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7227 | 1.0000 | +0.9994 | 56 | 56 | 0.11 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7461 | 1.0000 | +0.9997 | 62 | 62 | 0.11 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7578 | 1.0000 | +0.9995 | 65 | 65 | 0.11 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7578 | 1.0000 | +0.9996 | 65 | 65 | 0.11 |

