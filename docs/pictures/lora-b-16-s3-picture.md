# Variant B — stencil mask, native

model: /box/models/qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 16x16 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: bag (all grid tokens share one position id)
mask: prefix (causal) + self + 8 neighbors (dense)
head: sliced to the two answer tokens
forward: f32 training model (pure Burn ops), lora artifacts/lora-b-16-s3.bin
prefix: 68 tokens (rules only)
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed 1

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.8086 | 0.4674 | 0.5181 | 0.8281 | 1.0000 | +0.4357 | 83 | 52 | 0.76 |
| 2 | 0.7344 | 0.3645 | 0.5735 | 0.7383 | 0.9412 | +0.3666 | 68 | 78 | 0.11 |
| 3 | 0.8086 | 0.4432 | 0.5342 | 0.7891 | 1.0000 | +0.4670 | 73 | 54 | 0.11 |
| 4 | 0.7930 | 0.4479 | 0.6232 | 0.7734 | 1.0000 | +0.4500 | 69 | 70 | 0.11 |
| 5 | 0.8242 | 0.4944 | 0.6111 | 0.7852 | 1.0000 | +0.4717 | 72 | 61 | 0.11 |

## seed 2

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.8516 | 0.4933 | 0.5139 | 0.7852 | 1.0000 | +0.4848 | 72 | 40 | 0.11 |
| 2 | 0.7539 | 0.3368 | 0.5424 | 0.7344 | 1.0000 | +0.4046 | 59 | 68 | 0.11 |
| 3 | 0.7891 | 0.3165 | 0.4237 | 0.7344 | 1.0000 | +0.4385 | 59 | 45 | 0.11 |
| 4 | 0.8242 | 0.3836 | 0.5000 | 0.7227 | 1.0000 | +0.4573 | 56 | 45 | 0.11 |
| 5 | 0.8359 | 0.3333 | 0.4667 | 0.6797 | 1.0000 | +0.4630 | 45 | 39 | 0.11 |

## seed 3

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7852 | 0.4022 | 0.4684 | 0.8125 | 1.0000 | +0.4112 | 79 | 50 | 0.11 |
| 2 | 0.7188 | 0.2800 | 0.5000 | 0.7227 | 1.0000 | +0.3698 | 56 | 72 | 0.11 |
| 3 | 0.8242 | 0.3750 | 0.4355 | 0.7461 | 1.0000 | +0.4740 | 62 | 37 | 0.11 |
| 4 | 0.8008 | 0.4000 | 0.5231 | 0.7578 | 1.0000 | +0.4576 | 65 | 54 | 0.11 |
| 5 | 0.8398 | 0.5000 | 0.6308 | 0.7578 | 1.0000 | +0.4769 | 65 | 58 | 0.11 |

