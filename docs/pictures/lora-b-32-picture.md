# Variant B — stencil mask, native

model: qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 32x32 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: bag (all grid tokens share one position id)
mask: prefix (causal) + self + 8 neighbors (dense)
head: sliced to the two answer tokens
forward: f32 training model (pure Burn ops), lora artifacts/lora-b-32.bin
prefix: 68 tokens (rules only)
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed 1

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.8008 | 1.0000 | +0.9992 | 307 | 307 | 3.06 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7461 | 1.0000 | +0.9988 | 251 | 251 | 0.26 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7490 | 1.0000 | +0.9993 | 254 | 254 | 0.26 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7285 | 1.0000 | +0.9989 | 233 | 233 | 0.26 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7344 | 1.0000 | +0.9992 | 239 | 239 | 0.26 |

## seed 2

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.8594 | 1.0000 | +0.9992 | 367 | 367 | 0.26 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7676 | 1.0000 | +0.9974 | 273 | 273 | 0.26 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7686 | 1.0000 | +0.9991 | 274 | 274 | 0.26 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7480 | 1.0000 | +0.9990 | 253 | 253 | 0.26 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7441 | 1.0000 | +0.9991 | 249 | 249 | 0.26 |

## seed 3

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.7959 | 1.0000 | +0.9992 | 302 | 302 | 0.26 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7363 | 1.0000 | +0.9986 | 241 | 241 | 0.26 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7529 | 1.0000 | +0.9992 | 258 | 258 | 0.26 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7471 | 1.0000 | +0.9990 | 252 | 252 | 0.26 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7480 | 1.0000 | +0.9988 | 253 | 253 | 0.26 |

