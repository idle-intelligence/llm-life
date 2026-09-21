# Variant A — packed per-cell prompts, native

model: qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 32x32 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: restart at the prefix for every cell
mask: block-diagonal — prefix (causal) + the cell's own prompt
head: sliced to the two answer tokens
forward: f32 training model (pure Burn ops), no persistent KV cache — full prefix+chunk forward every chunk
adapter: artifacts/lora-a-norules-300.bin, prefix: 13 tokens (norules)
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed 1

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.8027 | 1.0000 | +0.9598 | 307 | 307 | 12.81 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7461 | 1.0000 | +0.9558 | 251 | 251 | 10.05 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7490 | 1.0000 | +0.9491 | 254 | 254 | 10.05 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7295 | 1.0000 | +0.9603 | 233 | 233 | 10.05 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7354 | 1.0000 | +0.9478 | 239 | 239 | 10.05 |

## seed 2

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.8594 | 1.0000 | +0.9598 | 367 | 367 | 10.02 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7676 | 1.0000 | +0.9671 | 273 | 273 | 10.08 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7686 | 1.0000 | +0.9600 | 274 | 274 | 10.07 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7480 | 1.0000 | +0.9615 | 253 | 253 | 10.08 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7451 | 1.0000 | +0.9641 | 249 | 249 | 10.07 |

## seed 3

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.7959 | 1.0000 | +0.9528 | 302 | 302 | 10.09 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7363 | 1.0000 | +0.9478 | 241 | 241 | 10.09 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7549 | 1.0000 | +0.9425 | 258 | 258 | 10.10 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7471 | 1.0000 | +0.9563 | 252 | 252 | 10.10 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7480 | 1.0000 | +0.9546 | 253 | 253 | 10.11 |

