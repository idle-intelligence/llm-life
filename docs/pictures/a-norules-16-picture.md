# Variant A — packed per-cell prompts, native

model: /box/models/qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 16x16 (torus)
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
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.8281 | 1.0000 | +0.9580 | 83 | 83 | 15.06 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7695 | 1.0000 | +0.9692 | 68 | 68 | 2.48 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7891 | 1.0000 | +0.9695 | 73 | 73 | 2.47 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7734 | 1.0000 | +0.9779 | 69 | 69 | 2.48 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7852 | 1.0000 | +0.9593 | 72 | 72 | 2.47 |

## seed 2

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.7852 | 1.0000 | +0.9559 | 72 | 72 | 2.48 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7344 | 1.0000 | +0.9642 | 59 | 59 | 2.49 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7383 | 1.0000 | +0.9431 | 59 | 59 | 2.49 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7227 | 1.0000 | +0.9145 | 56 | 56 | 2.49 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.6836 | 1.0000 | +0.9341 | 45 | 45 | 2.49 |

## seed 3

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0.8125 | 1.0000 | +0.9581 | 79 | 79 | 2.47 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7227 | 1.0000 | +0.9568 | 56 | 56 | 2.49 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.7461 | 1.0000 | +0.9568 | 62 | 62 | 2.49 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.7578 | 1.0000 | +0.9594 | 65 | 65 | 2.49 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.7578 | 1.0000 | +0.9510 | 65 | 65 | 2.50 |

