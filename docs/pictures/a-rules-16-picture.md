# Variant A — packed per-cell prompts, native

model: qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 16x16 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: restart at the prefix for every cell
mask: block-diagonal — prefix (causal) + the cell's own prompt
head: sliced to the two answer tokens
forward: f32 training model (pure Burn ops), no persistent KV cache — full prefix+chunk forward every chunk
adapter: artifacts/lora-a-rules-300.bin, prefix: 79 tokens (rules)
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed 1

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9844 | 0.9529 | 0.9759 | 0.8281 | 1.0000 | +0.8900 | 83 | 83 | 6.62 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7695 | 1.0000 | +0.9331 | 68 | 68 | 2.57 |
| 3 | 0.9883 | 0.9595 | 0.9726 | 0.7930 | 1.0000 | +0.9081 | 73 | 72 | 2.57 |
| 4 | 0.9961 | 0.9855 | 0.9855 | 0.7734 | 1.0000 | +0.9452 | 69 | 68 | 2.57 |
| 5 | 0.9961 | 0.9861 | 0.9861 | 0.7852 | 1.0000 | +0.9271 | 72 | 71 | 2.57 |

## seed 2

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9766 | 0.9221 | 0.9861 | 0.7852 | 1.0000 | +0.8638 | 72 | 76 | 2.57 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7344 | 1.0000 | +0.9402 | 59 | 59 | 2.58 |
| 3 | 0.9883 | 0.9500 | 0.9661 | 0.7344 | 1.0000 | +0.8972 | 59 | 58 | 2.58 |
| 4 | 0.9922 | 0.9649 | 0.9821 | 0.7227 | 1.0000 | +0.8962 | 56 | 56 | 2.57 |
| 5 | 0.9922 | 0.9574 | 1.0000 | 0.6797 | 1.0000 | +0.9069 | 45 | 47 | 2.58 |

## seed 3

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9727 | 0.9146 | 0.9494 | 0.8125 | 1.0000 | +0.8676 | 79 | 78 | 2.58 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 0.7227 | 1.0000 | +0.9352 | 56 | 56 | 2.59 |
| 3 | 0.9922 | 0.9688 | 1.0000 | 0.7461 | 1.0000 | +0.9006 | 62 | 64 | 2.59 |
| 4 | 0.9961 | 0.9848 | 1.0000 | 0.7578 | 1.0000 | +0.9369 | 65 | 66 | 2.58 |
| 5 | 0.9883 | 0.9552 | 0.9846 | 0.7617 | 1.0000 | +0.9143 | 65 | 66 | 2.58 |

