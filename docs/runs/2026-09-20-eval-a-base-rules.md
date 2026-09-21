# Variant A LoRA fine-tune — a-rules (B3/S23)

machine: Darwin 25.3.0 arm64
model: qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
adapter: a-rules (rules in the prefix)
LoRA: rank 8, alpha 16, q/k/v/o
steps: 0, batch 64, lr 0.00003, seed 1
trainable parameters: 1081344
wall clock: 1655.3s

## Base (step 0)

loss 0.9874, accuracy 0.4199 (215/512)

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 56 | 56 | 1.0000 |
| survive-2 | 28 | 1 | 0.0357 |
| survive-3 | 56 | 1 | 0.0179 |
| death-lonely (<2) | 9 | 9 | 1.0000 |
| death-crowded (>3) | 163 | 148 | 0.9080 |
| stay-dead | 200 | 0 | 0.0000 |

## Loss

| step | loss | s |
|---|---|---|

## Held-out (every eval-every steps)

## Best-by-accuracy checkpoint (saved to `/tmp/lora-a-base-rules-discard.bin`)

accuracy 0.4199 (215/512), IoU 0.0554, alive recall 0.5152, dead recall 0.0593

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 56 | 56 | 1.0000 |
| survive-2 | 28 | 1 | 0.0357 |
| survive-3 | 56 | 1 | 0.0179 |
| death-lonely (<2) | 9 | 9 | 1.0000 |
| death-crowded (>3) | 163 | 148 | 0.9080 |
| stay-dead | 200 | 0 | 0.0000 |

