# Forward-pass scaling, native — 2026-09-18

machine: Apple M2, macOS 26.3.1, Metal via Burn 0.20 / wgpu. **GPU uncontended**
(no other GPU job running; these timings are not provisional).
commit: `3e3a193`
model: `qwen2.5-0.5b-instruct-q4_0.gguf`
(Qwen2.5-0.5B-Instruct: 24 layers, hidden 896, 14 heads / 2 KV heads, intermediate 4864, vocab 151936, ctx 32768)
tokenizer: `tokenizer.json`

command:

```
cargo build --release --bin llm-life
./target/release/llm-life bench-forward \
  --gguf  qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokenizer tokenizer.json
```

Defaults: `--sizes 16,32,45,64 --chunks 16,32,64,128 --reps 3 --density 0.28`, no
`--fewshot`. One warm-up forward per row, then 3 timed repetitions; the table
reports the median. Timing brackets the whole forward: `forward_hidden_spec`
(prefill with the caller-supplied mask and positions) + sliced lm-head +
`logits_to_vec` readback.

## Variant B — one full forward (prefill + sliced-head readback)

Stencil mask (prefix causal, each cell token sees the prefix + itself + its 8
neighbors), bag positions (every grid token shares one position id).
T = 68-token rules prefix + one token per cell.

| grid | cells | tokens T | median s | s/token |
|---|---|---|---|---|
| 16x16 | 256 | 324 | 1.241 | 0.003830 |
| 32x32 | 1024 | 1092 | 3.086 | 0.002826 |
| 45x45 | 2025 | 2093 | 7.797 | 0.003725 |
| 64x64 | 4096 | 4164 | 21.903 | 0.005260 |

## Variant A — one chunk's forward against the resident prefix

Block-diagonal mask, positions restarting at the prefix for every cell, 79-token
rules prefix resident in the KV cache (`snapshot`/`restore` around each chunk),
28 tokens per cell. T = 79 + 28 x cells (the KV length the pass attends over).

| chunk cells | tokens T | median s | s/token |
|---|---|---|---|
| 16 | 527 | 1.686 | 0.003200 |
| 32 | 975 | 3.180 | 0.003261 |
| 64 | 1871 | 5.877 | 0.003141 |
| 128 | 3663 | 17.352 | 0.004737 |

## Verdict

Between T = 1092 and T = 4164 variant B's forward grows as T^1.46 (3.81x the
tokens, 7.10x the time) — clearly superlinear, roughly halfway between linear
and quadratic, and variant A shows the same knee (T^1.28 over 975 -> 3663, flat
s/token up to 1871 then a jump at 3663), so the dense O(T^2) attention/mask term
overtakes the O(T) matmul term somewhere around T ~ 2000 on this model.

---

**Superseded for the "after" numbers**: `2026-09-18-forward-profile.md` broke
this table down by component, and `2026-09-18-forward-scaling-after.md`
re-measures both tables after the sparse-attention kernel and the tiled GEMM
landed. The numbers above stand as the before.
