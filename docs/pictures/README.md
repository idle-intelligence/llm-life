# Pictures

Output of `llm-life picture` (variant B) and `llm-life picture-a` (variant A),
both native. Four PGMs per seed at generation 1, plus a `<tag>-picture.md`
with the per-generation table.

| file | what it is |
|---|---|
| `<tag>-<seed>-gen1-palive.pgm` | the model's p(alive) per cell, as gray — the interesting pixel |
| `<tag>-<seed>-gen1-argmax.pgm` | p(alive) thresholded — what the model "says" |
| `<tag>-<seed>-gen1-true.pgm` | true Life's next generation from the same input |
| `<tag>-<seed>-gen1-diff.pgm` | cells where the thresholded grid and true Life disagree |

`<tag>` is `b` for variant B, `a` for variant A, and `a-<attempt>` for the
attempts at making `1` reachable (below).

PGM (binary P5, 8-bit gray, black = alive) because it needs no dependency and
every viewer and every Python script reads it. `zoom` is baked in so a 64x64
grid is legible without a zooming viewer.

Accuracy alone is a bad headline number: a Life grid is mostly dead, so
answering `0` everywhere already scores ~90%. `first-picture.md` therefore
also reports live recall (accuracy restricted to cells true Life says are
alive) and the confidence gap (mean p(alive) on truly-live cells minus mean
p(alive) on truly-dead cells, i.e. whether the model separates them at all).

## Variant A

`a-picture.md` is variant A at 64x64: one templated prompt per cell
(`Neighbors: … / Self: … / Next: `, 28 Qwen tokens — the tokenizer splits every
digit and every space into its own token, which is why it is not the ~10 of
CONCEPT.md §2), packed 64 cells to a chunk behind a shared rules prefix that
stays resident in the KV cache. 64 chunks per generation at 64x64;
114k tokens per generation against variant B's 4.2k.

Chunk size is the one knob that matters: the chunk's attention is quadratic in
its token count, so 256 cells/chunk (7168 tokens, one chunk at 16x16) measured
*slower* than 4 chunks of 64 cells for the same work. 64 it is.

Both variants are scored at two thresholds now. `p(alive) >= 0.5` is what the
model says; the **grid median** of p(alive) is the same ranking calibrated to
the grid, equivalent to thresholding the logit difference `1`-`0`. The median
is handed the live fraction for free, so its live recall is an upper bound on
what the model has ordered correctly, not an accuracy claim.

### Variant A vs variant B, 64x64, Qwen2.5-0.5B-Instruct Q4_0, teacher-forced

Native, `--chunk-cells 64`, 64 chunks/generation. Timings **provisional**: the
Metal GPU was shared with another repo's training job throughout, and the
per-generation time swung between 519 s and 957 s for identical work.

| variant | seed | gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | model live / 4096 | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| B | glider | 1 | 0.9988 | 0.000 | — | — | +0.113 | ~0 | — |
| B | 1 | 1 | 0.6841 | 0.000 | — | — | +0.023 | ~0 | — |
| B | 2 | 1 | 0.6814 | 0.000 | — | — | +0.021 | ~0 | — |
| A | glider | 1 | 0.0010 | 0.400 | 0.7170 | 0.200 | -0.2013 | 4091 | 956 |
| A | glider | 2 | 0.0010 | 0.400 | 0.7173 | 0.400 | -0.2273 | 4091 | 950 |
| A | glider | 3 | 0.0010 | 0.400 | 0.7170 | 0.200 | -0.2240 | 4091 | 519 |
| A | 1 | 1 | 0.2998 | 0.560 | 0.4524 | 0.423 | -0.0945 | 3022 | 569 |
| A | 1 | 2 | 0.2888 | 0.475 | 0.4263 | 0.335 | -0.1193 | 2860 | 828 |
| A | 1 | 3 | 0.2439 | 0.484 | 0.4224 | 0.335 | -0.1382 | 3061 | 947 |
| A | 2 | 1 | 0.2864 | 0.517 | 0.4285 | 0.387 | -0.1136 | 2968 | 947 |
| A | 2 | 2 | 0.2866 | 0.471 | 0.4170 | 0.333 | -0.1246 | 2856 | 943 |
| A | 2 | 3 | 0.2490 | 0.469 | 0.4175 | 0.334 | -0.1409 | 3008 | 942 |

Three generations per seed, not ten: at ~15 minutes per generation, ten would
have been eight hours of a contended GPU.

**`1` is reachable — too reachable.** Variant B's instruct model answered `0`
on 4089–4096 of 4096 cells; variant A's answers `1` on 2860–4091. Accuracy is
therefore *worse* than variant B's, because variant B was accidentally right
about the dead majority and variant A is accidentally wrong about it. The
explicit `Neighbors: … / Self: … / Next: ` template is what moved the answer,
and it overshot.

The confidence gap is **negative** for every variant A row: p(alive) is on
average lower on the cells true Life says are alive. Variant B's gap was small
but positive. So variant A has not merely mis-thresholded a good ranking — its
ranking is anti-correlated with the rule, and the median threshold (which
fixes any pure threshold error) still only reaches 0.42–0.45 accuracy on the
random seeds. Nothing here is Life; it is the model's prior over digit
continuations, rendered per cell.

### Making `1` reachable — three attempts (64x64, seed 1, generation 1)

| attempt | accuracy | live recall | acc (median) | live recall (median) | confidence gap | model live / 4096 | s/gen |
|---|---|---|---|---|---|---|---|
| A, rules only | 0.2998 | 0.560 | 0.4524 | 0.423 | -0.0945 | 3022 | 569 |
| (a) A + 6-example few-shot prefix | **0.6252** | **0.574** | 0.5906 | **0.643** | **+0.0400** | 1727 | 849 |
| (b) base (non-instruct) Qwen2.5-0.5B Q4_0 | — | — | — | — | — | — | — |
| (c) median threshold instead of 0.5 | see the "(median)" columns — it is computed for every row | | | | | | |

True live count on that grid is 1294.

**(a) few-shot is what worked.** Six worked neighborhood→next examples covering
birth, survival (2 and 3 neighbors), overcrowding, loneliness and a dead cell
with 2 neighbors — 255 prefix tokens instead of 79. Accuracy doubles
(0.30 → 0.63), the model's live count drops from 3022 to 1727 against a true
1294, and the **confidence gap turns positive for the first time in variant A**
(-0.094 → +0.040): p(alive) is now higher on the cells true Life says are
alive. The ranking, not just the threshold, moved. Variant B's gap on the same
seed was +0.023 with live recall 0.000, so few-shot variant A is the first
configuration in this repo that both orders cells correctly *and* answers `1`.

**(b) blocked, not skipped.** Base Q4_0 GGUFs of Qwen2.5-0.5B do exist —
`QuantFactory/Qwen2.5-0.5B-GGUF` and `RichardErkhov/Qwen_-_Qwen2.5-0.5B-gguf`
were both downloaded — but neither loads:

```
Error: Expected Q4_0 for 'token_embd.weight', got Q8_0
```

llama.cpp keeps the token embedding at Q8_0 in a "Q4_0" build of a model this
small; the official `Qwen/Qwen2.5-0.5B-Instruct-GGUF` q4_0 happens to be the
exception (Q4_0 embedding, Q8_0 `output.weight`, which is what the first worker
already handled). Loading a base model needs Q8_0 *embedding* support in
llm-web's `Q4ModelLoader` — a real engine change, not header parsing — and that
was not attempted tonight.

**(c) median threshold is not enough on its own.** It is reported for every row
above. On rules-only variant A it lifts accuracy 0.30 → 0.45 and no further,
because the underlying ranking is anti-correlated; once few-shot fixes the
ranking, the median and 0.5 agree to within 0.04.
