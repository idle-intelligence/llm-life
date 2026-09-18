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

| variant | seed | gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | model live / 4096 | s/gen | acc (otsu) | live recall (otsu) | model live (otsu) | acc (z2) | live recall (z2) | model live (z2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | glider | 1 | 0.9988 | 0.000 | 0.5125 | 1.000 | +0.1266 | 0 | 53 | 0.9980 | 1.000 | 13 | 0.9958 | 1.000 | 22 |
| B | 1 | 1 | 0.6841 | 0.000 | 0.8162 | 1.000 | +0.0455 | 0 | 53 | 0.8252 | 0.9104 | 1778 | 0.6372 | 0.000 | 192 |
| B | 2 | 1 | 0.6814 | 0.000 | 0.8188 | 1.000 | +0.0435 | 0 | 53 | 0.7251 | 0.6559 | 1533 | 0.6316 | 0.000 | 204 |
| B + few-shot | glider | 1 | 0.9988 | 0.000 | 0.5137 | 1.000 | +0.1940 | 0 | 52 | 0.9993 | 1.000 | 8 | 0.9958 | 1.000 | 22 |
| B + few-shot | 1 | 1 | 0.6768 | 0.000 | 0.8162 | 1.000 | +0.0701 | 30 | 50 | 0.6807 | 0.4529 | 1186 | 0.6482 | 0.000 | 147 |
| B + few-shot | 2 | 1 | 0.6704 | 0.000 | 0.8188 | 1.000 | +0.0694 | 45 | 56 | 0.6741 | 0.4943 | 1320 | 0.6348 | 0.000 | 191 |
| A | glider | 1 | 0.0010 | 0.400 | 0.7170 | 0.200 | -0.2013 | 4091 | 956 | 0.0012 | 0.600 | 4092 | 0.9980 | 0.200 | 5 |
| A | glider | 2\* | 0.0010 | 0.400 | 0.7173 | 0.400 | -0.2273 | 4091 | 950 | — | — | — | — | — | — |
| A | glider | 3\* | 0.0010 | 0.400 | 0.7170 | 0.200 | -0.2240 | 4091 | 519 | — | — | — | — | — | — |
| A | 1 | 1 | 0.2998 | 0.560 | 0.4524 | 0.423 | -0.0945 | 3022 | 569 | 0.3018 | 0.5549 | 3002 | 0.6841 | 0.000 | 0 |
| A | 1 | 2\* | 0.2888 | 0.475 | 0.4263 | 0.335 | -0.1193 | 2860 | 828 | — | — | — | — | — | — |
| A | 1 | 3\* | 0.2439 | 0.484 | 0.4224 | 0.335 | -0.1382 | 3061 | 947 | — | — | — | — | — | — |
| A | 2 | 1 | 0.2864 | 0.517 | 0.4285 | 0.387 | -0.1136 | 2968 | 947 | 0.2866 | 0.5134 | 2957 | 0.6814 | 0.000 | 0 |
| A | 2 | 2\* | 0.2866 | 0.471 | 0.4170 | 0.333 | -0.1246 | 2856 | 943 | — | — | — | — | — | — |
| A | 2 | 3\* | 0.2490 | 0.469 | 0.4175 | 0.334 | -0.1409 | 3008 | 942 | — | — | — | — | — | — |

\* generation > 1 has no PGM on disk (`run_pictures` only writes one at `gen == 1`); re-scoring it needs a rerun, which was not done tonight — see CLAUDE.md GPU rule and the "not run" note below.

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

**Label-free thresholds (Otsu, z-score k=2), re-scored from the `palive`/
`true` PGMs already on disk — no rerun.** On variant B, where the confidence
gap is positive, Otsu **recovers the glider almost exactly**: 13 cells alive
against a true 5 (0.9980 accuracy, full recall), and few-shot B does even
better, 8 against 5 (0.9993). z-score (k=2) is more conservative on the
glider — 22 cells, still full recall — because it looks for outliers above
mean+2·std rather than a histogram gap, and the glider's five true cells sit
inside a wider blob of moderately-elevated p(alive) that Otsu's gap-finder
catches and z-score's higher bar does not. On B's random seeds, Otsu also
beats 0.5 and the median (0.83/0.73 vs 0.68/0.68 at 0.5), while z-score
answers close to nothing alive (192/204 cells, live recall 0.000) — its
threshold sits above almost the whole positive-but-flat p(alive) distribution
outside the object. On variant A, where the ranking is anti-correlated with
the rule, both thresholds behave exactly as the diagnosis predicts: Otsu
finds *a* gap in a near-bimodal distribution but the gap is on the wrong side
(4092/4096 alive on the glider, accuracy 0.0012), while z-score's outlier
criterion happens to land close to true on the glider (5 alive, 0.9980
accuracy) by picking out the handful of cells the inverted ranking pushes
highest — not a rule recovery, a coincidence of which cells the anti-
correlation ranks extreme. Few-shot repairs A's ranking (CLAUDE.md "Code is
ground truth" — the confidence-gap sign is the tell): Otsu then gets live
recall 0.88 on both random seeds, its best number of any threshold in this
document, at the cost of overshooting live count (~2970–2985 against a true
~1300).

### Making `1` reachable — three attempts (64x64, seed 1, generation 1)

| attempt | accuracy | live recall | acc (median) | live recall (median) | confidence gap | model live / 4096 | s/gen | acc (otsu) | live recall (otsu) | model live (otsu) | acc (z2) | live recall (z2) | model live (z2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A, rules only | 0.2998 | 0.560 | 0.4524 | 0.423 | -0.0945 | 3022 | 569 | 0.3018 | 0.5549 | 3002 | 0.6841 | 0.000 | 0 |
| (a) A + 6-example few-shot prefix | **0.6252** | **0.574** | 0.5906 | **0.643** | **+0.0400** | 1727 | 849 | 0.5176 | **0.8841** | 2970 | 0.6641 | 0.006 | 98 |
| (b) base (non-instruct) Qwen2.5-0.5B Q4_0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| (c) median threshold instead of 0.5 | see the "(median)" columns — it is computed for every row | | | | | | | | | | | | |

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

### Few-shot variant A across all three seeds (64x64, generation 1)

| seed | accuracy | live recall | acc (median) | live recall (median) | confidence gap | model live | true live | s/gen | acc (otsu) | live recall (otsu) | model live (otsu) | acc (z2) | live recall (z2) | model live (z2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glider | 0.9985 | 0.600 | 0.5911 | 1.000 | +0.1843 | 7 | 5 | 861 | 0.9961 | 1.000 | 21 | 0.9961 | 1.000 | 21 |
| 1 | 0.6252 | 0.574 | 0.5906 | 0.643 | +0.0400 | 1727 | 1294 | 849 | 0.5176 | 0.8841 | 2970 | 0.6641 | 0.006 | 98 |
| 2 | 0.6172 | 0.575 | 0.5986 | 0.654 | +0.0380 | 1763 | 1305 | 847 | 0.5083 | 0.8720 | 2985 | 0.6604 | 0.011 | 114 |

The confidence gap is positive on every seed. On the glider — a grid that is
4091/4096 dead with one five-cell object — the model puts 7 cells alive against
a true 5 and gets 3 of the 5 right, which is the first output in this repo that
looks like a Life step rather than a constant. On the random seeds it still
over-predicts alive by about a third (1727 and 1763 against 1294 and 1305),
which is where the fine-tune of CONCEPT.md §5 has to do its work.

Files: `a-fewshot-1-gen1-*.pgm` (seed 1) and `a-fewshot2-{glider,2}-gen1-*.pgm`.

## Few-shot on variant B

Variant A's six worked examples, verbatim, inserted into variant B's prefix
just before `Grid:` (68 tokens → 244). They cannot be rewritten in the form B's
cells take — a cell there is one bare digit token whose self/neighbor roles are
carried by the stencil mask and by nothing else, so no piece of text *is* one
cell. This is the closest form, and it is the same text that moved variant A.

Both runs below are 64x64, Qwen2.5-0.5B-Instruct Q4_0, teacher-forced, 3 seeds
x 3 generations, native, same commit, same session. The rules-only run is a
re-run, not the first picture's numbers: it exists to fill the median columns
the first run predates. It reproduces the first picture's generation-1
accuracies (0.9988 / 0.6841 / 0.6814) and its 12 PGMs are byte-identical to
`b-*.pgm`, so the pipeline is unchanged. Timings **provisional** — the Metal
GPU was shared with another repo's training job.

| prefix | seed | gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen | acc (otsu) | live recall (otsu) | model live (otsu) | acc (z2) | live recall (z2) | model live (z2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rules only | glider | 1 | 0.9988 | 0.0000 | 0.5125 | 1.0000 | +0.1266 | 5 | 0 | 53.5 | 0.9980 | 1.0000 | 13 | 0.9958 | 1.0000 | 22 |
| rules only | glider | 2\* | 0.9988 | 0.0000 | 0.5164 | 1.0000 | +0.1127 | 5 | 0 | 52.0 | — | — | — | — | — | — |
| rules only | glider | 3\* | 0.9988 | 0.0000 | 0.5181 | 1.0000 | +0.1266 | 5 | 0 | 51.7 | — | — | — | — | — | — |
| rules only | 1 | 1 | 0.6841 | 0.0000 | 0.8162 | 1.0000 | +0.0455 | 1294 | 0 | 53.1 | 0.8252 | 0.9104 | 1778 | 0.6372 | 0.0000 | 192 |
| rules only | 1 | 2\* | 0.7368 | 0.0000 | 0.6670 | 0.8183 | +0.0234 | 1073 | 5 | 52.8 | — | — | — | — | — | — |
| rules only | 1 | 3\* | 0.7344 | 0.0000 | 0.7659 | 1.0000 | +0.0490 | 1088 | 0 | 51.9 | — | — | — | — | — | — |
| rules only | 2 | 1 | 0.6814 | 0.0000 | 0.8188 | 1.0000 | +0.0435 | 1305 | 0 | 53.1 | 0.7251 | 0.6559 | 1533 | 0.6316 | 0.0000 | 204 |
| rules only | 2 | 2\* | 0.7253 | 0.0000 | 0.6462 | 0.7670 | +0.0210 | 1120 | 5 | 52.2 | — | — | — | — | — | — |
| rules only | 2 | 3\* | 0.7288 | 0.0000 | 0.7698 | 1.0000 | +0.0432 | 1104 | 7 | 52.9 | — | — | — | — | — | — |
| few-shot | glider | 1 | 0.9988 | 0.0000 | 0.5137 | 1.0000 | +0.1940 | 5 | 0 | 52.5 | 0.9993 | 1.0000 | 8 | 0.9958 | 1.0000 | 22 |
| few-shot | glider | 2\* | 0.9988 | 0.0000 | 0.5090 | 1.0000 | +0.1699 | 5 | 0 | 49.7 | — | — | — | — | — | — |
| few-shot | glider | 3\* | 0.9988 | 0.0000 | 0.5156 | 1.0000 | +0.1940 | 5 | 0 | 50.4 | — | — | — | — | — | — |
| few-shot | 1 | 1 | 0.6768 | 0.0000 | 0.8162 | 1.0000 | +0.0701 | 1294 | 30 | 50.4 | 0.6807 | 0.4529 | 1186 | 0.6482 | 0.0000 | 147 |
| few-shot | 1 | 2\* | 0.6687 | 0.0000 | 0.6670 | 0.8183 | +0.0400 | 1073 | 284 | 50.2 | — | — | — | — | — | — |
| few-shot | 1 | 3\* | 0.7075 | 0.0000 | 0.7659 | 1.0000 | +0.0777 | 1088 | 110 | 50.5 | — | — | — | — | — | — |
| few-shot | 2 | 1 | 0.6704 | 0.0000 | 0.8188 | 1.0000 | +0.0694 | 1305 | 45 | 56.1 | 0.6741 | 0.4943 | 1320 | 0.6348 | 0.0000 | 191 |
| few-shot | 2 | 2\* | 0.6626 | 0.0000 | 0.6467 | 0.7679 | +0.0373 | 1120 | 262 | 50.6 | — | — | — | — | — | — |
| few-shot | 2 | 3\* | 0.6975 | 0.0000 | 0.7698 | 1.0000 | +0.0697 | 1104 | 135 | 50.4 | — | — | — | — | — | — |

\* generation > 1 has no PGM on disk; re-scoring it needs a rerun (not done tonight).

**Few-shot does not move variant B's live recall off zero.** It answers `1` on
0–284 of 4096 cells instead of 0–7, and every one of those is wrong: live
recall at 0.5 is 0.0000 on all 18 rows of both runs. Accuracy at 0.5 is
slightly *worse* than rules-only on generations 2 and 3 (0.66–0.71 against
0.72–0.74), for the same reason variant A's was: the extra `1`s land on dead
cells.

**The confidence gap was already positive in B, and few-shot widens it** —
+0.0455 → +0.0701 (seed 1 gen 1), +0.0435 → +0.0694 (seed 2 gen 1), +0.127 →
+0.194 (glider). That is the same direction as variant A's -0.094 → +0.040,
but in A few-shot *created* a correct ordering where there was an
anti-correlated one, and in B it only sharpens an ordering that was already
right.

**And the ordering itself does not change.** The median-threshold accuracy is
identical between the two runs to four decimals on every random-seed row
(0.8162 / 0.6670 / 0.7659 and 0.8188 / 0.646x / 0.7698) and its live recall is
1.0000 wherever it is 1.0000 in the control. Every cell true Life says is alive
is already in the top half of B's p(alive) *without* few-shot; the examples
scale the logit difference without reordering the cells. So few-shot buys
variant B a wider margin and nothing that a threshold could not have bought —
unlike variant A, where it was the ranking that moved.

Files: `b-fewshot-{glider,1,2}-gen1-*.pgm`, `b-fewshot-picture.md`, and the
control `b-rules-picture.md` (whose PGMs are byte-identical to `b-*.pgm` and
were not duplicated).
