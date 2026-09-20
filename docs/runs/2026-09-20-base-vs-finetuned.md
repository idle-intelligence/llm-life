# Base model vs fine-tuned, both variants

Comparing the base Qwen2.5-0.5B-Instruct Q4_0 (`models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf`) against the LoRA fine-tunes, for both prompt shapes this repo has (variant A: packed per-cell prompt; variant B: whole grid as text).

## Held-out sets (two different conventions, one per variant, held fixed across every row of that variant)

- **Variant A** (`crates/llm-life/src/train/run_a.rs`): `accuracy` and the six per-case columns are over the full exhaustive 512 `(neighbors, self)` lookup table — every possible cell configuration, not a sample. `IoU`/`alive recall`/`dead recall` are over `run_a::real_grids(16)`: a glider, a blinker, and one `Grid::random(16,16,999_999,0.28)` board, one generation each, teacher-forced.
- **Variant B** (`crates/llm-life/src/train/data.rs::held_out`): variant B has no exhaustive-lookup mode (the grid, not a lookup table, is the unit of supervision), so every column is over `held_out(16,4)`: a glider, a blinker, and two `Grid::random(16,16,1_000_000|1_000_001,0.28)` boards — the same 4-grid set `docs/runs/2026-09-20-ft-b-2.md` and `docs/runs/2026-09-20-ft-b-16-s3-300.md` use.

These two sets are not identical (different random seeds, and A additionally has the 512-case exhaustive table), because the two variants' native eval code doesn't share one; each variant's own row uses its own fixed set consistently, rather than forcing a common set of grids through code neither variant's evaluator currently supports.

## Commands

Variant A base rows (steps 0 = LoRA at zero = base model, `b` matrix is zero-initialized so the delta is exactly zero regardless of `a`'s random init):
```
llm-life train-a --gguf <gguf> --tokenizer <tok> --steps 0 --eval-cases 512 --eval-grids 3 [--norules] \
  --out /tmp/discard.bin --run-doc docs/runs/2026-09-20-eval-a-base-{rules,norules}.md
```
Variant A fine-tuned rows: existing training run docs, `docs/runs/2026-09-20-ft-a-norules-300.md` / `-ft-a-rules-300.md`, "Best-by-accuracy checkpoint" section (`artifacts/lora-a-norules-300.bin` / `-a-rules-300.bin`).

Variant B base and fine-tuned rows: reused as-is from `docs/runs/2026-09-20-ft-b-16-s3-300.md` ("before"/"after (300 steps)" sections) — same held-out set, same code path, `artifacts/lora-b-16-s3-300.bin` (rank 8, alpha 16, q/k/v/o, 300 steps, lr 5e-5, seed 3). No new run needed since the set and code are identical to what this doc wants.

A CLI flag was added to make an equivalent "load a pretrained checkpoint and eval-only" run possible for variant B (`train --adapter <path> --steps 0`, `crates/llm-life/src/train/run.rs` / `crates/llm-life/src/bin/llm-life.rs`), but it was not exercised for this doc's B-fine-tuned row since `ft-b-16-s3-300.md` already had the number with `lora-b-16-s3-300.bin`. One earlier attempt to use it to run variant B with variant A's `lora-a-norules-300.bin` adapter was started, then killed before it produced output (to avoid overlapping with the still-running variant-A base-norules eval — one GPU job at a time on this machine) — it is **not** in the table below, since it never finished.

**The page currently applies the A adapter (`lora-a-norules-300.bin`) in both LLM modes.** `web/worker.js`'s `load()` handler calls `engine.loadAdapter(...)` with `LLM_MODEL.adapterUrl` unconditionally, regardless of whether the page is in variant-A or variant-B mode, and `LifeEngine::apply_lora` (`crates/llm-life/src/web.rs`) mutates the shared Q4Attention weights used by both variants' forwards. So today, the page's variant-B ("LLM batched"/whole-grid) numbers are actually the base Q4 model **plus** the variant-A adapter applied to variant B's whole-grid forward under variant B's own rules prefix — not a bare base model, and not `lora-b-16-s3-300.bin` either. This is being fixed to per-mode adapters (variant B should load its own `lora-b-16-s3-300.bin`); until that lands, the table below's "B fine-tuned" row is the *intended* fine-tune, not what the page currently shows.

## Results

| variant | prompt | model | held-out accuracy | IoU | alive recall | dead recall | birth / survive-2 / survive-3 / lonely / crowded / stay-dead recall |
|---|---|---|---|---|---|---|---|
| A | rules stated | base | 0.4199 (215/512) | 0.0554 | 0.5152 | 0.0593 | 1.0000 / 0.0357 / 0.0179 / 1.0000 / 0.9080 / 0.0000 |
| A | no rules | base | 0.3809 (195/512) | 0.1354 | 0.2951 | 0.8534 | 0.6250 / 0.0357 / 0.1964 / 1.0000 / 0.5399 / 0.2550 |
| A | rules stated | fine-tuned (`lora-a-rules-300.bin`) | 0.9941 (509/512) | 0.9909 | ~0.9705† | ~0.9814† | 1.0000 / 0.9286 / 1.0000 / 0.8889 / 1.0000 / 1.0000 |
| A | no rules | fine-tuned (`lora-a-norules-300.bin`) | 1.0000 (512/512) | 1.0000 | 1.0000‡ | 1.0000‡ | 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| B | rules stated (whole grid) | base | 0.8232 | 0.0000 | 0.0000 | 1.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 / 1.0000 / 1.0000 |
| B | rules stated (whole grid) | fine-tuned (`lora-b-16-s3-300.bin`) | 1.0000 | 1.0000 | 1.0000‡ | 1.0000‡ | 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 |

† Not printed by `ft-a-rules-300.md`'s "Best-by-accuracy" section (an older doc format that only logs accuracy/IoU there). Cited instead from `docs/runs/2026-09-20-a-rollout.md`'s generation-1 numbers for the same checkpoint, on a different machine (RTX 3080, Vulkan) and a different 3-seed rollout, not the norules-300/base rows' 16×16 held-out set: seed 1 alive 0.9759/dead 0.9884, seed 2 alive 0.9861/dead 0.9728, seed 3 alive 0.9494/dead 0.9831 → means 0.9705/0.9814.

‡ Not printed as a separate number in the source doc; derived by construction from `score::score`'s definitions — IoU = 1.0000 requires the predicted and true alive sets to be identical, which forces both alive recall and dead recall to 1.0000 (the same reasoning `docs/LADDER.md` already uses for every IoU = 1.000 rung).

## What the base model actually outputs

Derived from the full per-case breakdown above (all 512 exhaustive cases for A, all 1024 held-out cells for B) rather than a fresh sample of generations — this machine was already running one eval tonight and running more generations was avoidable extra GPU load. The per-case counts are exhaustive, not anecdotal, so this is exact, not a guess from a handful of samples.

- **B, base, rules stated**: answers 0 (dead) for every cell. All three true-alive classes (birth, survive-2, survive-3) score 0% correct — every one of those 181 held-out cells is answered "dead" when the truth is "alive" — while all three true-dead classes score 100%. There is no case where the base model predicts alive.
- **A, base, rules stated**: ignores neighbor count and mirrors the cell's own current state. Both classes where the cell starts **dead** (birth, stay-dead) are answered "1" almost regardless of the correct target: birth's target is 1, so it reads 100% correct; stay-dead's target is 0, so it reads 0% correct — same "always answer 1 when currently dead" behavior scored two different ways. Both classes where the cell starts **alive** (survive-2, survive-3) are answered "0" almost regardless of target for the same reason, alongside lonely/crowded (target 0, correctly answered 100%/90.8%). Net: answers "not-self" — dead cells become alive, alive cells become dead — independent of the neighbor count the prompt states.
- **A, base, no rules**: no comparably clean rule. Birth 62.5% correct, survive-2 3.6%, survive-3 19.6%, lonely 100%, crowded 54.0%, stay-dead 25.5% — closer to noisy guessing than a single heuristic, and worse than the rules-stated base on IoU's headline number (0.1354 vs 0.0554) only because its alive/dead recall split (0.2951/0.8534) happens to sit closer to the held-out grids' own live fraction, not because it is more accurate per-case.
