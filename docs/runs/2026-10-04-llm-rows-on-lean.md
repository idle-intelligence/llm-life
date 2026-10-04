# Compare page, 16x16, all twelve blog-table rows on lean (2026-10-04)

The five LLM rows of the compare page now run on lean (`lean-migration`
313d22a, BUILD `2026-10-04-release-01`). `docs/runs/2026-10-02-lean-migration.md`
only had native timings and a SwiftShader run for them; the Burn-era browser
numbers quoted in the trucs.ai tech post (`blog/llm-of-life.md`) are from
`docs/runs/2026-09-22-compare-tab.md`. This run replaces those twelve rows
with real-GPU browser numbers against the current build.

Playwright's bundled Chromium (chromium-1243, Chrome for Testing), launched
with `--enable-unsafe-webgpu --enable-features=WebGPU --use-angle=metal
--ignore-gpu-blocklist` (real Metal WebGPU, not SwiftShader), headless,
against `web/compare/?local=1` served from this worktree on
`http://127.0.0.1:8797` (`web/serve.py`). Grid set to 16x16, "run every
method" clicked, same session's board both times (not re-randomized between
runs). Two full runs, one browser session, sequentially. No other GPU job
ran at the same time.

BERT of Life, 9 numbers to centre and stencil (grid to grid) still run on
the Burn engine (`pkg-llm`); only the five LLM rows run on lean (`pkg-lean`)
in this build. Game of Life (rule) and lookup table are plain JS, no engine.

## Results

| method | run 1 (s/gen) | run 2 (s/gen) | parameters | cells correct |
|---|---|---|---|---|
| Game of Life (rule) | 1.15e-5 | 5.40e-6 | no parameters | 256 / 256 |
| lookup table | 4.64e-6 | 3.23e-6 | 512-entry table | 256 / 256 |
| LLM per cell (base) | 42.12 | 41.13 | 0.5B (no adapter) | 156 / 256 |
| LLM per cell (trained) | 16.52 | 17.09 | 0.5B (+ adapter) | 256 / 256 |
| LLM per cell (trained, batched) | 7.97 | 8.48 | 0.5B (+ adapter) | 256 / 256 |
| LLM whole grid (base) | 0.4833 | 0.4831 | 0.5B (no adapter) | 173 / 256 |
| LLM whole grid (trained) | 0.4979 | 0.4971 | 0.5B (+ adapter) | 256 / 256 |
| BERT of Life | 3.85 | 0.4035 | 3,490 | 256 / 256 |
| BERT of Life (batched) | 0.1344 | 0.0173 | 3,490 | 256 / 256 |
| 9 numbers to centre | 0.1864 | 0.1846 | 1,442 | 256 / 256 |
| 9 numbers to centre (batched) | 0.1150 | 0.0112 | 1,442 | 256 / 256 |
| stencil (grid to grid) | 0.8921 | 0.9596 | 3,329 | 256 / 256 |

Both runs agree on cells correct for every row. Console errors: none in
either run.

## Which run the blog table uses

Run 2's numbers (the table above, second column) are what went into the
tech post's table, figure and prose: it is the steadier of the two
(`Game of Life (rule)` and `lookup table` land close to the 2026-09-22
Burn-era CPU numbers, 5.43e-6 and 2.98e-6, which is a sanity check since
those two rows have no model and should not move much), and its BERT /
9-numbers-batched rows are not inflated by a first-use pipeline-compile
cost (see Observations).

- LLM per cell (trained): 17.09 s -> 0.0668 s per cell.
- LLM whole grid (trained): 0.4971 s.
- Batching the per-cell LLM (8.48 s) is now about 2x *faster* than one
  forward per cell (17.09 s), on both runs (7.97 s vs 16.52 s in run 1).
  This reverses the 2026-09-22 Burn-era result, where batching was
  slightly slower (31.97 s vs 28.46 s).

## Observations

- The Burn-engine rows (BERT of Life, BERT of Life batched, 9 numbers to
  centre batched) show a large run-to-run gap that the lean/LLM rows and
  the two CPU rows do not. The compare page's own measurement code returns
  the single first call's time unaveraged whenever that first call takes
  2 s or more (`measureWorkerStep` in `web/compare/index.html`), to avoid
  waiting through two more slow calls; otherwise it takes the median of
  three. In run 1, BERT of Life's first call took longer than 2 s (3.85 s
  returned as-is); in run 2 it did not (0.4035 s, a median of three). The
  likely cause is a one-time WebGPU pipeline-compile cost for the Burn
  engine's kernels, which Chrome's on-disk shader cache can warm from a
  previous page load in the same browser profile (both runs shared one
  browser instance). This is a measurement artifact of repeated page loads
  in one session, not a claim about the Burn engine's own behaviour.
- `stencil (grid to grid)` (Burn engine, unchanged code since
  2026-09-22) measured 0.89-0.96 s per generation here, against 0.0928 s
  in the 2026-09-22 Burn-era run. Both of this run's numbers are the
  median of three calls (not the single-first-call path above), so this
  is not the same first-call artifact. Reported as observed; not
  investigated further here.
- LLM per cell (base) stays wrong on the same 100 of 256 cells both runs
  (156 / 256 correct), and LLM whole grid (base) stays wrong on the same
  83 (173 / 256): the base model without the adapter is deterministic
  (greedy decoding) and board-dependent, so this is expected, not noise.

## Served bytes

Same build as `docs/runs/2026-10-02-lean-migration.md`'s successor:
`BUILD = '2026-10-04-release-01'` on both `web/compare/index.html` and
`web/worker.js`, served from this worktree's `web/serve.py` on port 8797.
