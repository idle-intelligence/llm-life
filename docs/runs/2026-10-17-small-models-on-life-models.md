# BERT of Life, 9 numbers and stencil on the life-models engine: speed and correctness against main (2026-10-17)

The three small models of the compare page used to run on the Burn engine (`pkg-llm`). On this branch they run in `crates/life-small` (`pkg-small`, the life-models engine of the jacobi2000 repo, WebGPU or one CPU thread, safetensors checkpoints). The question: do the rows give the same boards, and did any get slower.

## Parameters

- Machine: Apple M2, 16 GB. Both builds made with `scripts/build.sh` from fresh clones, the CI commands run first on the branch (`cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, both green).
- Main: commit 81fa290, build tag 3f0bf92. Branch: main plus the four branch commits, build tag 7e7b3ed. Pages served by `scripts/serve.py` (COOP/COEP), `?local=1`, the branch loading the three safetensors checkpoints from `crates/life-small/tests/fixtures/`, main loading its `.bin` checkpoints.
- Browsers: Playwright's bundled Chromium (headless, Chrome 153, `--use-angle=metal`, adapter metal-3) and bundled WebKit (Safari 26.6 build, WebGPU adapter apple). One fresh browser per run, one run at a time.
- Rows: the compare page's own steps (`measureWorkerStep`: median of 3 calls, or the first call alone when it takes 2 s or more), driven through a hook injected into the served page for the five small-model rows only; the repository files were not changed. Correctness is the page's own cell-by-cell comparison with one generation of the rule from the same random board.
- Sizes 16x16 and 64x64. Chromium: 3 runs per build per size, median given. WebKit: 1 run per build per size (its timer resolution is 1 ms).
- Full page: "run every method" at 16x16 on the branch in Chromium, all rows, screenshot `shots/br-chromium-16-all.png` in the job directory.
- Wasm size: main `pkg-llm` 17,680,220 bytes; branch `pkg-small` 224,919 bytes. `pkg-lean` is byte-identical in both (2,573,871 bytes). All three branch wasm files keep +simd128 (build check).

## Results, Chromium, seconds per generation (median of 3)

| row | size | main | branch | cells correct (main / branch) |
|---|---|---|---|---|
| BERT of Life | 16x16 | 0.393 | 0.121 | 256 / 256, 256 / 256 |
| 9 numbers to centre | 16x16 | 0.152 | 0.0843 | 256 / 256, 256 / 256 |
| BERT of Life, batched | 16x16 | 0.0178 | 0.000445 | 256 / 256, 256 / 256 |
| 9 numbers to centre, batched | 16x16 | 0.0168 | 0.00036 | 256 / 256, 256 / 256 |
| stencil (grid to grid) | 16x16 | 0.0980 | 0.000505 | 256 / 256, 256 / 256 |
| BERT of Life | 64x64 | 7.04 | 1.64 | 4096 / 4096, 4096 / 4096 |
| 9 numbers to centre | 64x64 | 2.41 | 1.49 | 4096 / 4096, 4096 / 4096 |
| BERT of Life, batched | 64x64 | 0.0711 | 0.00090 | 4096 / 4096, 4096 / 4096 |
| 9 numbers to centre, batched | 64x64 | 0.0612 | 0.00081 | 4096 / 4096, 4096 / 4096 |
| stencil (grid to grid) | 64x64 | 0.0901 | 0.00105 | 4096 / 4096, 4096 / 4096 |

The three runs of each cell: branch 9 numbers at 64x64 gave 2.55, 1.49 and 1.40 s; every other branch row varied by less than 10 percent between runs except the batched and stencil rows at 64x64 (0.0009 to 0.0011 s).

## Results, WebKit, seconds per generation (1 run)

| row | size | main | branch | cells correct (main / branch) |
|---|---|---|---|---|
| BERT of Life | 16x16 | 7.50 | 0.123 | 256 / 256, 256 / 256 |
| 9 numbers to centre | 16x16 | 0.188 | 0.089 | 256 / 256, 256 / 256 |
| BERT of Life, batched | 16x16 | 0.040 | 0.001 | 256 / 256, 256 / 256 |
| 9 numbers to centre, batched | 16x16 | 0.049 | 0.001 | 256 / 256, 256 / 256 |
| stencil (grid to grid) | 16x16 | 0.021 | 0.001 | 256 / 256, 256 / 256 |
| BERT of Life | 64x64 | 8.08 | 1.65 | 4096 / 4096, 4096 / 4096 |
| 9 numbers to centre | 64x64 | 2.80 | 2.37 | 4096 / 4096, 4096 / 4096 |
| BERT of Life, batched | 64x64 | 0.029 | 0.001 | 4096 / 4096, 4096 / 4096 |
| 9 numbers to centre, batched | 64x64 | 0.037 | 0.001 | 4096 / 4096, 4096 / 4096 |
| stencil (grid to grid) | 64x64 | 0.023 | 0.002 | 4096 / 4096, 4096 / 4096 |

## Results, full page on the branch, 16x16, Chromium (one run)

| row | s / generation | cells correct |
|---|---|---|
| LLM per cell (base) | 41.3 | 156 / 256 |
| LLM per cell (trained) | 17.6 | 256 / 256 |
| BERT of Life | 0.0946 | 256 / 256 |
| 9 numbers to centre | 0.0757 | 256 / 256 |
| LLM whole grid (base) | 0.488 | 173 / 256 |
| LLM whole grid (trained) | 0.500 | 256 / 256 |
| LLM per cell (trained, batched) | 7.98 | 256 / 256 |
| BERT of Life (batched) | 0.00158 | 256 / 256 |
| 9 numbers to centre (batched) | 0.000355 | 256 / 256 |
| stencil (grid to grid) | 0.00041 | 256 / 256 |

The other rows of the page ran too and matched the rule where they apply (ternary, binary, lookup, GPU lookup, 2-step); no console errors. The LLM rows agree with `2026-10-04-llm-rows-on-lean.md` (17.09, 8.48, 0.497 s).

## Native

- `cargo test --release -p life-small`: 3 passed (CPU boards match the rule, GPU matches CPU, per-cell equals whole board), on the three checkpoints.
- `cargo tree` for `life-small`, `llm-life-lean`, `life` and `life-lut-bench` lists no Burn crate. `llm-life` (the native Qwen "picture" driver) and its `llm-wasm` dependency still depend on Burn; the three small models are not in it.
- The native CLI no longer has a `bench-ladder` command, the small models have no native command line, and the tests are their native check.

## Observations

- No row gets slower, in either browser, at either size. Every row is exact (256 / 256 and 4096 / 4096) on both builds.
- The batched and stencil ratios above (40x to 200x in Chromium) compare main's warm-up with a warm branch: the compare page reports the median of the first 3 calls on a fresh page, and on main those calls fall inside the Burn engine's warm-up. Warmed up, the ratios are 2x to 8x; see the next section.
- The per-cell rows gain less (BERT 3x to 4x at 16x16 and 4x at 64x64; 9 numbers 1.8x at 16x16, 1.6x at 64x64 in Chromium). They are one worker round trip per cell on both builds, so they scale with the number of cells (about 0.4 ms per cell for BERT, 0.36 ms for 9 numbers at 64x64 on the branch).
- The 9-numbers per-cell row at 64x64 in WebKit is 2.37 s on the branch against 2.80 s on main, the closest pair of the table; one run each, so it is not distinguished from equal.
- The built wasm for the three models drops from 17.7 MB to 225 KB.
- Not run: the LLM rows at 64x64 and the full page in WebKit; the LLM code path is unchanged by the branch (`pkg-lean` is byte-identical).

## Warm-up against steady state (16x16, Chromium)

Same builds, one fresh page per build, 25 calls per row (6 for the per-cell rows), every call 256 / 256 correct. "First 3" is what the compare page reports; "steady" is the median of the last 10 calls.

| row | main, first 3 | main, steady | branch, steady | steady ratio |
|---|---|---|---|---|
| stencil (grid to grid) | 35 ms | 2.0 ms | 0.55 ms | 3.7x |
| BERT of Life, batched | 92 ms | 3.9 ms | 0.51 ms | 7.6x |
| 9 numbers to centre, batched | 10.7 ms | 0.86 ms | 0.40 ms | 2.2x |
| BERT of Life, per cell | 476 ms | 469 ms | 125 ms | 3.8x |
| 9 numbers to centre, per cell | 205 ms | 199 ms | 90 ms | 2.2x |

One steady-state stencil generation, counted by wrapping the WebGPU calls: main issues 73 dispatches, 73 bind groups, 4 submits, 159 buffer writes (including the 9,216-byte neighbour table, rebuilt each call) and one 16 KB read-back; the branch issues 1 dispatch, 1 submit, 2 writes (1 KB of cells, 16 bytes of dimensions) and one 1 KB read-back.

- On main, the first call creates about 50 compute pipelines, and calls 2 to 5 issue up to 66 read-backs each (inferred to be kernel autotuning; not confirmed by switching it off). Steady state arrives after about 8 calls. The branch's first call takes about 6 ms, the later ones under 1 ms.
- On the branch, a per-cell call (0.35 to 0.49 ms) costs about as much as a whole-board call (0.40 to 0.55 ms): what remains is one submit and one read-back (inferred to be the browser's round trip on this machine).
- Not measured here: WebKit and 64x64 in steady state, per-dispatch GPU time.
