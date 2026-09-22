# llm-life

A language model as the update rule of a cellular automaton. One pixel is one
cell; one cell's next state is read out of the model's logits over a two-token
answer vocabulary. The picture is the model's confidence per cell, and the
drift from true Life is the model's character rendered as an image.

[Try the demo](https://idle-intelligence.github.io/llm-life/web/) and the
[compare page](https://idle-intelligence.github.io/llm-life/web/compare/),
which runs every method on the same grid, one tab. Both need WebGPU; without
it, the page says so: "doesn't work without WebGPU, yet".

## The rules, Martin Gardner, 1970

> Conway's genetic laws are delightfully simple. First note that each cell of
> the checkerboard (assumed to be an infinite plane) has eight neighboring
> cells, four adjacent orthogonally, four adjacent diagonally. The rules are:
>
> **Survivals.** Every counter with two or three neighboring counters
> survives for the next generation.
>
> **Deaths.** Each counter with four or more neighbors dies (is removed) from
> overpopulation. Every counter with one neighbor or none dies from
> isolation.
>
> **Births.** Each empty cell adjacent to exactly three neighbors, no more,
> no fewer, is a birth cell. A counter is placed on it at the next move.

Martin Gardner, "Mathematical Games: The fantastic combinations of John
Conway's new solitaire game 'life'", Scientific American 223 (October 1970).

## Six ways to compute one generation

- **Game of Life (rule).** The classical rule, as a for loop.
- **LLM per cell.** Give a language model the rules and current cell's
  state, *as text*, and ask if it lives or dies. Repeat for each and every
  cell. Of course, it's very slow, and an off the shelf model doesn't work.
  Fine tuning a small LoRa adapter gets us perfect results, but it's still
  slow... Does it speed up if we batch the inference?
- **LLM whole grid.** Look, there are many optimisation we could do. That's
  not the game we are playing. Yet, cell by cell _is_ stupid. Instead, we ask
  the model for the whole grid in one prompt, one token per cell. It's much
  faster, but training a 64x64 adapter gets expensive, and 256x256 doesn't
  fit in this small model's context. Another LoRa adapter gets us to 100%
  accuracy.
- **BERT of Life.** Alive or Dead is a classification. What would a BERT do?
  A BERT-shaped classifier, one-layer encoder over the 9 cells as tokens,
  3,490 parameters trained from scratch; it works.
- **9 numbers to centre.** Why do we even use a language model? Can we fit a
  "modern" model architecture that actually learns the rules? A 2-layer MLP
  over the 9 cell values, 1,442 parameters; We run it cell by cell, pretty
  fast, it works.
- **Stencil (grid to grid).** Cell by cell is nice, but we can do better: One
  attention layer over the whole grid, each cell masked to its 3x3 stencil.
  3,329 parameters, trained at 16x16, it scales to whatever grid size.

## Models

[llm-of-life-lora](https://huggingface.co/idle-intelligence/llm-of-life-lora)
holds four LoRA adapters on top of Qwen2.5-0.5B-Instruct, for the per-cell
and whole-grid LLM methods above.
[stencil-life](https://huggingface.co/idle-intelligence/stencil-life) holds
the three from-scratch models: BERT of Life (3,490 parameters), the 9-number
MLP (1,442 parameters) and the stencil attention model (3,329 parameters).
The base model is
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
by the Qwen team, Apache-2.0.

## Results

Measured in a browser tab at 16x16 on an M2.

| method | s / generation | parameters | cells correct |
|---|---|---|---|
| Game of Life (rule) | 5.43e-6 | no parameters | 256 / 256 |
| lookup table | 2.98e-6 | 512-entry table | 256 / 256 |
| LLM per cell (base) | 30.92 | 0.5B (no adapter) | 156 / 256 |
| LLM per cell (trained) | 28.46 | 0.5B (+ adapter) | 256 / 256 |
| LLM per cell (trained, batched) | 31.97 | 0.5B (+ adapter) | 256 / 256 |
| LLM whole grid (base) | 0.5740 | 0.5B (no adapter) | 173 / 256 |
| LLM whole grid (trained) | 0.6364 | 0.5B (+ adapter) | 256 / 256 |
| BERT of Life | 1.06 | 3,490 | 256 / 256 |
| BERT of Life (batched) | 0.0195 | 3,490 | 256 / 256 |
| 9 numbers to centre | 0.7794 | 1,442 | 256 / 256 |
| 9 numbers to centre (batched) | 0.0253 | 1,442 | 256 / 256 |
| stencil (grid to grid) | 0.0928 | 3,329 | 256 / 256 |

Two open questions: "Why is batching slower with the LLM? Why is batching
faster with the other models?"

## Layout

```
crates/life/        classical B/S cellular automaton (the ground truth), + wasm wrapper
crates/llm-life/    variant A packing (per-cell prompts, block-diagonal mask),
                    variant B packing (stencil mask), scoring, PGM output
                    src/bin/llm-life.rs   native driver ("picture")
                    src/web.rs            wasm-bindgen LifeEngine for the tab
web/                the demo page (STATUS / INPUT / OUTPUT / PERFORMANCE) and the compare page
scripts/headless/   Playwright (bundled Chromium only) verification
docs/pictures/      first-picture output + accuracy tables
```

## Build and run

Native tests, no GPU and no model files needed:

```bash
cargo test --workspace
```

Native tests including the CPU-backend (ndarray) vector-model equivalence
test:

```bash
cargo test --workspace --features cpu
```

Native release build of the CLI:

```bash
cargo build --release -p llm-life --features cpu
```

The classical Game of Life CLI:

```bash
cargo run -p life --bin life -- --glider --size 8 --generations 3
cargo run -p life --bin life -- --size 32 --seed 42 --rule B3/S23 --generations 10
```

The native "picture" driver, variant B, against Qwen2.5-0.5B-Instruct Q4_0:

```bash
cargo run --release -p llm-life -- picture \
  --gguf ~/Code/idle-intelligence/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokenizer ~/Code/idle-intelligence/models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json \
  --size 64 --generations 10
```

Building the demo page's wasm (slow, several minutes, run one build at a
time):

```bash
wasm-pack build crates/life --target web --out-dir ../../web/pkg --features web
wasm-pack build crates/llm-life --target web --out-dir ../../web/pkg-llm --no-default-features --features web
python3 web/serve.py            # http://127.0.0.1:8010/
```

`crates/llm-life` depends on `llm-wasm` as a git dependency
(https://github.com/idle-intelligence/llm-web, pinned `rev`), fetched
automatically by Cargo; no local checkout of `llm-web` is needed. If a
corporate proxy blocks Cargo's built-in git fetcher, set
`CARGO_NET_GIT_FETCH_WITH_CLI=true` to shell out to the system `git`
instead.

On the live page, everything loads from Hugging Face. Adding `?local=1` to
the page's URL switches it to local files instead: a `web/models` symlink to
a local models directory for the base GGUF and tokenizer, and the LoRA,
BERT, MLP and stencil `.bin` files sitting next to `web/index.html` itself.

## Headless checks

Set `PLAYWRIGHT_MODULE` to a local `playwright/index.mjs` and
`CHROMIUM_PATH` to a Chromium executable; both are required, there are no
defaults. Start the local server first with `python3 web/serve.py`.
Screenshots land in `scripts/headless/out/` unless `--screenshot` overrides
it.

```bash
PLAYWRIGHT_MODULE=/path/to/node_modules/playwright/index.mjs \
CHROMIUM_PATH=/path/to/chrome \
node scripts/headless/llm.mjs --url http://127.0.0.1:8010/
```

## Publish

`tools/publish-pages.sh` builds both wasm engines and republishes the
committed `web/` tree to an orphan `gh-pages` branch. It refuses to publish
if any model weights ended up in the export. It never pushes; push the
result with `git push --force-with-lease origin gh-pages`.

## Engine

The inference engine is `llm-wasm` from
[llm-web](https://github.com/idle-intelligence/llm-web), pulled in as a git
dependency pinned to a `rev`. A small patch this repo needs is vendored
under `patches/`.

## License

MIT. The base model, Qwen2.5-0.5B-Instruct, is Apache-2.0.
