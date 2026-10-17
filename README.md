# llm-life

A language model as the update rule of a cellular automaton. One pixel is one
cell; one cell's next state is read out of the model's logits over a two-token
answer vocabulary. The picture is the model's confidence per cell, and the
drift from true Life is the model's character rendered as an image.

This started as a (stupid) experiment: what if we ask an LLM to play Game of
Life, cell by cell? It grew into a comparison of every way we found to compute
one generation of the game, from a for loop to a 0.5B-parameter model to a
45-parameter network, all checked against the classical rule on the same board.

[Try the demo](https://idle-intelligence.github.io/llm-life/web/) and the
[compare page](https://idle-intelligence.github.io/llm-life/web/compare/),
which runs every method on the same grid, one tab. Both need WebGPU.

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

All boards are tori, and the rule is B3/S23. The classical implementation is
`crates/life`; every other method is scored cell by cell against it.

## Ways to compute one generation

The first six are the original story, in the order the compare page runs
them. The rest were added later and are on the same page.

**Game of Life (rule).** The classical rule, as a for loop.

**LLM per cell.** Give a language model the rules and current cell's state,
*as text*, and ask if it lives or dies. Repeat for each and every cell. Of
course, it's very slow, and an off the shelf model doesn't work. Fine tuning a
small LoRA adapter gets us perfect results, but it's still slow. Does it speed
up if we batch the inference? Batching packs 64 cells into one sequence with a
block-diagonal attention mask.

**BERT of Life.** Alive or Dead is a classification. What would a BERT do? A
BERT-shaped classifier, one-layer encoder over the 9 cells as tokens, 3,490
parameters trained from scratch; it works.

**9 numbers to centre.** Why do we even use a language model? Can we fit a
"modern" model architecture that actually learns the rules? A 2-layer MLP over
the 9 cell values, 1,442 parameters. We run it cell by cell, pretty fast, it
works.

**LLM whole grid.** Look, there are many optimisation we could do. That's not
the game we are playing. Yet, cell by cell *is* stupid. Instead, we ask the
model for the whole grid in one prompt, one token per cell. It's much faster,
but training a 64x64 adapter gets expensive, and 256x256 doesn't fit in this
small model's context. Another LoRA adapter gets us to 100% accuracy.

**Stencil (grid to grid).** One attention layer over the whole grid, each cell
masked to its 3x3 stencil. 3,329 parameters, trained at 16x16, it scales to
whatever grid size.

**Lookup table.** The same rule as a 512-entry table indexed by the 9
neighbourhood bits. The fastest way to compute it on a CPU, and the thing every
model above is measured against.

**GPU lookup.** The same table, looked up by a WebGPU compute shader: one
thread per cell, one lookup each. In the first variant the board is read back to
the CPU every generation. In the resident variant the kernel runs back to back
with ping-pong buffers and no readback in between, and only the final board
comes back.

**Bitwise (GPU, resident).** 32 cells packed into one 32-bit word, and a
bitwise adder network counts all 32 cells' neighbours at once instead of
looking each one up individually. Same ping-pong, one readback at the end.

**Ternary model (45 parameters).** A tiny convolutional network trained from
scratch with ternary ({-1, 0, 1}) weights that reproduces Conway's rule exactly:
a 3x3 convolution to 4 channels, a ReLU, and a 1x1 readout, so 40 ternary
weights and 5 float biases. Quantization-aware training with a straight-through
estimator and the absmean scheme of BitNet b1.58, one float scale per layer. The
compare page runs it three ways: as an ordinary network with dequantized weights
(a float multiply per weight), using only integer adds and subtractions of the
0/1 cell values, and bit-packed, compiled to bit operations on 32 cells per
word.

**Binary model (177 parameters).** The same idea with binary ({-1, 1}) weights
instead of ternary. It needs roughly four times the parameters to still
reproduce the rule exactly, since a binary weight cannot ignore an input. Run as
float multiplies and as integer adds and subtractions.

**Ternary model, compiled (89 parameters).** A 9 to 8 to 1 ternary network with
step activations, turned into a straight-line program of 92 AND, OR, XOR and NOT
operations over 32-bit words by a generic bit-slice compiler, rather than
written by hand like the bitwise kernel. It is exact on all 512 neighbourhoods.
The compiler (tritsliced) is private; the page runs the gate list it emits
(`web/compare/bitslice_life_compiled.wgsl`).

**Two steps in one pass.** A 32-channel ternary network (9,601 parameters, two
3x3 convolutions and a 1x1 readout, step activations) trained to predict two
generations at once and compiled the same way, 12,794 operations. It is exact on
all 2^25 = 33,554,432 possible 5x5 neighbourhoods. The page also runs it as a
float network and with integer adds, and compares it with a 2-step lookup table
that stores every 5x5 neighbourhood, one bit each, precomputed from the rule
applied twice.

Two more pages sit next to the demo: `web/fullscreen/` runs the bitwise kernel at
one cell per pixel of your screen, the board never leaving the GPU, and
`web/cellspeed/` times the per-cell LLM on a fixed 16x16 board.

## Results

Accuracy is cells correct against the classical rule on the same board.
Speeds are from a browser tab on an Apple M2 (WebGPU on Metal), driven by
Playwright's bundled Chromium. Every number below is copied from the run
document named next to it.

### One generation at 16x16

Source: `docs/runs/2026-10-04-llm-rows-on-lean.md`, second of two runs (seconds
per generation). The first run agrees on cells correct for every row.

| method | s / generation | parameters | cells correct |
|---|---|---|---|
| Game of Life (rule) | 5.40e-6 | no parameters | 256 / 256 |
| lookup table | 3.23e-6 | 512-entry table | 256 / 256 |
| LLM per cell (base) | 41.13 | 0.5B (no adapter) | 156 / 256 |
| LLM per cell (trained) | 17.09 | 0.5B (+ adapter) | 256 / 256 |
| LLM per cell (trained, batched) | 8.48 | 0.5B (+ adapter) | 256 / 256 |
| LLM whole grid (base) | 0.4831 | 0.5B (no adapter) | 173 / 256 |
| LLM whole grid (trained) | 0.4971 | 0.5B (+ adapter) | 256 / 256 |
| BERT of Life | 0.4035 | 3,490 | 256 / 256 |
| BERT of Life (batched) | 0.0173 | 3,490 | 256 / 256 |
| 9 numbers to centre | 0.1846 | 1,442 | 256 / 256 |
| 9 numbers to centre (batched) | 0.0112 | 1,442 | 256 / 256 |
| stencil (grid to grid) | 0.9596 | 3,329 | 256 / 256 |

The base model fails on both prompts: per cell it gets 156 of 256, and for the
whole grid it answers dead everywhere (173 is the board's dead count). Batching
the per-cell LLM is about twice as fast as one forward per cell in this build.
In the earlier build of the same page (`docs/runs/2026-09-22-compare-tab.md`)
batching was slightly slower, 31.97 s against 28.46 s. The stencil row measured
0.0928 s in that earlier build; the later figure is reported as observed and
was not investigated. The BERT of Life and 9-number rows run on the Burn engine
and include a first-use pipeline-compile cost in some runs, which is why the two
runs of that doc differ for them.

### The from-scratch models at larger grids

Source: `docs/runs/2026-09-22-sizes.md`. Native, 5 generations, seeds 1 to 3,
random boards at density 0.28. The stencil model, BERT of Life and the 9-number
MLP made 0 wrong cells in every generation at 64x64, 128x128 and 256x256. The
stencil was trained at 16x16: the mask is the rule, and the grid size is not.

### How small can an exact network be

Sources: `docs/runs/2026-09-30-minimal-life.md` (float) and
`docs/runs/2026-10-01-life-1bit.md` (ternary, binary). Convolutional family,
one generation, 10 seeds per cell; "exact" means all 512 neighbourhoods and 100
fresh random boards, evaluated with the quantized weights only.

| parameters | float CNN | ternary CNN | binary CNN |
|---:|---:|---:|---:|
| 23 | 6 / 10 | 0 / 10 | 0 / 10 |
| 45 | 7 / 10 | 2 / 10 | 0 / 10 |
| 89 | 10 / 10 | 5 / 10 | 0 / 10 |
| 177 | 10 / 10 | 10 / 10 | 6 / 10 |

The float row at 177 parameters is the c=16 CNN of the float sweep. The
single-hidden-layer MLP on the 9 neighbourhood values behaves the same way in
the ternary sweep (2, 5 and 10 of 10 at 45, 89 and 177 parameters). Quantizing
costs reliability at a fixed width, and the fix is to go wider. Several
non-converged binary runs plateau at exactly 456 / 512 cases, the rule "alive
iff exactly 3 of the 9 cells are alive", which misses a live cell with 3
neighbours. The ternary model's zero weight is what lets a unit ignore its own
cell. A converged ternary MLP is readable: one hidden unit fires when more than
3 neighbours are alive, one fires when fewer than 3 cells of the block (self
included) are alive, and the readout says alive when neither fires. Of the 40
ternary weights of the smallest exact CNN, 26 are nonzero, which gives 26 adds
or subtractions plus 5 bias adds, 31 integer operations per cell.

The two-step network is harder: in float, untied two-layer CNNs converge 5 of 5
at 1,257 and 18,849 parameters, and in ternary only 2 of 5 seeds at 32 channels
converged (`docs/runs/2026-10-02-bitslice-2step.md`); binary weights, 0 of 5.
A plain MLP given the 5x5 window and no convolution prior reached no exact
model in `docs/runs/2026-10-02-oneshot-nstep.md`.

### Cells per second against grid size

Source: `docs/runs/2026-10-12-size-sweep.md`. Square grids from 16x16 to
16384x16384, Apple M2 (16 GB), random board at density 0.28. The CPU rows are
from the first and the larger-grid sections of that doc, the GPU rows from its
last section, where the timer stops when the GPU has finished the last
generation and the final board read-back is outside it. Every point was checked
against the CPU rule.

| method | 64x64 | 1024x1024 | 16384x16384 |
|---|---:|---:|---:|
| CPU rule | 5.93e+07 | 5.43e+07 | 4.86e+07 |
| CPU lookup table | 1.17e+08 | 1.25e+08 | 1.23e+08 |
| GPU lookup, kept on GPU | 1.36e+08 | 3.38e+09 | 2.29e+09 |
| bitwise GPU (hand-written) | 1.36e+08 | 3.32e+10 | 9.76e+10 |
| compiled gates GPU (89-parameter ternary network) | 1.33e+08 | 3.27e+10 | 9.05e+10 |
| 45-parameter ternary network, bit-packed GPU | 1.34e+08 | 3.29e+10 | 9.81e+10 |

- The best CPU method is the lookup table, flat at 1.1e8 to 1.25e8 cells per
  second at every size. The GPU lookup with a read-back every generation never
  passes it (peak 4.4e7 at 1024x1024).
- Below 256x256 every resident GPU method takes about 30 microseconds per
  generation, so the rate grows with the number of cells: that is the cost of
  one submit per generation, not the kernel. The GPU methods pass the CPU lookup
  table at 64x64.
- At 16384x16384 the bitwise kernel is 794 times the CPU lookup table. The
  compiled 89-parameter gates run at 0.93 to 1.00 of the hand-written bitwise
  kernel at every size from 32x32 up, and the bit-packed 45-parameter network at
  0.99 to 1.01 from 128x128 up.
- The stencil model peaks at 1.7e6 cells per second (1024x1024) and never
  reaches the CPU rule.
- The LLM rows run at 5 to 30 cells per second per cell and 5e2 to 7e2 for the
  whole grid, at 16x16 and 32x32 only.

Other run documents: `docs/runs/2026-09-29-life-lut-bench.md` (the
`life-lut-bench` crate, native), `docs/runs/2026-09-30-browser-lookup-bitwise.md`,
`docs/runs/2026-10-02-phone-compare.md` (the compare page on a phone), and the
fine-tuning runs `docs/runs/2026-09-19-*` and `2026-09-20-*`.

## Models

[llm-of-life-lora](https://huggingface.co/idle-intelligence/llm-of-life-lora)
holds four LoRA adapters on top of Qwen2.5-0.5B-Instruct (`lora-a-norules-300`,
`lora-a-rules-300`, `lora-b-16-s3-300`, `lora-b-32`), for the per-cell and
whole-grid LLM methods above.
[stencil-life](https://huggingface.co/idle-intelligence/stencil-life) holds the
three small models: BERT of Life (`bert-d16-L1`, 3,490 parameters), the 9-number
MLP (`mlp2-32-lrfix`, 1,442 parameters) and the stencil attention model
(`stencil-d16-L1`, 3,329 parameters). The base model is
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
by the Qwen team, Apache-2.0.

The ternary, binary and 2-step networks are small enough to ship in the repo, as
`web/compare/life-ternary-45.json`,
`web/compare/life-binary-177.json` and `web/compare/life-2step-9601.json`, with the compiled kernels next to them as WGSL files.
The trained 89-parameter and 2-step networks are in `pytorch/bitslice_models/`.

## Run it

Native tests, no GPU and no model files needed (this is what CI runs, with
`cargo clippy --workspace --all-targets -- -D warnings`):

```bash
cargo test --workspace
```

Including the CPU (ndarray) vector-model equivalence test:

```bash
cargo test --workspace --features cpu
```

The classical Game of Life CLI:

```bash
cargo run -p life --bin life -- --glider --size 8 --generations 3
cargo run -p life --bin life -- --size 32 --seed 42 --rule B3/S23 --generations 10
```

The native "picture" driver, the whole grid against Qwen2.5-0.5B-Instruct Q4_0
(download the GGUF and `tokenizer.json` from the Qwen repositories linked above;
`picture-a` is the per-cell variant, and `bench-ladder` times the small methods
on this machine):

```bash
cargo run --release -p llm-life -- picture \
  --gguf models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokenizer models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json \
  --size 64 --generations 10
```

The lookup-table and bit-packed benchmark, native:

```bash
cargo run --release -p life-lut-bench -- correctness
cargo run --release -p life-lut-bench -- bench --sizes 16,64,256
```

The demo and the compare page. `scripts/build.sh` is CI's own build and
assembles the site into `_site`; it needs `wasm-pack`:

```bash
scripts/build.sh
python3 scripts/serve.py        # http://127.0.0.1:8030/web/
```

While iterating on one engine's wasm against the `web/` sources directly (slow;
run one build at a time):

```bash
wasm-pack build crates/life --target web --out-dir ../../web/pkg --features web
wasm-pack build crates/llm-life --target web --out-dir ../../web/pkg-llm --no-default-features --features web
python3 web/serve.py            # http://127.0.0.1:8010/
```

On the live pages, everything loads from Hugging Face. Adding `?local=1` to a
page's URL switches it to local files: a `web/models` symlink to a local models
directory for the base GGUF and tokenizer, and the adapter and small-model `.bin`
files next to `web/index.html`. The checkpoints are not tracked in this repo:

```bash
hf download idle-intelligence/stencil-life --local-dir artifacts
hf download idle-intelligence/llm-of-life-lora --local-dir artifacts
```

## Train it

Training is Python and PyTorch, in a virtual environment (`torch`,
`transformers`, `safetensors`, `gguf`, `numpy`). Rust owns the reference
implementation, the scoring and the inference; weights cross over as safetensors.
`pytorch/life.py` reproduces the Rust random number generator and grid step bit
for bit. Run the scripts from inside `pytorch/`:

```bash
cd pytorch
python train_stencil_models.py --out-dir ../out        # BERT of Life, 9 numbers, stencil
python train_variant_b.py --gguf-dir <dir> --tokenizer-dir <dir> --steps 300   # whole-grid adapter
python train_variant_a.py --gguf-dir <dir> --tokenizer-dir <dir> --steps 300   # per-cell adapter (--norules for the other)
python minimal_life.py --out ../minimal.json           # how small can an exact float network be
python bitslice_life.py --out ../bitslice.json         # ternary weights, step activations
```

The 2-step trainer is `pytorch/bitslice_life2.py`. `docs/pytorch-training.md`
explains how the PyTorch models were checked against the original Rust
training, and `tools/` holds the LoRA merge and the Hugging Face PEFT parity
check. The runs behind the tables above are written up in `docs/runs/`, one
file per run.

## Layout

```
crates/life/           classical B/S cellular automaton (the ground truth), + wasm wrapper
crates/llm-life/       per-cell and whole-grid packing, scoring, PGM output, the small
                       models (bert/, vector/), native driver src/bin/llm-life.rs
crates/llm-life-lean/  the language-model methods on the lean engine (native and wasm)
crates/life-lut-bench/ lookup-table and bit-packed Life, CPU and GPU, native benchmark
web/                   demo page, compare/ (every method), fullscreen/, cellspeed/
pytorch/               training, data generator, the trained bit-slice networks
scripts/               build.sh (CI's own build, also runnable locally) and serve.py
docs/runs/             one document per experiment: parameters, results, observations
docs/pictures/         first-picture output + accuracy tables
CONCEPT.md             the design note the project started from
```

## Engine

The language-model methods run on `lean`, the project's wgpu and WGSL engine,
from [llm-web](https://github.com/idle-intelligence/llm-web), pulled in as a git
dependency pinned to a `rev`. BERT of Life, the 9-number MLP and the stencil
model still run on Burn. A small patch this repo needs is vendored under
`patches/`. If a corporate proxy blocks Cargo's built-in git fetcher, set
`CARGO_NET_GIT_FETCH_WITH_CLI=true`.

## Related

The BERT of Life, 9 numbers to centre and stencil models also run in the engine
of [jacobi2000](https://github.com/ilnmtlbnm/jacobi2000), a personal physics
project on grids.

Prior work on learning Life with small networks: Springer and Kenyon, "It's Hard
for Neural Networks to Learn the Game of Life" (arXiv:2009.01398); BitNet b1.58,
Ma et al. (arXiv:2402.17764); Miotti et al., "Differentiable Logic Cellular
Automata" (arXiv:2506.04912).

## License

MIT. The base model, Qwen2.5-0.5B-Instruct, is Apache-2.0.
