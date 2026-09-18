# llm-life

A language model as the update rule of a cellular automaton. One pixel is one
cell; one cell's next state is read out of the model's logits over a two-token
answer vocabulary. The picture is the model's confidence per cell, and the
drift from true Life is the model's character rendered as an image.

`CONCEPT.md` is the design. `CLAUDE.md` is how this repo works.

## Layout

```
crates/life/        classical B/S cellular automaton (the ground truth), + wasm wrapper
crates/llm-life/    variant A packing (per-cell prompts, block-diagonal mask),
                    variant B packing (stencil mask), scoring, PGM output
                    src/bin/llm-life.rs   native driver ("picture")
                    src/web.rs            wasm-bindgen LifeEngine for the tab
web/                the demo page (STATUS / INPUT / OUTPUT / PERFORMANCE)
scripts/headless/   Playwright (bundled Chromium only) verification
docs/pictures/      first-picture output + accuracy tables
```

The engine is `llm-wasm` from a dedicated worktree and branch of `llm-web`:

```
git -C ~/Code/idle-intelligence/llm-web worktree add .claude/worktrees/llm-life -b llm-life mcp-agent
```

llm-web's Cargo manifests are frozen; the engine changes this repo needs live
on that `llm-life` branch, in `crates/llm-wasm/src/`.

## Build and run

```bash
# classical demo
wasm-pack build crates/life --target web --out-dir ../../web/pkg --features web
ln -s ~/Code/idle-intelligence/models web/models   # once, gitignored — LLM mode reads it same-origin
python3 web/serve.py            # http://127.0.0.1:8010/
node scripts/headless/run.mjs

# LLM mode (variant A + B) — sparse-mask attention and the tiled prefill
# GEMM live in llm-wasm and are both compiled in unconditionally (no cargo
# feature or env var to set): `ForwardSpec::sparse` when the caller supplies
# a `SparseMask`, `TILED_PREFILL_MATMUL` (crates/llm-wasm/src/gguf.rs) at
# `const … = true`. Rebuild after changing crates/llm-life/src/{variant_b,web}.rs:
wasm-pack build crates/llm-life --target web --out-dir ../../web/pkg-llm --no-default-features --features web
node scripts/headless/llm.mjs --url http://127.0.0.1:8010/ --mode llm --seed glider --grid 64

# native first picture (variant B, Qwen2.5-0.5B-Instruct Q4_0)
cargo run --release -p llm-life -- picture \
  --gguf ~/Code/idle-intelligence/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokenizer ~/Code/idle-intelligence/models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json \
  --size 64 --generations 10

# variant A (packed per-cell prompts, chunked against the resident prefix)
cargo run --release -p llm-life -- picture-a \
  --gguf ~/Code/idle-intelligence/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokenizer ~/Code/idle-intelligence/models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json \
  --size 64 --generations 10 --chunk-cells 128
```

Models live under `~/Code/idle-intelligence/models/`; weights are never
committed.
