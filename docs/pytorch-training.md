# PyTorch training

v1 trained everything in Rust: a hand-rolled f32 Qwen2 forward in Burn
(`crates/llm-life/src/train/*`) just to backpropagate, LoRA saved in a
hand-rolled `LLMLIFE2` format, and the three tiny from-scratch nets (BERT of
Life, the 9-number MLP, the stencil attention model) in Burn's
`BinBytesRecorder` format. This move puts training in Python + PyTorch,
following the stack rule: Rust owns data generation and inference, Python
trains, weights cross the boundary as safetensors + config.

Training code lives under `pytorch/` (a `venv`, not committed). The Rust
side keeps everything it had — `crates/life`, `crates/llm-life`'s packing,
scoring, and the native/web engines are unchanged. Nothing was deleted:
`crates/llm-life/src/train/*`, `bert/train.rs` and `vector/train.rs` still
build and still produce the same checkpoints; this migration adds a second,
parallel path rather than removing the first (that deletion is a separate,
later decision once the published Hub weights point at the PyTorch output).

## The bridge: `tools export`

`crates/llm-life/src/bin/export.rs` (features `native,cpu,export`, NdArray
backend, no GPU) is a one-shot tool, not part of the ongoing build. It reads
the published Burn checkpoints and writes:

- the three tiny models to safetensors, native tensor names (Burn's own
  `Linear`/`LayerNorm`/`Embedding` field names, e.g. `mha.query.weight`,
  `norm_1.gamma`) — the PyTorch loader (`pytorch/load_stencil.py`) transposes
  every `*.weight` that is a `Linear` (Burn stores `[in, out]`, PyTorch
  `[out, in]`) and renames LayerNorm's `gamma`/`beta` to `weight`/`bias`;
- each `LLMLIFE2` LoRA adapter to PEFT layout (`lora_A.weight` `[r, in]`,
  `lora_B.weight` `[out, r]` — a transpose of the Rust file's `[in, r]` /
  `[r, out]`, the same delta either way) plus an `adapter_config.json`;
- fixed-input oracle logits for every model, read straight out of the loaded
  Burn module before any PyTorch code runs, so the PyTorch port has an exact
  target to match, not just an accuracy claim.

Run once per checkpoint set:

```bash
cargo run --release -p llm-life --bin export --no-default-features \
  --features native,cpu,export -- \
  --stencil-life <dir with bert-d16-L1.bin, mlp2-32-lrfix.bin, stencil-d16-L1.bin> \
  --lora <dir with lora-*.bin> \
  --gguf <qwen2.5-0.5b-instruct-q4_0.gguf> \
  --tokenizer <tokenizer.json> \
  --out export-out
```

## Parity

### Stencil-life (BERT of Life, the 9-number MLP, the stencil model)

`pytorch/stencil_models.py` ports each architecture line-for-line off the
Burn source (`burn-nn` 0.20.1's `TransformerEncoderLayer`/`MultiHeadAttention`
for BERT of Life — post-norm, separate q/k/v/out `Linear`s, no fused
in-proj; `vector::model::StencilBlock` for the stencil model — pre-norm,
neighbour-gather attention instead of a dense mask).
`pytorch/test_stencil_parity.py` loads the exported safetensors and checks
every model's logits against the oracle on the exhaustive 512 neighbourhood
cases (BERT, MLP) or the held-out 16x16 grid (stencil):

max |logit diff| against the Burn oracle, float32:

| model | max abs diff |
|---|---|
| BERT of Life | 1.85e-6 |
| 9-number MLP | 7.63e-6 |
| stencil (grid-to-grid) | 5.72e-6 |

`pytorch/life.py`'s xorshift64 port (`Grid.random`, `Rng`, `sample_grid`,
`held_out`) is checked bit-for-bit against the Rust RNG via the stencil
oracle's fixed grid (seed 1,000,000, density 0.28): identical cells.

### LoRA on Qwen2.5-0.5B-Instruct (variant B, whole grid)

The base model loads through `transformers`' own GGUF loader
(`AutoModelForCausalLM.from_pretrained(dir, gguf_file=...)`), dequantizing
the same Q4_0 blocks Burn's `weights::dequant_q4_0` dequantizes — same
numbers, different code. `pytorch/qwen_life.py`'s `QwenLife` wraps that
model with a from-scratch attention pass (reusing the model's own
`input_layernorm`/`post_attention_layernorm`/`mlp`/`norm` submodules
unchanged, since those are frozen and untouched by the adapter) because
`train/model.rs`'s reasons for reimplementing attention apply here too:
caller-supplied boolean mask (the stencil, not causal) and caller-supplied
positions (RoPE "bag" mode — every grid cell shares one position id).

Critically, the head is **not** `model.lm_head`: this GGUF's metadata reads
`tie_word_embeddings=False` and the loader gives it an untied `lm_head`, but
`TrainModel` always reads a two-row slice of the tied token embedding at the
answer tokens ('0', '1'). Using `lm_head` here would silently train and
score against a different head than the one every published number was
measured with.

`pytorch/test_lora_parity.py` loads the published `lora-b-16-s3-300`
adapter (converted to PEFT layout by `tools export`) and runs the exact
16x16 grid (seed 1,000,000, density 0.28) the export tool ran through
`TrainModel` on NdArray (CPU):

| | max abs diff vs Burn oracle |
|---|---|
| p(alive) per cell | 1.43e-7 |
| logits, all 324 positions | 1.72e-4 |

The logit diff (float32 accumulated over 24 layers, two independent
implementations of RMSNorm/RoPE/GQA/attention) is far below anything that
would change an argmax; p(alive), the number the demo and the accuracy
tables read, agrees to seven digits.

## What this does not yet prove

Parity here is Burn `TrainModel` (NdArray, CPU) vs PyTorch `QwenLife`
(transformers + custom attention, CPU) on the same converted weights —
"same math, same weights, same answer." It is not yet PyTorch vs the `lean`
inference engine llm-web is building (CONCEPT.md's ultimate target):
proving the converted safetensors adapter loads and produces the same
logits in `lean` is inference-engine work, out of scope for a training
agent, and depends on `lean`'s runtime-LoRA support landing first.

## Retraining from scratch

Same data generator (`pytorch/life.py`, xorshift64 ported bit-for-bit), same
packing (`pytorch/variant_b.py`, `pytorch/train_variant_a.py`'s per-cell
packer), same hyperparameters as the run that produced the published
adapters (`docs/runs/2026-09-20-ft-b-16-s3-300.md`,
`docs/runs/2026-09-20-ft-a-{rules,norules}-300.md`). See
`docs/runs/2026-09-29-pytorch-retrain.md` for the numbers.

A sanity check worth recording: the published Burn run and this port's
retrain, run with the same seed, produce **the exact same step-1 loss**
(0.2138) even though neither run's LoRA `a` matrix is seeded (Burn's
`Tensor::random` and this port's `torch.randn` are both left to their
default per-process RNG). This is expected, not a coincidence: `b` starts
at zero in both, so `delta(x) = (x @ a) @ b * scale` is exactly zero until
the first optimizer step regardless of `a`'s random values — step 1's loss
is the base model's loss on the first sampled grid, and the data RNG *is*
seeded identically in both frameworks. It is the cleanest evidence that the
packing, masking, and forward math agree end to end, independent of any
LoRA-specific randomness.
