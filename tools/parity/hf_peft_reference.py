"""HF transformers + PEFT reference for llm-life's language-model methods.

Writes the `[dead, alive]` answer logits that the lean engine
(crates/llm-life-lean) must reproduce token-exactly, greedy (the answer is
the argmax of the two):

  * per cell ("variant A"): all 512 (8 neighbours, self) cases against
    four resident prefixes: the base model with rules + six worked examples
    (the demo's base row), the base model with rules only (the 215/512 run),
    and the two per-cell adapters with the prefix each was trained with.
    Plain causal prompts (prefix, then one cell prompt), the prefix run
    once and its KV cache reused for every cell.
  * whole grid ("variant B"): fixed random 16x16 and 32x32 grids, base and
    the whole-grid adapter of that size: one forward, a custom 4D attention
    mask (prefix causal, each cell token sees the prefix, itself and its 8
    toroidal neighbours) and custom position ids (every cell token at the
    position after the prefix).

Base weights: the same Q4_0 GGUF, dequantized by transformers' GGUF loader.
Adapters: the published LLMLIFE2 files, converted in memory to PEFT's
layout (`lora_A` = a^T, `lora_B` = b^T) and loaded with
`set_peft_model_state_dict` on a `get_peft_model` LoRA model (r, alpha
from the file header, q/k/v/o). Head: the token embedding rows of '0' and
'1' applied to the final normed hidden state, the head the adapters were
trained against (the GGUF also carries a separately quantized
`output.weight`, which is not it).

Usage (a venv with tools/parity/requirements.txt):

  python tools/parity/hf_peft_reference.py --gguf <q4_0.gguf> \
      --hf-dir <Qwen2.5-0.5B-Instruct dir with tokenizer.json> \
      --adapters <dir with lora-*.bin> \
      --out crates/llm-life-lean/tests/fixtures/hf_peft_reference.json
"""

import argparse
import copy
import json
import struct
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

RULE = "B3/S23"

# Text identical to crates/llm-life/src/variant_a.rs / variant_b.rs (the
# Rust gate also checks token ids against the ones recorded here).
RULES_A = (
    f"Cellular automaton, rule {RULE}. Each cell is 0 (dead) or 1 (alive).\n"
    "A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n"
    "A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n"
    "For each cell, answer with one digit: its next state.\n"
)
NORULES_A = "For each cell, answer with one digit: its next state.\n"
RULES_B = (
    f"Cellular automaton, rule {RULE}. Each cell is 0 (dead) or 1 (alive).\n"
    "A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n"
    "A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n"
    "Grid:\n"
)


def cell_prompt(nb, me):
    return "Neighbors:" + "".join(" " + str(n) for n in nb) + f" / Self: {me} / Next: "


def fewshot():
    cases = [
        ([1, 1, 1, 0, 0, 0, 0, 0], 0, 1),
        ([1, 1, 0, 1, 0, 0, 0, 0], 1, 1),
        ([1, 1, 0, 0, 0, 0, 0, 0], 1, 1),
        ([1, 1, 1, 1, 0, 0, 0, 0], 1, 0),
        ([1, 0, 0, 0, 0, 0, 0, 0], 1, 0),
        ([1, 1, 0, 0, 0, 0, 0, 0], 0, 0),
    ]
    return "Examples:\n" + "".join(cell_prompt(nb, me) + f"{nxt}\n" for nb, me, nxt in cases)


def case_cells(k):
    return [(k >> b) & 1 for b in range(8)], (k >> 8) & 1


def neighbor_indices(w, h, i):
    """crates/life/src/grid.rs `neighbor_indices`: toroidal, row-major,
    (dy, dx) in -1..=1 x -1..=1 skipping (0, 0)."""
    x, y = i % w, i // w
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            out.append(((y + dy) % h) * w + (x + dx) % w)
    return out


def parse_llmlife2(path):
    data = Path(path).read_bytes()
    assert data[:8] == b"LLMLIFE2", path
    rank, alpha, mlp = struct.unpack_from("<IfB", data, 8)
    assert mlp == 0
    (n,) = struct.unpack_from("<I", data, 17)
    off, tensors = 21, []
    for _ in range(n):
        rows, cols = struct.unpack_from("<II", data, off)
        off += 8
        t = np.frombuffer(data, dtype="<f4", count=rows * cols, offset=off).reshape(rows, cols)
        off += rows * cols * 4
        tensors.append(torch.from_numpy(t.copy()))
    assert off == len(data)
    return rank, alpha, tensors


def load_model(args, adapter):
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, gguf_file=args.gguf, dtype=torch.float32, attn_implementation="eager", device_map="cpu"
    )
    model.eval()
    if adapter is None:
        return model, model.model
    rank, alpha, tensors = parse_llmlife2(Path(args.adapters) / adapter)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    peft = get_peft_model(model, cfg)
    sd, it = {}, iter(tensors)
    for layer in range(model.config.num_hidden_layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            a, b = next(it), next(it)  # a: [in, r], b: [r, out]
            p = f"base_model.model.model.layers.{layer}.self_attn.{name}"
            sd[f"{p}.lora_A.weight"] = a.T.contiguous()
            sd[f"{p}.lora_B.weight"] = b.T.contiguous()
    res = set_peft_model_state_dict(peft, sd)
    assert not res.unexpected_keys, res.unexpected_keys[:4]
    assert not [k for k in res.missing_keys if "lora_" in k], "LoRA weights not all loaded"
    peft.eval()
    return peft, peft.get_base_model().model


def head(model, tok):
    ids = [tok.encode("0", add_special_tokens=False)[0], tok.encode("1", add_special_tokens=False)[0]]
    return ids, model.get_input_embeddings().weight[ids].detach()


@torch.no_grad()
def per_cell(args, tok, prefix_text, adapter, batch=64):
    model, inner = load_model(args, adapter)
    ids, rows = head(model, tok)
    prefix = tok.encode(prefix_text, add_special_tokens=False)
    prompts = [tok.encode("\n" + cell_prompt(*case_cells(k)), add_special_tokens=False) for k in range(512)]
    assert len({len(p) for p in prompts}) == 1, "per-cell prompts must share one length"
    out = inner(input_ids=torch.tensor([prefix]), use_cache=True)
    prefix_cache = out.past_key_values
    logits = []
    for s in range(0, 512, batch):
        chunk = prompts[s : s + batch]
        cache = copy.deepcopy(prefix_cache)
        cache.batch_repeat_interleave(len(chunk))
        h = inner(input_ids=torch.tensor(chunk), past_key_values=cache, use_cache=True).last_hidden_state
        logits += (h[:, -1, :] @ rows.T).tolist()
    # The shared-prefix cache must be invisible: two cases, forwarded whole.
    for k in (0, 511):
        h = inner(input_ids=torch.tensor([prefix + prompts[k]]), use_cache=False).last_hidden_state
        d = (h[0, -1, :] @ rows.T - torch.tensor(logits[k])).abs().max().item()
        assert d < 1e-3, f"prefix cache changed case {k} by {d}"
    return {"prefix_ids": prefix, "prompt_ids": prompts, "dead_alive_ids": ids, "logits": logits}


@torch.no_grad()
def whole_grid(args, tok, adapter, w, h, cells):
    model, inner = load_model(args, adapter)
    ids, rows = head(model, tok)
    prefix = tok.encode(RULES_B, add_special_tokens=False)
    p, n = len(prefix), w * h
    t = p + n
    tokens = prefix + [ids[1] if c else ids[0] for c in cells]
    positions = list(range(p)) + [p] * n
    allowed = torch.zeros(t, t, dtype=torch.bool)
    for i in range(p):
        allowed[i, : i + 1] = True
    for c in range(n):
        allowed[p + c, :p] = True
        allowed[p + c, p + c] = True
        for j in neighbor_indices(w, h, c):
            allowed[p + c, p + j] = True
    mask = torch.zeros(1, 1, t, t)
    mask.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    hs = inner(
        input_ids=torch.tensor([tokens]),
        position_ids=torch.tensor([positions]),
        attention_mask=mask,
        use_cache=False,
    ).last_hidden_state
    return {"prefix_ids": prefix, "logits": (hs[0, p:, :] @ rows.T).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.hf_dir)

    a_configs = [
        ("base-fewshot", RULES_A + fewshot(), None),
        ("base-rules", RULES_A, None),
        ("a-norules-300", NORULES_A, "lora-a-norules-300.bin"),
        ("a-rules-300", RULES_A, "lora-a-rules-300.bin"),
    ]
    out = {"source": "transformers GGUF-loaded base + PEFT LoRA, see tools/parity/hf_peft_reference.py", "per_cell": [], "whole_grid": []}
    for name, prefix, adapter in a_configs:
        r = per_cell(args, tok, prefix, adapter)
        r.update(name=name, adapter=adapter)
        out["per_cell"].append(r)
        print(f"per-cell {name}: done", flush=True)

    rng = np.random.default_rng(20261002)
    grids = [(16, 16, rng.random(256) < 0.35), (16, 16, rng.random(256) < 0.35), (32, 32, rng.random(1024) < 0.35)]
    for w, h, g in grids:
        cells = [int(c) for c in g]
        adapter = {16: "lora-b-16-s3-300.bin", 32: "lora-b-32.bin"}[w]
        for a in (None, adapter):
            r = whole_grid(args, tok, a, w, h, cells)
            r.update(name=f"{'base' if a is None else a[:-4]}-{w}x{h}", adapter=a, width=w, height=h, cells=cells)
            out["whole_grid"].append(r)
            print(f"whole grid {r['name']}: done", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
