"""Parity: PyTorch variant-A packing/forward against
`tools/export`'s `lora-{a-rules,a-norules}-300-variant-a-oracle.json` — the
exact `TrainModel` forward through `run_a::full_sequence`'s block-diagonal
causal packing on the first 64 of the 512 exhaustive cases, with the
published (trained) adapter loaded."""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qwen_life import LoraSpec, QwenLife, load_base, load_lora_state
from train_variant_a import build_batch


def main(gguf_dir, gguf_file, tokenizer_dir, oracle_path, lora_path, norules):
    from transformers import AutoTokenizer

    oracle = json.load(open(oracle_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    text = (
        "For each cell, answer with one digit: its next state.\n"
        if norules
        else (
            "Cellular automaton, rule B3/S23. Each cell is 0 (dead) or 1 (alive).\n"
            "A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n"
            "A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n"
            "For each cell, answer with one digit: its next state.\n"
        )
    )
    prefix_ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(prefix_ids) == oracle["prefix_len"], (len(prefix_ids), oracle["prefix_len"])

    hf = load_base(gguf_dir, gguf_file).to("cpu")
    hf.eval()
    model = QwenLife(hf, [oracle["dead_token"], oracle["alive_token"]], LoraSpec(8, 16.0))
    load_lora_state(model, lora_path)
    model.eval()

    batch = [(c["neighbors"], c["self"], c["target"]) for c in oracle["cases"]]
    tokens, positions, m, rows = build_batch(tokenizer, prefix_ids, batch, "cpu")
    assert tokens.shape[0] == oracle["total_tokens"], (tokens.shape[0], oracle["total_tokens"])
    # `export.rs` wrote `Chunk::answer_rows()` verbatim (chunk-relative, not
    # counting the prefix); the forward ran on `prefix ++ chunk`, so the
    # logits index is `row + prefix_len`, exactly what `run_a::evaluate_a`
    # itself adds before reading `cell_cross_entropy`'s rows.
    oracle_rows = [r + oracle["prefix_len"] for r in oracle["answer_rows"]]
    assert rows == oracle_rows, "answer row indices diverge from the Rust packer"

    with torch.no_grad():
        logits = model(tokens, positions, m)
    got = logits[0].numpy()
    want = np.array(oracle["logits_all"]).reshape(-1, 2)
    diff = np.abs(got - want).max()
    print(f"variant A forward parity ({'norules' if norules else 'rules'}): max |logit diff| = {diff:.3e}")

    row_idx = torch.tensor(rows, dtype=torch.long)
    pred = logits[0, row_idx, :].argmax(-1).numpy()
    targets = np.array(oracle["targets"])
    acc = (pred == targets).mean()
    print(f"accuracy on this 64-case chunk vs published targets: {acc:.4f}")
    return diff


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--gguf-file", default="qwen2.5-0.5b-instruct-q4_0.gguf")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--norules", action="store_true")
    args = ap.parse_args()
    diff = main(args.gguf_dir, args.gguf_file, args.tokenizer_dir, args.oracle, args.lora, args.norules)
    assert diff < 1e-2, f"variant A parity failed: {diff}"
    print("VARIANT A FORWARD PARITY PASSED")
