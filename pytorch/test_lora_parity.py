"""Parity: the PyTorch QwenLife forward (base = same Q4_0 GGUF via
transformers, LoRA converted to PEFT layout) against `tools/export`'s
`lora-b-16-s3-300-forward-oracle.json` — the exact `TrainModel::forward`
(Burn, NdArray, CPU) on the published `lora-b-16-s3-300` adapter, on a fixed
16x16 grid (seed 1_000_000, density 0.28)."""

import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from life import Grid, Rule
from qwen_life import LoraSpec, QwenLife, load_base, load_lora_state
from variant_b import mask_out, p_alive, pack, rules_prefix


def main(gguf_dir, gguf_file, tokenizer_dir, oracle_path, lora_path):
    oracle = json.load(open(oracle_path))
    rule = Rule.life()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    prefix = tokenizer.encode(rules_prefix(rule), add_special_tokens=False)
    assert len(prefix) == oracle["prefix_len"], (len(prefix), oracle["prefix_len"])
    dead = tokenizer.encode("0", add_special_tokens=False)[0]
    alive = tokenizer.encode("1", add_special_tokens=False)[0]
    assert dead == oracle["dead_token"] and alive == oracle["alive_token"]

    hf = load_base(gguf_dir, gguf_file).to("cpu")
    hf.eval()
    model = QwenLife(hf, [dead, alive], LoraSpec(8, 16.0))
    load_lora_state(model, lora_path)
    model.eval()

    g = Grid.random(oracle["grid_size"], oracle["grid_size"], oracle["seed"], oracle["density"])
    packed = pack(g, prefix, dead, alive)
    m = mask_out(packed)
    with torch.no_grad():
        logits = model(packed.tokens, packed.positions, m)
    pa = p_alive(logits, packed.grid_start, oracle["grid_size"] ** 2).numpy()

    want_pa = np.array(oracle["p_alive"])
    diff = np.abs(pa - want_pa).max()
    print(f"variant B forward parity (lora-b-16-s3-300): max |p_alive diff| = {diff:.3e}")

    want_logits = np.array(oracle["logits_all"]).reshape(-1, 2)
    got_logits = logits[0].numpy()
    ldiff = np.abs(got_logits - want_logits).max()
    print(f"max |logit diff| (all {got_logits.shape[0]} positions) = {ldiff:.3e}")

    pred = (pa >= 0.5).astype("uint8")
    truth = np.array(oracle["truth"]) if "truth" in oracle else None
    print(f"p_alive range: [{pa.min():.4f}, {pa.max():.4f}]")
    return diff, ldiff


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--gguf-file", default="qwen2.5-0.5b-instruct-q4_0.gguf")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--lora", required=True)
    args = ap.parse_args()
    diff, ldiff = main(args.gguf_dir, args.gguf_file, args.tokenizer_dir, args.oracle, args.lora)
    assert ldiff < 1e-2, f"logit parity failed: {ldiff}"
    print("LORA FORWARD PARITY PASSED")
