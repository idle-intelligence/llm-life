"""Parity: PyTorch ports of the three stencil-life models against
`tools/export`'s oracle dumps (exact weights and exact forward inputs the
Burn model used). Run after `cargo run --bin export` has written
`export-out/`.
"""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from load_stencil import load_bert, load_mlp2, load_stencil
from stencil_models import stencil_neighbors


def check_bert(out_dir: str) -> None:
    model = load_bert(f"{out_dir}/bert-d16-L1.safetensors")
    oracle = json.load(open(f"{out_dir}/bert-oracle.json"))
    n = oracle["shape"][0]
    from life import Rule

    rule = Rule.life()
    cases = []
    for k in range(512):
        nb = [(k >> b) & 1 for b in range(8)]
        self_state = (k >> 8) & 1
        cases.append(nb + [self_state])
    tokens = torch.tensor(cases, dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens).numpy()
    want = np.array(oracle["logits"]).reshape(oracle["shape"])
    diff = np.abs(logits - want).max()
    print(f"bert-of-life: max abs logit diff = {diff:.3e} ({n} cases)")
    pred = logits.argmax(-1)
    targets = np.array(oracle["targets"])
    acc = (pred == targets).mean()
    print(f"bert-of-life: accuracy on oracle targets = {acc:.4f}")
    assert diff < 1e-3, "BERT parity failed"


def check_mlp2(out_dir: str) -> None:
    model = load_mlp2(f"{out_dir}/mlp2-32-lrfix.safetensors")
    oracle = json.load(open(f"{out_dir}/mlp2-oracle.json"))
    cases = []
    for k in range(512):
        nb = [(k >> b) & 1 for b in range(8)]
        self_state = (k >> 8) & 1
        cases.append(nb + [self_state])
    bits = torch.tensor(cases, dtype=torch.float32)
    with torch.no_grad():
        logits = model(bits).numpy()
    want = np.array(oracle["logits"]).reshape(oracle["shape"])
    diff = np.abs(logits - want).max()
    print(f"mlp2: max abs logit diff = {diff:.3e}")
    assert diff < 1e-3, "MLP2 parity failed"


def check_stencil(out_dir: str) -> None:
    model = load_stencil(f"{out_dir}/stencil-d16-L1.safetensors", n_layers=1)
    oracle = json.load(open(f"{out_dir}/stencil-oracle.json"))
    from life import Grid

    g = Grid.random(16, 16, 1_000_000, 0.28)
    cells = torch.tensor(g.cells, dtype=torch.float32).unsqueeze(0)
    neighbors = stencil_neighbors(16, 16)
    with torch.no_grad():
        logits = model(cells, neighbors).squeeze(0).numpy()
    want = np.array(oracle["logits"])
    diff = np.abs(logits - want).max()
    print(f"stencil: max abs logit diff = {diff:.3e}")
    truth_rust = np.array(oracle["targets"])
    truth_py = g.step(__import__("life").Rule.life()).cells
    assert (truth_rust == truth_py).all(), "Grid.random/step port does not match Rust RNG"
    print("stencil: Python Grid.random/step matches Rust bit-for-bit")
    assert diff < 1e-3, "Stencil parity failed"


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "export-out"
    check_bert(out_dir)
    check_mlp2(out_dir)
    check_stencil(out_dir)
    print("ALL PARITY CHECKS PASSED")
