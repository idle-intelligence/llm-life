"""Retrain the three stencil-life models from scratch in PyTorch, matching
crates/llm-life/src/{bert,vector}/train.rs: same exhaustive 512-case split
(`bert::data::split_512`, deterministic stride, not random), same Adam
default (burn's `AdamConfig::new()` = betas 0.9/0.999, eps 1e-8, no weight
decay), same held-out rollout convention (5 teacher-forced generations,
seeds 1..=3, at 16x16, density 0.28).

Published configs (config.json in the staged HF package):
  BERT of Life:  d_model 16, 1 layer, 1 head           -> 3,490 parameters
  9-numbers MLP: 2-layer MLP, hidden 32                -> 1,442 parameters
  Stencil:       d_model 16, 1 layer, 1 head            -> 3,329 parameters
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from life import Grid, Rule
from score import score
from stencil_models import BertOfLife, Mlp2OfLife, StencilOfLife, stencil_neighbors


def all_cases(rule: Rule):
    cases = []
    for k in range(512):
        nb = [(k >> b) & 1 for b in range(8)]
        self_state = (k >> 8) & 1
        n = sum(nb)
        y = 1 if rule.next(self_state != 0, n) else 0
        cases.append((nb + [self_state], y))
    return cases


def split_512(rule: Rule, held_out_n: int):
    all_c = all_cases(rule)
    stride = 512 // max(held_out_n, 1)
    train, held = [], []
    for k, case in enumerate(all_c):
        if k % stride == 0 and len(held) < held_out_n:
            held.append(case)
        else:
            train.append(case)
    return train, held


def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def grid_cases(g: Grid, rule: Rule):
    out = []
    for i in range(g.width * g.height):
        nb = [int(g.cells[j]) for j in g.neighbor_indices(i)]
        self_state = int(g.cells[i])
        n = sum(nb)
        y = 1 if rule.next(self_state != 0, n) else 0
        out.append((nb + [self_state], y))
    return out


def rollout_score(logits_fn, rule: Rule, seed: int, size: int = 16):
    grid = Grid.random(size, size, seed, 0.28)
    scores = []
    for gen in range(1, 6):
        truth = grid.step(rule)
        cases = grid_cases(grid, rule)
        p = logits_fn(cases)
        pred = (np.asarray(p) >= 0.5).astype("uint8")
        scores.append(score(truth, pred, np.asarray(p)))
        grid = truth
    return scores


def train_classifier(model, forward_fn, cases_to_tensor, train, held, all_cases_, steps, lr, log_every=5):
    train_x, train_y = cases_to_tensor(train)
    all_x, all_y = cases_to_tensor(all_cases_)
    held_x, held_y = cases_to_tensor(held)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    steps_to_512 = None
    for step in range(1, steps + 1):
        model.train()
        logits = forward_fn(model, train_x)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if steps_to_512 is None and step % log_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (forward_fn(model, all_x).argmax(-1) == all_y).float().mean().item()
            if acc >= 1.0:
                steps_to_512 = step
                break
    model.eval()
    with torch.no_grad():
        held_acc = (forward_fn(model, held_x).argmax(-1) == held_y).float().mean().item()
    return steps_to_512, held_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bert-steps", type=int, default=2000)
    ap.add_argument("--bert-lr", type=float, default=2e-3)
    ap.add_argument("--mlp2-steps", type=int, default=800)
    ap.add_argument("--mlp2-lr", type=float, default=2e-3)
    ap.add_argument("--stencil-steps", type=int, default=2000)
    ap.add_argument("--stencil-lr", type=float, default=2e-3)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    rule = Rule.life()
    report = {}

    # BERT of Life
    train, held = split_512(rule, 64)
    all_c = all_cases(rule)

    def tok_tensor(cases):
        x = torch.tensor([c for c, _ in cases], dtype=torch.long)
        y = torch.tensor([y for _, y in cases], dtype=torch.long)
        return x, y

    bert = BertOfLife(d_model=16, n_heads=1, d_ff=64)
    steps_to_512, held_acc = train_classifier(
        bert, lambda m, x: m(x), tok_tensor, train, held, all_c, args.bert_steps, args.bert_lr
    )
    scores = []
    for seed in (1, 2, 3):
        scores.extend(rollout_score(
            lambda cases: F.softmax(bert(torch.tensor([c for c, _ in cases], dtype=torch.long)), dim=-1)[:, 1].detach().numpy(),
            rule, seed,
        ))
    report["bert"] = {
        "parameters": num_params(bert), "steps_to_512": steps_to_512, "held_out_acc": held_acc,
        "mean_iou": float(np.mean([s["iou"] for s in scores])),
        "mean_accuracy": float(np.mean([s["accuracy"] for s in scores])),
    }
    from safetensors.torch import save_file
    sd = {k: v.detach() for k, v in bert.state_dict().items()}
    sd = {k.replace(".weight", ".gamma") if "norm" in k and k.endswith(".weight") else k: v for k, v in sd.items()}
    sd = {k.replace(".bias", ".beta") if "norm" in k and k.endswith(".bias") else k: v for k, v in sd.items()}
    save_file(bert.state_dict(), f"{args.out_dir}/bert-d16-L1-retrained.safetensors")
    print("bert:", report["bert"])

    # 9-numbers MLP (2-layer, hidden 32)
    def bits_tensor(cases):
        x = torch.tensor([c for c, _ in cases], dtype=torch.float32)
        y = torch.tensor([y for _, y in cases], dtype=torch.long)
        return x, y

    mlp2 = Mlp2OfLife(hidden=32)
    steps_to_512, held_acc = train_classifier(
        mlp2, lambda m, x: m(x), bits_tensor, train, held, all_c, args.mlp2_steps, args.mlp2_lr
    )
    scores = []
    for seed in (1, 2, 3):
        scores.extend(rollout_score(
            lambda cases: F.softmax(mlp2(torch.tensor([c for c, _ in cases], dtype=torch.float32)), dim=-1)[:, 1].detach().numpy(),
            rule, seed,
        ))
    report["mlp2"] = {
        "parameters": num_params(mlp2), "steps_to_512": steps_to_512, "held_out_acc": held_acc,
        "mean_iou": float(np.mean([s["iou"] for s in scores])),
        "mean_accuracy": float(np.mean([s["accuracy"] for s in scores])),
    }
    save_file(mlp2.state_dict(), f"{args.out_dir}/mlp2-32-retrained.safetensors")
    print("mlp2:", report["mlp2"])

    # Stencil (whole-grid, trained at 16x16, generalizes by construction: no
    # absolute position embedding)
    neighbors16 = stencil_neighbors(16, 16)
    stencil = StencilOfLife(d_model=16, n_layers=1, n_heads=1)
    opt = torch.optim.Adam(stencil.parameters(), lr=args.stencil_lr, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(0)
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    for step in range(1, args.stencil_steps + 1):
        stencil.train()
        xs, ys = [], []
        for d in densities:
            seed = int(rng.integers(0, 1_000_000))
            g = Grid.random(16, 16, seed, d)
            nxt = g.step(rule)
            xs.append(g.cells.astype("float32"))
            ys.append(nxt.cells.astype("float32"))
        x = torch.tensor(np.stack(xs))
        y = torch.tensor(np.stack(ys))
        logits = stencil(x, neighbors16)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    stencil.eval()
    scores = []
    for seed in (1, 2, 3):
        scores.extend(rollout_score(
            lambda cases: torch.sigmoid(stencil(
                torch.tensor([[c[-1]] for c, _ in cases], dtype=torch.float32).new_zeros(1, 256),
                neighbors16,
            )).detach().numpy().reshape(-1) if False else None,
            rule, seed,
        )) if False else None
    # Whole-grid rollout (stencil takes a full grid, not per-case batches).
    def stencil_rollout(seed):
        g = Grid.random(16, 16, seed, 0.28)
        out = []
        for gen in range(1, 6):
            truth = g.step(rule)
            x = torch.tensor(g.cells.astype("float32")).unsqueeze(0)
            with torch.no_grad():
                p = torch.sigmoid(stencil(x, neighbors16)).squeeze(0).numpy()
            pred = (p >= 0.5).astype("uint8")
            out.append(score(truth, pred, p))
            g = truth
        return out

    scores = []
    for seed in (1, 2, 3):
        scores.extend(stencil_rollout(seed))
    report["stencil"] = {
        "parameters": num_params(stencil),
        "mean_iou": float(np.mean([s["iou"] for s in scores])),
        "mean_accuracy": float(np.mean([s["accuracy"] for s in scores])),
    }
    save_file(stencil.state_dict(), f"{args.out_dir}/stencil-d16-L1-retrained.safetensors")
    print("stencil:", report["stencil"])

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
