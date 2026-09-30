"""How small can a network be and still learn Game of Life exactly?

Reuses the repo's own data generator and torus handling (life.py's Grid,
xorshift64-seeded, toroidal boundary, ported bit-for-bit from crates/life)
and its scoring convention (score.py), same as pytorch/train_stencil_models.py.

Reference: Springer & Kenyon, "It's Hard for Neural Networks To Learn the
Game of Life" (arXiv:2009.01398). Its abstract states the theoretical
minimum is a 2n+1 layer CNN for n steps, that minimal architectures rarely
converge in practice, that near-minimal nets are sign-flip-fragile, and
that there is a training density d0 that strongly affects convergence odds.
The abstract does not give concrete parameter counts or layer widths for a
minimal n=1 net (checked via WebFetch on 2026-09-30; full text not fetched).
The CNN family below is therefore defined here, not copied from the paper:

    Conv3x3(1 -> c, circular padding, bias) -> ReLU -> Conv1x1(c -> 1, bias) -> sigmoid

with c = 2 (m=1x, the smallest c for which a 3x3-to-1x1 net can even
represent a neighbour count and a self-state test) then c in {4, 8, 16}
(x2/x4/x8 widths). Params = 10c + (c + 1) = 11c + 1.

Exact game-of-life rule for reference: alive iff n==3, or (self and n==2).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from life import Grid, Rule
from score import score

RULE = Rule.life()


def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- models --

class MinimalCNN(nn.Module):
    """Conv3x3(1->c, circular) -> ReLU -> Conv1x1(c->1) -> sigmoid logits."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        # x: [b, h, w] -> [b, 1, h, w]
        x = x.unsqueeze(1)
        h = F.relu(self.conv1(x))
        return self.conv2(h).squeeze(1)


class DeepCNN(nn.Module):
    """n=2 in one pass: two stacked 3x3-circular conv blocks (5x5 receptive
    field), ReLU between, 1x1 readout. width multiplies both layers' channels."""

    def __init__(self, channels: int, depth: int = 2):
        super().__init__()
        chans = [1] + [channels] * depth
        self.convs = nn.ModuleList([
            nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, padding_mode="circular")
            for i in range(depth)
        ])
        self.readout = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        h = x.unsqueeze(1)
        for conv in self.convs:
            h = F.relu(conv(h))
        return self.readout(h).squeeze(1)


class TinyMLP(nn.Module):
    """9 neighbourhood values (8 neighbours + self, life.py's
    neighbor_indices order + self) -> hidden -> 1 sigmoid logit. Smallest
    hidden width tried below, for comparison with the repo's published
    1,442-param 2-layer/hidden-32 MLP (stencil_models.Mlp2OfLife)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(9, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, bits):
        return self.fc2(F.relu(self.fc1(bits))).squeeze(-1)


# ------------------------------------------------------------- data/eval --

def random_batch(rng: np.random.Generator, batch: int, size: int, density: float, n_steps: int = 1):
    xs, ys = [], []
    for _ in range(batch):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        x = g.cells.astype("float32").reshape(size, size)
        for _ in range(n_steps):
            g = g.step(RULE)
        y = g.cells.astype("float32").reshape(size, size)
        xs.append(x)
        ys.append(y)
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


_NEIGHBOR_TABLE_CACHE: dict[int, np.ndarray] = {}


def _neighbor_table(size: int) -> np.ndarray:
    if size not in _NEIGHBOR_TABLE_CACHE:
        g = Grid(size, size)
        _NEIGHBOR_TABLE_CACHE[size] = np.array([g.neighbor_indices(i) for i in range(size * size)])
    return _NEIGHBOR_TABLE_CACHE[size]


def random_mlp_batch(rng: np.random.Generator, batch: int, size: int, density: float, n_steps: int):
    """Same boards as random_batch, but unrolled into per-cell [9] rows for
    the MLP, using a cached neighbour-index table instead of a Python loop
    per cell per board."""
    nb_table = _neighbor_table(size)
    xs, ys = [], []
    for _ in range(batch):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        flat = g.cells.astype("float32")
        truth = g
        for _ in range(n_steps):
            truth = truth.step(RULE)
        nb = flat[nb_table]
        xs.append(np.concatenate([nb, flat[:, None]], axis=1))
        ys.append(truth.cells.astype("float32"))
    x = torch.tensor(np.concatenate(xs, axis=0))
    y = torch.tensor(np.concatenate(ys, axis=0))
    return x, y


def all_512_grid(n_steps: int = 1):
    """All 512 3x3 neighbourhoods, each embedded as the centre cell of a
    padded grid the CNN can run on directly (circular padding on a >=3x3
    grid does not reach past the immediate neighbours for a single 3x3 conv
    at the centre, so an odd-sized small torus reproduces the exact 512
    lookup table for n=1)."""
    cases = []
    for k in range(512):
        nb = [(k >> b) & 1 for b in range(8)]  # NW,N,NE,W,E,SW,S,SE
        self_state = (k >> 8) & 1
        cases.append((nb, self_state))
    return cases


_ORDER = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _all_512_grids_and_truth():
    """All 512 3x3-neighbourhood cases as one [512, 3, 3] float32 array plus
    the rule's truth for the centre cell, built once (cached at module load
    via functools-free memoisation through a mutable default)."""
    grids = np.zeros((512, 3, 3), dtype="float32")
    truth = np.zeros(512, dtype="int64")
    for k in range(512):
        nb = [(k >> b) & 1 for b in range(8)]
        self_state = (k >> 8) & 1
        grids[k, 1, 1] = self_state
        for (dy, dx), v in zip(_ORDER, nb):
            grids[k, (1 + dy) % 3, (1 + dx) % 3] = v
        n = sum(nb)
        truth[k] = 1 if RULE.next(self_state != 0, n) else 0
    return grids, truth


_ALL_512_GRIDS, _ALL_512_TRUTH = _all_512_grids_and_truth()


def cnn_512_eval(model) -> float:
    """Exact match on all 512 3x3 neighbourhoods (n=1 only: a single step
    depends only on the 3x3 neighbourhood, so this table fully specifies the
    rule), one batched forward pass. Not meaningful for n=2 (a 2-step
    transition depends on the 5x5 neighbourhood, 2**25 cases); n=2
    conditions are scored on fresh-board cell accuracy only."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(_ALL_512_GRIDS)
        logits = model(x)
        pred = (torch.sigmoid(logits[:, 1, 1]) >= 0.5).long().numpy()
    return float((pred == _ALL_512_TRUTH).mean())


def mlp_512_eval(model) -> float:
    model.eval()
    nb_bits = ((np.arange(512)[:, None] >> np.arange(9)[None, :]) & 1).astype("float32")
    with torch.no_grad():
        x = torch.tensor(nb_bits)
        pred = (torch.sigmoid(model(x)) >= 0.5).long().numpy()
    return float((pred == _ALL_512_TRUTH).mean())


def fresh_boards_eval(model, rng: np.random.Generator, n_boards: int, size: int, density: float, n_steps: int, is_mlp: bool):
    model.eval()
    xs, ys = [], []
    for _ in range(n_boards):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        truth = g
        for _ in range(n_steps):
            truth = truth.step(RULE)
        xs.append(g.cells.astype("float32").reshape(size, size))
        ys.append(truth.cells.astype("int64").reshape(-1))
    want = np.stack(ys)
    with torch.no_grad():
        if is_mlp:
            neighbor_table = _neighbor_table(size)
            cases = []
            for x_grid in xs:
                flat = x_grid.reshape(-1)
                nb = flat[neighbor_table]
                cases.append(np.concatenate([nb, flat[:, None]], axis=1))
            x = torch.tensor(np.stack(cases).astype("float32")).reshape(-1, 9)
            pred = (torch.sigmoid(model(x)) >= 0.5).long().numpy().reshape(n_boards, -1)
        else:
            x = torch.tensor(np.stack(xs))
            logits = model(x)
            pred = (torch.sigmoid(logits) >= 0.5).long().numpy().reshape(n_boards, -1)
    correct_cells = int((pred == want).sum())
    total_cells = want.size
    all_exact = bool(np.all(np.all(pred == want, axis=1)))
    return correct_cells / total_cells, all_exact


# --------------------------------------------------------------- training --

def train_one(make_model, is_mlp: bool, seed: int, steps: int, density: float,
              board_size: int, batch: int, n_steps: int, lr: float, log_every: int = 25):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    converged_at = None
    best_cells_correct = 0.0
    for step in range(1, steps + 1):
        model.train()
        if is_mlp:
            x, y = random_mlp_batch(rng, batch, board_size, density, n_steps)
            logits = model(x)
        else:
            x, y = random_batch(rng, batch, board_size, density, n_steps)
            logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            if n_steps == 1:
                table_acc = mlp_512_eval(model) if is_mlp else cnn_512_eval(model)
            else:
                table_acc = None  # not defined for n>1; see cnn_512_eval docstring
            eval_rng = np.random.default_rng(seed * 1_000_003 + step)
            cells_acc, boards_exact_all = fresh_boards_eval(
                model, eval_rng, 100, board_size, density, n_steps, is_mlp
            )
            best_cells_correct = max(best_cells_correct, cells_acc)
            table_ok = table_acc is None or table_acc >= 1.0
            if table_ok and cells_acc >= 1.0:
                converged_at = step
                break
    return converged_at, best_cells_correct, num_params(model)


def run_condition(name, make_model, is_mlp, seeds, steps, density, board_size, batch, n_steps, lr):
    results = []
    t0 = time.time()
    for seed in seeds:
        converged_at, best_cells, params = train_one(
            make_model, is_mlp, seed, steps, density, board_size, batch, n_steps, lr
        )
        results.append({"seed": seed, "converged_at": converged_at, "best_cells_correct": best_cells})
    dt = time.time() - t0
    n_conv = sum(1 for r in results if r["converged_at"] is not None)
    conv_steps = [r["converged_at"] for r in results if r["converged_at"] is not None]
    median_steps = float(np.median(conv_steps)) if conv_steps else None
    best_non_conv = max((r["best_cells_correct"] for r in results if r["converged_at"] is None), default=None)
    return {
        "name": name,
        "params": params,
        "density": density,
        "n_steps": n_steps,
        "converged": n_conv,
        "seeds": len(seeds),
        "median_steps_to_converge": median_steps,
        "best_cells_correct_non_converged": best_non_conv,
        "wall_seconds": dt,
        "per_seed": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--board-size", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n2-steps", type=int, default=None, help="steps budget for n=2 conditions (default: same as --steps)")
    ap.add_argument("--skip-n2", action="store_true")
    args = ap.parse_args()
    n2_steps = args.n2_steps if args.n2_steps is not None else args.steps

    seeds = list(range(args.seeds))
    default_density = 0.375
    conditions = []

    # 1. Minimal CNN (c=2) + width sweep, n=1, at default density
    widths = {"minimal (c=2, x1)": 2, "x2 (c=4)": 4, "x4 (c=8)": 8, "x8 (c=16)": 16}
    for label, c in widths.items():
        conditions.append(dict(
            name=f"CNN {label}",
            make_model=(lambda c=c: MinimalCNN(c)),
            is_mlp=False, density=default_density, n_steps=1,
        ))

    # density sweep, minimal size only
    for d0 in (0.2, 0.38, 0.5):
        conditions.append(dict(
            name=f"CNN minimal (c=2, x1) d0={d0}",
            make_model=(lambda: MinimalCNN(2)),
            is_mlp=False, density=d0, n_steps=1,
        ))

    # 2. Tiny MLP on 9 values, smallest hidden width, for comparison with
    # the repo's 1,442-param Mlp2OfLife (hidden=32, 2 layers, 2-way softmax)
    for hidden in (2, 4, 8):
        conditions.append(dict(
            name=f"MLP-9 hidden={hidden}",
            make_model=(lambda hidden=hidden: TinyMLP(hidden)),
            is_mlp=True, density=default_density, n_steps=1,
        ))

    # 3. n=2 in one pass, if time remains: smallest CNN width that reliably
    # learned n=1 (picked after the loop below at report time we just try c=2
    # then double/quadruple), depth-2 stack
    if not args.skip_n2:
        for label, c in {"x1 (c=2)": 2, "x2 (c=4)": 4, "x4 (c=8)": 8}.items():
            conditions.append(dict(
                name=f"n=2 DeepCNN {label}",
                make_model=(lambda c=c: DeepCNN(c, depth=2)),
                is_mlp=False, density=default_density, n_steps=2,
                steps=n2_steps,
            ))

    report = []
    for cond in conditions:
        steps = cond.get("steps", args.steps)
        r = run_condition(
            cond["name"], cond["make_model"], cond["is_mlp"], seeds,
            steps, cond["density"], args.board_size, args.batch,
            cond["n_steps"], args.lr,
        )
        print(json.dumps({k: v for k, v in r.items() if k != "per_seed"}), flush=True)
        report.append(r)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
