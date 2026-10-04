"""QAT for the generic bit-slice-compiler target: binary inputs, ternary
weights, binary activations between layers, integer-foldable thresholds.

Differs from minimal_life_1bit.py (which this branch forks from) in one
place: the hidden-layer activation. minimal_life_1bit.py uses float ReLU
between the quantized conv layers, so only the weights are low-bit; the
compiler needs *every* intermediate value to be a single bit too, so that
each layer is exactly "integer sum of selected binary inputs, then integer
threshold" and the whole forward pass compiles to bitwise ops. So here the
hidden activation is a hard step function (Heaviside at 0) trained with a
straight-through estimator: forward value is the step, backward gradient is
passed through a hardtanh-shaped window (gradient 1 where the pre-activation
is within +-1 of the threshold, 0 outside), the standard recipe from
Binarized Neural Network literature (same family as BitNet b1.58,
arXiv:2402.17764, and XNOR-Net, arXiv:1603.05279, use for weights; applying
it to activations too is what "Differentiable Logic Cellular Automata",
arXiv:2506.04912, and Petersen et al.'s differentiable logic gate networks
(NeurIPS 2022) do for their own thresholded/logic cells).

Weight quantization is the same absmean ternary scheme as
minimal_life_1bit.py (BitNet b1.58 style): gamma = mean(|W|), Wq =
clip(round(W/gamma), -1, 1). The per-layer bias stays full precision during
training; after training it is folded with the scale into a single integer
threshold per output unit (see the bit-slice compiler), so nothing but
ternary weights and integer thresholds survive into the compiled kernel.

Rules: Conway's Life (B3/S23) and HighLife (B36/S23), both via the ported
`life.Rule`/`life.Grid` (life.py), matching the Rust ground truth bit for
bit.
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


def quantize_ternary(w: torch.Tensor):
    scale = w.detach().abs().mean().clamp(min=1e-8)
    wq = torch.clamp(torch.round(w / scale), -1, 1)
    w_ste = w + (wq - w).detach()
    return w_ste, scale


class StepSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y):
        ctx.save_for_backward(y)
        return (y >= 0).to(y.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        # Soft surrogate gradient (derivative of y / (1 + |y|)): unlike a
        # hard box window this never fully zeroes out, which keeps units
        # that are currently wrong on one side of the threshold from
        # getting permanently stuck -- the "slope annealing" style fix for
        # straight-through estimator training on hard step activations.
        surrogate = 1.0 / (1.0 + y.abs()) ** 2
        return grad_out * surrogate


step_ste = StepSTE.apply


class QuantThreshold(nn.Module):
    """y = scale * (x @ Wq^T) + bias, Wq ternary, x binary -> integer sum."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        wq, scale = quantize_ternary(self.weight)
        return scale * F.linear(x, wq) + self.bias

    def quantized(self):
        with torch.no_grad():
            wq, scale = quantize_ternary(self.weight)
            return (
                wq.detach().cpu().numpy().astype("int64"),
                float(scale),
                self.bias.detach().cpu().numpy().astype("float64"),
            )


class ThresholdNet(nn.Module):
    """in_features -> hidden[0] -> ... -> hidden[-1] -> 1, every inter-layer
    activation a binary step; the final layer's raw value is the logit used
    for the training loss (sigmoid(logit) >= 0.5 at eval, same as
    step(logit >= 0))."""

    def __init__(self, in_features: int, hidden: list[int]):
        super().__init__()
        sizes = [in_features] + hidden + [1]
        self.layers = nn.ModuleList(
            QuantThreshold(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = step_ste(layer(h))
        return self.layers[-1](h).squeeze(-1)

    def quantized_layers(self):
        return [layer.quantized() for layer in self.layers]


# ------------------------------------------------------------- windowing --


def window_offsets(w: int):
    r = (w - 1) // 2
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


_NEIGHBOR_CACHE: dict[tuple[int, int], np.ndarray] = {}


def window_table(size: int, w: int) -> np.ndarray:
    key = (size, w)
    if key not in _NEIGHBOR_CACHE:
        offsets = window_offsets(w)
        idx = np.empty((size * size, len(offsets)), dtype=np.int64)
        for i in range(size * size):
            x, y = i % size, i // size
            for k, (dy, dx) in enumerate(offsets):
                idx[i, k] = ((y + dy) % size) * size + ((x + dx) % size)
        _NEIGHBOR_CACHE[key] = idx
    return _NEIGHBOR_CACHE[key]


def apply_steps(g: Grid, rule: Rule, steps: int) -> Grid:
    for _ in range(steps):
        g = g.step(rule)
    return g


def make_batch(rng: np.random.Generator, batch: int, size: int, density: float,
                window: int, steps: int, rule: Rule):
    table = window_table(size, window)
    xs, ys = [], []
    for _ in range(batch):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        flat = g.cells.astype("float32")
        truth = apply_steps(g, rule, steps).cells.astype("float32")
        xs.append(flat[table])
        ys.append(truth)
    x = torch.tensor(np.concatenate(xs, axis=0))
    y = torch.tensor(np.concatenate(ys, axis=0))
    return x, y


def all_grids_bits_truth(window: int, steps: int, rule: Rule):
    """Exhaustive enumeration, only tractable for window=3 (2**9 = 512)."""
    n = window * window
    assert n <= 10, "exhaustive enumeration only for small windows"
    total = 1 << n
    bits = np.zeros((total, n), dtype="float32")
    truth = np.zeros(total, dtype="int64")
    r = (window - 1) // 2
    size = window + 2  # pad so the window never wraps during enumeration
    for k in range(total):
        vals = [(k >> b) & 1 for b in range(n)]
        bits[k] = vals
        g = Grid(size, size)
        for idx, (dy, dx) in enumerate(window_offsets(window)):
            g.set(r + r + dx, r + r + dy, vals[idx])
        out = apply_steps(g, rule, steps)
        truth[k] = out.get(r + r, r + r)
    return bits, truth


def eval_fresh(model, rng, n_boards, size, density, window, steps, rule, device):
    model.eval()
    table = window_table(size, window)
    xs, ys = [], []
    for _ in range(n_boards):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        flat = g.cells.astype("float32")
        truth = apply_steps(g, rule, steps).cells.astype("int64")
        xs.append(flat[table])
        ys.append(truth)
    x = torch.tensor(np.concatenate(xs, axis=0), device=device)
    want = np.concatenate(ys, axis=0)
    with torch.no_grad():
        pred = (torch.sigmoid(model(x)) >= 0.5).long().cpu().numpy()
    return float((pred == want).mean())


def train_one(window, steps, rule, hidden, seed, train_steps, density, board_size, batch,
              lr, device, log_every, plateau_window, plateau_eps, exhaustive):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ThresholdNet(window * window, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if exhaustive:
        ex_bits, ex_truth = all_grids_bits_truth(window, steps, rule)
        ex_bits_t = torch.tensor(ex_bits, device=device)

    converged_at = None
    best = 0.0
    history = []
    last_step = 0
    for step in range(1, train_steps + 1):
        last_step = step
        model.train()
        x, y = make_batch(rng, batch, board_size, density, window, steps, rule)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == train_steps:
            if exhaustive:
                model.eval()
                with torch.no_grad():
                    pred = (torch.sigmoid(model(ex_bits_t)) >= 0.5).long().cpu().numpy()
                table_acc = float((pred == ex_truth).mean())
            else:
                table_acc = 1.0
            eval_rng = np.random.default_rng(seed * 1_000_003 + step)
            cells_acc = eval_fresh(model, eval_rng, 50, board_size, density, window, steps,
                                    rule, device)
            score = min(table_acc, cells_acc)
            best = max(best, score)
            history.append(score)
            if table_acc >= 1.0 and cells_acc >= 1.0:
                converged_at = step
                break
            if len(history) > plateau_window and (best - history[-plateau_window - 1]) < plateau_eps:
                break
    return model, {"converged_at": converged_at, "best": best, "steps_run": last_step}


def dump_model(model: ThresholdNet) -> dict:
    layers = []
    for wq, scale, bias in model.quantized_layers():
        layers.append({"weight": wq.tolist(), "scale": scale, "bias": bias.tolist()})
    return {"layers": layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--steps-rule", type=int, default=1, help="CA generations per forward pass")
    ap.add_argument("--rule", default="life", choices=["life", "highlife"])
    ap.add_argument("--hidden", default="4")
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--board-size", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--density", type=float, default=0.38)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--plateau-window", type=int, default=15)
    ap.add_argument("--plateau-eps", type=float, default=0.002)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = [int(h) for h in args.hidden.split(",") if h]
    rule = Rule.life() if args.rule == "life" else Rule({3, 6}, {2, 3})
    exhaustive = (args.window * args.window <= 10) and args.steps_rule == 1

    best_model, best_info = None, None
    t0 = time.time()
    for seed in range(args.seeds):
        model, info = train_one(args.window, args.steps_rule, rule, hidden, seed,
                                 args.train_steps, args.density, args.board_size, args.batch,
                                 args.lr, device, args.log_every, args.plateau_window,
                                 args.plateau_eps, exhaustive)
        print(json.dumps({"seed": seed, **info}), flush=True)
        if info["converged_at"] is not None:
            best_model, best_info = model, info
            break
        if best_info is None or info["best"] > best_info["best"]:
            best_model, best_info = model, info

    out = {
        "window": args.window,
        "steps_rule": args.steps_rule,
        "rule": args.rule,
        "hidden": hidden,
        "info": best_info,
        "wall_seconds": time.time() - t0,
        "model": dump_model(best_model),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
