"""QAT for a single network that predicts Game of Life two generations
ahead in one forward pass, architecture made compilable by the generic
bit-slice compiler: binary input, ternary weights
(BitNet b1.58 absmean, same scheme as bitslice_life.py), binary
inter-layer activations via the same soft-surrogate straight-through
estimator that fixed convergence in bitslice_life.py (1/(1+|y|)^2, not the
hard box window -- see docs/runs/2026-10-02-bitslice-compiler.md's
Observations).

Architecture (`BitsliceDeepCNN`): Conv3x3(1->c, circular, ternary) -> step
-> Conv3x3(c->c, circular, ternary) -> step -> Conv1x1(c->1, ternary) ->
step. Exactly the 5x5 receptive field a 2-step Life prediction needs (no
more, no less) -- copied from `minimal_life_full.py`'s `DeepCNN` training
recipe (data generation, pattern-mixed boards, plateau stopping) but with
`DeepCNN`'s depth fixed at 2 conv layers + 1x1 readout (not depth=n+1) and
float ReLU replaced by ternary weights + binary step activations, so the
whole thing compiles.

Training setup copied from `minimal_life_full.py`'s `DeepCNN n=2` condition
(docs/runs/2026-09-30-minimal-life.md, section "Results, n = 2"): same
`make_batch_mixed`/`fresh_boards_eval` (random board + vec_step applied
twice, pattern_frac=0.25 by default -- that run found pattern-mixing sped
convergence, 1500 vs 2000 median steps, at the same float architecture),
same Adam optimizer, same plateau-based stopping rule. What's new here is
only the layer internals (ternary + step instead of float + ReLU) and the
fixed depth (2 conv layers, not n+1).
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
from minimal_life_nstep import vec_step, verify_vec_step, fresh_boards_eval
from minimal_life_full import make_batch_mixed
from minimal_life_patterns import build_pattern_dataset, eval_patterns

RULE = Rule.life()


def quantize_ternary(w: torch.Tensor):
    scale = w.detach().abs().mean().clamp(min=1e-8)
    wq = torch.clamp(torch.round(w / scale), -1, 1)
    w_ste = w + (wq - w).detach()
    return w_ste, scale


def quantize_binary(w: torch.Tensor):
    """XNOR-Net style (arXiv:1603.05279): wq = sign(w) in {-1, +1} (never
    0), scale = mean(|w|). Step 5's optional binary-weight variant: the
    bit-slice compiler's split into a positive/negative set needs no
    change, a weight of 0 simply never occurs."""
    scale = w.detach().abs().mean().clamp(min=1e-8)
    wq = torch.sign(w)
    wq = torch.where(wq == 0, torch.ones_like(wq), wq)
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
        surrogate = 1.0 / (1.0 + y.abs()) ** 2
        return grad_out * surrogate


step_ste = StepSTE.apply


class QuantConv2d(nn.Module):
    """y = scale * circular_conv2d(x, Wq) + bias, Wq ternary, x binary ->
    per-output-location integer sum (the bit-slice compiler's
    compile_conv_layer assumes exactly this: one scalar scale per layer,
    full-precision bias folded into an integer threshold per output
    channel after training)."""

    def __init__(self, in_c: int, out_c: int, k: int, quant: str = "ternary"):
        super().__init__()
        self.k = k
        self.pad = (k - 1) // 2
        self.quant_fn = quantize_binary if quant == "binary" else quantize_ternary
        self.weight = nn.Parameter(torch.empty(out_c, in_c, k, k))
        self.bias = nn.Parameter(torch.zeros(out_c))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        wq, scale = self.quant_fn(self.weight)
        if self.pad > 0:
            x = F.pad(x, (self.pad,) * 4, mode="circular")
        y = F.conv2d(x, wq)
        return scale * y + self.bias.view(1, -1, 1, 1)

    def quantized(self):
        with torch.no_grad():
            wq, scale = self.quant_fn(self.weight)
            return (
                wq.detach().cpu().numpy().astype("int64"),
                float(scale),
                self.bias.detach().cpu().numpy().astype("float64"),
            )


class BitsliceDeepCNN(nn.Module):
    def __init__(self, channels: int, quant: str = "ternary"):
        super().__init__()
        self.conv1 = QuantConv2d(1, channels, 3, quant)
        self.conv2 = QuantConv2d(channels, channels, 3, quant)
        self.readout = QuantConv2d(channels, 1, 1, quant)

    def forward(self, x):
        h = x.unsqueeze(1)
        h = step_ste(self.conv1(h))
        h = step_ste(self.conv2(h))
        return self.readout(h).squeeze(1).squeeze(1)

    def dump(self) -> dict:
        def layer_dict(conv):
            wq, scale, bias = conv.quantized()
            return {"weight": wq.tolist(), "scale": scale, "bias": bias.tolist()}
        return {
            "conv1": layer_dict(self.conv1),
            "conv2": layer_dict(self.conv2),
            "readout": layer_dict(self.readout),
        }


def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def train_one(channels, seed, steps, density, board_size, batch, lr, device, log_every,
              plateau_window, plateau_eps, pattern_frac, quant="ternary"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BitsliceDeepCNN(channels, quant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    converged_at = None
    stopped_early = None
    history = []
    best = 0.0
    last_step = 0
    t0 = time.time()
    for step in range(1, steps + 1):
        last_step = step
        model.train()
        x, y = make_batch_mixed(rng, batch, board_size, density, 2, device, pattern_frac)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            eval_rng = np.random.default_rng(seed * 1_000_003 + step)
            cells_acc, baseline = fresh_boards_eval(model, eval_rng, 100, board_size, density, 2, device)
            best = max(best, cells_acc)
            history.append(cells_acc)
            print(json.dumps({"seed": seed, "step": step, "cells_acc": cells_acc,
                               "baseline": baseline, "elapsed": time.time() - t0}), flush=True)
            if cells_acc >= 1.0:
                converged_at = step
                break
            if len(history) > plateau_window and (best - history[-plateau_window - 1]) < plateau_eps:
                stopped_early = "plateau"
                break
    model.eval()
    final_rng = np.random.default_rng(seed * 7 + 1)
    final_cells_acc, final_baseline = fresh_boards_eval(model, final_rng, 200, board_size, density, 2, device)
    return model, {
        "converged_at": converged_at,
        "best_cells_correct": best,
        "final_cells_correct": final_cells_acc,
        "final_baseline": final_baseline,
        "stopped_early": stopped_early,
        "steps_run": last_step,
        "params": num_params(model),
        "wall_seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--board-size", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--density", type=float, default=0.38)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--plateau-window", type=int, default=20)
    ap.add_argument("--plateau-eps", type=float, default=0.002)
    ap.add_argument("--pattern-frac", type=float, default=0.25)
    ap.add_argument("--pattern-eval-size", type=int, default=32)
    ap.add_argument("--pattern-eval-positions", type=int, default=4)
    ap.add_argument("--quant", default="ternary", choices=["ternary", "binary"])
    args = ap.parse_args()

    device = "cpu"
    assert verify_vec_step(), "vec_step disagrees with life.py Grid.step; refusing to run"
    seeds = [int(s) for s in args.seeds.split(",") if s]
    pattern_eval_ds = build_pattern_dataset(args.pattern_eval_size, args.pattern_eval_positions, seed=12345)

    results = []
    t0 = time.time()
    for seed in seeds:
        model, info = train_one(args.channels, seed, args.train_steps, args.density, args.board_size,
                                 args.batch, args.lr, device, args.log_every, args.plateau_window,
                                 args.plateau_eps, args.pattern_frac, args.quant)
        pattern_results = eval_patterns(model, pattern_eval_ds, 2, device)
        info["patterns"] = {k: v for k, v in pattern_results.items()}
        print(json.dumps({"seed": seed, "summary": {k: v for k, v in info.items() if k != "patterns"}}),
              flush=True)
        results.append({"seed": seed, **info, "model": model.dump() if info["converged_at"] else None})

    out = {
        "channels": args.channels,
        "seeds": seeds,
        "n_converged": sum(1 for r in results if r["converged_at"] is not None),
        "wall_seconds": time.time() - t0,
        "per_seed": [{k: v for k, v in r.items() if k != "model"} for r in results],
        "models": {r["seed"]: r["model"] for r in results if r["model"] is not None},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
