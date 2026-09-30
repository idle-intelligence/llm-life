"""Follow-up to minimal_life.py: how small a network learns n>1 steps of
Game of Life, for n in {2, 3, 10}, asked after seeing the n=1/n=2 sweep in
docs/runs/2026-09-30-minimal-life.md.

Two model families, both evaluated with the same exactness criterion as
minimal_life.py (100% cells correct on 100 fresh random 16x16 boards after
n steps; the 512-neighbourhood table check does not apply for n>1, see
minimal_life.py's cnn_512_eval docstring):

A. Untied one-pass deep CNN (DeepCNN, imported from minimal_life.py): L =
   n+1 stacked Conv3x3(circular)+ReLU layers, then a Conv1x1 readout, one
   forward pass predicts the n-step-ahead state directly. Widths c in
   {8, 32, 64}.

B. Weight-tied recurrent: the n=1 minimal-reliable CNN (MinimalCNN,
   channels=8, 89 params, the smallest n=1 width that converged 10/10 in
   the previous sweep) applied n times with shared weights, trained
   end-to-end on the n-step target only (no 1-step supervision anywhere).
   Intermediate states are pushed through a straight-through estimator
   (STE): the forward value at each intermediate step is the hard
   threshold of the model's sigmoid output (so the recurrence really is
   the binary board-to-board dynamics Game of Life needs, not a
   continuous relaxation of it), and the backward pass uses the sigmoid's
   own gradient (`hard + (probs - probs.detach())`), so gradients reach
   every one of the n applications despite the threshold. Eval uses the
   same forward function (model.eval() gives numerically the same hard
   values; the STE trick only changes what .backward() sees). After
   training, the tied core is checked against the exact rule on all 512
   3x3 neighbourhoods (`minimal_life.cnn_512_eval`) to see whether it
   discovered the Game of Life update rule itself, not just some n-step
   composite it never has to apply out of context.

Data generation: same Grid.random (xorshift64, ported bit-for-bit from
crates/life) for the initial board as minimal_life.py / life.py. Applying
the rule n times to build the n-step target is done with a vectorized
numpy stepper (`vec_step`, toroidal neighbour sum via np.roll) rather than
life.py's pure-Python per-cell Grid.step loop, because generating n-step
targets for a whole training batch, n times, for n up to 10, is the
bottleneck once models are batched on GPU. `verify_vec_step` below checks
`vec_step` bit-for-bit against `Grid.step` before any run trusts it.

Early stopping for hopeless conditions: if, after --hopeless-after-steps
steps, best-cells-correct-so-far is not at least --hopeless-margin above
the trivial baseline (predict every cell dead, the majority class once a
board has evolved for a few generations), the condition is stopped and
marked `"stopped_early": "hopeless"` rather than spending the rest of its
step budget. This is what "if n=10 untied is clearly hopeless early, stop
it" is implemented as, applied uniformly to every condition (cheap
insurance for the whole sweep's wall-clock budget, not just n=10).
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
from minimal_life import DeepCNN, MinimalCNN, num_params, cnn_512_eval

RULE = Rule.life()


def vec_step(boards: np.ndarray) -> np.ndarray:
    """Vectorized B3/S23 update, toroidal boundary, batched over the first
    axis: boards is [B, H, W] of 0/1. Same rule as life.py's Rule.life()
    applied cell-by-cell via Grid.step; verify_vec_step checks the two
    agree bit-for-bit."""
    b = boards.astype(np.int16)
    n = np.zeros_like(b)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            n += np.roll(np.roll(b, dy, axis=1), dx, axis=2)
    alive = b != 0
    born = (~alive) & (n == 3)
    survive = alive & ((n == 2) | (n == 3))
    return (born | survive).astype(boards.dtype)


def verify_vec_step(trials: int = 20, size: int = 16, seed0: int = 999) -> bool:
    for i in range(trials):
        g = Grid.random(size, size, seed0 + i, 0.38)
        want = g.step(RULE).cells.reshape(size, size)
        got = vec_step(g.cells.reshape(1, size, size))[0]
        if not np.array_equal(want, got):
            return False
    return True


class TiedRecurrentCNN(nn.Module):
    """MinimalCNN(channels) applied n_steps times, shared weights, trained
    end-to-end on the n-step target only. See module docstring for the STE."""

    def __init__(self, channels: int, n_steps: int):
        super().__init__()
        self.core = MinimalCNN(channels)
        self.n_steps = n_steps

    def forward(self, x):
        logits = None
        for i in range(self.n_steps):
            logits = self.core(x)
            if i < self.n_steps - 1:
                probs = torch.sigmoid(logits)
                hard = (probs >= 0.5).float()
                x = hard + (probs - probs.detach())
        return logits


def make_batch(rng: np.random.Generator, batch: int, size: int, density: float, n_steps: int, device):
    xs = np.empty((batch, size, size), dtype=np.float32)
    for i in range(batch):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        xs[i] = g.cells.reshape(size, size).astype(np.float32)
    ys = xs.astype(np.uint8)
    for _ in range(n_steps):
        ys = vec_step(ys)
    x = torch.tensor(xs, device=device)
    y = torch.tensor(ys.astype(np.float32), device=device)
    return x, y


def fresh_boards_eval(model, rng: np.random.Generator, n_boards: int, size: int, density: float, n_steps: int, device):
    model.eval()
    xs = np.empty((n_boards, size, size), dtype=np.float32)
    for i in range(n_boards):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        xs[i] = g.cells.reshape(size, size).astype(np.float32)
    ys = xs.astype(np.uint8)
    for _ in range(n_steps):
        ys = vec_step(ys)
    want = ys.reshape(n_boards, -1).astype(np.int64)
    with torch.no_grad():
        x = torch.tensor(xs, device=device)
        logits = model(x)
        pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().reshape(n_boards, -1)
    correct = int((pred == want).sum())
    total = want.size
    baseline = float((want == 0).mean())  # trivial "predict all dead" accuracy
    return correct / total, baseline


def train_one(make_model, seed: int, steps: int, density: float, board_size: int,
              batch: int, n_steps: int, lr: float, device, log_every: int,
              hopeless_after: int, hopeless_margin: float, deadline: float | None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    converged_at = None
    best_cells_correct = 0.0
    stopped_early = None
    last_step = 0
    for step in range(1, steps + 1):
        last_step = step
        model.train()
        x, y = make_batch(rng, batch, board_size, density, n_steps, device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            eval_rng = np.random.default_rng(seed * 1_000_003 + step)
            cells_acc, baseline = fresh_boards_eval(model, eval_rng, 100, board_size, density, n_steps, device)
            best_cells_correct = max(best_cells_correct, cells_acc)
            if cells_acc >= 1.0:
                converged_at = step
                break
            if step >= hopeless_after and (best_cells_correct - baseline) < hopeless_margin:
                stopped_early = "hopeless"
                break
            if deadline is not None and time.time() >= deadline:
                stopped_early = "deadline"
                break
    return {
        "converged_at": converged_at,
        "best_cells_correct": best_cells_correct,
        "stopped_early": stopped_early,
        "steps_run": last_step,
        "params": num_params(model),
    }, model


def run_condition(name, make_model, seeds, steps, density, board_size, batch, n_steps, lr,
                   device, log_every, hopeless_after, hopeless_margin, deadline,
                   check_rule_discovery=False):
    results = []
    t0 = time.time()
    params = None
    rule_discovery = []
    for seed in seeds:
        if deadline is not None and time.time() >= deadline:
            break
        r, model = train_one(make_model, seed, steps, density, board_size, batch, n_steps, lr,
                              device, log_every, hopeless_after, hopeless_margin, deadline)
        params = r["params"]
        results.append({"seed": seed, **{k: v for k, v in r.items() if k != "params"}})
        if check_rule_discovery:
            acc512 = cnn_512_eval(model.core.to("cpu"))
            rule_discovery.append(acc512)
    dt = time.time() - t0
    n_conv = sum(1 for r in results if r["converged_at"] is not None)
    conv_steps = [r["converged_at"] for r in results if r["converged_at"] is not None]
    median_steps = float(np.median(conv_steps)) if conv_steps else None
    best_non_conv = max((r["best_cells_correct"] for r in results if r["converged_at"] is None), default=None)
    n_hopeless = sum(1 for r in results if r["stopped_early"] == "hopeless")
    out = {
        "name": name,
        "params": params,
        "n_steps": n_steps,
        "density": density,
        "converged": n_conv,
        "seeds": len(seeds),
        "seeds_run": len(results),
        "median_steps_to_converge": median_steps,
        "best_cells_correct_non_converged": best_non_conv,
        "stopped_early_hopeless": n_hopeless,
        "wall_seconds": dt,
        "per_seed": results,
    }
    if check_rule_discovery:
        out["rule_discovery_512_acc_per_seed"] = rule_discovery
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--board-size", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--density", type=float, default=0.38)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--hopeless-after", type=int, default=3000)
    ap.add_argument("--hopeless-margin", type=float, default=0.03)
    ap.add_argument("--ns", default="2,3,10")
    ap.add_argument("--widths", default="8,32,64")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--max-wall-seconds", type=float, default=None,
                     help="global deadline (from process start); once passed, no new seed "
                          "starts and any seed mid-training stops at its next eval")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert verify_vec_step(), "vec_step disagrees with life.py Grid.step; refusing to run"

    ns = [int(x) for x in args.ns.split(",")]
    widths = [int(x) for x in args.widths.split(",")]
    seeds = list(range(args.seeds))
    deadline = (time.time() + args.max_wall_seconds) if args.max_wall_seconds else None

    report = []

    def run_and_save(cond, **kw):
        if deadline is not None and time.time() >= deadline:
            print(json.dumps({"name": cond, "skipped": "deadline"}), flush=True)
            report.append({"name": cond, "skipped": "deadline"})
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)
            return
        r = run_condition(cond, seeds=seeds, steps=args.steps, density=args.density,
                           board_size=args.board_size, batch=args.batch, lr=args.lr,
                           device=device, log_every=args.log_every,
                           hopeless_after=args.hopeless_after, hopeless_margin=args.hopeless_margin,
                           deadline=deadline, **kw)
        print(json.dumps({k: v for k, v in r.items() if k not in ("per_seed",)}), flush=True)
        report.append(r)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    if not args.skip_a:
        for n in ns:
            for c in widths:
                name = f"A untied DeepCNN n={n} c={c}"
                run_and_save(name, make_model=(lambda c=c, n=n: DeepCNN(c, depth=n + 1)), n_steps=n)

    if not args.skip_b:
        for n in ns:
            name = f"B tied MinimalCNN(c=8) n={n}"
            run_and_save(name, make_model=(lambda n=n: TiedRecurrentCNN(8, n)), n_steps=n,
                         check_rule_discovery=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
