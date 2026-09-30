"""Comprehensive follow-up, budget lifted: for n in {2, 3, 10}, the full
untied-width sweep (c in {8, 32, 64}, plus a requested wider c=128 for n in
{3, 10}), the weight-tied recurrent model (MinimalCNN c=8 applied n times,
see minimal_life_nstep.py's docstring for the straight-through estimator),
and a structured-pattern training variant (25% of training boards are a
known Game of Life pattern in a random orientation/position instead of
random soup) applied to the best-converging pure-random-trained width for
that n. Every condition, every seed, reports both the random-soup exactness
numbers (with the trivial all-dead baseline) and the structured-pattern
numbers (block/beehive/blinker/toad/beacon/glider/lwss/r_pentomino; see
minimal_life_patterns.py for the LifeWiki-sourced coordinates and the
dihedral-transform placement).

Stopping rule (replaces the flat step-count ceiling from
minimal_life_nstep.py now that there is no tight wall-clock cap): train up
to --steps (default 100,000) steps, evaluated every --log-every steps on
100 fresh boards; stop immediately on exact 100% match, or when the best
cells-correct score has not improved by --plateau-eps over the last
--plateau-window evals ("stop a run early when flat"). A run stuck at the
trivial baseline plateaus immediately by this rule, so no separate
"hopeless" check is needed (minimal_life_nstep.py's was a hard wall-clock
safety valve for a tighter budget that no longer applies).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from life import Grid, Rule
from minimal_life import MinimalCNN, num_params, cnn_512_eval
from minimal_life_nstep import DeepCNN, TiedRecurrentCNN, vec_step, verify_vec_step, fresh_boards_eval
from minimal_life_patterns import PATTERNS, dihedral_variants, build_pattern_dataset, eval_patterns

RULE = Rule.life()

_PATTERN_VARIANTS = {name: dihedral_variants(cells) for name, cells in PATTERNS.items()}
_PATTERN_NAMES = list(_PATTERN_VARIANTS.keys())


def place_pattern_on(cells, size: int, rng: np.random.Generator) -> np.ndarray:
    board = np.zeros((size, size), dtype=np.uint8)
    ox = int(rng.integers(0, size))
    oy = int(rng.integers(0, size))
    for x, y in cells:
        board[(oy + y) % size, (ox + x) % size] = 1
    return board


def make_batch_mixed(rng: np.random.Generator, batch: int, size: int, density: float,
                      n_steps: int, device, pattern_frac: float):
    xs = np.empty((batch, size, size), dtype=np.float32)
    for i in range(batch):
        if pattern_frac > 0 and rng.random() < pattern_frac:
            name = _PATTERN_NAMES[int(rng.integers(0, len(_PATTERN_NAMES)))]
            variants = _PATTERN_VARIANTS[name]
            cells = variants[int(rng.integers(0, len(variants)))]
            xs[i] = place_pattern_on(cells, size, rng).astype(np.float32)
        else:
            seed = int(rng.integers(0, 2**31 - 1))
            g = Grid.random(size, size, seed, density)
            xs[i] = g.cells.reshape(size, size).astype(np.float32)
    ys = xs.astype(np.uint8)
    for _ in range(n_steps):
        ys = vec_step(ys)
    x = torch.tensor(xs, device=device)
    y = torch.tensor(ys.astype(np.float32), device=device)
    return x, y


def train_one(make_model, seed: int, steps: int, density: float, board_size: int, batch: int,
              n_steps: int, lr: float, device, log_every: int, plateau_window: int,
              plateau_eps: float, pattern_frac: float, pattern_eval_ds: dict):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    converged_at = None
    stopped_early = None
    history = []
    best_cells_correct = 0.0
    last_step = 0
    for step in range(1, steps + 1):
        last_step = step
        model.train()
        x, y = make_batch_mixed(rng, batch, board_size, density, n_steps, device, pattern_frac)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            eval_rng = np.random.default_rng(seed * 1_000_003 + step)
            cells_acc, _baseline = fresh_boards_eval(model, eval_rng, 100, board_size, density, n_steps, device)
            best_cells_correct = max(best_cells_correct, cells_acc)
            history.append(cells_acc)
            if cells_acc >= 1.0:
                converged_at = step
                break
            if len(history) > plateau_window and (best_cells_correct - history[-plateau_window - 1]) < plateau_eps:
                stopped_early = "plateau"
                break

    model.eval()
    final_rng = np.random.default_rng(seed * 7 + 1)
    final_cells_correct, final_baseline = fresh_boards_eval(model, final_rng, 200, board_size, density, n_steps, device)
    pattern_results = eval_patterns(model, pattern_eval_ds, n_steps, device)
    return {
        "converged_at": converged_at,
        "best_cells_correct": best_cells_correct,
        "final_cells_correct": final_cells_correct,
        "final_baseline": final_baseline,
        "stopped_early": stopped_early,
        "steps_run": last_step,
        "params": num_params(model),
        "patterns": pattern_results,
    }, model


def run_condition(name, make_model, seeds, steps, density, board_size, batch, n_steps, lr, device,
                   log_every, plateau_window, plateau_eps, pattern_frac, pattern_eval_ds,
                   check_rule_discovery=False):
    results = []
    t0 = time.time()
    params = None
    rule_discovery = []
    for seed in seeds:
        r, model = train_one(make_model, seed, steps, density, board_size, batch, n_steps, lr, device,
                              log_every, plateau_window, plateau_eps, pattern_frac, pattern_eval_ds)
        params = r["params"]
        results.append({"seed": seed, **{k: v for k, v in r.items() if k != "params"}})
        if check_rule_discovery:
            rule_discovery.append(cnn_512_eval(model.core.to("cpu")))
    dt = time.time() - t0
    n_conv = sum(1 for r in results if r["converged_at"] is not None)
    conv_steps = [r["converged_at"] for r in results if r["converged_at"] is not None]
    median_steps = float(np.median(conv_steps)) if conv_steps else None
    best_non_conv = max((r["best_cells_correct"] for r in results if r["converged_at"] is None), default=None)

    pattern_agg = {}
    for pname in pattern_eval_ds:
        exact = sum(r["patterns"][pname]["exact_boards"] for r in results)
        total = sum(r["patterns"][pname]["total_boards"] for r in results)
        masked = float(np.mean([r["patterns"][pname]["masked_cells_correct"] for r in results]))
        pattern_agg[pname] = {"exact_boards": exact, "total_boards": total, "mean_masked_cells_correct": masked}

    out = {
        "name": name,
        "params": params,
        "n_steps": n_steps,
        "density": density,
        "pattern_frac": pattern_frac,
        "converged": n_conv,
        "seeds": len(seeds),
        "median_steps_to_converge": median_steps,
        "best_cells_correct_non_converged": best_non_conv,
        "mean_final_cells_correct": float(np.mean([r["final_cells_correct"] for r in results])),
        "mean_final_baseline": float(np.mean([r["final_baseline"] for r in results])),
        "pattern_agg": pattern_agg,
        "wall_seconds": dt,
        "per_seed": results,
    }
    if check_rule_discovery:
        out["rule_discovery_512_acc_per_seed"] = rule_discovery
    return out


def _cond_score(r):
    med = r["median_steps_to_converge"]
    med_val = med if med is not None else float("inf")
    return (r["converged"], -med_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--board-size", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--density", type=float, default=0.38)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--plateau-window", type=int, default=10)
    ap.add_argument("--plateau-eps", type=float, default=0.005)
    ap.add_argument("--ns", default="2,3,10")
    ap.add_argument("--widths", default="8,32,64")
    ap.add_argument("--wide-width", type=int, default=128)
    ap.add_argument("--wide-ns", default="3,10")
    ap.add_argument("--pattern-frac", type=float, default=0.25)
    ap.add_argument("--pattern-eval-positions", type=int, default=4)
    ap.add_argument("--pattern-eval-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert verify_vec_step(), "vec_step disagrees with life.py Grid.step; refusing to run"

    ns = [int(x) for x in args.ns.split(",")]
    widths = [int(x) for x in args.widths.split(",")]
    wide_ns = set(int(x) for x in args.wide_ns.split(","))
    seeds = list(range(args.seeds))
    pattern_eval_ds = build_pattern_dataset(args.pattern_eval_size, args.pattern_eval_positions, seed=12345)
    print(json.dumps({"pattern_variant_counts": {k: len(v) for k, v in pattern_eval_ds.items()}}), flush=True)

    report = []

    def save():
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    def run_and_save(cond, **kw):
        r = run_condition(cond, seeds=seeds, steps=args.steps, density=args.density,
                           board_size=args.board_size, batch=args.batch, lr=args.lr, device=device,
                           log_every=args.log_every, plateau_window=args.plateau_window,
                           plateau_eps=args.plateau_eps, pattern_eval_ds=pattern_eval_ds, **kw)
        summary = {k: v for k, v in r.items() if k not in ("per_seed", "pattern_agg")}
        summary["pattern_agg_summary"] = {
            k: (v["exact_boards"], v["total_boards"], round(v["mean_masked_cells_correct"], 4))
            for k, v in r["pattern_agg"].items()
        }
        print(json.dumps(summary), flush=True)
        report.append(r)
        save()
        return r

    for n in ns:
        w_list = list(widths) + ([args.wide_width] if n in wide_ns else [])
        cond_results = []
        for c in w_list:
            r = run_and_save(f"A untied DeepCNN n={n} c={c}",
                              make_model=(lambda c=c, n=n: DeepCNN(c, depth=n + 1)),
                              n_steps=n, pattern_frac=0.0)
            cond_results.append((c, r))

        run_and_save(f"B tied MinimalCNN(c=8) n={n}",
                     make_model=(lambda n=n: TiedRecurrentCNN(8, n)),
                     n_steps=n, pattern_frac=0.0, check_rule_discovery=True)

        best_c, _ = max(cond_results, key=lambda t: _cond_score(t[1]))
        run_and_save(f"A untied DeepCNN n={n} c={best_c} +25%patterns",
                     make_model=(lambda c=best_c, n=n: DeepCNN(c, depth=n + 1)),
                     n_steps=n, pattern_frac=args.pattern_frac)

        print(f"N_DONE {n}", flush=True)

    save()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
