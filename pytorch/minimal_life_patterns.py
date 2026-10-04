"""Follow-up to minimal_life_nstep.py: random 16x16 soups mostly die out by
n=10, so an all-dead prediction scores misleadingly high there. This adds a
structured pattern test (known Game of Life still
lifes/oscillators/spaceships/methuselah, LifeWiki-standard coordinates:
https://conwaylife.com/wiki/) on top of the random-soup numbers, and
reports the trivial all-dead baseline explicitly next to both.

Scope, given the shared wall-clock budget with minimal_life_nstep.py: one
representative model per (n, family) at width c=8 (the cheapest width
already characterized as reliable-ish for n=1/n=2 and the only width n=10
was checked at), trained fresh here (single seed, same hyperparameters),
then evaluated on both fresh random boards (for the all-dead baseline) and
the pattern set. Not a re-run of the full 5-seed/3-width sweep from
minimal_life_nstep.py -- that sweep's own results
(docs/runs/2026-09-30-minimal-life-nstep-results.json) already cover
convergence rates per width; this script answers "how does a
representative trained model do on structured, still-alive-by-construction
inputs" rather than "how many seeds converge".

n=1 has no tied/untied distinction (applying a model once is the same
either way), so it gets one model (MinimalCNN c=8, 89 params, the
reliable-10/10 width from the n=1 sweep) instead of an A/B pair.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from life import Grid, Rule
from minimal_life import MinimalCNN, num_params
from minimal_life_nstep import DeepCNN, TiedRecurrentCNN, vec_step, verify_vec_step, train_one, fresh_boards_eval

RULE = Rule.life()

# Standard LifeWiki coordinates (https://conwaylife.com/wiki/), one cell
# list per pattern, arbitrary anchor.
PATTERNS = {
    "block": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "beehive": [(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)],
    "blinker": [(0, 0), (1, 0), (2, 0)],
    "toad": [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)],
    "beacon": [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 2), (2, 3), (3, 3)],
    "glider": [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
    "lwss": [(1, 0), (4, 0), (0, 1), (0, 2), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3)],
    "r_pentomino": [(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],
}


def dihedral_variants(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """All 8 dihedral-group transforms (4 rotations x mirror), deduplicated
    after normalizing each to a non-negative, origin-anchored coordinate
    set -- symmetric patterns (block, beehive, blinker, toad, beacon)
    naturally collapse to fewer than 8 distinct boards; chiral ones
    (glider, lwss, r_pentomino) keep all 8."""
    seen = set()
    out = []
    pts = list(cells)
    for _ in range(4):
        pts = [(y, -x) for x, y in pts]  # rotate 90 degrees
        for mirror in (False, True):
            p2 = [(-x, y) for x, y in pts] if mirror else pts
            xs = [x for x, y in p2]
            ys = [y for x, y in p2]
            minx, miny = min(xs), min(ys)
            norm = tuple(sorted((x - minx, y - miny) for x, y in p2))
            if norm not in seen:
                seen.add(norm)
                out.append(list(norm))
    return out


def place_pattern(cells: list[tuple[int, int]], size: int, rng: np.random.Generator) -> np.ndarray:
    board = np.zeros((size, size), dtype=np.uint8)
    ox = int(rng.integers(0, size))
    oy = int(rng.integers(0, size))
    for x, y in cells:
        board[(oy + y) % size, (ox + x) % size] = 1
    return board


def build_pattern_dataset(size: int, n_positions: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for name, cells in PATTERNS.items():
        boards = []
        for variant in dihedral_variants(cells):
            for _ in range(n_positions):
                boards.append(place_pattern(variant, size, rng))
        out[name] = np.stack(boards)
    return out


def eval_patterns(model, datasets: dict[str, np.ndarray], n_steps: int, device) -> dict:
    model.eval()
    results = {}
    with torch.no_grad():
        for name, boards in datasets.items():
            xs = boards.astype(np.float32)
            ys = boards.astype(np.uint8)
            for _ in range(n_steps):
                ys = vec_step(ys)
            want = ys.reshape(len(boards), -1).astype(np.int64)
            x = torch.tensor(xs, device=device)
            logits = model(x)
            pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().reshape(len(boards), -1)
            exact = int(np.all(pred == want, axis=1).sum())
            total = len(boards)
            union_mask = (want == 1) | (pred == 1)
            num = int(((pred == want) & union_mask).sum())
            den = int(union_mask.sum())
            masked_acc = 1.0 if den == 0 else num / den
            results[name] = {"exact_boards": exact, "total_boards": total, "masked_cells_correct": masked_acc}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--board-size", type=int, default=16)
    ap.add_argument("--pattern-board-size", type=int, default=32)
    ap.add_argument("--pattern-positions", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--density", type=float, default=0.38)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--hopeless-after", type=int, default=2000)
    ap.add_argument("--hopeless-margin", type=float, default=0.03)
    ap.add_argument("--ns", default="1,2,3,10")
    ap.add_argument("--max-wall-seconds", type=float, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert verify_vec_step(), "vec_step disagrees with life.py Grid.step; refusing to run"

    ns = [int(x) for x in args.ns.split(",")]
    pattern_ds = build_pattern_dataset(args.pattern_board_size, args.pattern_positions, seed=12345)
    print(json.dumps({"pattern_variant_counts": {k: len(v) for k, v in pattern_ds.items()}}), flush=True)

    deadline = (time.time() + args.max_wall_seconds) if args.max_wall_seconds else None
    report = []

    def maybe_stop():
        return deadline is not None and time.time() >= deadline

    def run_one(cond_name, make_model, n_steps):
        if maybe_stop():
            report.append({"name": cond_name, "skipped": "deadline"})
            print(json.dumps(report[-1]), flush=True)
            return
        t0 = time.time()
        r, model = train_one(make_model, args.seed, args.steps, args.density, args.board_size,
                              args.batch, n_steps, args.lr, device, args.log_every,
                              args.hopeless_after, args.hopeless_margin, deadline)
        eval_rng = np.random.default_rng(999_000 + args.seed)
        soup_acc, soup_baseline = fresh_boards_eval(model, eval_rng, 200, args.board_size,
                                                      args.density, n_steps, device)
        pattern_results = eval_patterns(model, pattern_ds, n_steps, device)
        out = {
            "name": cond_name,
            "n_steps": n_steps,
            "params": r["params"],
            "converged_at": r["converged_at"],
            "stopped_early": r["stopped_early"],
            "steps_run": r["steps_run"],
            "random_soup_cells_correct": soup_acc,
            "random_soup_all_dead_baseline": soup_baseline,
            "patterns": pattern_results,
            "wall_seconds": time.time() - t0,
        }
        report.append(out)
        print(json.dumps({k: v for k, v in out.items() if k != "patterns"} | {"patterns_summary":
              {k: (v["exact_boards"], v["total_boards"], round(v["masked_cells_correct"], 4)) for k, v in pattern_results.items()}}),
              flush=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    for n in ns:
        if n == 1:
            run_one("n=1 MinimalCNN(c=8)", lambda: MinimalCNN(8), 1)
        else:
            run_one(f"A untied DeepCNN n={n} c=8", (lambda n=n: DeepCNN(8, depth=n + 1)), n)
            run_one(f"B tied MinimalCNN(c=8) n={n}", (lambda n=n: TiedRecurrentCNN(8, n)), n)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
