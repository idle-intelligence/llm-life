"""Fine-tune a trained bitslice_life2.py seed with hard examples drawn from a
known set of failing 5x5 patterns.

Loads the dumped (ternary-weight) model back into a BitsliceDeepCNN (raw
float params initialized to wq*scale, which re-quantizes to the identical
wq/scale on the first forward pass), then continues the same QAT recipe
(ternary weights, binary step activations, soft-surrogate STE) with the
known failing 5x5 patterns (and their 8 dihedral symmetries) mixed into
every training batch as hard examples, alongside normal random/pattern
boards (same make_batch_mixed as the original run).

Hard-example construction: each failing pattern is a 25-bit 5x5 window.
Since the composed 2-step centre value depends only on this window (the two
interior 3x3 one-step neighbourhoods never reach outside it), embedding the
pattern (one of its 8 dihedral transforms, at a random board offset) into an
otherwise-random board of the normal training size and masking the training
loss to that one centre pixel gives an exact, context-independent training
signal for that failing case, without touching board size or breaking
translation invariance."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from bitslice_life2 import BitsliceDeepCNN, step_ste, quantize_ternary
from minimal_life_nstep import vec_step, verify_vec_step, fresh_boards_eval
from minimal_life_full import make_batch_mixed


def window_offsets(window: int):
    r = (window - 1) // 2
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


OFFSETS5 = window_offsets(5)
INNER3 = window_offsets(3)


def load_model_from_dump(channels: int, dump: dict, device) -> BitsliceDeepCNN:
    """Reconstructs raw float params that re-quantize (via quantize_ternary's
    scale = w.abs().mean(), wq = clamp(round(w/scale), -1, 1)) to exactly the
    dumped wq ints AND the dumped per-layer scale -- not just w = wq*scale,
    which changes the *recomputed* scale to scale_orig * (nnz/total) since
    quantize_ternary's mean is taken over ALL weights, zeros included. Each
    nonzero entry is instead given magnitude scale_orig * total/nnz (same
    sign as wq), so mean(|w|) over the whole tensor works out to scale_orig
    exactly; zero entries stay 0 (round(0/scale) = 0 regardless of scale)."""
    model = BitsliceDeepCNN(channels, "ternary").to(device)
    with torch.no_grad():
        for name in ("conv1", "conv2", "readout"):
            conv = getattr(model, name)
            layer = dump[name]
            wq = torch.tensor(layer["weight"], dtype=torch.float32, device=device)
            scale = layer["scale"]
            total = wq.numel()
            nnz = int((wq != 0).sum().item())
            mag = scale * total / nnz if nnz > 0 else scale
            conv.weight.copy_(wq * mag)
            conv.bias.copy_(torch.tensor(layer["bias"], dtype=torch.float32, device=device))
    return model


def pattern_index_to_grid(i: int) -> np.ndarray:
    grid = np.zeros((5, 5), dtype=np.uint8)
    for k, (dy, dx) in enumerate(OFFSETS5):
        r, c = dy + 2, dx + 2
        grid[r, c] = (i >> k) & 1
    return grid


def dihedral_variants(grid: np.ndarray) -> list[np.ndarray]:
    variants = []
    g = grid
    for _ in range(4):
        g = np.rot90(g)
        variants.append(g)
        variants.append(np.fliplr(g))
    # dedup identical variants (symmetric patterns)
    uniq = []
    for v in variants:
        if not any(np.array_equal(v, u) for u in uniq):
            uniq.append(v.copy())
    return uniq


def make_hard_batch(rng, batch: int, board_size: int, density: float, device, variants: list[np.ndarray]):
    boards = (rng.random((batch, board_size, board_size)) < density).astype(np.float32)
    centers = []
    for b in range(batch):
        variant = variants[rng.integers(len(variants))]
        oy = int(rng.integers(board_size))
        ox = int(rng.integers(board_size))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                r = (oy + dy) % board_size
                c = (ox + dx) % board_size
                boards[b, r, c] = variant[dy + 2, dx + 2]
        centers.append((b, oy, ox))
    y_np = vec_step(vec_step(boards))
    x = torch.tensor(boards, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    mask = torch.zeros_like(y)
    for b, oy, ox in centers:
        mask[b, oy, ox] = 1.0
    return x, y, mask


def dump_model(model: BitsliceDeepCNN) -> dict:
    return model.dump()


def run_finetune(channels, seed, base_model_dump, failing_patterns, steps, board_size, density,
                  batch, hard_frac, lr, device, log_every, time_cap_s):
    torch.manual_seed(seed + 999_000)
    rng = np.random.default_rng(seed + 999_000)
    model = load_model_from_dump(channels, base_model_dump, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    all_variants = []
    for i in failing_patterns:
        grid = pattern_index_to_grid(i)
        all_variants.extend(dihedral_variants(grid))
    print(f"hard examples: {len(failing_patterns)} failing pattern(s), {len(all_variants)} dihedral variants total", flush=True)

    hard_batch = max(1, int(round(batch * hard_frac)))
    normal_batch = batch - hard_batch

    t0 = time.time()
    step = 0
    for step in range(1, steps + 1):
        if time.time() - t0 > time_cap_s:
            print(f"time cap reached at step {step}", flush=True)
            break
        model.train()
        x_n, y_n = make_batch_mixed(rng, normal_batch, board_size, density, 2, device, 0.25)
        mask_n = torch.ones_like(y_n)
        x_h, y_h, mask_h = make_hard_batch(rng, hard_batch, board_size, density, device, all_variants)
        x = torch.cat([x_n, x_h], dim=0)
        y = torch.cat([y_n, y_h], dim=0)
        mask = torch.cat([mask_n, mask_h], dim=0)

        logits = model(x)
        loss_per_cell = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (loss_per_cell * mask).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            eval_rng = np.random.default_rng(seed * 1_000_003 + step + 999_000)
            cells_acc, baseline = fresh_boards_eval(model, eval_rng, 100, board_size, density, 2, device)
            print(json.dumps({"step": step, "cells_acc": cells_acc, "baseline": baseline,
                               "elapsed": time.time() - t0}), flush=True)

    model.eval()
    final_rng = np.random.default_rng(seed * 7 + 1 + 999_000)
    final_cells_acc, final_baseline = fresh_boards_eval(model, final_rng, 200, board_size, density, 2, device)
    return dump_model(model), {"final_cells_correct": final_cells_acc, "final_baseline": final_baseline,
                                "steps_run": step, "wall_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seed-select", type=int, required=True)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--failing-patterns", required=True, help="comma-separated pattern indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--board-size", type=int, default=16)
    ap.add_argument("--density", type=float, default=0.38)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--time-cap-s", type=float, default=600)
    args = ap.parse_args()

    device = "cpu"
    assert verify_vec_step()
    with open(args.models) as f:
        data = json.load(f)
    base_dump = data["models"][str(args.seed_select)]
    failing_patterns = [int(s) for s in args.failing_patterns.split(",") if s]

    best_dump, info = run_finetune(args.channels, args.seed_select, base_dump, failing_patterns,
                                    args.steps, args.board_size, args.density, args.batch,
                                    args.hard_frac, args.lr, device, args.log_every, args.time_cap_s)
    out = {"seed": args.seed_select, "failing_patterns": failing_patterns, "info": info,
           "model": best_dump}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(info, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
