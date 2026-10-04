"""N=2, one-shot MLP, trained on the FULL set of 2**25 5x5-window patterns
(not sampled boards) and verified exhaustively against the same independent
ground truth as docs/runs/2026-10-02-bitslice-2step.md's compiled-circuit
check: `bitslice_life2_exhaustive.ground_truth_2step`, a from-scratch
bitwise popcount/Life-step-twice reference (not reusing bitslice_ops'
popcount primitives), itself cross-checked against life.py/vec_step on
20,000 random windows before being trusted there.

Model: oneshot_model.make_mlp (plain MLP depth<=4, residual+LayerNorm MLP
depth>8) on the flattened 25-bit window (bitslice_compiler.window_offsets(5)
order) -- no convolution, no recurrence, no weight tying, depth not derived
from N.

Training: full-dataset epochs over all 1,048,576 32-pattern words (33,554,432
patterns), in randomly-ordered chunks of `--chunk-words`, BCEWithLogitsLoss
with a global pos_weight (computed once by one exhaustive pass counting
positive labels) to correct the class imbalance in the full window space.
Exactness (0 / 33,554,432 mismatches) is checked by the identical
chunked-exhaustive procedure every `--eval-every-epochs` epochs; training
stops the moment a model is exact.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from bitslice_life2_exhaustive import (
    OFFSETS5, TOTAL_WORDS, make_planes_chunk, ground_truth_2step,
)
from oneshot_model import make_mlp, num_params

ARANGE32 = np.arange(32, dtype=np.uint32)


def words_to_bits_2d(words: np.ndarray) -> np.ndarray:
    """uint32[n] -> float32[n, 32], bit b of word w at [w, b] (pattern index
    w*32+b), matching bitslice_life2_exhaustive's word/lane convention."""
    return ((words[:, None] >> ARANGE32) & np.uint32(1)).astype(np.float32)


def chunk_to_xy(w_start: int, w_end: int) -> tuple[np.ndarray, np.ndarray]:
    planes = make_planes_chunk(w_start, w_end)
    n = w_end - w_start
    x = np.empty((n * 32, 25), dtype=np.float32)
    for k, off in enumerate(OFFSETS5):
        x[:, k] = words_to_bits_2d(planes[off]).reshape(-1)
    truth_words = ground_truth_2step(planes)
    y = words_to_bits_2d(truth_words).reshape(-1)
    return x, y


def compute_global_pos_weight(chunk_words: int) -> float:
    pos = 0
    total = 0
    for w_start in range(0, TOTAL_WORDS, chunk_words):
        w_end = min(w_start + chunk_words, TOTAL_WORDS)
        planes = make_planes_chunk(w_start, w_end)
        truth_words = ground_truth_2step(planes)
        pos += int(np.unpackbits(truth_words.view(np.uint8)).sum())
        total += (w_end - w_start) * 32
    neg = total - pos
    return neg / max(pos, 1), pos, total


def exhaustive_eval(model: nn.Module, chunk_words: int, device) -> dict:
    mismatches = 0
    pos_pred = 0
    total = 0
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for w_start in range(0, TOTAL_WORDS, chunk_words):
            w_end = min(w_start + chunk_words, TOTAL_WORDS)
            x, y = chunk_to_xy(w_start, w_end)
            xt = torch.tensor(x, device=device)
            logits = model(xt)
            pred = (logits >= 0).float().cpu().numpy()
            mismatches += int(np.sum(pred != y))
            pos_pred += int(np.sum(pred))
            total += len(y)
    return {"mismatches": mismatches, "total": total, "pos_pred": pos_pred,
             "wall_seconds": time.time() - t0}


def train_one(depth: int, width: int, seed: int, pos_weight: float, device,
              max_epochs: int, train_chunk_words: int, eval_chunk_words: int,
              eval_every_epochs: int, lr: float):
    if device == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(seed)
    model = make_mlp(25, width, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    rng = np.random.default_rng(seed + 10000)
    n_chunks = (TOTAL_WORDS + train_chunk_words - 1) // train_chunk_words
    chunk_starts = [i * train_chunk_words for i in range(n_chunks)]

    converged_epoch = None
    final_eval = None
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        rng.shuffle(chunk_starts)
        model.train()
        epoch_loss = 0.0
        for w_start in chunk_starts:
            w_end = min(w_start + train_chunk_words, TOTAL_WORDS)
            x, y = chunk_to_xy(w_start, w_end)
            xt = torch.tensor(x, device=device)
            yt = torch.tensor(y, device=device)
            logits = model(xt)
            loss = loss_fn(logits, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(y)
        epoch_loss /= TOTAL_WORDS * 32

        if epoch % eval_every_epochs == 0 or epoch == max_epochs:
            ev = exhaustive_eval(model, eval_chunk_words, device)
            final_eval = ev
            print(json.dumps({"depth": depth, "width": width, "seed": seed, "epoch": epoch,
                               "epoch_loss": epoch_loss, "mismatches": ev["mismatches"],
                               "elapsed_s": round(time.time() - t0, 1)}), flush=True)
            if ev["mismatches"] == 0:
                converged_epoch = epoch
                break

    if final_eval is None:
        final_eval = exhaustive_eval(model, eval_chunk_words, device)

    return {
        "depth": depth, "width": width, "seed": seed,
        "params": num_params(model),
        "converged_epoch": converged_epoch,
        "exact": converged_epoch is not None,
        "final_mismatches": final_eval["mismatches"],
        "final_total": final_eval["total"],
        "epochs_run": epoch,
        "wall_seconds": time.time() - t0,
    }, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="2,4,8,16")
    ap.add_argument("--widths", default="128,512")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--eval-every-epochs", type=int, default=3)
    ap.add_argument("--train-chunk-words", type=int, default=2048)
    ap.add_argument("--eval-chunk-words", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stop-on-first-exact", action="store_true",
                     help="skip remaining seeds for a (depth,width) once one seed is exact, and skip larger widths/depths once the smallest exact config is found")
    ap.add_argument("--save-models-dir", default=None,
                     help="if set, save every trained model's state_dict here as depth{d}_width{w}_seed{s}.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depths = [int(x) for x in args.depths.split(",")]
    widths = [int(x) for x in args.widths.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print("computing global class balance over all 2**25 patterns...", flush=True)
    pos_weight, pos, total = compute_global_pos_weight(args.eval_chunk_words)
    print(json.dumps({"global_pos": pos, "global_total": total, "pos_weight": pos_weight}), flush=True)

    report = {"pos_weight": pos_weight, "pos": pos, "total": total, "runs": []}

    def save():
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    found_exact = False
    for depth in depths:
        for width in widths:
            if args.stop_on_first_exact and found_exact:
                print(json.dumps({"skip": f"depth={depth} width={width}", "reason": "smaller exact config already found"}), flush=True)
                continue
            for seed in seeds:
                r, model = train_one(depth, width, seed, pos_weight, device,
                                      args.max_epochs, args.train_chunk_words, args.eval_chunk_words,
                                      args.eval_every_epochs, args.lr)
                print(json.dumps(r), flush=True)
                report["runs"].append(r)
                save()
                if args.save_models_dir:
                    import os
                    os.makedirs(args.save_models_dir, exist_ok=True)
                    path = f"{args.save_models_dir}/depth{depth}_width{width}_seed{seed}.pt"
                    torch.save(model.state_dict(), path)
                if r["exact"]:
                    found_exact = True

    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
