"""Dump every failing 5x5 window for one trained one-shot N=2 MLP, with its
live-cell count and the direction of the error (model says 1/truth 0 or
vice versa) -- cheap post-hoc analysis of a model's exhaustive error set,
reusing the same chunked generator/ground-truth as oneshot_n2_exhaustive.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from life_2step_truth import OFFSETS5, TOTAL_WORDS, make_planes_chunk, ground_truth_2step
from oneshot_n2_exhaustive import words_to_bits_2d
from oneshot_model import make_mlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-state", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-words", type=int, default=4096)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_mlp(25, args.width, args.depth).to(device)
    model.load_state_dict(torch.load(args.model_state, map_location=device))
    model.eval()

    failures = []
    live_counts_all = {"model_1_truth_0": [], "model_0_truth_1": []}
    with torch.no_grad():
        for w_start in range(0, TOTAL_WORDS, args.chunk_words):
            w_end = min(w_start + args.chunk_words, TOTAL_WORDS)
            planes = make_planes_chunk(w_start, w_end)
            n = w_end - w_start
            x = np.empty((n * 32, 25), dtype=np.float32)
            for k, off in enumerate(OFFSETS5):
                x[:, k] = words_to_bits_2d(planes[off]).reshape(-1)
            truth_words = ground_truth_2step(planes)
            y = words_to_bits_2d(truth_words).reshape(-1)
            xt = torch.tensor(x, device=device)
            pred = (model(xt) >= 0).float().cpu().numpy()
            mism = np.nonzero(pred != y)[0]
            for idx in mism:
                global_i = w_start * 32 + idx
                live = int(x[idx].sum())
                direction = "model_1_truth_0" if pred[idx] == 1 else "model_0_truth_1"
                live_counts_all[direction].append(live)
                failures.append({"pattern_index": int(global_i), "live_cells_5x5": live, "direction": direction})

    summary = {
        "depth": args.depth, "width": args.width,
        "n_failures": len(failures),
        "live_count_histogram_model1_truth0": dict(sorted(
            {int(k): int((np.array(live_counts_all["model_1_truth_0"]) == k).sum())
             for k in set(live_counts_all["model_1_truth_0"])}.items())),
        "live_count_histogram_model0_truth1": dict(sorted(
            {int(k): int((np.array(live_counts_all["model_0_truth_1"]) == k).sum())
             for k in set(live_counts_all["model_0_truth_1"])}.items())),
        "failures": failures,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
