"""Linear probes on every hidden layer of a trained one-shot N=2 MLP: can a
logistic-regression probe recover the centre cell's intermediate state at
t+1 (one Life step, not two) from that layer's activations? Probing only,
no interpretation beyond the reported numbers.

Ground truth for the probe label reuses bitslice_life2_exhaustive's
`life_step_planes` applied to the centre 3x3 sub-window of the 5x5 input
(offsets INNER3 are exactly the centre cell's own 1-step neighbourhood,
since the centre cell sits at (0,0) of the window) -- the same function
`ground_truth_2step` itself calls internally for its `gen1[(0,0)]` term, so
the probe target is the exact intermediate value the 2-step ground truth
computes on its way to the final answer, not a re-derivation.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from bitslice_life2_exhaustive import INNER3, make_planes_chunk, life_step_planes
from oneshot_n2_exhaustive import words_to_bits_2d
from oneshot_model import make_mlp


def sample_dataset(n_words: int, seed: int):
    rng = np.random.default_rng(seed)
    w_starts = rng.integers(0, (1 << 25) // 32, size=n_words)
    xs = []
    ys = []
    for w in w_starts:
        planes = make_planes_chunk(int(w), int(w) + 1)
        x = np.empty((32, 25), dtype=np.float32)
        from bitslice_life2_exhaustive import OFFSETS5
        for k, off in enumerate(OFFSETS5):
            x[:, k] = words_to_bits_2d(planes[off]).reshape(-1)
        centre_sub = {off: planes[off] for off in INNER3}
        t1_word = life_step_planes(centre_sub)
        y = words_to_bits_2d(t1_word).reshape(-1)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_logreg(x: np.ndarray, y: np.ndarray, device, steps: int = 500, lr: float = 0.1):
    xt = torch.tensor(x, device=device)
    yt = torch.tensor(y, device=device)
    d = x.shape[1]
    w = torch.zeros(d, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = xt @ w + b
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits = xt @ w + b
        pred = (logits >= 0).float()
        acc = (pred == yt).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-state", required=True, help="torch state_dict .pt file")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train-words", type=int, default=4000)
    ap.add_argument("--n-test-words", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_mlp(25, args.width, args.depth).to(device)
    model.load_state_dict(torch.load(args.model_state, map_location=device))
    model.eval()

    x_train, y_train = sample_dataset(args.n_train_words, seed=1)
    x_test, y_test = sample_dataset(args.n_test_words, seed=2)

    with torch.no_grad():
        hs_train = model.hidden_states(torch.tensor(x_train, device=device))
        hs_test = model.hidden_states(torch.tensor(x_test, device=device))

    results = []
    for layer_idx, (ht, hv) in enumerate(zip(hs_train, hs_test)):
        acc = train_logreg(ht.cpu().numpy(), y_train, device)
        # evaluate the same probe weights on held-out test activations by refitting is avoided;
        # instead fit on train, score on test, using a fresh LBFGS fit on train only then eval on test.
        xt_tr = ht
        xt_te = hv
        w = torch.zeros(xt_tr.shape[1], device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        yt_tr = torch.tensor(y_train, device=device)
        opt = torch.optim.LBFGS([w, b], lr=0.1, max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = xt_tr @ w + b
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yt_tr)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            train_acc = ((xt_tr @ w + b >= 0).float() == yt_tr).float().mean().item()
            yt_te = torch.tensor(y_test, device=device)
            test_acc = ((xt_te @ w + b >= 0).float() == yt_te).float().mean().item()
        results.append({"layer": layer_idx, "train_acc": train_acc, "test_acc": test_acc})
        print(json.dumps(results[-1]), flush=True)

    out = {"depth": args.depth, "width": args.width, "n_train": len(y_train), "n_test": len(y_test),
           "probe_target": "centre cell at t+1 (one Life step)", "per_layer": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
