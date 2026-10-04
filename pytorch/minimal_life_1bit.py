"""Does Game of Life survive 1-bit weights? Quantization-aware training
(QAT) of the n=1 minimal-network families from minimal_life.py, now with
binary (+-1) or ternary (-1/0/+1, BitNet b1.58 style) weights.

Quantization scheme, following BitNet b1.58 (Ma et al., "The Era of
1-bit LLMs: All Large Language Models are in 1.58 Bits", arXiv:2402.17764)
for the ternary case and the XNOR-Net binary-weight scheme (Rastegari et
al., arXiv:1603.05279) for the binary case:

  ternary:  gamma = mean(|W|) (absmean, per layer, a single scalar);
            Wq = clip(round(W / gamma), -1, 1) in {-1, 0, 1};
            layer output = gamma * (x @ Wq^T) + bias
  binary:   gamma = mean(|W|) (absmean, matches XNOR-Net's per-filter
            scaling factor collapsed to one scalar since these layers are
            tiny); Wq = sign(W) in {-1, +1} (ties broken to +1);
            layer output = gamma * (x @ Wq^T) + bias

A full-precision latent weight is kept as the trainable nn.Parameter;
the forward pass quantizes it on the fly with a straight-through
estimator (STE): the forward value is exactly Wq (so training sees the
real quantization noise), the backward gradient is the identity (so the
latent weight still gets a gradient signal). This is the standard
BitNet/XNOR-Net QAT recipe, not a novel scheme.

Bias stays full precision (not quantized) -- standard practice in both
cited papers, and biases are a tiny fraction of the parameter count here
(1-2 per layer).

At eval, the model is used exactly as trained (model.eval(), same
forward): no separate "export to int8/pack bits" step was built, since
the forward already only ever uses Wq + the one scale float per layer,
which is what "using only the quantized weights at eval" means here --
nothing about the latent full-precision weight reaches the output.

Data generation, scoring and the 512-neighbourhood exactness check are
copied from minimal_life.py / life.py (n=1 only; this experiment does not
touch n=2/3/10).
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

RULE = Rule.life()


def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------ quantization --

def quantize(w: torch.Tensor, mode: str):
    """Returns (w_ste, scale). w_ste's forward value is the quantized
    weight (ternary in {-1,0,1} or binary in {-1,+1}); its gradient is the
    identity w.r.t. the latent weight w (straight-through estimator)."""
    scale = w.detach().abs().mean().clamp(min=1e-8)
    if mode == "ternary":
        wq = torch.clamp(torch.round(w / scale), -1, 1)
    elif mode == "binary":
        wq = torch.sign(w)
        wq = torch.where(wq == 0, torch.ones_like(wq), wq)
    else:
        raise ValueError(mode)
    w_ste = w + (wq - w).detach()
    return w_ste, scale


class QuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, mode: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.mode = mode

    def forward(self, x):
        wq, scale = quantize(self.weight, self.mode)
        return scale * F.linear(x, wq) + self.bias

    def quantized_weight(self):
        with torch.no_grad():
            wq, scale = quantize(self.weight, self.mode)
            return wq.detach().cpu().numpy(), float(scale)


class QuantConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, mode: str,
                 padding: int = 0, padding_mode: str = "zeros"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.mode = mode
        self.padding = padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size

    def forward(self, x):
        wq, scale = quantize(self.weight, self.mode)
        if self.padding_mode == "circular" and self.padding > 0:
            x = F.pad(x, [self.padding] * 4, mode="circular")
            pad = 0
        else:
            pad = self.padding
        return scale * F.conv2d(x, wq, padding=pad) + self.bias.view(1, -1, 1, 1)

    def quantized_weight(self):
        with torch.no_grad():
            wq, scale = quantize(self.weight, self.mode)
            return wq.detach().cpu().numpy(), float(scale)


# ---------------------------------------------------------------- models --

class QuantCNN(nn.Module):
    """Conv3x3(1->c, circular, quantized) -> ReLU -> Conv1x1(c->1, quantized)."""

    def __init__(self, channels: int, mode: str):
        super().__init__()
        self.conv1 = QuantConv2d(1, channels, 3, mode, padding=1, padding_mode="circular")
        self.conv2 = QuantConv2d(channels, 1, 1, mode)

    def forward(self, x):
        x = x.unsqueeze(1)
        h = F.relu(self.conv1(x))
        return self.conv2(h).squeeze(1)


class QuantMLP(nn.Module):
    """9 neighbourhood values -> hidden -> 1, quantized weights."""

    def __init__(self, hidden: int, mode: str):
        super().__init__()
        self.fc1 = QuantLinear(9, hidden, mode)
        self.fc2 = QuantLinear(hidden, 1, mode)

    def forward(self, bits):
        return self.fc2(F.relu(self.fc1(bits))).squeeze(-1)


# ------------------------------------------------------------- data/eval --

def random_batch(rng: np.random.Generator, batch: int, size: int, density: float):
    xs, ys = [], []
    for _ in range(batch):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        xs.append(g.cells.astype("float32").reshape(size, size))
        ys.append(g.step(RULE).cells.astype("float32").reshape(size, size))
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


_NEIGHBOR_TABLE_CACHE: dict[int, np.ndarray] = {}


def neighbor_table(size: int) -> np.ndarray:
    if size not in _NEIGHBOR_TABLE_CACHE:
        g = Grid(size, size)
        _NEIGHBOR_TABLE_CACHE[size] = np.array([g.neighbor_indices(i) for i in range(size * size)])
    return _NEIGHBOR_TABLE_CACHE[size]


def mlp_batch(rng: np.random.Generator, batch: int, size: int, density: float):
    nb_table = neighbor_table(size)
    xs, ys = [], []
    for _ in range(batch):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        flat = g.cells.astype("float32")
        truth = g.step(RULE)
        nb = flat[nb_table]
        xs.append(np.concatenate([nb, flat[:, None]], axis=1))
        ys.append(truth.cells.astype("float32"))
    x = torch.tensor(np.concatenate(xs, axis=0))
    y = torch.tensor(np.concatenate(ys, axis=0))
    return x, y


def all_512_grids_and_truth():
    order = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    grids = np.zeros((512, 3, 3), dtype="float32")
    bits = np.zeros((512, 9), dtype="float32")
    truth = np.zeros(512, dtype="int64")
    for k in range(512):
        nb = [(k >> b) & 1 for b in range(8)]
        self_state = (k >> 8) & 1
        grids[k, 1, 1] = self_state
        bits[k] = nb + [self_state]
        for (dy, dx), v in zip(order, nb):
            grids[k, (1 + dy) % 3, (1 + dx) % 3] = v
        n = sum(nb)
        truth[k] = 1 if RULE.next(self_state != 0, n) else 0
    return grids, bits, truth


_ALL_512_GRIDS, _ALL_512_BITS, _ALL_512_TRUTH = all_512_grids_and_truth()


def cnn_512_eval(model, device) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(_ALL_512_GRIDS, device=device)
        logits = model(x)
        pred = (torch.sigmoid(logits[:, 1, 1]) >= 0.5).long().cpu().numpy()
    return float((pred == _ALL_512_TRUTH).mean())


def mlp_512_eval(model, device) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(_ALL_512_BITS, device=device)
        pred = (torch.sigmoid(model(x)) >= 0.5).long().cpu().numpy()
    return float((pred == _ALL_512_TRUTH).mean())


def fresh_boards_eval(model, rng, n_boards, size, density, device, is_mlp):
    model.eval()
    xs, ys = [], []
    for _ in range(n_boards):
        seed = int(rng.integers(0, 2**31 - 1))
        g = Grid.random(size, size, seed, density)
        xs.append(g.cells.astype("float32").reshape(size, size))
        ys.append(g.step(RULE).cells.astype("int64").reshape(-1))
    want = np.stack(ys)
    with torch.no_grad():
        if is_mlp:
            nb_table = neighbor_table(size)
            cases = []
            for x_grid in xs:
                flat = x_grid.reshape(-1)
                nb = flat[nb_table]
                cases.append(np.concatenate([nb, flat[:, None]], axis=1))
            x = torch.tensor(np.stack(cases).astype("float32"), device=device).reshape(-1, 9)
            pred = (torch.sigmoid(model(x)) >= 0.5).long().cpu().numpy().reshape(n_boards, -1)
        else:
            x = torch.tensor(np.stack(xs), device=device)
            logits = model(x)
            pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().reshape(n_boards, -1)
    correct = int((pred == want).sum())
    return correct / want.size


# --------------------------------------------------------------- training --

def train_one(make_model, is_mlp, seed, steps, density, board_size, batch, lr, device,
              log_every, plateau_window, plateau_eps):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    converged_at = None
    stopped_early = None
    best_cells_correct = 0.0
    history = []
    last_step = 0
    for step in range(1, steps + 1):
        last_step = step
        model.train()
        if is_mlp:
            x, y = mlp_batch(rng, batch, board_size, density)
        else:
            x, y = random_batch(rng, batch, board_size, density)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            table_acc = mlp_512_eval(model, device) if is_mlp else cnn_512_eval(model, device)
            eval_rng = np.random.default_rng(seed * 1_000_003 + step)
            cells_acc = fresh_boards_eval(model, eval_rng, 100, board_size, density, device, is_mlp)
            score = min(table_acc, cells_acc)
            best_cells_correct = max(best_cells_correct, score)
            history.append(score)
            if table_acc >= 1.0 and cells_acc >= 1.0:
                converged_at = step
                break
            if len(history) > plateau_window and (best_cells_correct - history[-plateau_window - 1]) < plateau_eps:
                stopped_early = "plateau"
                break
    return {
        "converged_at": converged_at,
        "best_cells_correct": best_cells_correct,
        "stopped_early": stopped_early,
        "steps_run": last_step,
        "params": num_params(model),
    }, model


def run_condition(name, make_model, is_mlp, seeds, steps, density, board_size, batch, lr,
                   device, log_every, plateau_window, plateau_eps, mode, dump_weights):
    results = []
    t0 = time.time()
    params = None
    weight_dump = None
    for seed in seeds:
        r, model = train_one(make_model, is_mlp, seed, steps, density, board_size, batch, lr,
                              device, log_every, plateau_window, plateau_eps)
        params = r["params"]
        results.append({"seed": seed, **{k: v for k, v in r.items() if k != "params"}})
        if dump_weights and r["converged_at"] is not None and weight_dump is None:
            layers = {}
            for lname, layer in model.named_children():
                wq, scale = layer.quantized_weight()
                layers[lname] = {
                    "weight": wq.tolist(),
                    "scale": scale,
                    "bias": layer.bias.detach().cpu().numpy().tolist(),
                }
            weight_dump = {"seed": seed, "layers": layers}
    dt = time.time() - t0
    n_conv = sum(1 for r in results if r["converged_at"] is not None)
    conv_steps = [r["converged_at"] for r in results if r["converged_at"] is not None]
    median_steps = float(np.median(conv_steps)) if conv_steps else None
    best_non_conv = max((r["best_cells_correct"] for r in results if r["converged_at"] is None), default=None)
    bits_per_weight = 1.0 if mode == "binary" else (np.log2(3))
    out = {
        "name": name,
        "mode": mode,
        "params": params,
        "bits_per_weight": bits_per_weight,
        "density": density,
        "converged": n_conv,
        "seeds": len(seeds),
        "median_steps_to_converge": median_steps,
        "best_cells_correct_non_converged": best_non_conv,
        "wall_seconds": dt,
        "per_seed": results,
    }
    if weight_dump is not None:
        out["weight_dump"] = weight_dump
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--board-size", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--density", type=float, default=0.38)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--plateau-window", type=int, default=10)
    ap.add_argument("--plateau-eps", type=float, default=0.005)
    ap.add_argument("--cnn-widths", default="2,4,8,16")
    ap.add_argument("--mlp-hiddens", default="4,8,16,32")
    ap.add_argument("--modes", default="ternary,binary")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    widths = [int(x) for x in args.cnn_widths.split(",")]
    hiddens = [int(x) for x in args.mlp_hiddens.split(",")]
    modes = args.modes.split(",")
    seeds = list(range(args.seeds))

    report = []

    def save():
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    def run_and_save(cond, **kw):
        r = run_condition(cond, seeds=seeds, steps=args.steps, density=args.density,
                           board_size=args.board_size, batch=args.batch, lr=args.lr, device=device,
                           log_every=args.log_every, plateau_window=args.plateau_window,
                           plateau_eps=args.plateau_eps, **kw)
        summary = {k: v for k, v in r.items() if k not in ("per_seed", "weight_dump")}
        print(json.dumps(summary), flush=True)
        report.append(r)
        save()

    for mode in modes:
        for c in widths:
            run_and_save(f"QuantCNN c={c} [{mode}]",
                         make_model=(lambda c=c, mode=mode: QuantCNN(c, mode)),
                         is_mlp=False, mode=mode, dump_weights=True)
        for h in hiddens:
            run_and_save(f"QuantMLP hidden={h} [{mode}]",
                         make_model=(lambda h=h, mode=mode: QuantMLP(h, mode)),
                         is_mlp=True, mode=mode, dump_weights=True)

    save()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
