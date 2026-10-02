"""Plain MLP family for the one-shot N-step Life question: no convolution,
no recurrence, no weight tying, depth not tied to N. Input is the flattened
(2N+1)x(2N+1) window around a cell (the light-cone prior only); output is a
single logit for that cell's state N steps later.

depth in {2, 4}: a plain stack, Linear(in, w) -> ReLU -> [Linear(w,w) ->
ReLU] * (depth-1) -> Linear(w, 1). No normalization, no residual -- "say
what you used": plain MLP for the shallow conditions.

depth in {8, 16}: input-projected to width w, then `depth` pre-LN residual
blocks (x = x + Linear(ReLU(Linear(LayerNorm(x)))) two-layer MLP block per
residual unit so the block itself has some nonlinwear capacity, with its own
hidden expansion fixed at w, i.e. no bottleneck/widen), then a final
LayerNorm and Linear(w, 1) readout. Residual + LayerNorm only -- no conv, no
weight sharing across blocks (every block has its own parameters).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PlainMLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int):
        super().__init__()
        assert depth >= 1
        layers = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        self.depth = depth
        self.width = width
        self.in_dim = in_dim

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h).squeeze(-1)

    def hidden_states(self, x):
        """Returns the post-ReLU activation after each hidden Linear layer,
        for probing. List of length `depth`."""
        outs = []
        h = x
        for i in range(0, len(self.trunk), 2):
            h = self.trunk[i](h)
            h = self.trunk[i + 1](h)
            outs.append(h)
        return outs


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        h = self.ln(x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
        self.out_ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        self.depth = depth
        self.width = width
        self.in_dim = in_dim

    def forward(self, x):
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        h = self.out_ln(h)
        return self.head(h).squeeze(-1)

    def hidden_states(self, x):
        """Returns the residual stream after each block, for probing."""
        outs = []
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
            outs.append(h)
        return outs


def make_mlp(in_dim: int, width: int, depth: int) -> nn.Module:
    if depth <= 4:
        return PlainMLP(in_dim, width, depth)
    return ResidualMLP(in_dim, width, depth)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
