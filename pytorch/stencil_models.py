"""PyTorch ports of the three stencil-life models
(crates/llm-life/src/{bert,vector}/model.rs).

Each module's parameter names match `tools/export`'s safetensors dump
exactly, so `load_state_dict` on the converted checkpoint is a straight
name match with no remapping.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertOfLife(nn.Module):
    """bert::model::BertOfLife — burn-nn 0.20's `TransformerEncoder`,
    1 layer / 1 head / d_model 16, over 9 tokens (8 neighbours + self),
    post-norm (burn-nn's `norm_first: false` default), no attention mask.

    Architecture pinned from burn-nn-0.20.1's
    `TransformerEncoderLayer::forward` (post-norm branch):
        x = x + MHA(x)
        x = norm_1(x)
        x = x + PWFF(x)          # linear_inner -> GELU -> linear_outer
        x = norm_2(x)
    and `MultiHeadAttention::forward`: separate q/k/v/out Linear(d,d) with
    bias, `attn = softmax(q @ k^T / sqrt(d_k)) @ v`, heads split
    `[b, t, h, d_k]` then `swap_dims(1, 2)` (standard head-major layout).
    """

    SEQ_LEN = 9

    def __init__(self, d_model: int = 16, n_heads: int = 1, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.tok_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(self.SEQ_LEN, d_model)
        self.mha = nn.ModuleDict({
            "query": nn.Linear(d_model, d_model),
            "key": nn.Linear(d_model, d_model),
            "value": nn.Linear(d_model, d_model),
            "output": nn.Linear(d_model, d_model),
        })
        self.pwff = nn.ModuleDict({
            "linear_inner": nn.Linear(d_model, d_ff),
            "linear_outer": nn.Linear(d_ff, d_model),
        })
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(t, device=tokens.device))
        q = self._split_heads(self.mha["query"](x))
        k = self._split_heads(self.mha["key"](x))
        v = self._split_heads(self.mha["value"](x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).reshape(b, t, self.d_model)
        attn = self.mha["output"](attn)
        x = self.norm_1(x + attn)
        ff = self.pwff["linear_outer"](F.gelu(self.pwff["linear_inner"](x)))
        x = self.norm_2(x + ff)
        center = x[:, -1, :]
        return self.head(center)


class Mlp2OfLife(nn.Module):
    """vector::model::Mlp2OfLife — 9 -> hidden -> hidden -> 2, ReLU."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(9, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(bits))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def stencil_neighbors(width: int, height: int) -> torch.Tensor:
    """vector::model::stencil_neighbors: flat `[n*9]` gather indices, row
    `i`'s taps at `[i*9:i*9+9]` = 8 neighbours (`Grid.neighbor_indices`
    order) then `i` itself last."""
    from life import Grid

    g = Grid(width, height)
    n = width * height
    idx = torch.empty(n * 9, dtype=torch.long)
    for i in range(n):
        nb = g.neighbor_indices(i)
        idx[i * 9:i * 9 + 8] = torch.tensor(nb)
        idx[i * 9 + 8] = i
    return idx


class StencilBlock(nn.Module):
    """vector::model::StencilBlock — pre-norm attention + pre-norm MLP,
    both residual, attention restricted to each query's 9 stencil taps
    (self + 8 neighbours) via a gather, never a dense mask."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        heads, d_k = self.n_heads, self.d_k

        def split(y):
            return y.reshape(b, t, heads, d_k).permute(0, 2, 1, 3)

        q = split(self.q(h))
        k = split(self.k(h))
        v = split(self.v(h))
        # gather each query's 9 taps: k/v [b, heads, t, d_k] -> [b, heads, t, 9, d_k]
        k_nb = k[:, :, neighbors, :].reshape(b, heads, t, 9, d_k)
        v_nb = v[:, :, neighbors, :].reshape(b, heads, t, 9, d_k)
        q = q.reshape(b, heads, t, 1, d_k)
        scores = (q * k_nb).sum(-1) / math.sqrt(d_k)
        w = F.softmax(scores, dim=-1).unsqueeze(-1)
        a = (w * v_nb).sum(3).permute(0, 2, 1, 3).reshape(b, t, d)
        x = x + self.proj(a)
        h = self.norm2(x)
        return x + self.ff2(F.gelu(self.ff1(h)))


class StencilOfLife(nn.Module):
    def __init__(self, d_model: int = 16, n_layers: int = 1, n_heads: int = 1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([StencilBlock(d_model, d_model * 4, n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, cells: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        b, n = cells.shape
        x = self.embed(cells.reshape(b, n, 1))
        for blk in self.blocks:
            x = blk(x, neighbors)
        return self.head(x).reshape(b, n)
