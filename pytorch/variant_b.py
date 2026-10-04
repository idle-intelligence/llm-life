"""Python port of crates/llm-life/src/variant_b.rs: the whole-grid packing
(one token per cell after a shared rules prefix, stencil mask, bag
positions) that both the Rust training loop and this port's fine-tune
consume."""

from __future__ import annotations

import torch

from life import Grid, Rule


def rules_prefix(rule: Rule) -> str:
    return (
        f"Cellular automaton, rule {rule.to_rulestring()}. Each cell is 0 (dead) or 1 (alive).\n"
        "A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n"
        "A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n"
        "Grid:\n"
    )


class Packed:
    __slots__ = ("tokens", "positions", "allowed", "grid_start")

    def __init__(self, tokens, positions, allowed, grid_start):
        self.tokens = tokens
        self.positions = positions
        self.allowed = allowed
        self.grid_start = grid_start


def pack(grid: Grid, prefix_tokens: list[int], dead: int, alive: int) -> Packed:
    p = len(prefix_tokens)
    n = grid.width * grid.height
    t = p + n

    tokens = list(prefix_tokens) + [alive if c else dead for c in grid.cells.tolist()]
    positions = list(range(p)) + [p] * n

    allowed = torch.zeros(t, t, dtype=torch.bool)
    for i in range(p):
        allowed[i, : i + 1] = True
    for c in range(n):
        row = p + c
        allowed[row, :p] = True
        allowed[row, p + c] = True
        for j in grid.neighbor_indices(c):
            allowed[row, p + j] = True

    return Packed(
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
        allowed,
        p,
    )


def mask_out(packed: Packed) -> torch.Tensor:
    """llm-life's `mask_out` convention: True where attention IS masked."""
    return ~packed.allowed


def p_alive(logits: torch.Tensor, grid_start: int, n_cells: int) -> torch.Tensor:
    """Softmax over the two answer columns at the cell rows only."""
    cell_logits = logits[0, grid_start:grid_start + n_cells, :]
    return torch.softmax(cell_logits, dim=-1)[:, 1]
