"""Python port of crates/life (the classical CA, ground truth) and the
xorshift64 RNG crates/llm-life/src/train/data.py's Rust twin uses.

Ported, not wrapped: llm-life's Rust `life` crate has no Python bindings, and
training needs the exact same grids and rule the Rust CLI evaluates the
published adapters on, cell-for-cell and bit-for-bit, so the two frameworks'
numbers are comparable. Every function here is a line-for-line translation of
its Rust counterpart; where that matters (RNG, neighbour order, boundary),
the docstring says which Rust file it mirrors.
"""

from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1


def _wmul(a: int, b: int) -> int:
    return (a * b) & MASK64


class Grid:
    """crates/life/src/grid.rs `Grid`. Row-major, toroidal boundary."""

    __slots__ = ("width", "height", "cells")

    def __init__(self, width: int, height: int, cells: np.ndarray | None = None):
        self.width = width
        self.height = height
        self.cells = cells if cells is not None else np.zeros(width * height, dtype=np.uint8)

    @staticmethod
    def random(width: int, height: int, seed: int, density: float) -> "Grid":
        """`Grid::random`: xorshift64, splitmix-scrambled seed."""
        s = _wmul(seed & MASK64, 0x9E3779B97F4A7C15) | 1
        n = width * height
        cells = np.empty(n, dtype=np.uint8)
        for i in range(n):
            s ^= (s << 13) & MASK64
            s ^= s >> 7
            s ^= (s << 17) & MASK64
            s &= MASK64
            u = (s >> 11) / float(1 << 53)
            cells[i] = 1 if u < density else 0
        return Grid(width, height, cells)

    def place_glider(self, x: int, y: int) -> None:
        for dx, dy in [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]:
            self.set((x + dx) % self.width, (y + dy) % self.height, 1)

    def set(self, x: int, y: int, v: int) -> None:
        self.cells[y * self.width + x] = v

    def get(self, x: int, y: int) -> int:
        return int(self.cells[y * self.width + x])

    def neighbor_indices(self, i: int) -> list[int]:
        """NW, N, NE, W, E, SW, S, SE — `Grid::neighbor_indices`."""
        w, h = self.width, self.height
        x, y = i % w, i // w
        xm, xp = (x - 1) % w, (x + 1) % w
        ym, yp = (y - 1) % h, (y + 1) % h
        return [
            ym * w + xm, ym * w + x, ym * w + xp,
            y * w + xm, y * w + xp,
            yp * w + xm, yp * w + x, yp * w + xp,
        ]

    def live_neighbors(self, i: int) -> int:
        return sum(1 for j in self.neighbor_indices(i) if self.cells[j] != 0)

    def step(self, rule: "Rule") -> "Grid":
        n = self.width * self.height
        out = np.empty(n, dtype=np.uint8)
        for i in range(n):
            out[i] = 1 if rule.next(self.cells[i] != 0, self.live_neighbors(i)) else 0
        return Grid(self.width, self.height, out)

    def live_count(self) -> int:
        return int(self.cells.sum())


class Rule:
    """crates/life/src/rule.rs. Only `B3/S23` (Conway's Life) is used."""

    def __init__(self, birth: set[int], survive: set[int]):
        self.birth = birth
        self.survive = survive

    @staticmethod
    def life() -> "Rule":
        return Rule({3}, {2, 3})

    def next(self, alive: bool, neighbors: int) -> bool:
        return (neighbors in self.survive) if alive else (neighbors in self.birth)

    def to_rulestring(self) -> str:
        b = "".join(str(n) for n in sorted(self.birth))
        s = "".join(str(n) for n in sorted(self.survive))
        return f"B{b}/S{s}"


class Rng:
    """crates/llm-life/src/train/data.rs `Rng` — the training sampler's own
    xorshift64, seeded `seed | 1` (no splitmix scramble, unlike `Grid::random`)."""

    def __init__(self, seed: int):
        self.s = seed | 1

    def next_u64(self) -> int:
        x = self.s
        x ^= (x << 13) & MASK64
        x ^= x >> 7
        x ^= (x << 17) & MASK64
        x &= MASK64
        self.s = x
        return x

    def unit(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53)


def sample_grid(rng: Rng, size: int) -> Grid:
    """`train::data::sample_grid`: mostly random density in [0.1, 0.4], with
    a small share of seeded gliders/blinkers."""
    roll = rng.unit()
    if roll < 0.10:
        g = Grid(size, size)
        x = int(rng.unit() * size)
        y = int(rng.unit() * size)
        g.place_glider(min(x, size - 3), min(y, size - 3))
        return g
    if roll < 0.15:
        g = Grid(size, size)
        x = int(rng.unit() * (size - 3)) + 1
        y = int(rng.unit() * (size - 3)) + 1
        g.set(x - 1, y, 1)
        g.set(x, y, 1)
        g.set(x + 1, y, 1)
        return g
    density = 0.1 + rng.unit() * 0.3
    return Grid.random(size, size, rng.next_u64(), density)


def held_out(size: int, n: int) -> list[Grid]:
    """`train::data::held_out`: never drawn from the training RNG."""
    out = []
    g = Grid(size, size)
    g.place_glider(size // 2, size // 2)
    out.append(g)
    b = Grid(size, size)
    b.set(size // 2 - 1, size // 2, 1)
    b.set(size // 2, size // 2, 1)
    b.set(size // 2 + 1, size // 2, 1)
    out.append(b)
    for seed in range(max(0, n - 2)):
        out.append(Grid.random(size, size, 1_000_000 + seed, 0.28))
    return out


def targets(grid: Grid, rule: Rule) -> np.ndarray:
    return grid.step(rule).cells
