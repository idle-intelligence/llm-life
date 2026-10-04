"""Compiler-independent ground truth for two Life steps on a 5x5 neighbourhood,
bit-sliced over 32-lane words.

Bit convention: pattern index i in [0, 2**25); for k in OFFSETS5 (row-major
over the 5x5 window, (dy,dx) for dy in -2..2, dx in -2..2), bit k of i is the
state of cell OFFSETS5[k]. Packed into u32 words: word w, lane b -> pattern
index i = w*32 + b. Since b spans exactly 5 bits (0..31), bit k of i for k < 5
depends only on lane b (so plane_k is a FIXED word constant for k < 5,
broadcastable); for k >= 5, bit k of i depends only on bit (k-5) of w (so
plane_k is all-ones or all-zeros per word, the same for every lane) -- this
lets every one of the 2**25 patterns be generated without ever materializing
a (2**25)-length Python/NumPy loop.

2**25 words / 32-per-word = 1,048,576 words total; callers process in chunks
to bound memory.

The reference below (life_step_planes / ground_truth_2step) is plain bitwise
popcount arithmetic -- half/full adders written directly here, not reusing
any shared primitives -- so a bug in a compiled circuit built some other way
cannot cancel out against this reference by sharing the same bug."""

from __future__ import annotations

import numpy as np

OFFSETS5 = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]  # 25 entries, row-major
assert len(OFFSETS5) == 25
INNER3 = [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)]  # 9 entries, (dy,dx) in -1..1

TOTAL_PATTERNS = 1 << 25
TOTAL_WORDS = TOTAL_PATTERNS // 32  # 1,048,576

# Fixed low-bit-plane constants (k = 0..4): bit b of the word is bit k of b.
PLANE_LOW = []
for k in range(5):
    val = 0
    for b in range(32):
        if (b >> k) & 1:
            val |= 1 << b
    PLANE_LOW.append(np.uint32(val))


def make_planes_chunk(w_start: int, w_end: int) -> dict[tuple[int, int], np.ndarray]:
    """Returns {(dy,dx): uint32 array of length (w_end-w_start)} for all 25
    offsets in OFFSETS5, for word indices [w_start, w_end)."""
    n = w_end - w_start
    w_arr = np.arange(w_start, w_end, dtype=np.uint64).astype(np.uint32)
    planes: dict[tuple[int, int], np.ndarray] = {}
    for k, (dy, dx) in enumerate(OFFSETS5):
        if k < 5:
            planes[(dy, dx)] = np.full(n, PLANE_LOW[k], dtype=np.uint32)
        else:
            shift = k - 5
            bit = (w_arr >> np.uint32(shift)) & np.uint32(1)
            planes[(dy, dx)] = (np.uint32(0) - bit).astype(np.uint32)  # 0 -> 0x00000000, 1 -> 0xFFFFFFFF
    return planes


def _half_adder(a, b):
    return a ^ b, a & b


def _full_adder(a, b, c):
    s = a ^ b ^ c
    carry = (a & b) | (b & c) | (a & c)
    return s, carry


def life_step_planes(window9: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """window9: {(dy,dx): uint32 array} for (dy,dx) in INNER3 (3x3, self =
    (0,0)). Returns the next-generation state of the centre cell, bit-sliced,
    via a direct ripple increment-by-bit popcount of the 8 neighbours (4-bit
    counter, max value 8 fits exactly)."""
    self_bit = window9[(0, 0)]
    neighbours = [window9[(dy, dx)] for dy, dx in INNER3 if not (dy == 0 and dx == 0)]
    assert len(neighbours) == 8
    count = [np.zeros_like(self_bit) for _ in range(4)]
    for nb in neighbours:
        carry = nb
        for i in range(4):
            s, c = _half_adder(count[i], carry)
            count[i] = s
            carry = c
        # carry out of bit 3 is always 0 here since max count across all additions stays <= 8
    c0, c1, c2, c3 = count
    eq3 = c0 & c1 & (~c2) & (~c3)
    eq2 = (~c0) & c1 & (~c2) & (~c3)
    next_state = (self_bit & (eq2 | eq3)) | ((~self_bit) & eq3)
    return next_state.astype(np.uint32)


def ground_truth_2step(planes25: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """planes25: {(dy,dx): array} for (dy,dx) in OFFSETS5 (5x5). Computes the
    9 interior one-step-ahead cells (each from its own 3x3 sub-window, which
    lies entirely inside the 5x5 outer window), then applies the step once
    more to the centre using that 3x3 block."""
    gen1: dict[tuple[int, int], np.ndarray] = {}
    for oy, ox in INNER3:
        sub = {(dy, dx): planes25[(oy + dy, ox + dx)] for dy, dx in INNER3}
        gen1[(oy, ox)] = life_step_planes(sub)
    return life_step_planes(gen1)


def words_to_bits(words: np.ndarray) -> np.ndarray:
    """Unpack a 1D uint32 array into a 1D uint8 bit array (LSB = lowest
    pattern index in the word), for reporting individual failing patterns."""
    n = words.shape[0]
    out = np.zeros(n * 32, dtype=np.uint8)
    for b in range(32):
        out[b::32] = ((words >> np.uint32(b)) & np.uint32(1)).astype(np.uint8)
    return out


def pattern_grid(i: int) -> list[list[int]]:
    grid = [[0] * 5 for _ in range(5)]
    for k, (dy, dx) in enumerate(OFFSETS5):
        r, c = dy + 2, dx + 2
        grid[r][c] = (i >> k) & 1
    return grid
