"""Q4_0 / Q8_0 dequantize + quantize, matching llama.cpp's block layout and
llm-wasm's reader (crates/llm-wasm/src/gguf.rs) byte-for-byte.

venv: tools/merge/.venv

Block layout (both formats, block size 32 elements):
  Q4_0: 18 bytes = f16 scale `d` + 16 bytes of paired nibbles. Byte `j`'s low
        nibble is element `j`, high nibble is element `j+16` (llm-life's
        `weights.rs::dequant_q4_0` pairing, which matches llama.cpp's
        `y[j] = xi0 | (xi1 << 4)` with `xi0` from `x[j]`, `xi1` from
        `x[j+16]`). Dequant: `(nibble - 8) * d`.
  Q8_0: 34 bytes = f16 scale `d` + 32 signed int8 quants. Dequant: `q * d`.

Quantization (llama.cpp `quantize_row_q{4,8}_0_reference`):
  Q4_0: `amax`/`max` over the block keeping sign; `d = max / -8`,
        `id = 1/d` (or 0 if `d == 0`); `nibble = round(x*id + 8)` clamped to
        [0, 15].
  Q8_0: `amax = max(|x|)`; `d = amax / 127`, `id = 1/d` (or 0);
        `q = round(x*id)` clamped to [-127, 127].
"""

import numpy as np

QK = 32


def _f16_to_f32(u16arr: np.ndarray) -> np.ndarray:
    return u16arr.astype("<u2").view("<f2").astype(np.float32)


def _f32_to_f16(f32arr: np.ndarray) -> np.ndarray:
    return f32arr.astype(np.float32).astype("<f2").view("<u2")


def dequant_q4_0(raw: bytes, n_elements: int) -> np.ndarray:
    blocks = n_elements // QK
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, 18)
    d = _f16_to_f32(buf[:, 0:2].copy().view("<u2").reshape(blocks))
    nib = buf[:, 2:18]  # [blocks, 16]
    lo = (nib & 0x0F).astype(np.float32) - 8.0
    hi = ((nib >> 4) & 0x0F).astype(np.float32) - 8.0
    out = np.empty((blocks, QK), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None]
    out[:, 16:32] = hi * d[:, None]
    return out.reshape(-1)[:n_elements]


def dequant_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    blocks = n_elements // QK
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, 34)
    d = _f16_to_f32(buf[:, 0:2].copy().view("<u2").reshape(blocks))
    q = buf[:, 2:34].view(np.int8).astype(np.float32)
    out = q * d[:, None]
    return out.reshape(-1)


def quantize_q4_0(x: np.ndarray) -> bytes:
    n = x.shape[-1]
    assert n % QK == 0, f"length {n} not a multiple of {QK}"
    blocks = n // QK
    xb = x.reshape(blocks, QK).astype(np.float32)
    idx = np.argmax(np.abs(xb), axis=1)
    amax_signed = xb[np.arange(blocks), idx]
    d = amax_signed / -8.0
    id_ = np.where(d != 0, 1.0 / np.where(d == 0, 1.0, d), 0.0)
    q = np.rint(xb * id_[:, None] + 8.0)
    q = np.clip(q, 0, 15).astype(np.uint8)
    lo = q[:, 0:16]
    hi = q[:, 16:32]
    packed = (lo | (hi << 4)).astype(np.uint8)
    dh = _f32_to_f16(d).astype("<u2").tobytes()
    out = np.empty((blocks, 18), dtype=np.uint8)
    out[:, 0:2] = np.frombuffer(dh, dtype=np.uint8).reshape(blocks, 2)
    out[:, 2:18] = packed
    return out.tobytes()


def quantize_q8_0(x: np.ndarray) -> bytes:
    n = x.shape[-1]
    assert n % QK == 0, f"length {n} not a multiple of {QK}"
    blocks = n // QK
    xb = x.reshape(blocks, QK).astype(np.float32)
    amax = np.max(np.abs(xb), axis=1)
    d = amax / 127.0
    id_ = np.where(d != 0, 1.0 / np.where(d == 0, 1.0, d), 0.0)
    q = np.rint(xb * id_[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    dh = _f32_to_f16(d).astype("<u2").tobytes()
    out = np.empty((blocks, 34), dtype=np.uint8)
    out[:, 0:2] = np.frombuffer(dh, dtype=np.uint8).reshape(blocks, 2)
    out[:, 2:34] = q.view(np.uint8)
    return out.tobytes()
