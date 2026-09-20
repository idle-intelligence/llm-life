"""Reader for llm-life's on-disk LoRA format (crates/llm-life/src/train/lora_io.rs).

File layout:
  8 bytes  magic "LLMLIFE2"
  u32 le   rank
  f32 le   alpha
  u8       mlp (adapts gate/up too, in addition to q/k/v/o)
  u32 le   n tensors
  per tensor: u32 rows, u32 cols, rows*cols f32 le, row-major

Tensor order (TrainModel::lora_params, model.rs): per layer 0..num_layers,
for each of [q, k, v, o] always, then [gate, up] iff `mlp`, in that order:
`a` ([in_features, rank]) then `b` ([rank, out_features])`. `down` is never
adapted.
"""

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"LLMLIFE2"


@dataclass
class LoraSpec:
    rank: int
    alpha: float
    mlp: bool


def load(path: str):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == MAGIC, f"not a LoRA file: {path}"
    rank = struct.unpack_from("<I", data, 8)[0]
    alpha = struct.unpack_from("<f", data, 12)[0]
    mlp = data[16] != 0
    n = struct.unpack_from("<I", data, 17)[0]
    off = 21
    tensors = []
    for _ in range(n):
        r, c = struct.unpack_from("<II", data, off)
        off += 8
        count = r * c
        arr = np.frombuffer(data, dtype="<f4", count=count, offset=off).reshape(r, c)
        off += count * 4
        tensors.append(arr)
    assert off == len(data), f"trailing bytes: {len(data) - off}"
    return LoraSpec(rank, alpha, mlp), tensors


def per_layer_matrices(spec: LoraSpec, tensors, num_layers: int):
    """Split the flat tensor list into per-layer dicts of {name: (a, b)}."""
    names = ["q", "k", "v", "o"] + (["gate", "up"] if spec.mlp else [])
    per_layer = 2 * len(names)
    assert len(tensors) == num_layers * per_layer, (
        f"expected {num_layers * per_layer} LoRA tensors ({num_layers} layers x "
        f"{len(names)} adapted linears x 2), got {len(tensors)}"
    )
    out = []
    it = iter(tensors)
    for _ in range(num_layers):
        layer = {}
        for name in names:
            a = next(it)
            b = next(it)
            layer[name] = (a, b)
        out.append(layer)
    return out
