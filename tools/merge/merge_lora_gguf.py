#!/usr/bin/env python3
"""Merge an llm-life variant-A LoRA adapter into the base Qwen2.5-0.5B-Instruct
GGUF weights and write two whole-model requantizations of the result.

venv: tools/merge/.venv (`python3 -m venv .venv && .venv/bin/pip install
numpy gguf torch --index-url https://download.pytorch.org/whl/cpu` for
torch, plain `pip install numpy gguf` otherwise — this script only needs
numpy + gguf).

Base precision: only a Q4_0 GGUF of Qwen2.5-0.5B-Instruct is available on
this machine (no safetensors/bf16 copy under models/hf/). The base weights
are therefore dequantized from Q4_0 to f32, which is the highest precision
actually available here — see docs/runs/2026-09-20-merge.md for what that
means for the -q8_0 output's accuracy ceiling (it cannot recover precision
Q4_0 already discarded; it only avoids *adding* further rounding at merge
time, and reduces rounding in the delta add itself).

Merge math (crates/llm-life/src/train/model.rs `Lora::delta`,
`TrainLinear::forward`; crates/llm-life/src/train/weights.rs layout note):
  - TrainModel weights are `[in_features, out_features]` (GGUF/Q4Linear are
    `[out_features, in_features]`, transposed once at load).
  - `delta(x) = (x @ a) @ b * (alpha / rank)`, `a: [in, rank]`,
    `b: [rank, out]` -- added to `x @ w`.
  - So in TrainModel layout: `w' = w + scale * (a @ b)`, `scale = alpha/rank`.
  - Transposing back to GGUF's `[out, in]`: `W' = W + scale * (a @ b)^T
    = W + scale * b^T @ a^T`.
  - Adapted projections (crates/llm-life/src/train/load.rs `build`): q, k, v,
    o always; gate, up iff the adapter's `LoraSpec.mlp` is set; `down` is
    never adapted.

Tensor naming (ggml/llama.cpp convention, unchanged from the base file):
  blk.{i}.attn_{q,k,v,output}.weight, blk.{i}.attn_{q,k,v}.bias,
  blk.{i}.ffn_{gate,up,down}.weight, blk.{i}.{attn,ffn}_norm.weight,
  output_norm.weight, token_embd.weight (+ an unused `output.weight` some
  base GGUFs ship alongside the tied embedding; copied through untouched).
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ggml_quant import dequant_q4_0, dequant_q8_0, quantize_q4_0, quantize_q8_0
import lora_io

ADAPTED_ALWAYS = ["q", "k", "v", "o"]
ADAPTED_MLP = ["gate", "up"]
TENSOR_NAME = {
    "q": "attn_q.weight",
    "k": "attn_k.weight",
    "v": "attn_v.weight",
    "o": "attn_output.weight",
    "gate": "ffn_gate.weight",
    "up": "ffn_up.weight",
    "down": "ffn_down.weight",
}
QUANTIZE = {"q4_0": (quantize_q4_0, gguf.GGMLQuantizationType.Q4_0), "q8_0": (quantize_q8_0, gguf.GGMLQuantizationType.Q8_0)}
DEQUANTIZE = {
    gguf.GGMLQuantizationType.Q4_0: dequant_q4_0,
    gguf.GGMLQuantizationType.Q8_0: dequant_q8_0,
}


def dequant_tensor(reader: gguf.GGUFReader, t) -> np.ndarray:
    """Dequantize a 2D GGUF tensor to f32, shaped `[out_features, in_features]`
    (`t.shape` is `[in, out]`, GGUF's fastest-varying-first `ne`)."""
    in_f, out_f = int(t.shape[0]), int(t.shape[1])
    n = in_f * out_f
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        flat = t.data.view(np.float32).reshape(-1)[:n].astype(np.float32)
    else:
        deq = DEQUANTIZE.get(t.tensor_type)
        if deq is None:
            raise ValueError(f"tensor '{t.name}' has unsupported dtype {t.tensor_type}")
        flat = deq(t.data.tobytes(), n)
    return flat.reshape(out_f, in_f)


def copy_kv(writer: gguf.GGUFWriter, field: gguf.ReaderField) -> None:
    if field.name.startswith("GGUF."):
        return  # magic/version/tensor_count/kv_count: GGUFWriter derives these itself
    if field.name == "general.architecture":
        return  # GGUFWriter(..., arch="qwen2") already wrote this in __init__
    if field.types[0] == gguf.GGUFValueType.ARRAY:
        val = field.contents()
        if len(val) == 0:
            return
        writer.add_key_value(field.name, val, gguf.GGUFValueType.ARRAY, sub_type=field.types[-1])
    else:
        writer.add_key_value(field.name, field.contents(), field.types[0])


def merge_and_write(base_path: str, adapter_path: str, out_prefix: str) -> None:
    reader = gguf.GGUFReader(base_path)
    spec, flat_tensors = lora_io.load(adapter_path)
    num_layers = int(reader.fields["qwen2.block_count"].contents())
    per_layer = lora_io.per_layer_matrices(spec, flat_tensors, num_layers)

    print(f"adapter {adapter_path}: rank {spec.rank}, alpha {spec.alpha}, "
          f"mlp={spec.mlp} ({'q/k/v/o + gate/up' if spec.mlp else 'q/k/v/o only'}), "
          f"{num_layers} layers")

    tensors_by_name = {t.name: t for t in reader.tensors}
    adapted_names = set(ADAPTED_ALWAYS) | (set(ADAPTED_MLP) if spec.mlp else set())

    # Merge once, in f32, independent of target quant format.
    merged = {}  # gguf tensor name -> f32 ndarray [out, in]
    scale = spec.alpha / spec.rank
    total_relnorm = []
    for i in range(num_layers):
        for key in adapted_names:
            name = f"blk.{i}.{TENSOR_NAME[key]}"
            t = tensors_by_name[name]
            w = dequant_tensor(reader, t)
            a, b = per_layer[i][key]  # a: [in, rank], b: [rank, out]
            delta = scale * (b.T @ a.T)  # [out, in]
            assert delta.shape == w.shape, f"{name}: delta {delta.shape} vs w {w.shape}"
            merged[name] = w + delta
            total_relnorm.append((name, float(np.linalg.norm(delta) / np.linalg.norm(w))))

    worst = max(total_relnorm, key=lambda x: x[1])
    print(f"merged {len(merged)} tensors; ||delta||/||W|| ranges up to {worst[1]:.4f} ({worst[0]})")

    # Requantize the full linear stack (adapted + untouched) into each target
    # format, so both output files are self-consistent whole-model quants,
    # not "mostly Q4_0 with a few merged tensors".
    all_linear_names = []
    for i in range(num_layers):
        for key in list(ADAPTED_ALWAYS) + ADAPTED_MLP + ["down"]:
            all_linear_names.append(f"blk.{i}.{TENSOR_NAME[key]}")

    for fmt, (quant_fn, dtype_enum) in QUANTIZE.items():
        out_path = f"{out_prefix}-{fmt}.gguf"
        writer = gguf.GGUFWriter(out_path, arch="qwen2")
        for field in reader.fields.values():
            copy_kv(writer, field)

        for t in reader.tensors:
            if t.name in merged:
                f32 = merged[t.name]
            elif t.name in all_linear_names:
                f32 = dequant_tensor(reader, t)
            else:
                # Untouched: norms, biases, token embedding, the unused
                # `output.weight` — copy the original bytes/dtype verbatim.
                writer.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=None if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16) else t.tensor_type)
                continue
            raw = quant_fn(f32.reshape(-1))
            raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(f32.shape[0], -1)
            writer.add_tensor(t.name, raw_arr, raw_shape=raw_arr.shape, raw_dtype=dtype_enum)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        size = Path(out_path).stat().st_size
        print(f"wrote {out_path} ({size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base Q4_0 GGUF")
    ap.add_argument("--adapter", required=True, help="llm-life LoRA .bin (lora_io.rs format)")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>-q4_0.gguf and <prefix>-q8_0.gguf")
    args = ap.parse_args()
    merge_and_write(args.base, args.adapter, args.out_prefix)


if __name__ == "__main__":
    main()
