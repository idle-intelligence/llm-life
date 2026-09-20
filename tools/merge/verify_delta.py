#!/usr/bin/env python3
"""Load a merged GGUF back, dequantize its adapted tensors, and check two
things against an independently-recomputed f32 merge (base dequant + the
adapter's own `scale * b^T @ a^T`, not reusing `merge_lora_gguf.py`'s output):

  1. requantization error: ||dequant(merged_quantized) - merged_f32|| /
     ||merged_f32|| per tensor — this is the number the task's "Q8 < 1%,
     Q4 a few %" bar is about, since it is the ordinary weight-quantization
     error applied to the post-merge weight.
  2. delta survival: ||dequant(merged_quantized) - dequant(base_quantized)||
     / ||delta|| — how much of the LoRA delta itself is still visible after
     requantization, which is a much noisier number because the delta
     (a few % of ||W||, see merge_lora_gguf.py's printed norms) is itself
     comparable in magnitude to one Q4_0 quantization step. A low number
     here is expected and is not a bug; it says Q4_0 partly swallows the
     fine-tune's signal, which is exactly the risk of merging a LoRA delta
     into a 4-bit base instead of a higher-precision one.

venv: tools/merge/.venv
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from merge_lora_gguf import ADAPTED_ALWAYS, ADAPTED_MLP, TENSOR_NAME, dequant_tensor
import lora_io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged", required=True, nargs="+", help="one or more merged GGUFs to check")
    args = ap.parse_args()

    base_reader = gguf.GGUFReader(args.base)
    base_tensors = {t.name: t for t in base_reader.tensors}
    spec, flat = lora_io.load(args.adapter)
    num_layers = int(base_reader.fields["qwen2.block_count"].contents())
    per_layer = lora_io.per_layer_matrices(spec, flat, num_layers)
    scale = spec.alpha / spec.rank
    adapted = list(ADAPTED_ALWAYS) + (ADAPTED_MLP if spec.mlp else [])

    for merged_path in args.merged:
        m_reader = gguf.GGUFReader(merged_path)
        m_tensors = {t.name: t for t in m_reader.tensors}
        quant_err, delta_survival = [], []
        for i in range(num_layers):
            for key in adapted:
                name = f"blk.{i}.{TENSOR_NAME[key]}"
                w_base = dequant_tensor(base_reader, base_tensors[name])
                w_merged_q = dequant_tensor(m_reader, m_tensors[name])
                a, b = per_layer[i][key]
                delta = scale * (b.T @ a.T)
                w_merged_f32 = w_base + delta

                quant_err.append(np.linalg.norm(w_merged_q - w_merged_f32) / np.linalg.norm(w_merged_f32))
                delta_survival.append(
                    np.linalg.norm(w_merged_q - w_base) / max(np.linalg.norm(delta), 1e-12)
                )
        quant_err, delta_survival = np.array(quant_err), np.array(delta_survival)
        print(f"{Path(merged_path).name}: {len(quant_err)} adapted tensors")
        print(f"  requant error vs f32 merge:  mean {quant_err.mean():.4%}, max {quant_err.max():.4%}")
        print(f"  delta survival (||recovered delta||/||intended delta||): "
              f"mean {delta_survival.mean():.2f}, min {delta_survival.min():.2f}, max {delta_survival.max():.2f}")


if __name__ == "__main__":
    main()
