"""Load the `tools/export`-produced safetensors into the PyTorch stencil
models. The only name mismatch is LayerNorm: Burn calls its two parameters
`gamma`/`beta`, PyTorch calls them `weight`/`bias`."""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from stencil_models import BertOfLife, Mlp2OfLife, StencilOfLife


_EMBEDDING_KEYS = {"tok_emb.weight", "pos_emb.weight"}


def _transpose_linears(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Burn's `Linear` stores weight `[in, out]` (`x.matmul(weight)`);
    PyTorch's `nn.Linear` stores `[out, in]` (`x @ weight.T`). Every 2D
    `*.weight` that is not an embedding table needs the transpose; biases,
    LayerNorm gamma/beta and embedding weights are unaffected."""
    out = dict(sd)
    for k, v in sd.items():
        if k.endswith(".weight") and k not in _EMBEDDING_KEYS and v.dim() == 2:
            out[k] = v.t().contiguous()
    return out


def _remap_ln(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    out = dict(sd)
    if f"{prefix}.gamma" in out:
        out[f"{prefix}.weight"] = out.pop(f"{prefix}.gamma")
    if f"{prefix}.beta" in out:
        out[f"{prefix}.bias"] = out.pop(f"{prefix}.beta")
    return out


def load_bert(path: str) -> BertOfLife:
    sd = load_file(path)
    sd = _transpose_linears(sd)
    sd = _remap_ln(sd, "norm_1")
    sd = _remap_ln(sd, "norm_2")
    model = BertOfLife(d_model=16, n_heads=1, d_ff=64)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    return model


def load_mlp2(path: str) -> Mlp2OfLife:
    sd = load_file(path)
    sd = _transpose_linears(sd)
    model = Mlp2OfLife(hidden=32)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    return model


def load_stencil(path: str, n_layers: int = 1) -> StencilOfLife:
    sd = load_file(path)
    sd = _transpose_linears(sd)
    for i in range(n_layers):
        sd = _remap_ln(sd, f"blocks.{i}.norm1")
        sd = _remap_ln(sd, f"blocks.{i}.norm2")
    model = StencilOfLife(d_model=16, n_layers=n_layers, n_heads=1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    return model
