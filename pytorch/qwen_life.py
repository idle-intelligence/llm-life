"""Qwen2.5-0.5B-Instruct as llm-life's cellular-automaton update rule, in
PyTorch + transformers, matching crates/llm-life/src/train/model.rs
`TrainModel` operation-for-operation:

  * base weights: the same Q4_0 GGUF, dequantized by transformers' own GGUF
    loader (`AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`) —
    the same block dequant Burn's `weights::dequant_q4_0` does, so the two
    frameworks start from numerically identical float32 weights.
  * norms, MLP: `layer.input_layernorm` / `post_attention_layernorm` /
    `layer.mlp` and the final `model.norm` are used unchanged (frozen,
    untouched by any adapter) — same RMSNorm/SwiGLU formula either way.
  * attention: reimplemented here rather than calling `self_attn` directly,
    because llm-life needs a caller-supplied boolean mask (the stencil, not
    a causal one) and caller-supplied position ids restarting per cell
    (CONCEPT.md's "bag" positions) — transformers' own causal-mask machinery
    has no hook for either.
  * head: **not** `model.lm_head`. `TrainModel` reads a two-row slice of the
    tied token embedding at the answer tokens ('0', '1') and lm_head is a
    separately-initialized tensor in this GGUF (`tie_word_embeddings` reads
    False from the GGUF metadata, but the Burn trainer always ties). Using
    lm_head here would silently diverge from what train/model.rs and the
    published adapters were trained against.
  * LoRA: q/k/v/o only (this repo's adapters never touch gate/up), PEFT
    layout (`lora_A` `[r, in]`, `lora_B` `[out, r]`), delta =
    `lora_B(lora_A(x)) * alpha/r`, added to the frozen projection's output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


@dataclass
class LoraSpec:
    rank: int
    alpha: float
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoraAdapter(torch.nn.Module):
    """One projection's LoRA pair, PEFT layout. `lora_A`: `[r, in]` (a
    `Linear(in, r, bias=False)` weight), `lora_B`: `[out, r]`, initialized to
    an identity delta (`lora_B` = 0) unless loaded from a checkpoint."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * (1.0 / rank) ** 0.5)
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class QwenLife(torch.nn.Module):
    """Wraps a loaded `Qwen2ForCausalLM` (from the Q4_0 GGUF) with optional
    LoRA on q/k/v/o and a sliced, tied-embedding head."""

    def __init__(self, hf_model, answer_tokens: list[int], lora: LoraSpec | None = None):
        super().__init__()
        self.hf = hf_model
        cfg = hf_model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.rope_theta = cfg.rope_parameters["rope_theta"]
        self.eps = cfg.rms_norm_eps

        for p in self.hf.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            embed = self.hf.model.embed_tokens.weight
            self.head = torch.nn.Parameter(embed[answer_tokens].clone(), requires_grad=False)

        self.lora: dict[tuple[int, str], LoraAdapter] = {}
        if lora is not None:
            mods = torch.nn.ModuleDict()
            proj_dim = {"q_proj": (cfg.hidden_size, self.n_heads * self.head_dim),
                        "k_proj": (cfg.hidden_size, self.n_kv * self.head_dim),
                        "v_proj": (cfg.hidden_size, self.n_kv * self.head_dim),
                        "o_proj": (self.n_heads * self.head_dim, cfg.hidden_size)}
            for layer in range(self.n_layers):
                for name in lora.target_modules:
                    in_f, out_f = proj_dim[name]
                    a = LoraAdapter(in_f, out_f, lora.rank, lora.alpha)
                    mods[f"{layer}_{name}"] = a
                    self.lora[(layer, name)] = a
            self.lora_modules = mods

    def _lora(self, layer: int, name: str, x: torch.Tensor) -> torch.Tensor:
        a = self.lora.get((layer, name))
        return a(x) if a is not None else 0.0

    def _rope(self, positions: torch.Tensor, device, dtype):
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2 / self.head_dim))
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [1, T, H, Dh]; cos/sin: [T, Dh] -> [1, T, 1, Dh]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        b, n_kv, t, d = x.shape
        return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor, mask_out: torch.Tensor) -> torch.Tensor:
        """`token_ids`, `positions`: `[T]` long. `mask_out`: `[T, T]` bool,
        True where attention must be masked out (llm-life convention).
        Returns `[1, T, K]` logits over the answer tokens."""
        device = self.head.device
        t = token_ids.shape[0]
        x = self.hf.model.embed_tokens(token_ids.to(device)).unsqueeze(0)
        cos, sin = self._rope(positions.to(device), device, x.dtype)

        for i, layer in enumerate(self.hf.model.layers):
            residual = x
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(h) + self._lora(i, "q_proj", h)
            k = attn.k_proj(h) + self._lora(i, "k_proj", h)
            v = attn.v_proj(h) + self._lora(i, "v_proj", h)
            q = q.view(1, t, self.n_heads, self.head_dim)
            k = k.view(1, t, self.n_kv, self.head_dim)
            v = v.view(1, t, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
            q = self._apply_rope(q, cos, sin).permute(0, 2, 1, 3)
            k = self._apply_rope(k, cos, sin).permute(0, 2, 1, 3)
            n_rep = self.n_heads // self.n_kv
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)
            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(mask_out.to(device)[None, None, :, :], float("-inf"))
            probs = F.softmax(scores, dim=-1)
            ctx = probs @ v
            ctx = ctx.permute(0, 2, 1, 3).reshape(1, t, self.n_heads * self.head_dim)
            attn_out = attn.o_proj(ctx) + self._lora(i, "o_proj", ctx)
            x = residual + attn_out

            residual = x
            h = layer.post_attention_layernorm(x)
            x = residual + layer.mlp(h)

        x = self.hf.model.norm(x)
        logits = x @ self.head.T
        return logits


def load_base(gguf_dir: str, gguf_file: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(gguf_dir, gguf_file=gguf_file)


def load_lora_state(model: QwenLife, safetensors_path: str) -> None:
    sd = load_file(safetensors_path)
    for (layer, name), adapter in model.lora.items():
        prefix = f"base_model.model.model.layers.{layer}.self_attn.{name}"
        adapter.lora_A.data.copy_(sd[f"{prefix}.lora_A.weight"])
        adapter.lora_B.data.copy_(sd[f"{prefix}.lora_B.weight"])
