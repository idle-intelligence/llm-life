"""A from-scratch Qwen2 forward pass in torch (CPU, f32), reading weights
straight out of a GGUF via `gguf.GGUFReader` + `ggml_quant`.

This exists only to functionally score the merged GGUFs without touching any
GPU on this laptop (no ndarray/CPU backend exists in the Rust crate — see
docs/runs/2026-09-20-merge.md — so this is a from-scratch reimplementation,
matched to crates/llm-life/src/train/model.rs's `TrainModel` piece by piece:
RMSNorm, rotate-half RoPE, GQA attention, SwiGLU MLP, tied-embedding head
sliced to two answer tokens).

venv: tools/merge/.venv
"""

import sys
from pathlib import Path

import gguf
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ggml_quant import dequant_q4_0, dequant_q8_0

DEQUANT = {
    gguf.GGMLQuantizationType.Q4_0: dequant_q4_0,
    gguf.GGMLQuantizationType.Q8_0: dequant_q8_0,
}


class Qwen2Forward:
    def __init__(self, gguf_path: str):
        r = gguf.GGUFReader(gguf_path)
        self.r = r
        tensors = {t.name: t for t in r.tensors}
        self.tensors = tensors

        self.num_layers = int(r.fields["qwen2.block_count"].contents())
        self.hidden = int(r.fields["qwen2.embedding_length"].contents())
        self.n_heads = int(r.fields["qwen2.attention.head_count"].contents())
        self.n_kv = int(r.fields["qwen2.attention.head_count_kv"].contents())
        self.rms_eps = float(r.fields["qwen2.attention.layer_norm_rms_epsilon"].contents())
        self.rope_theta = float(r.fields["qwen2.rope.freq_base"].contents())
        self.head_dim = self.hidden // self.n_heads

        def f32(name):
            t = tensors[name]
            assert t.tensor_type == gguf.GGMLQuantizationType.F32, (name, t.tensor_type)
            return torch.from_numpy(t.data.astype(np.float32).copy())

        def weight_in_out(name):
            """Dequantize a 2D projection and transpose to [in, out], the
            TrainModel convention (weights.rs), so forward is `x @ w`."""
            t = tensors[name]
            in_f, out_f = int(t.shape[0]), int(t.shape[1])
            n = in_f * out_f
            deq = DEQUANT.get(t.tensor_type)
            if deq is None:
                raise ValueError(f"{name}: unsupported dtype {t.tensor_type}")
            flat = deq(t.data.tobytes(), n)
            w_out_in = flat.reshape(out_f, in_f)  # GGUF/Q4Linear layout [out, in]
            return torch.from_numpy(np.ascontiguousarray(w_out_in.T))  # -> [in, out]

        self.layers = []
        for i in range(self.num_layers):
            p = f"blk.{i}"
            self.layers.append(dict(
                attn_norm=f32(f"{p}.attn_norm.weight"),
                q_w=weight_in_out(f"{p}.attn_q.weight"), q_b=f32(f"{p}.attn_q.bias"),
                k_w=weight_in_out(f"{p}.attn_k.weight"), k_b=f32(f"{p}.attn_k.bias"),
                v_w=weight_in_out(f"{p}.attn_v.weight"), v_b=f32(f"{p}.attn_v.bias"),
                o_w=weight_in_out(f"{p}.attn_output.weight"),
                ffn_norm=f32(f"{p}.ffn_norm.weight"),
                gate_w=weight_in_out(f"{p}.ffn_gate.weight"),
                up_w=weight_in_out(f"{p}.ffn_up.weight"),
                down_w=weight_in_out(f"{p}.ffn_down.weight"),
            ))
        self.out_norm = f32("output_norm.weight")

        embd = tensors["token_embd.weight"]
        self.embd_t = embd
        self.embd_in = int(embd.shape[0])   # hidden
        self.embd_deq = DEQUANT[embd.tensor_type]
        self.embd_bytes_per_row = embd.data.shape[1]  # byte-shape: [vocab, bytes]

        half = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(half) * 2.0 / self.head_dim))
        self.inv_freq = torch.from_numpy(inv_freq.astype(np.float32))

    def embed_row(self, token_id: int) -> torch.Tensor:
        row_bytes = self.embd_t.data[token_id].tobytes()
        return torch.from_numpy(self.embd_deq(row_bytes, self.embd_in).copy())

    def embed_tokens(self, ids):
        return torch.stack([self.embed_row(i) for i in ids])  # [T, hidden]

    def rms_norm(self, x, gamma):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_eps)
        return (x / rms) * gamma

    def rope(self, x, positions):
        # x: [T, H, Dh]
        t = x.shape[0]
        pos = torch.tensor(positions, dtype=torch.float32).unsqueeze(1)  # [T,1]
        freqs = pos * self.inv_freq.unsqueeze(0)  # [T, half]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, Dh]
        cos = emb.cos().unsqueeze(1)  # [T,1,Dh]
        sin = emb.sin().unsqueeze(1)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def forward_logits(self, token_ids, answer_token_ids):
        """Causal forward over one sequence, returns logits at the last
        position for `answer_token_ids` (tied-embedding head, broadcast dot
        product == `TrainModel::head_sliced`)."""
        t = len(token_ids)
        positions = list(range(t))
        x = self.embed_tokens(token_ids)  # [T, hidden]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)  # True = masked out

        for layer in self.layers:
            normed = self.rms_norm(x, layer["attn_norm"])
            q = (normed @ layer["q_w"] + layer["q_b"]).view(t, self.n_heads, self.head_dim)
            k = (normed @ layer["k_w"] + layer["k_b"]).view(t, self.n_kv, self.head_dim)
            v = (normed @ layer["v_w"] + layer["v_b"]).view(t, self.n_kv, self.head_dim)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
            n_rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(n_rep, dim=1)  # [T, n_heads, Dh]
            v = v.repeat_interleave(n_rep, dim=1)
            q = q.transpose(0, 1)  # [H, T, Dh]
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale  # [H, T, T]
            scores = scores.masked_fill(causal, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            attn = probs @ v  # [H, T, Dh]
            attn = attn.transpose(0, 1).reshape(t, self.hidden)
            x = x + attn @ layer["o_w"]

            normed2 = self.rms_norm(x, layer["ffn_norm"])
            gate = torch.nn.functional.silu(normed2 @ layer["gate_w"])
            up = normed2 @ layer["up_w"]
            x = x + (gate * up) @ layer["down_w"]

        x = self.rms_norm(x, self.out_norm)
        last = x[-1]  # [hidden]
        head = torch.stack([self.embed_row(i) for i in answer_token_ids])  # [K, hidden]
        return head @ last  # [K]
