"""A small BERT-style encoder, from scratch (no transformers library
module), for the "bert-text" task: read the LLM's own per-cell prompt text
and return a 2-way (dead/alive) decision in one forward pass from a
prepended CLS token - non-autoregressive, a classifier head on a pooled
token, not a generation loop. This is the shape described for "Jev"
(TypeSafe AI, Sept 2026): text in, a typed decision with probabilities out,
in one pass - noted here as context, not reused code.

Architecture matches the repo's "BERT of Life" ladder rung
(crates/llm-life/src/bert/model.rs: token embedding + learned position
embedding, a plain post-LN transformer encoder, a 2-class linear head) but
reads text tokens instead of the 9 raw neighbourhood bits, and is
reimplemented here in plain PyTorch (manual multi-head attention) rather
than calling burn-nn or torch.nn.TransformerEncoder, so every parameter is
accounted for by hand.

Reference: Devlin et al. 2018, "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding", arXiv:1810.04805 - encoder-only,
bidirectional attention, classification from a CLS token.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                return_attn: bool = False):
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            # key_padding_mask: [b, t] True where PAD (masked out)
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, t, d)
        out = self.o_proj(out)
        x = self.ln1(x + out)

        ff = self.fc2(F.relu(self.fc1(x)))
        x = self.ln2(x + ff)

        if return_attn:
            return x, attn
        return x


class BertText(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 max_seq_len: int, cls_id: int, pad_id: int):
        super().__init__()
        self.cls_id = cls_id
        self.pad_id = pad_id
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        d_ff = d_model * 4
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, 2)

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor, return_attn: bool = False):
        """token_ids/attn_mask: [B, T] (already includes CLS at position 0,
        attn_mask 1 for real tokens incl. CLS, 0 for PAD)."""
        b, t = token_ids.shape
        x = self.tok_emb(token_ids)
        pos_ids = torch.arange(t, device=token_ids.device).unsqueeze(0).expand(b, t)
        x = x + self.pos_emb(pos_ids)
        key_padding_mask = attn_mask == 0

        attn_maps = []
        for layer in self.layers:
            if return_attn:
                x, attn = layer(x, key_padding_mask, return_attn=True)
                attn_maps.append(attn)
            else:
                x = layer(x, key_padding_mask)

        cls_out = x[:, 0, :]
        logits = self.head(cls_out)
        if return_attn:
            return logits, attn_maps
        return logits

    def num_params(self) -> dict:
        emb = sum(p.numel() for p in [self.tok_emb.weight, self.pos_emb.weight])
        rest = sum(p.numel() for n, p in self.named_parameters()
                   if not n.startswith("tok_emb") and not n.startswith("pos_emb"))
        return {"embedding": emb, "rest": rest, "total": emb + rest}


def make_batch(prompts: list[str], vocab, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompts with the reduced vocab, prepend CLS, right-pad with
    PAD to the batch's max length, return (token_ids, attn_mask)."""
    seqs = [[vocab.CLS] + vocab.encode(p) for p in prompts]
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), vocab.PAD, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)
