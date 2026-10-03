"""Reduce Qwen2.5-0.5B-Instruct's tokenizer vocabulary to the ids that
actually occur in a set of training prompts, remapped to 0..V-1 plus
PAD/CLS/UNK. Loaded with the `tokenizers` library (not `transformers`),
same tokenizer.json the Rust/Burn/transformers sides of this repo all read.
"""

from __future__ import annotations

import os

from tokenizers import Tokenizer

# Path to Qwen2.5-0.5B-Instruct's tokenizer.json, downloaded with `hf
# download` into this repo's models directory (never ~/models). Set
# QWEN_TOKENIZER_PATH to override; no default is baked in here since the
# models directory is machine-local, not part of the repo.
QWEN_TOKENIZER_PATH = os.environ.get("QWEN_TOKENIZER_PATH")
FULL_VOCAB_SIZE = 151936  # config.json vocab_size (embedding row count incl. padding rows)


class ReducedVocab:
    def __init__(self, tok: Tokenizer, prompts: list[str]):
        self.base_tok = tok
        used_ids: set[int] = set()
        for p in prompts:
            used_ids.update(tok.encode(p).ids)
        ordered = sorted(used_ids)

        self.PAD = 0
        self.CLS = 1
        self.UNK = 2
        self.id_map: dict[int, int] = {}
        next_id = 3
        for old_id in ordered:
            self.id_map[old_id] = next_id
            next_id += 1
        self.size = next_id  # PAD, CLS, UNK + len(ordered)
        self.n_base_tokens = len(ordered)

    def encode(self, prompt: str) -> list[int]:
        ids = self.base_tok.encode(prompt).ids
        return [self.id_map.get(i, self.UNK) for i in ids]


def load_qwen_tokenizer() -> Tokenizer:
    if not QWEN_TOKENIZER_PATH:
        raise RuntimeError(
            "Set QWEN_TOKENIZER_PATH to the local Qwen2.5-0.5B-Instruct "
            "tokenizer.json (downloaded with `hf download`)."
        )
    return Tokenizer.from_file(QWEN_TOKENIZER_PATH)
