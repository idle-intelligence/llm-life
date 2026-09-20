#!/usr/bin/env python3
"""Functional check: run the from-scratch Python Qwen2 forward
(`qwen2_forward.py`) over all 512 `(neighbors, self)` cases — the exhaustive
lookup table variant A is trained on (`crates/llm-life/src/train/run_a.rs
all_cases`) — and report accuracy against true Conway life (B3/S23).

Each case is scored as its own full causal sequence (prefix + "\n" + the
cell's prompt), not packed into a shared-prefix chunk the way the training
loop does. Per case this is mathematically identical to the Rust chunk-packed
forward: `run_a.rs::full_sequence`'s mask gives every suffix row the full
resident prefix plus ordinary causal attention within its own block and
nothing else — exactly plain causal attention over `[prefix, this case's
suffix]` with no cross-case leakage (the block-diagonal packing exists only
to amortize the shared-prefix compute across cases in one batched matmul, not
to change what any one case attends to).

Also runs the *base* (unmerged) GGUF, as an oracle check: crates/llm-life's
own run docs record the base model's pre-training accuracy on the full
512-case set (before-training "step 0" row), and this script's tokenization
+ forward should reproduce those numbers before trusting the merged-model
numbers at all.

venv: tools/merge/.venv
"""

import argparse
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from qwen2_forward import Qwen2Forward

RULES_PREFIX = (
    "Cellular automaton, rule B3/S23. Each cell is 0 (dead) or 1 (alive).\n"
    "A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n"
    "A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n"
    "For each cell, answer with one digit: its next state.\n"
)
NORULES_PREFIX = "For each cell, answer with one digit: its next state.\n"


def rule_next(alive: bool, n: int) -> bool:
    return n in (2, 3) if alive else n == 3


def all_cases():
    out = []
    for k in range(512):
        nb = [(k >> b) & 1 for b in range(8)]
        self_state = (k >> 8) & 1
        n = sum(nb)
        out.append((nb, self_state, int(rule_next(bool(self_state), n))))
    return out


def cell_prompt(nb, self_state) -> str:
    s = "Neighbors:" + "".join(f" {n}" for n in nb)
    s += f" / Self: {self_state} / Next: "
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--norules", action="store_true")
    ap.add_argument("--limit", type=int, default=512, help="score only the first N of 512 cases")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    prefix_text = NORULES_PREFIX if args.norules else RULES_PREFIX
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False).ids

    dead_ids = tok.encode("0", add_special_tokens=False).ids
    alive_ids = tok.encode("1", add_special_tokens=False).ids
    assert len(dead_ids) == 1 and len(alive_ids) == 1, (dead_ids, alive_ids)
    answer_ids = [dead_ids[0], alive_ids[0]]

    model = Qwen2Forward(args.gguf)

    cases = all_cases()[: args.limit]
    correct = 0
    t0 = time.time()
    for nb, self_state, target in cases:
        suffix_text = "\n" + cell_prompt(nb, self_state)
        suffix_ids = tok.encode(suffix_text, add_special_tokens=False).ids
        ids = prefix_ids + suffix_ids
        logits = model.forward_logits(ids, answer_ids)
        pred = int(logits.argmax().item())
        if pred == target:
            correct += 1
    dt = time.time() - t0
    n = len(cases)
    print(f"{Path(args.gguf).name} ({'norules' if args.norules else 'rules'} prefix, "
          f"{n}/512 cases, {dt:.1f}s): accuracy {correct}/{n} = {correct/n:.4f}")


if __name__ == "__main__":
    main()
