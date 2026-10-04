"""Variant 1: "no rules, Conway only". Train a from-scratch BERT-style
encoder on the LLM's own per-cell prompt text (norules format,
`prompts.v1_examples`) for all 512 neighbourhoods, labels from Conway's
rule (B3/S23). Sweep tiny sizes, several seeds, report the smallest
configuration exact on all 512.
"""

from __future__ import annotations

import itertools
import json
import time

import torch

from bert_text.model import BertText, make_batch
from bert_text import prompts
from bert_text.vocab import FULL_VOCAB_SIZE, ReducedVocab, load_qwen_tokenizer

DEVICE = torch.device("cpu")
LR = 2e-3
MAX_STEPS = 600


def build_data():
    all_prompts, labels = prompts.v1_examples()
    tok = load_qwen_tokenizer()
    vocab = ReducedVocab(tok, all_prompts)
    ids, mask = make_batch(all_prompts, vocab, DEVICE)
    y = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    return ids, mask, y, vocab


def train_one(ids, mask, y, vocab, d_model, n_layers, n_heads, seed):
    torch.manual_seed(seed)
    max_seq_len = ids.shape[1]
    model = BertText(vocab.size, d_model, n_layers, n_heads, max_seq_len,
                      cls_id=vocab.CLS, pad_id=vocab.PAD).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    steps_to_exact = None
    for step in range(1, MAX_STEPS + 1):
        model.train()
        logits = model(ids, mask)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(ids, mask).argmax(dim=-1)
            correct = (pred == y).sum().item()
        if correct == 512:
            steps_to_exact = step
            break
    params = model.num_params()
    return steps_to_exact, params


def main():
    t0 = time.time()
    ids, mask, y, vocab = build_data()
    print(f"V={vocab.size} (base tokens used={vocab.n_base_tokens}, "
          f"+PAD/CLS/UNK), max_seq_len={ids.shape[1]}, full_vocab={FULL_VOCAB_SIZE}")

    configs = []
    for d_model in (16, 32, 64):
        for n_layers in (1, 2):
            for n_heads in (1, 2, 4):
                if d_model % n_heads != 0:
                    continue
                configs.append((d_model, n_layers, n_heads))

    seeds = list(range(1, 11))
    results = []
    for d_model, n_layers, n_heads in configs:
        converged = 0
        steps_list = []
        params = None
        for seed in seeds:
            steps_to_exact, p = train_one(ids, mask, y, vocab, d_model, n_layers, n_heads, seed)
            params = p
            if steps_to_exact is not None:
                converged += 1
                steps_list.append(steps_to_exact)
        results.append({
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "params": params,
            "converged": converged,
            "total_seeds": len(seeds),
            "steps_to_exact_min": min(steps_list) if steps_list else None,
            "steps_to_exact_median": sorted(steps_list)[len(steps_list) // 2] if steps_list else None,
        })
        print(results[-1])

    out = {
        "vocab_size_v1": vocab.size,
        "n_base_tokens_v1": vocab.n_base_tokens,
        "full_vocab_size": FULL_VOCAB_SIZE,
        "max_seq_len_v1": ids.shape[1],
        "lr": LR,
        "max_steps": MAX_STEPS,
        "seeds": seeds,
        "results": results,
        "wall_clock_s": time.time() - t0,
    }
    with open("pytorch/bert_text/results_v1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wall clock: {out['wall_clock_s']:.1f}s")


if __name__ == "__main__":
    main()
