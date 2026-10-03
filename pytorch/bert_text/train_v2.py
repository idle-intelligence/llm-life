"""Variant 2: "rule written in the prompt". Prepend the rule in B/S
notation ("Rule: B3/S23\\n") to the same per-cell text, train on a mixture
of outer-totalistic rules, and evaluate both exactness on the training
rules and generalisation to rules never seen in training. This is the real
question: does the encoder learn to read the rule out of the prompt, or
does it memorise per training rule?
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F

from bert_text.model import BertText, make_batch
from bert_text import prompts
from bert_text.vocab import FULL_VOCAB_SIZE, ReducedVocab, load_qwen_tokenizer

torch.set_num_threads(8)

DEVICE = torch.device("cpu")
LR = 1e-3
MAX_STEPS = 800
CHECK_EVERY = 100


def build_rule_sets():
    train_rules = dict(prompts.NAMED_TRAIN_RULES)
    train_rules.update(prompts.random_rules(seed=12345, count=8))
    held_rules = dict(prompts.NAMED_HELDOUT_RULES)
    held_rules.update(prompts.random_rules(seed=999, count=5, exclude=set(train_rules.keys())))
    assert not (set(train_rules) & set(held_rules))
    return train_rules, held_rules


def build_data(train_rules, held_rules):
    train_prompts, train_labels, train_names = prompts.v2_examples(train_rules)
    held_prompts, held_labels, held_names = prompts.v2_examples(held_rules)

    tok = load_qwen_tokenizer()
    vocab = ReducedVocab(tok, train_prompts)  # reduced vocab built from TRAINING prompts only

    train_ids, train_mask = make_batch(train_prompts, vocab, DEVICE)
    held_ids, held_mask = make_batch(held_prompts, vocab, DEVICE)
    train_y = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)
    held_y = torch.tensor(held_labels, dtype=torch.long, device=DEVICE)

    n_unk_held = sum(1 for p in held_prompts for i in vocab.base_tok.encode(p).ids
                      if i not in vocab.id_map)
    return (train_ids, train_mask, train_y, train_names,
            held_ids, held_mask, held_y, held_names, vocab, n_unk_held)


def per_rule_accuracy(pred, y, names):
    from collections import defaultdict
    correct = defaultdict(int)
    total = defaultdict(int)
    for p, t, n in zip(pred.tolist(), y.tolist(), names):
        total[n] += 1
        if p == t:
            correct[n] += 1
    return {n: correct[n] / total[n] for n in total}


def train_one(d_model, n_layers, n_heads, seed, data):
    (train_ids, train_mask, train_y, train_names,
     held_ids, held_mask, held_y, held_names, vocab, n_unk_held) = data

    torch.manual_seed(seed)
    max_seq_len = max(train_ids.shape[1], held_ids.shape[1])
    model = BertText(vocab.size, d_model, n_layers, n_heads, max_seq_len,
                      cls_id=vocab.CLS, pad_id=vocab.PAD).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Full-batch gradient descent (6656 training examples fit easily in one
    # forward pass on CPU; mini-batching only added mini-batch-loop overhead
    # here without changing the plateau behaviour observed in exploration).
    for step in range(1, MAX_STEPS + 1):
        model.train()
        logits = model(train_ids, train_mask)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % CHECK_EVERY == 0 or step == MAX_STEPS:
            model.eval()
            with torch.no_grad():
                pred = model(train_ids, train_mask).argmax(dim=-1)
                acc = (pred == train_y).float().mean().item()
            print(f"  step {step} loss {loss.item():.4f} train_acc {acc:.4f}", flush=True)
            if acc == 1.0:
                break

    model.eval()
    with torch.no_grad():
        train_pred = model(train_ids, train_mask).argmax(dim=-1)
        held_pred = model(held_ids, held_mask).argmax(dim=-1)

    train_per_rule = per_rule_accuracy(train_pred, train_y, train_names)
    held_per_rule = per_rule_accuracy(held_pred, held_y, held_names)
    held_cells_correct = (held_pred == held_y).float().mean().item()
    held_exact_rules = sum(1 for v in held_per_rule.values() if v == 1.0)

    return {
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "seed": seed,
        "steps_run": step,
        "params": model.num_params(),
        "train_per_rule_acc": train_per_rule,
        "train_exact_rules": sum(1 for v in train_per_rule.values() if v == 1.0),
        "train_total_rules": len(train_per_rule),
        "held_per_rule_acc": held_per_rule,
        "held_cells_correct": held_cells_correct,
        "held_exact_rules": held_exact_rules,
        "held_total_rules": len(held_per_rule),
        "n_unk_tokens_in_held_prompts": n_unk_held,
    }


def main():
    t0 = time.time()
    train_rules, held_rules = build_rule_sets()
    data = build_data(train_rules, held_rules)
    vocab = data[8]
    print(f"V={vocab.size} (base tokens used={vocab.n_base_tokens}), "
          f"train rules={len(train_rules)}, held rules={len(held_rules)}, "
          f"unk tokens in held prompts={data[9]}")

    configs = [(32, 2, 4), (64, 2, 4)]
    seeds = [1, 2]
    results = []
    for d_model, n_layers, n_heads in configs:
        for seed in seeds:
            r = train_one(d_model, n_layers, n_heads, seed, data)
            results.append(r)
            print({k: v for k, v in r.items()
                   if k not in ("train_per_rule_acc", "held_per_rule_acc")})

    out = {
        "train_rulestrings": {k: v.to_rulestring() for k, v in train_rules.items()},
        "held_rulestrings": {k: v.to_rulestring() for k, v in held_rules.items()},
        "vocab_size_v2": vocab.size,
        "n_base_tokens_v2": vocab.n_base_tokens,
        "full_vocab_size": FULL_VOCAB_SIZE,
        "lr": LR,
        "max_steps": MAX_STEPS,
        "n_unk_tokens_in_held_prompts": data[9],
        "results": results,
        "wall_clock_s": time.time() - t0,
    }
    with open("pytorch/bert_text/results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wall clock: {out['wall_clock_s']:.1f}s")


if __name__ == "__main__":
    main()
