"""PyTorch retrain of variant A adapters (per-cell, exhaustive 512-case
table), matching crates/llm-life/src/train/run_a.rs: two adapters differing
only in the prefix (rules text vs none), rank 8 alpha 16 on q/k/v/o,
batch 64, lr 3e-5, seed 1 (docs/runs/2026-09-20-ft-a-{rules,norules}-300.md).

Packing here is simpler than variant B's stencil: each of the `batch`
per-cell prompts is independent text after the shared prefix, so a causal
mask over `prefix ++ concatenated prompts` with each prompt block only
seeing the prefix and itself reproduces variant_a::pack_chunk without the
KV-chunking Rust's engine needed (this port never runs 4096 cells per
generation, only a training batch at a time).
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from life import Grid, Rule, Rng
from qwen_life import LoraSpec, QwenLife, load_base
from score import score


def rules_prefix(rule: Rule) -> str:
    return (
        f"Cellular automaton, rule {rule.to_rulestring()}. Each cell is 0 (dead) or 1 (alive).\n"
        "A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n"
        "A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n"
        "For each cell, answer with one digit: its next state.\n"
    )


def norules_prefix() -> str:
    return "For each cell, answer with one digit: its next state.\n"


def cell_prompt(neighbors: list[int], self_state: int) -> str:
    s = "Neighbors:"
    for n in neighbors:
        s += f" {1 if n else 0}"
    s += f" / Self: {1 if self_state else 0} / Next: "
    return s


def all_cases(rule: Rule):
    out = []
    for k in range(512):
        nb = [(k >> b) & 1 for b in range(8)]
        self_state = (k >> 8) & 1
        n = sum(nb)
        nxt = 1 if rule.next(self_state != 0, n) else 0
        out.append((nb, self_state, nxt))
    return out


def build_batch(tokenizer, prefix_ids: list[int], batch, device):
    """Pack `prefix ++ per-cell prompt` blocks, each block causal within
    itself and seeing the shared prefix, nothing else across blocks."""
    p = len(prefix_ids)
    blocks = []
    for nb, self_state, _ in batch:
        text = "\n" + cell_prompt(nb, self_state)
        ids = tokenizer.encode(text, add_special_tokens=False)
        blocks.append(ids)
    total = p + sum(len(b) for b in blocks)
    tokens = list(prefix_ids)
    positions = list(range(p))
    answer_rows = []
    starts = []
    cur = p
    for ids in blocks:
        starts.append(cur)
        tokens.extend(ids)
        positions.extend(range(p, p + len(ids)))
        answer_rows.append(cur + len(ids) - 1)
        cur += len(ids)
    mask_out = torch.ones(total, total, dtype=torch.bool)
    for i in range(p):
        mask_out[i, : i + 1] = False
    for start, ids in zip(starts, blocks):
        for i in range(len(ids)):
            row = start + i
            mask_out[row, :p] = False
            mask_out[row, start : row + 1] = False
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    positions_t = torch.tensor(positions, dtype=torch.long)
    return tokens_t, positions_t, mask_out, answer_rows


@torch.no_grad()
def evaluate(model, tokenizer, prefix_ids, cases, device, chunk=64):
    # Bare eval-time forwards build a full autograd graph too (`lora_A`/
    # `lora_B` require grad and every base-model activation feeds them) even
    # though nothing ever calls `.backward()` on the result. Never freed,
    # that graph accumulates across every chunk of a 512-case eval and OOMs
    # a 10 GB card long before the training loop itself would — the PyTorch
    # analog of the Burn trainer's `evaluate_a`'s own `drop(l.backward())`
    # workaround for the same retained-graph problem.
    correct = 0
    loss_sum = 0.0
    done = 0
    for i in range(0, len(cases), chunk):
        batch = cases[i:i + chunk]
        tokens, positions, m, rows = build_batch(tokenizer, prefix_ids, batch, device)
        logits = model(tokens, positions, m)
        row_idx = torch.tensor(rows, dtype=torch.long)
        cell_logits = logits[0, row_idx, :]
        targets = torch.tensor([t for _, _, t in batch], dtype=torch.long)
        loss = F.cross_entropy(cell_logits, targets)
        loss_sum += loss.item() * len(batch)
        pred = cell_logits.argmax(-1)
        correct += int((pred == targets).sum())
        done += len(batch)
    return {"loss": loss_sum / done, "accuracy": correct / done, "n": done}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--gguf-file", default="qwen2.5-0.5b-instruct-q4_0.gguf")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--norules", action="store_true")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-doc", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rule = Rule.life()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    text = norules_prefix() if args.norules else rules_prefix(rule)
    prefix_ids = tokenizer.encode(text, add_special_tokens=False)
    dead = tokenizer.encode("0", add_special_tokens=False)[0]
    alive = tokenizer.encode("1", add_special_tokens=False)[0]
    print(f"variant A {'a-norules' if args.norules else 'a-rules'}: prefix {len(prefix_ids)} tokens")

    t0 = time.time()
    hf = load_base(args.gguf_dir, args.gguf_file).to(device)
    hf.eval()
    model = QwenLife(hf, [dead, alive], LoraSpec(args.rank, args.alpha)).to(device)
    print(f"loaded in {time.time() - t0:.1f}s")

    params = list(model.lora_modules.parameters())
    n_trainable = sum(p.numel() for p in params)
    print(f"{len(params)} LoRA tensors, {n_trainable} trainable parameters")

    cases = all_cases(rule)
    before = evaluate(model, tokenizer, prefix_ids, cases, device)
    print(f"step 0 (base): {before}")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    rng = Rng(args.seed)
    order = list(range(len(cases)))
    cursor = len(order)
    log = []
    evals = []
    start = time.time()
    best = None
    stopped_full = False
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            for i in range(len(order) - 1, 0, -1):
                j = int(rng.unit() * (i + 1))
                order[i], order[j] = order[j], order[i]
            cursor = 0
        idx = order[cursor:cursor + args.batch]
        cursor += args.batch
        batch = [cases[i] for i in idx]

        tokens, positions, m, rows = build_batch(tokenizer, prefix_ids, batch, device)
        logits = model(tokens, positions, m)
        row_idx = torch.tensor(rows, dtype=torch.long)
        cell_logits = logits[0, row_idx, :]
        targets = torch.tensor([t for _, _, t in batch], dtype=torch.long)
        loss = F.cross_entropy(cell_logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        secs = time.time() - start
        log.append((step, loss.item(), secs))
        print(f"step {step:4d} loss {loss.item():.4f}  {secs:.1f}s")

        if args.eval_every and step % args.eval_every == 0:
            e = evaluate(model, tokenizer, prefix_ids, cases, device)
            print(f"  held-out (full 512): {e}")
            evals.append((step, e))
            is_best = best is None or e["accuracy"] > best[0]
            if is_best:
                best = (e["accuracy"], {k: v.detach().clone() for k, v in model.lora_modules.state_dict().items()})
            if e["accuracy"] >= 1.0:
                print(f"512/512 correct at step {step}, stopping")
                stopped_full = True
                break

    if best is not None:
        model.lora_modules.load_state_dict(best[1])
    final = evaluate(model, tokenizer, prefix_ids, cases, device)
    print(f"final (best-by-accuracy) full 512: {final}")

    from safetensors.torch import save_file
    sd = {}
    for (layer, name), adapter in model.lora.items():
        prefix_k = f"base_model.model.model.layers.{layer}.self_attn.{name}"
        sd[f"{prefix_k}.lora_A.weight"] = adapter.lora_A.detach().cpu()
        sd[f"{prefix_k}.lora_B.weight"] = adapter.lora_B.detach().cpu()
    save_file(sd, args.out)
    print(f"wrote {args.out}")

    if args.run_doc:
        doc = {
            "hyperparameters": vars(args),
            "trainable_parameters": n_trainable,
            "wall_clock_s": time.time() - start,
            "before": before,
            "evals": evals,
            "final": final,
            "stopped_full": stopped_full,
        }
        with open(args.run_doc, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {args.run_doc}")


if __name__ == "__main__":
    main()
