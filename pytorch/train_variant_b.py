"""PyTorch retrain of variant B (whole-grid LoRA), matching
crates/llm-life/src/train/run.rs: same data generator (life.py, ported
xorshift64), same packing (variant_b.py), same LoRA target (q/k/v/o,
rank/alpha from the published run), AdamW, cross-entropy on the two answer
logits at every cell.

Published hyperparameters (docs/runs/2026-09-20-ft-b-16-s3-300.md):
  grid 16x16, rule B3/S23, LoRA rank 8 alpha 16 (q/k/v/o only),
  steps 300, lr 5e-5, seed 3.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from life import Rule, Rng, held_out, sample_grid, targets
from qwen_life import LoraSpec, QwenLife, load_base
from score import score
from variant_b import mask_out, p_alive, pack, rules_prefix


def build_prompt(tokenizer, rule: Rule):
    prefix = tokenizer.encode(rules_prefix(rule), add_special_tokens=False)
    dead = tokenizer.encode("0", add_special_tokens=False)
    alive = tokenizer.encode("1", add_special_tokens=False)
    assert len(dead) == 1 and len(alive) == 1, (dead, alive)
    return prefix, dead[0], alive[0]


def evaluate(model, prefix, dead, alive, grids, rule, device):
    losses, accs, ious, alive_recalls, dead_recalls = [], [], [], [], []
    for g in grids:
        packed = pack(g, prefix, dead, alive)
        m = mask_out(packed)
        logits = model(packed.tokens, packed.positions, m)
        n = g.width * g.height
        tgt = torch.tensor(targets(g, rule), dtype=torch.long, device=device)
        cell_logits = logits[0, packed.grid_start:packed.grid_start + n, :]
        loss = F.cross_entropy(cell_logits, tgt)
        losses.append(loss.item())

        pa = p_alive(logits, packed.grid_start, n).detach().cpu().numpy()
        pred = (pa >= 0.5).astype("uint8")
        truth = g.step(rule)
        s = score(truth, pred, pa)
        accs.append(s["accuracy"])
        ious.append(s["iou"])
        alive_recalls.append(s["alive_recall"])
        dead_recalls.append(s["dead_recall"])
    n = len(grids)
    return {
        "loss": sum(losses) / n,
        "accuracy": sum(accs) / n,
        "iou": sum(ious) / n,
        "alive_recall": sum(alive_recalls) / n,
        "dead_recall": sum(dead_recalls) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--gguf-file", default="qwen2.5-0.5b-instruct-q4_0.gguf")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-grids", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-doc", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rule = Rule.life()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    prefix, dead, alive = build_prompt(tokenizer, rule)
    print(f"prefix {len(prefix)} tokens, answer '0'={dead} '1'={alive}, grid {args.size}x{args.size}")

    t0 = time.time()
    hf = load_base(args.gguf_dir, args.gguf_file).to(device)
    hf.eval()
    model = QwenLife(hf, [dead, alive], LoraSpec(args.rank, args.alpha)).to(device)
    print(f"loaded in {time.time() - t0:.1f}s")

    params = [p for p in model.lora_modules.parameters()]
    n_trainable = sum(p.numel() for p in params)
    print(f"{len(params)} LoRA tensors, {n_trainable} trainable parameters")

    eval_grids = held_out(args.size, args.eval_grids)
    before = evaluate(model, prefix, dead, alive, eval_grids, rule, device)
    print(f"step 0 (base): {before}")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    rng = Rng(args.seed)
    log = []
    start = time.time()
    for step in range(1, args.steps + 1):
        grid = sample_grid(rng, args.size)
        packed = pack(grid, prefix, dead, alive)
        m = mask_out(packed)
        logits = model(packed.tokens, packed.positions, m)
        n = args.size * args.size
        tgt = torch.tensor(targets(grid, rule), dtype=torch.long, device=device)
        cell_logits = logits[0, packed.grid_start:packed.grid_start + n, :]
        loss = F.cross_entropy(cell_logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        secs = time.time() - start
        log.append((step, loss.item(), secs))
        print(f"step {step:4d} loss {loss.item():.4f}  {secs:.1f}s")

        if args.eval_every and step % args.eval_every == 0:
            e = evaluate(model, prefix, dead, alive, eval_grids, rule, device)
            print(f"  held-out {e}")

    after = evaluate(model, prefix, dead, alive, eval_grids, rule, device)
    print(f"final held-out: {after}")

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
            "after": after,
            "loss_log": log,
        }
        with open(args.run_doc, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {args.run_doc}")


if __name__ == "__main__":
    main()
