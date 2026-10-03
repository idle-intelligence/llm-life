"""Optional: train one Variant 1 (norules) model and dump the CLS token's
attention weights for a few example cases, to see whether the model attends
to the neighbour digits or to "Self" when deciding.
"""

from __future__ import annotations

import json

import torch

from bert_text.model import BertText, make_batch
from bert_text import prompts
from bert_text.vocab import ReducedVocab, load_qwen_tokenizer

DEVICE = torch.device("cpu")


def main():
    all_prompts, labels = prompts.v1_examples()
    tok = load_qwen_tokenizer()
    vocab = ReducedVocab(tok, all_prompts)
    ids, mask = make_batch(all_prompts, vocab, DEVICE)
    y = torch.tensor(labels, dtype=torch.long, device=DEVICE)

    torch.manual_seed(1)
    model = BertText(vocab.size, 16, 1, 4, ids.shape[1], vocab.CLS, vocab.PAD).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(1, 601):
        logits = model(ids, mask)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                acc = (model(ids, mask).argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                break
    print(f"trained to step {step}, train_acc {acc}")

    # Example cases: all-dead neighbourhood+self dead, a birth case (3
    # neighbours, dead self), a survival case (2 neighbours, alive self), an
    # overcrowding case (alive self, 4 neighbours).
    examples = {
        "all_dead": ([0, 0, 0, 0, 0, 0, 0, 0], 0),
        "birth_3": ([1, 1, 1, 0, 0, 0, 0, 0], 0),
        "survive_2": ([1, 1, 0, 0, 0, 0, 0, 0], 1),
        "overcrowd_4": ([1, 1, 1, 1, 0, 0, 0, 0], 1),
    }
    prefix = prompts.norules_prefix()
    out = {}
    model.eval()
    for name, (nb, self_state) in examples.items():
        p = prefix + prompts.cell_prompt(nb, self_state)
        tokens = tok.encode(p).tokens
        ids_e, mask_e = make_batch([p], vocab, DEVICE)
        with torch.no_grad():
            logits, attn_maps = model(ids_e, mask_e, return_attn=True)
        pred = logits.argmax(-1).item()
        # layer 0 attention, averaged over heads, CLS (position 0) row.
        cls_attn = attn_maps[0][0].mean(dim=0)[0].tolist()  # [T]
        seq_tokens = ["[CLS]"] + tokens
        out[name] = {
            "pred": pred,
            "tokens": seq_tokens,
            "cls_attn": cls_attn,
        }

    with open("pytorch/bert_text/attn_examples.json", "w") as f:
        json.dump(out, f, indent=2)

    for name, d in out.items():
        print(f"\n{name} (pred={d['pred']}):")
        pairs = sorted(zip(d["tokens"], d["cls_attn"]), key=lambda x: -x[1])
        for tok_s, w in pairs[:6]:
            print(f"  {tok_s!r}: {w:.3f}")


if __name__ == "__main__":
    main()
