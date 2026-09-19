"""Distil the teacher's calibrated soft labels into a small encoder with four heads.

    uv run distill/train_student.py [--encoder answerdotai/ModernBERT-base] [--epochs 4]

Trains on data/synthetic_states.jsonl joined with data/synthetic_labels/{small,tool}.jsonl.
5% of the synthetic set is held out for model selection; the 123 human-labeled rows are never
trained or selected on -- they are reported per epoch for information only and remain the test set.
Saves to models/student/.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import QUESTIONS, TOOLS  # noqa: E402

HEADS = {  # name -> output size; 1 means a single sigmoid logit (noul)
    "tool": len(TOOLS), "urgency": len(QUESTIONS["urgency"].criteria), "destructive": 1, "needs_confirmation": 1,
}
TOOL_LABELS = list(TOOLS)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Student(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)  # deberta ships fp16
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in HEADS.items()})

    def forward(self, **enc):
        hs = self.encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean pooling
        pooled = self.drop(pooled)
        return {k: head(pooled) for k, head in self.heads.items()}


def render(state: dict) -> str:
    return json.dumps(state, ensure_ascii=False)


def load_training_rows() -> list[dict]:
    states = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data" / "synthetic_states.jsonl").open()}
    labels: dict[int, dict] = {}
    for phase in ("small", "tool", "confirm", "destructive"):  # later phases override earlier ones per key
        p = ROOT / "data" / "synthetic_labels" / f"{phase}.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            if line.strip():
                d = json.loads(line)
                labels.setdefault(d["id"], {}).update({k: v for k, v in d.items() if k != "id"})
    rows = []
    for sid, lab in labels.items():
        if all(k in lab for k in HEADS) and sid in states:
            rows.append({
                "text": render(states[sid]["state"]),
                "tool": [lab["tool"].get(t, 0.0) for t in TOOL_LABELS],
                "urgency": [lab["urgency"].get(str(i), 0.0) for i in range(HEADS["urgency"])],
                "destructive": lab["destructive"]["p"],
                "needs_confirmation": lab["needs_confirmation"]["p"],
            })
    return rows


def load_human_rows() -> list[dict]:
    rows = []
    for line in (ROOT / "data" / "routing.jsonl").open():
        d = json.loads(line)
        rows.append({
            "text": render(d["state"]),
            "tool": [1.0 if t == d["tool"] else 0.0 for t in TOOL_LABELS],
            "urgency": [1.0 if i == d["urgency"] else 0.0 for i in range(HEADS["urgency"])],
            "destructive": float(d["destructive"]), "needs_confirmation": float(d["needs_confirmation"]),
        })
    return rows


def batches(rows, tok, bs, max_len, shuffle, rng):
    idx = list(range(len(rows)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        enc = tok([r["text"] for r in chunk], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        tgt = {
            "tool": torch.tensor([r["tool"] for r in chunk]),
            "urgency": torch.tensor([r["urgency"] for r in chunk]),
            "destructive": torch.tensor([r["destructive"] for r in chunk]),
            "needs_confirmation": torch.tensor([r["needs_confirmation"] for r in chunk]),
        }
        yield {k: v.to(DEVICE) for k, v in enc.items()}, {k: v.to(DEVICE) for k, v in tgt.items()}


def loss_fn(out, tgt):
    l_tool = -(tgt["tool"] * F.log_softmax(out["tool"], -1)).sum(-1).mean()
    l_urg = -(tgt["urgency"] * F.log_softmax(out["urgency"], -1)).sum(-1).mean()
    l_des = F.binary_cross_entropy_with_logits(out["destructive"].squeeze(-1), tgt["destructive"])
    l_con = F.binary_cross_entropy_with_logits(out["needs_confirmation"].squeeze(-1), tgt["needs_confirmation"])
    return l_tool + l_urg + l_des + l_con, (l_tool.item(), l_urg.item(), l_des.item(), l_con.item())


@torch.no_grad()
def evaluate(model, tok, rows, max_len) -> dict:
    """Loss plus argmax accuracy against whatever targets the rows carry (hard or soft)."""
    model.eval()
    tot, n, hits = 0.0, 0, {k: 0 for k in HEADS}
    for enc, tgt in batches(rows, tok, 32, max_len, False, None):
        out = model(**enc)
        l, _ = loss_fn(out, tgt)
        tot += l.item() * enc["input_ids"].shape[0]; n += enc["input_ids"].shape[0]
        hits["tool"] += (out["tool"].argmax(-1) == tgt["tool"].argmax(-1)).sum().item()
        hits["urgency"] += (out["urgency"].argmax(-1) == tgt["urgency"].argmax(-1)).sum().item()
        for k in ("destructive", "needs_confirmation"):
            hits[k] += ((out[k].squeeze(-1) > 0) == (tgt[k] > 0.5)).sum().item()
    model.train()
    return {"loss": tot / n, **{f"{k}_acc": v / n for k, v in hits.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "models" / "student"))
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = random.Random(args.seed)

    rows = load_training_rows()
    rng.shuffle(rows)
    n_dev = min(max(50, len(rows) // 20), len(rows) // 2)
    dev, train = rows[:n_dev], rows[n_dev:]
    human = load_human_rows()
    print(f"train {len(train)}  dev {len(dev)}  human-test {len(human)}  device {DEVICE}")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = Student(args.encoder).to(DEVICE)
    lens = [len(tok(r["text"])["input_ids"]) for r in train[:500]]
    print(f"token length p50 {sorted(lens)[len(lens) // 2]}  p95 {sorted(lens)[int(len(lens) * .95)]}  max_len {args.max_len}")

    enc_params = list(model.encoder.parameters()); head_params = list(model.heads.parameters())
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": 1e-3}], weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.bs); warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))

    best, step, t0 = float("inf"), 0, time.perf_counter()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        run = [0.0] * 4
        for enc, tgt in batches(train, tok, args.bs, args.max_len, True, rng):
            l, parts = loss_fn(model(**enc), tgt)
            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            run = [a + b for a, b in zip(run, parts)]
            if step % 20 == 0:
                print(f"\r  ep {epoch + 1} step {step}/{steps}  loss tool {run[0] / 20:.3f} urg {run[1] / 20:.3f} "
                      f"des {run[2] / 20:.3f} con {run[3] / 20:.3f}  {(time.perf_counter() - t0) / 60:.1f} min", end="", flush=True)
                run = [0.0] * 4
        d = evaluate(model, tok, dev, args.max_len); h = evaluate(model, tok, human, args.max_len)
        print(f"\nepoch {epoch + 1}: dev loss {d['loss']:.3f} tool {d['tool_acc']:.2f} | HUMAN tool {h['tool_acc']:.2f} "
              f"urg {h['urgency_acc']:.2f} des {h['destructive_acc']:.2f} con {h['needs_confirmation_acc']:.2f}")
        if d["loss"] < best:  # select on synthetic dev only; the human set stays a pure test set
            best = d["loss"]
            torch.save(model.state_dict(), out_dir / "student.pt")
            (out_dir / "config.json").write_text(json.dumps({
                "encoder": args.encoder, "max_len": args.max_len, "heads": HEADS, "tool_labels": TOOL_LABELS,
                "urgency_levels": list(QUESTIONS["urgency"].criteria), "epoch": epoch + 1, "dev_loss": best,
            }, indent=1))
            print(f"  saved (dev loss {best:.3f})")
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {out_dir}")
