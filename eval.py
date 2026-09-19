"""Evaluate a System One backend on data/routing.jsonl: accuracy and calibration.

    uv run eval.py --backend nli
    uv run eval.py --backend ollama --model phi3
    uv run eval.py --backend typesafe
    uv run eval.py --report          # re-print tables from cached results/*.jsonl

Predictions are cached in results/<backend>[-<model>].jsonl so analysis can be re-run offline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from demo import QUESTIONS, make_client

load_dotenv()

DATA = Path(__file__).with_name("data") / "routing.jsonl"
RESULTS = Path(__file__).with_name("results")
N_BINS = 10


# ---------------------------------------------------------------------------- inference

def run(backend: str, model: str | None, limit: int | None, resume: bool = False) -> Path:
    os.environ["SYSTEMONE_BACKEND"] = backend
    if model:
        os.environ["SYSTEMONE_MODEL"] = model
    client = make_client()
    rows = [json.loads(l) for l in DATA.open()][:limit]
    if backend == "student" and model:
        tag = Path(model).name if Path(model).exists() else f"student-{model.replace('/', '_')}"
    else:
        tag = backend + (f"-{model.replace(':', '_').replace('/', '_')}" if model else "")
    if backend == "ollama" and os.environ.get("SYSTEMONE_STYLE", "letters") != "letters":
        tag += "-" + os.environ["SYSTEMONE_STYLE"]
    out = RESULTS / f"{tag}.jsonl"
    RESULTS.mkdir(exist_ok=True)
    done: set[int] = set()
    if resume and out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                pass
    latencies = []
    with out.open("a" if resume else "w") as f:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            t0 = time.perf_counter()
            resp = client.system_one(state=row["state"], questions=QUESTIONS)
            latencies.append((time.perf_counter() - t0) * 1000)
            a = resp.answers
            f.write(json.dumps({
                "id": row["id"],
                "latency_ms": latencies[-1],
                "tool": {"probs": a["tool"].probabilities, "gold": row["tool"]},
                "urgency": {"probs": {str(k): v for k, v in a["urgency"].probabilities.items()},
                            "score": a["urgency"].score, "gold": row["urgency"]},
                "destructive": {"p": a["destructive"].noul, "gold": row["destructive"]},
                "needs_confirmation": {"p": a["needs_confirmation"].noul, "gold": row["needs_confirmation"]},
            }) + "\n")
            print(f"\r{tag}: {i + 1}/{len(rows)}  ({latencies[-1]:.0f}ms)", end="", flush=True)
    lat = sorted(latencies)
    print(f"\n{tag}: median latency {lat[len(lat) // 2]:.0f}ms, p90 {lat[int(len(lat) * .9)]:.0f}ms -> {out}")
    return out


# ---------------------------------------------------------------------------- metrics

def ece(confs: list[float], correct: list[bool]) -> tuple[float, list[tuple[float, float, int]]]:
    """Expected calibration error with equal-width bins; also returns (conf, acc, n) per bin."""
    bins = [[] for _ in range(N_BINS)]
    for c, ok in zip(confs, correct):
        bins[min(int(c * N_BINS), N_BINS - 1)].append((c, ok))
    total, table = 0.0, []
    for b in bins:
        if not b:
            continue
        mc = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        total += len(b) / len(confs) * abs(mc - acc)
        table.append((mc, acc, len(b)))
    return total, table


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def reliability(table: list[tuple[float, float, int]]) -> str:
    return "  ".join(f"{mc:.2f}->{acc:.2f}(n={n})" for mc, acc, n in table)


def report(path: Path) -> dict:
    rows = []
    for line in path.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # file still being written
            pass
    # Re-join gold labels from the data file so relabeling doesn't require re-running models.
    gold_by_id = {json.loads(l)["id"]: json.loads(l) for l in DATA.open()}
    for x in rows:
        g = gold_by_id[x["id"]]
        for k in ("tool", "urgency", "destructive", "needs_confirmation"):
            x[k]["gold"] = g[k]
    r: dict = {"backend": path.stem, "n": len(rows)}
    if not rows:
        return r
    lat = sorted(x["latency_ms"] for x in rows)
    r["latency_ms"] = lat[len(lat) // 2]

    # tool (choice): top-1 accuracy, ECE on the top-1 probability, NLL
    pred = [max(x["tool"]["probs"], key=x["tool"]["probs"].get) for x in rows]
    conf = [max(x["tool"]["probs"].values()) for x in rows]
    ok = [p == x["tool"]["gold"] for p, x in zip(pred, rows)]
    r["tool_acc"] = sum(ok) / len(ok)
    r["tool_ece"], r["tool_rel"] = ece(conf, ok)
    r["tool_nll"] = -sum(math.log(max(x["tool"]["probs"][x["tool"]["gold"]], 1e-9)) for x in rows) / len(rows)
    # confusion of gold -> most common wrong prediction
    wrong: dict[str, dict[str, int]] = {}
    for p, x in zip(pred, rows):
        if p != x["tool"]["gold"]:
            wrong.setdefault(x["tool"]["gold"], {}).setdefault(p, 0)
            wrong[x["tool"]["gold"]][p] += 1
    r["tool_confusions"] = {g: max(d, key=d.get) + f" x{max(d.values())}" for g, d in wrong.items()}

    # urgency (score): argmax accuracy, within-one, MAE of the expected score
    am = [int(max(x["urgency"]["probs"], key=x["urgency"]["probs"].get)) for x in rows]
    gold = [x["urgency"]["gold"] for x in rows]
    r["urg_acc"] = sum(a == g for a, g in zip(am, gold)) / len(rows)
    r["urg_within1"] = sum(abs(a - g) <= 1 for a, g in zip(am, gold)) / len(rows)
    r["urg_mae"] = sum(abs(x["urgency"]["score"] - g) for x, g in zip(rows, gold)) / len(rows)

    # nouls: accuracy at 0.5, Brier, AUROC, ECE
    for k in ("destructive", "needs_confirmation"):
        p = [x[k]["p"] for x in rows]
        g = [x[k]["gold"] for x in rows]
        r[f"{k}_acc"] = sum((pi >= 0.5) == bool(gi) for pi, gi in zip(p, g)) / len(rows)
        r[f"{k}_brier"] = sum((pi - gi) ** 2 for pi, gi in zip(p, g)) / len(rows)
        r[f"{k}_auroc"] = auroc(p, g)
        # calibration of the predicted class: conf = max(p, 1-p)
        r[f"{k}_ece"], r[f"{k}_rel"] = ece([max(pi, 1 - pi) for pi in p],
                                           [(pi >= 0.5) == bool(gi) for pi, gi in zip(p, g)])
    return r


def print_table(reports: list[dict]) -> None:
    cols = [
        ("backend", "{}"), ("n", "{}"), ("latency_ms", "{:.0f}"),
        ("tool_acc", "{:.2f}"), ("tool_ece", "{:.2f}"), ("tool_nll", "{:.2f}"),
        ("urg_acc", "{:.2f}"), ("urg_within1", "{:.2f}"), ("urg_mae", "{:.2f}"),
        ("destructive_acc", "{:.2f}"), ("destructive_auroc", "{:.2f}"), ("destructive_brier", "{:.2f}"), ("destructive_ece", "{:.2f}"),
        ("needs_confirmation_acc", "{:.2f}"), ("needs_confirmation_auroc", "{:.2f}"), ("needs_confirmation_brier", "{:.2f}"), ("needs_confirmation_ece", "{:.2f}"),
    ]
    short = {"needs_confirmation": "confirm", "destructive": "destr", "latency_ms": "ms"}
    hdr = [short.get(c.rsplit("_", 1)[0], c.rsplit("_", 1)[0]) + ("_" + c.rsplit("_", 1)[1] if "_" in c else "") if c.startswith(("needs", "destr")) else short.get(c, c) for c, _ in cols]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in reports:
        print("| " + " | ".join(fmt.format(r[c]) for c, fmt in cols) + " |")
    print()
    for r in reports:
        print(f"{r['backend']}")
        print(f"  tool reliability (conf->acc):  {reliability(r['tool_rel'])}")
        print(f"  tool confusions (gold: predicted): {r['tool_confusions']}")
        print(f"  destr reliability:   {reliability(r['destructive_rel'])}")
        print(f"  confirm reliability: {reliability(r['needs_confirmation_rel'])}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["nli", "ollama", "student", "typesafe"])
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report", action="store_true", help="only print tables from cached results")
    ap.add_argument("--resume", action="store_true", help="skip ids already present in the results file")
    args = ap.parse_args()
    if args.backend:
        run(args.backend, args.model, args.limit, args.resume)
    print_table([r for r in (report(p) for p in sorted(RESULTS.glob("*.jsonl"))) if r["n"]])
