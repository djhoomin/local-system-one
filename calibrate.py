"""Post-hoc calibration of cached backend outputs, evaluated with 5-fold cross-validation.

    uv run calibrate.py

Choice (tool): temperature scaling -- one scalar T, p_i ∝ p_i^(1/T).
Noul:          Platt scaling -- sigmoid(a * logit(p) + b), two scalars.
Fitting on cached probabilities is exact for temperature scaling because softmax(z/T) ∝ p^(1/T).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from eval import DATA, RESULTS, ece, auroc

K = 5
EPS = 1e-6


def load(path: Path) -> list[dict]:
    gold = {json.loads(l)["id"]: json.loads(l) for l in DATA.open()}
    rows = []
    for line in path.open():
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        g = gold[x["id"]]
        rows.append({
            "tool_probs": x["tool"]["probs"], "tool_gold": g["tool"],
            "destructive": (x["destructive"]["p"], g["destructive"]),
            "needs_confirmation": (x["needs_confirmation"]["p"], g["needs_confirmation"]),
        })
    return rows


def folds(n: int):
    idx = list(range(n))
    for k in range(K):
        test = idx[k::K]
        yield [i for i in idx if i not in set(test)], test


# ---------------------------------------------------------------- temperature scaling (choice)

def temper(probs: dict[str, float], t: float) -> dict[str, float]:
    logs = {k: math.log(max(v, EPS)) / t for k, v in probs.items()}
    m = max(logs.values())
    z = sum(math.exp(v - m) for v in logs.values())
    return {k: math.exp(v - m) / z for k, v in logs.items()}


def nll_choice(rows, t):
    return -sum(math.log(max(temper(r["tool_probs"], t)[r["tool_gold"]], EPS)) for r in rows) / len(rows)


def fit_t(rows) -> float:
    grid = [10 ** (i / 20) for i in range(-20, 41)]  # 0.1 .. 100
    return min(grid, key=lambda t: nll_choice(rows, t))


# ---------------------------------------------------------------- Platt scaling (noul)

def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def platt(p: float, a: float, b: float) -> float:
    return sigmoid(a * logit(p) + b)


def nll_noul(pairs, a, b):
    return -sum(math.log(max(platt(p, a, b) if g else 1 - platt(p, a, b), EPS)) for p, g in pairs) / len(pairs)


def fit_platt(pairs) -> tuple[float, float]:
    # Coarse-to-fine grid search; two parameters on ~100 points, no need for anything fancier.
    best = (0.0, 0.0)
    a_grid = [i / 20 for i in range(0, 41)]      # 0 .. 2  (a=0 collapses to the base rate)
    b_grid = [i / 4 for i in range(-24, 25)]     # -6 .. 6
    best = min(((a, b) for a in a_grid for b in b_grid), key=lambda ab: nll_noul(pairs, *ab))
    a0, b0 = best
    fine = [(a0 + da / 100, b0 + db / 40) for da in range(-5, 6) for db in range(-10, 11)]
    return min(fine, key=lambda ab: nll_noul(pairs, *ab))


# ---------------------------------------------------------------- evaluation

def evaluate(path: Path) -> None:
    rows = load(path)
    n = len(rows)
    if n < K:
        return
    print(f"=== {path.stem}  (n={n})")

    # choice
    raw_conf, raw_ok, cal_conf, cal_ok, ts, raw_nll, cal_nll = [], [], [], [], [], 0.0, 0.0
    for train, test in folds(n):
        t = fit_t([rows[i] for i in train])
        ts.append(t)
        for i in test:
            r = rows[i]
            pred = max(r["tool_probs"], key=r["tool_probs"].get)
            ok = pred == r["tool_gold"]
            raw_conf.append(r["tool_probs"][pred]); raw_ok.append(ok)
            cal = temper(r["tool_probs"], t)
            cal_conf.append(cal[pred]); cal_ok.append(ok)
            raw_nll -= math.log(max(r["tool_probs"][r["tool_gold"]], EPS))
            cal_nll -= math.log(max(cal[r["tool_gold"]], EPS))
    e_raw, _ = ece(raw_conf, raw_ok)
    e_cal, rel = ece(cal_conf, cal_ok)
    print(f"  tool   acc={sum(raw_ok) / n:.2f}  T={sum(ts) / K:.2f}   ECE {e_raw:.3f} -> {e_cal:.3f}   NLL {raw_nll / n:.2f} -> {cal_nll / n:.2f}")
    print("         calibrated reliability: " + "  ".join(f"{mc:.2f}->{acc:.2f}(n={k})" for mc, acc, k in rel))

    # nouls
    for key in ("destructive", "needs_confirmation"):
        pairs = [r[key] for r in rows]
        base = sum(g for _, g in pairs) / n
        base_brier = sum((base - g) ** 2 for _, g in pairs) / n
        raw_p, cal_p, golds, abs_ = [], [], [], []
        for train, test in folds(n):
            a, b = fit_platt([pairs[i] for i in train])
            abs_.append((a, b))
            for i in test:
                p, g = pairs[i]
                raw_p.append(p); cal_p.append(platt(p, a, b)); golds.append(g)
        brier = lambda ps: sum((p - g) ** 2 for p, g in zip(ps, golds)) / n
        e_raw, _ = ece([max(p, 1 - p) for p in raw_p], [(p >= .5) == bool(g) for p, g in zip(raw_p, golds)])
        e_cal, _ = ece([max(p, 1 - p) for p in cal_p], [(p >= .5) == bool(g) for p, g in zip(cal_p, golds)])
        a, b = (sum(x[0] for x in abs_) / K, sum(x[1] for x in abs_) / K)
        print(f"  {key:<19} AUROC={auroc(raw_p, golds):.2f}  Platt a={a:.2f} b={b:+.2f}   "
              f"Brier {brier(raw_p):.3f} -> {brier(cal_p):.3f} (base-rate {base_brier:.3f})   ECE {e_raw:.3f} -> {e_cal:.3f}")
    print()


# ---------------------------------------------------------------- final parameters for reuse

def fit_all(path: Path) -> dict:
    """Fit T (tool, urgency) and Platt (nouls) on every row; used to calibrate teacher labels."""
    gold = {json.loads(l)["id"]: json.loads(l) for l in DATA.open()}
    rows = [json.loads(l) for l in path.open()]
    tool_rows = [{"tool_probs": x["tool"]["probs"], "tool_gold": gold[x["id"]]["tool"]} for x in rows]
    urg_rows = [{"tool_probs": x["urgency"]["probs"], "tool_gold": str(gold[x["id"]]["urgency"])} for x in rows]
    out = {"tool_T": fit_t(tool_rows), "urgency_T": fit_t(urg_rows)}
    for key in ("destructive", "needs_confirmation"):
        out[f"{key}_platt"] = fit_platt([(x[key]["p"], gold[x["id"]][key]) for x in rows])
    return out


def write_calibration() -> None:
    params = {p.stem: fit_all(p) for p in sorted(RESULTS.glob("*.jsonl")) if sum(1 for _ in p.open()) >= K}
    (RESULTS / "calibration.json").write_text(json.dumps(params, indent=1))
    print(f"wrote {RESULTS / 'calibration.json'}")


if __name__ == "__main__":
    for p in sorted(RESULTS.glob("*.jsonl")):
        evaluate(p)
    write_calibration()