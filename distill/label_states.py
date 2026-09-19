"""Teacher-label synthetic states with calibrated soft targets.

    uv run distill/label_states.py --phase small     # gemma3n: urgency, destructive, needs_confirmation
    uv run distill/label_states.py --phase tool      # gemma3 12b: tool (serial; big model)
    uv run distill/label_states.py --phase confirm   # gemma3 12b: needs_confirmation (overrides small)
    uv run distill/label_states.py --phase destructive  # gemma3 12b: destructive (overrides small)

Each phase is resumable and writes into data/synthetic_labels/<phase>.jsonl. Probabilities are
calibrated with the parameters in results/calibration.json (temperature for choice/score, Platt
for nouls) so the student is distilled from calibrated targets, not raw logprob spikes.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "distill"))

from calibrate import platt, temper  # noqa: E402
from demo import QUESTIONS  # noqa: E402
from local_systemone import LocalClient  # noqa: E402

STATES = ROOT / "data" / "synthetic_states.jsonl"
OUT_DIR = ROOT / "data" / "synthetic_labels"
PHASES = {
    "small": {"model": "gemma3n:e4b", "workers": 3, "questions": ["urgency", "destructive", "needs_confirmation"]},
    "tool": {"model": "gemma3:12b-it-qat", "workers": 1, "questions": ["tool"]},
    # needs_confirmation re-labeled with the 12B: gemma3n's Platt fit squashed it to the base rate
    "confirm": {"model": "gemma3:12b-it-qat", "workers": 1, "questions": ["needs_confirmation"]},
    # destructive re-labeled with the 12B: gemma3n's Platt fit (a=0.31 on 9 positives) compressed it
    "destructive": {"model": "gemma3:12b-it-qat", "workers": 1, "questions": ["destructive"]},
}


def calibrated(name: str, answer, params: dict) -> dict:
    if name in ("tool", "urgency"):
        probs = {str(k): v for k, v in answer.probabilities.items()}
        return temper(probs, params[f"{name}_T"])
    a, b = params[f"{name}_platt"]
    return {"p": platt(getattr(answer, "noul"), a, b)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tail", type=int, help="only the last N states (for labeling a supplement)")
    args = ap.parse_args()
    phase = PHASES[args.phase]
    os.environ["SYSTEMONE_WORKERS"] = str(phase["workers"])
    params = json.load(open(ROOT / "results" / "calibration.json"))[
        "ollama-" + phase["model"].replace(":", "_").replace("/", "_")]

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{args.phase}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open() if l.strip()}
    rows = [json.loads(l) for l in STATES.open()]
    rows = rows[-args.tail :] if args.tail else rows[: args.limit]
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.phase}: {len(done)} done, {len(todo)} to label with {phase['model']}")

    client = LocalClient(backend="ollama", model=phase["model"])
    questions = {k: QUESTIONS[k] for k in phase["questions"]}
    t0 = time.perf_counter()
    with out.open("a") as f:
        for i, row in enumerate(todo):
            resp = client.system_one(state=row["state"], questions=questions)
            f.write(json.dumps({"id": row["id"], **{k: calibrated(k, a, params) for k, a in resp.answers.items()}}) + "\n")
            f.flush()
            el = time.perf_counter() - t0
            print(f"\r{i + 1}/{len(todo)}  {el / (i + 1):.1f}s/state  eta {(len(todo) - i - 1) * el / (i + 1) / 60:.0f} min",
                  end="", flush=True)
    print(f"\nwrote {out}")
