"""Generate synthetic agent states with a local LLM.

    uv run distill/gen_states.py [--model gemma3n:e4b] [--per-prompt 20]

Messages are generated per (intended tool, domain, mood) seed for coverage; the seed is NOT a
label -- the teacher labels everything afterwards. cwd / recent_tool_calls are sampled
independently of the message so the student cannot learn to route from context alone.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import TOOLS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_states.jsonl"

DOMAINS = {
    "a Python ML research repo": "/Users/dev/research/calib",
    "a TypeScript web app with a Postgres backend": "/Users/dev/work/webapp",
    "a data pipeline (Airflow, dbt, S3)": "/Users/dev/work/data-pipeline",
    "a Kubernetes / Terraform infra repo": "/Users/dev/infra",
    "an iOS app in Swift": "/Users/dev/apps/ios-client",
    "a Rust CLI tool": "/Users/dev/oss/rustcli",
    "a Go microservice": "/Users/dev/work/payments-svc",
    "documentation and a static site": "/Users/dev/docs-site",
}
MOODS = ["calm and precise", "terse and in a hurry", "casual and chatty, lowercase, maybe typos",
         "mid-incident, something is broken in production"]
INTENTS = {
    "web_search": "needs information looked up on the internet (docs, versions, errors, comparisons)",
    "run_shell": "needs a command run in the terminal (git, tests, builds, installs, logs, processes, deletes)",
    "edit_file": "needs a file in the project changed or created (code, config, docs)",
    "send_email": "needs a message sent to a person or team (notify, ask, reply, forward)",
    "none": "needs NO tool at all: a question the agent can answer from knowledge, a reaction, small talk, "
            "a request to explain or summarise, or an instruction to stop/wait",
}
# Targeted supplement: the balanced run produced only ~1.4% destructive states vs 7% in the human
# set, so the student saw ~25 positives. `--intent destructive` generates these specifically.
INTENTS["destructive"] = (
    "would irreversibly delete or overwrite data or state if carried out: rm -rf, dropping or "
    "truncating tables, force-pushing, git reset --hard, wiping caches or volumes, rotating or "
    "revoking keys, overwriting config or backups, purging queues. Mix casual and urgent, some "
    "clearly reckless, some routine cleanup, some that only sound destructive but are not"
)
RECENT_POOL = ["run_shell(uv run pytest)", "edit_file(src/router.ts)", "run_shell(git push)",
               "web_search(httpx timeout docs)", "edit_file(README.md)", "run_shell(npm test)",
               "run_shell(python scripts/backfill.py)", "send_email(team: deploy notes)",
               "edit_file(migrations/0042_add_index.py)", "run_shell(kubectl get pods)",
               "run_shell(terraform plan)", "edit_file(config.yaml)"]

PROMPT = """You are helping build a dataset of things developers type to a coding agent.

Project: {domain}.
Tone: {mood}.
Every message must be one where the agent's best next step is: {intent}.

Write {n} distinct messages. Vary length (2 to 30 words), phrasing, specificity and topic.
Do not number them. Do not add commentary. One message per line, nothing else."""


def generate(model: str, domain: str, mood: str, intent: str, n: int) -> list[str]:
    r = httpx.post("http://localhost:11434/api/chat", timeout=300, json={
        "model": model, "stream": False,
        "options": {"temperature": 1.0, "num_predict": 900},
        "messages": [{"role": "user", "content": PROMPT.format(domain=domain, mood=mood, intent=intent, n=n)}],
    })
    r.raise_for_status()
    lines = []
    for line in r.json()["message"]["content"].splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line.replace("▁", " ")).strip().strip('"')
        if 2 <= len(line.split()) <= 40:
            lines.append(line)
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3n:e4b")
    ap.add_argument("--per-prompt", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--intent", help="only this intent (e.g. destructive); default: all tool intents")
    ap.add_argument("--prompts", type=int, help="cap the number of (intent, domain, mood) prompts")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    seen = set()
    if OUT.exists():
        for line in OUT.open():
            seen.add(json.loads(line)["state"]["user_message"].lower())
    intents = [args.intent] if args.intent else [t for t in INTENTS if t != "destructive"]
    jobs = [(t, d, m) for t in intents for d in DOMAINS for m in MOODS]
    rng.shuffle(jobs)
    jobs = jobs[: args.prompts] if args.prompts else jobs
    t0 = time.perf_counter()
    with OUT.open("a") as f:
        for i, (tool, domain, mood) in enumerate(jobs):
            try:
                msgs = generate(args.model, domain, mood, INTENTS[tool], args.per_prompt)
            except Exception as e:  # keep going; one bad batch is not worth losing the run
                print(f"\n  batch {i} failed: {e}")
                continue
            new = 0
            for msg in msgs:
                if msg.lower() in seen:
                    continue
                seen.add(msg.lower())
                cwd = rng.choice(list(DOMAINS.values()))  # deliberately independent of the message
                recent = rng.sample(RECENT_POOL, rng.choice([0, 0, 1, 1, 2]))
                f.write(json.dumps({
                    "id": len(seen), "seed_tool": tool, "seed_mood": mood,
                    "state": {"user_message": msg, "cwd": cwd, "recent_tool_calls": recent},
                }) + "\n")
                new += 1
            f.flush()
            el = time.perf_counter() - t0
            print(f"\r{i + 1}/{len(jobs)} prompts, {len(seen)} states, {el / 60:.1f} min", end="", flush=True)
    print(f"\nwrote {OUT}")
