"""Native function-calling accuracy on the routing set (no distribution -- just which tool the
model calls, or none). The fair test for models post-trained on tool use.

    uv run eval_toolcall.py lfm2.5:1.2b [more models...]
"""

import json
import sys
import time
from collections import Counter

import httpx

from demo import TOOLS
from eval import DATA

SYSTEM = ("You are a coding agent. Given the current state, call the single most appropriate tool "
          "for the user's latest message. If no tool is needed, reply in plain text instead.")
tools = [{"type": "function", "function": {"name": k, "description": v,
          "parameters": {"type": "object", "properties": {}, "required": []}}}
         for k, v in TOOLS.items() if k != "none"]


NATURAL = "--natural" in sys.argv  # message as the user turn, context in the system prompt


def call(model: str, state: dict) -> str:
    if NATURAL:
        ctx = (f"Working directory: {state['cwd']}\n"
               f"Recent tool calls: {', '.join(state['recent_tool_calls']) or 'none'}")
        messages = [{"role": "system", "content": SYSTEM + "\n\n" + ctx},
                    {"role": "user", "content": state["user_message"]}]
    else:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(state)}]
    r = httpx.post("http://localhost:11434/api/chat", timeout=120, json={
        "model": model, "stream": False, "tools": tools, "think": False,
        "options": {"temperature": 0, "num_predict": 80},
        "messages": messages,
    }).json()
    calls = r.get("message", {}).get("tool_calls") or []
    return calls[0]["function"]["name"] if calls else "none"


for model in [a for a in sys.argv[1:] if not a.startswith("--")]:
    rows = [json.loads(l) for l in DATA.open()]
    t0 = time.perf_counter()
    preds = [call(model, row["state"]) for row in rows]
    ms = (time.perf_counter() - t0) * 1000 / len(rows)
    ok = [p == r["tool"] for p, r in zip(preds, rows)]
    errs = Counter((r["tool"], p) for p, r in zip(preds, rows) if p != r["tool"])
    print(f"{model:<16} {'natural' if NATURAL else 'json   '} native tool-call acc={sum(ok) / len(ok):.2f}  {ms:.0f}ms/state  "
          f"pred dist={dict(Counter(preds))}")
    print(f"{'':<16} top errors: " + ", ".join(f"{g}->{p} x{n}" for (g, p), n in errs.most_common(4)))
