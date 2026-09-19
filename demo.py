"""System One smoke test: tool routing, scoring, and guardrail checks in one call.

Backend is chosen by SYSTEMONE_BACKEND:
    typesafe  hosted Jev (needs TYPESAFE_API_KEY)
    nli       local zero-shot DeBERTa encoder (default when no key is set)
    ollama    local LLM via Ollama, first-token logprobs (SYSTEMONE_MODEL picks the model)

    uv run demo.py
    SYSTEMONE_BACKEND=ollama SYSTEMONE_MODEL=phi3 uv run demo.py
"""

import os
import time

from dotenv import load_dotenv
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

load_dotenv()

TOOLS = {
    "web_search": "Look something up on the internet",
    "run_shell": "Execute a command in the user's terminal",
    "edit_file": "Modify a file in the user's project",
    "send_email": "Send an email on the user's behalf",
    "none": "No tool needed; a plain reply is enough",
}

STATES = [
    {
        "user_message": "can you rm -rf the build dir and rerun the tests? prod is on fire",
        "cwd": "/Users/djhuman/Documents/jev",
        "recent_tool_calls": ["edit_file(tests/test_router.py)"],
    },
    {
        "user_message": "what's the latest stable version of uv?",
        "cwd": "/Users/djhuman/Documents/jev",
        "recent_tool_calls": [],
    },
    {
        "user_message": "thanks, that's all for today",
        "cwd": "/Users/djhuman/Documents/jev",
        "recent_tool_calls": ["run_shell(uv run pytest)"],
    },
]

QUESTIONS = {
    "tool": Choice(
        instructions="Which tool should the agent call next to satisfy the user?",
        criteria=TOOLS,
    ),
    "urgency": Score(
        instructions="How urgent is the user's request?",
        criteria=["not urgent", "somewhat urgent", "urgent", "critical / outage"],
    ),
    "destructive": Noul(
        instructions="Would fulfilling this request delete or overwrite data?",
    ),
    "needs_confirmation": Noul(
        instructions="Should the agent confirm with the user before acting?",
    ),
}


def make_client():
    backend = os.environ.get("SYSTEMONE_BACKEND") or (
        "typesafe" if os.environ.get("TYPESAFE_API_KEY") else "nli"
    )
    if backend == "typesafe":
        from typesafe_sdk import TypeSafeClient

        return TypeSafeClient()
    from local_systemone import LocalClient

    return LocalClient(backend=backend, model=os.environ.get("SYSTEMONE_MODEL"))


def fmt(ans) -> str:
    if isinstance(ans, ChoiceAnswer):
        dist = ", ".join(f"{k} {v:.2f}" for k, v in sorted(ans.probabilities.items(), key=lambda kv: -kv[1]))
        return f"{ans.choice:<12} conf={ans.confidence:.2f}   [{dist}]"
    if isinstance(ans, ScoreAnswer):
        dist = ", ".join(f"{ans.legend[k]} {v:.2f}" for k, v in ans.probabilities.items())
        return f"{ans.score:<12.2f} conf={ans.confidence:.2f}   [{dist}]"
    if isinstance(ans, NoulAnswer):
        return f"{ans.noul:.2f}"
    return repr(ans)


if __name__ == "__main__":
    t0 = time.perf_counter()
    client = make_client()
    print(f"backend ready in {time.perf_counter() - t0:.1f}s\n")

    for state in STATES:
        t0 = time.perf_counter()
        resp = client.system_one(state=state, questions=QUESTIONS)
        ms = (time.perf_counter() - t0) * 1000
        print(f'> "{state["user_message"]}"')
        print(f"  model={resp.model}  latency={ms:.0f}ms  input_tokens={resp.usage.input_tokens}")
        for name, ans in resp.answers.items():
            print(f"  {name:<20} {fmt(ans)}")
        print()
