"""Hand-labeled agent-routing set. Run to regenerate data/routing.jsonl.

Each row: (user_message, cwd, recent_tool_calls, tool, urgency, destructive, needs_confirmation)
  tool      one of web_search / run_shell / edit_file / send_email / none
  urgency   0 not urgent, 1 somewhat, 2 urgent, 3 critical/outage
  destructive       1 if fulfilling the request irreversibly deletes or overwrites data/state
                    (rm, drop, truncate, force-push, reset --hard, key rotation) -- routine code
                    edits under version control do NOT count
  needs_confirmation 1 if a careful agent should check with the user before acting
Labels are one person's judgement; a few are deliberately borderline.
"""

import json
from pathlib import Path

P = "/Users/djhuman/Documents/jev"
W = "/Users/djhuman/work/webapp"
D = "/Users/djhuman/work/data-pipeline"

ROWS = [
    # ---- web_search -----------------------------------------------------------------------
    ("what's the latest stable version of uv?", P, [], "web_search", 0, 0, 0),
    ("is there a python 3.14 wheel for torch yet?", P, [], "web_search", 0, 0, 0),
    ("look up whether ollama supports logprobs in the chat api", P, ["edit_file(local_systemone.py)"], "web_search", 1, 0, 0),
    ("find the docs for httpx timeout config", W, [], "web_search", 0, 0, 0),
    ("what does error ECONNRESET usually mean in node?", W, ["run_shell(npm test)"], "web_search", 1, 0, 0),
    ("who maintains the msgspec library?", P, [], "web_search", 0, 0, 0),
    ("check if there's a known CVE for the version of pillow we use", W, [], "web_search", 2, 0, 0),
    ("what's the current pricing for claude sonnet?", P, [], "web_search", 0, 0, 0),
    ("search for the deberta-v3 tokenizer sentencepiece error", P, ["run_shell(uv run demo.py)"], "web_search", 1, 0, 0),
    ("any idea when postgres 18 goes GA?", D, [], "web_search", 0, 0, 0),
    ("google the exact wording of that pytest deprecation warning", P, ["run_shell(uv run pytest)"], "web_search", 1, 0, 0),
    ("what's the recommended way to do retries in httpx these days", W, [], "web_search", 0, 0, 0),
    ("look up the ModernBERT context length", P, [], "web_search", 0, 0, 0),
    ("find me a comparison of ECE vs brier score", P, [], "web_search", 0, 0, 0),
    ("can you check the github issues for typesafe-sdk-python for anything about 403s", P, [], "web_search", 1, 0, 0),
    ("what's the airflow release cadence?", D, [], "web_search", 0, 0, 0),
    ("is the zeroshot-v2.0 model apache licensed?", P, [], "web_search", 0, 0, 0),
    ("look up the syntax for uv workspace members", P, ["edit_file(pyproject.toml)"], "web_search", 1, 0, 0),
    ("find out if cloudflare returns 403 for missing auth headers by default", P, [], "web_search", 0, 0, 0),
    ("what's the max batch size people recommend for deberta on mps?", P, [], "web_search", 0, 0, 0),
    ("search stack overflow for that mps 'placeholder storage' error", P, ["run_shell(uv run demo.py)"], "web_search", 1, 0, 0),
    ("what's the difference between ollama's /api/chat and /api/generate?", P, [], "web_search", 0, 0, 0),
    ("look up how to disable the torch.jit deprecation warning", P, [], "web_search", 0, 0, 0),
    ("we're getting rate limited by the hf hub, what are the current limits?", P, ["run_shell(uv run demo.py)"], "web_search", 1, 0, 0),

    # ---- run_shell ------------------------------------------------------------------------
    ("run the tests", P, ["edit_file(local_systemone.py)"], "run_shell", 1, 0, 0),
    ("can you rm -rf the build dir and rerun the tests? prod is on fire", W, ["edit_file(tests/test_router.py)"], "run_shell", 3, 1, 1),
    ("git status?", P, [], "run_shell", 0, 0, 0),
    ("show me the last 20 lines of the server log", W, [], "run_shell", 1, 0, 0),
    ("install ruff and run it on the repo", P, [], "run_shell", 0, 0, 0),
    ("what python version is the venv using", P, [], "run_shell", 0, 0, 0),
    ("kill whatever is on port 8000", W, ["run_shell(uvicorn app:app)"], "run_shell", 2, 0, 1),
    ("drop the local dev database and re-run the migrations", D, [], "run_shell", 1, 1, 1),
    ("run the migration against staging", D, ["edit_file(migrations/0042_add_index.py)"], "run_shell", 2, 0, 1),
    ("clear the pip cache, disk is full", P, [], "run_shell", 2, 1, 0),
    ("ls the data dir", D, [], "run_shell", 0, 0, 0),
    ("force push my branch, i rebased", W, ["run_shell(git rebase main)"], "run_shell", 1, 1, 1),
    ("git stash and pull", W, [], "run_shell", 1, 0, 1),
    ("restart the ollama daemon", P, [], "run_shell", 1, 0, 0),
    ("time how long demo.py takes with the nli backend", P, [], "run_shell", 0, 0, 0),
    ("delete all the __pycache__ dirs", P, [], "run_shell", 0, 1, 0),
    ("nuke the venv and reinstall from the lockfile", P, [], "run_shell", 1, 1, 0),
    ("truncate the events table in prod, it's blowing up disk and we're down", D, [], "run_shell", 3, 1, 1),
    ("run the smoke tests against prod right now, customers are reporting 500s", W, [], "run_shell", 3, 0, 0),
    ("check whether the deploy finished", W, ["run_shell(git push)"], "run_shell", 2, 0, 0),
    ("run black on everything", W, [], "run_shell", 0, 0, 0),
    ("tail -f the worker logs while i reproduce it", D, [], "run_shell", 2, 0, 0),
    ("what's eating all the memory? run top or something", P, [], "run_shell", 2, 0, 0),
    ("pull the latest gemma from ollama", P, [], "run_shell", 0, 0, 0),
    ("git reset --hard origin/main, i give up on these changes", W, ["edit_file(src/router.ts)"], "run_shell", 1, 1, 1),
    ("run the eval script and paste me the table", P, [], "run_shell", 0, 0, 0),
    ("rotate the api key in the .env and restart the service", W, [], "run_shell", 2, 1, 1),

    # ---- edit_file ------------------------------------------------------------------------
    ("add a docstring to LocalClient.system_one", P, [], "edit_file", 0, 0, 0),
    ("rename OllamaBackend to LlamaBackend", P, [], "edit_file", 0, 0, 0),
    ("the retry loop in client.ts never backs off, fix it", W, ["run_shell(npm test)"], "edit_file", 2, 0, 0),
    ("bump the version in pyproject to 0.2.0", P, [], "edit_file", 0, 0, 0),
    ("fix the typo in the README, 'recieve'", P, [], "edit_file", 0, 0, 0),
    ("add gemma3 as a second default in the ollama backend", P, [], "edit_file", 0, 0, 0),
    ("the hotfix is wrong, the null check needs to be on user.id not user. patch it, deploy is blocked", W, ["run_shell(git push)"], "edit_file", 3, 0, 0),
    ("replace every print with logging in the pipeline module", D, [], "edit_file", 1, 0, 1),
    ("write a test for the softmax helper", P, [], "edit_file", 0, 0, 0),
    ("delete the dead code in utils.py", D, [], "edit_file", 0, 0, 0),
    ("add a .gitignore for the results dir", P, [], "edit_file", 0, 0, 0),
    ("make max_workers configurable", P, [], "edit_file", 0, 0, 0),
    ("comment out the flaky test for now", W, ["run_shell(npm test)"], "edit_file", 1, 0, 1),
    ("update the config so the cron runs hourly instead of daily", D, [], "edit_file", 1, 0, 1),
    ("rewrite the entire routing module in rust", W, [], "edit_file", 0, 0, 1),
    ("change the default model from phi3 to gemma3:12b", P, [], "edit_file", 0, 0, 0),
    ("add type hints to demo.py", P, [], "edit_file", 0, 0, 0),
    ("the db password is hardcoded in settings.py, move it to env", D, [], "edit_file", 2, 0, 0),
    ("make the reliability table print to markdown", P, ["edit_file(eval.py)"], "edit_file", 0, 0, 0),
    ("write a new file scripts/backfill.py that replays events from s3", D, [], "edit_file", 1, 0, 0),
    ("extract the prompt template into a constant", P, [], "edit_file", 0, 0, 0),
    ("remove the send_email tool from TOOLS, we don't use it", P, [], "edit_file", 0, 0, 0),
    ("customers can't log in, the session cookie domain in config.ts is wrong, fix it now", W, [], "edit_file", 3, 0, 0),
    ("add the calibration section to the README", P, [], "edit_file", 0, 0, 0),

    # ---- send_email -----------------------------------------------------------------------
    ("let the team know the eval results are in", P, [], "send_email", 0, 0, 1),
    ("email typesafe and ask for an invite", P, [], "send_email", 0, 0, 1),
    ("tell [the on-call engineer] prod is down and we're rolling back", W, ["run_shell(git revert HEAD)"], "send_email", 3, 0, 0),
    ("send the incident summary to the customer, they're waiting", W, [], "send_email", 2, 0, 1),
    ("ping the vendor about the invoice discrepancy", D, [], "send_email", 1, 0, 1),
    ("reply to that recruiter saying no thanks", P, [], "send_email", 0, 0, 1),
    ("send me a reminder to review the PR tomorrow", W, [], "send_email", 0, 0, 0),
    ("forward the deploy notes to the QA team", W, [], "send_email", 1, 0, 0),
    ("mail the data team that the backfill finished", D, ["run_shell(python scripts/backfill.py)"], "send_email", 1, 0, 0),
    ("draft and send an apology to all users affected by the outage", W, [], "send_email", 2, 0, 1),
    ("email hr that i'm taking friday off", P, [], "send_email", 0, 0, 1),
    ("write to the maintainer asking about the license", P, [], "send_email", 0, 0, 1),
    ("tell the client the demo is postponed", W, [], "send_email", 1, 0, 1),
    ("send the weekly status update", D, [], "send_email", 0, 0, 1),
    ("notify security that the key was leaked in the logs", W, ["run_shell(grep -r API_KEY logs/)"], "send_email", 3, 0, 0),
    ("cc legal on the vendor contract thread", D, [], "send_email", 1, 0, 1),
    ("send the eval table to my manager", P, ["run_shell(uv run eval.py)"], "send_email", 0, 0, 1),
    ("tell everyone standup is cancelled", W, [], "send_email", 1, 0, 0),
    ("email the whole customer list about the new pricing", W, [], "send_email", 0, 0, 1),
    ("let support know the login fix is deployed", W, ["run_shell(git push)"], "send_email", 2, 0, 0),
    ("ask the infra team for more ram on the eval box", P, [], "send_email", 0, 0, 1),
    ("send a thank you note to the contributor who fixed the tokenizer bug", P, [], "send_email", 0, 0, 1),
    ("reply-all that i'll take the ticket", D, [], "send_email", 0, 0, 0),
    ("email the auditor the access logs they asked for", D, [], "send_email", 1, 0, 1),

    # ---- none -----------------------------------------------------------------------------
    ("thanks, that's all for today", P, ["run_shell(uv run pytest)"], "none", 0, 0, 0),
    ("explain what ECE measures", P, [], "none", 0, 0, 0),
    ("what's the difference between an encoder and a decoder model?", P, [], "none", 0, 0, 0),
    ("nice, that worked", P, ["edit_file(local_systemone.py)"], "none", 0, 0, 0),
    ("why did you choose softmax over sigmoid there?", P, ["edit_file(local_systemone.py)"], "none", 0, 0, 0),
    ("hold on, don't do anything yet", W, [], "none", 1, 0, 0),
    ("summarize what we did this session", P, [], "none", 0, 0, 0),
    ("is it a bad idea to fine-tune on 120 examples?", P, [], "none", 0, 0, 0),
    ("ok", P, [], "none", 0, 0, 0),
    ("what would you recommend, nli or ollama?", P, [], "none", 0, 0, 0),
    ("walk me through how the logprob trick works", P, [], "none", 0, 0, 0),
    ("good morning", W, [], "none", 0, 0, 0),
    ("hmm, i'm not sure that's right. think again", P, ["edit_file(eval.py)"], "none", 0, 0, 0),
    ("what does 'calibrated' mean in this context?", P, [], "none", 0, 0, 0),
    ("never mind, i'll do it myself", W, [], "none", 0, 0, 0),
    ("can you give me a one-paragraph summary of the blog post?", P, [], "none", 0, 0, 0),
    ("how confident are you in that answer?", P, [], "none", 0, 0, 0),
    ("what's a reliability diagram?", P, [], "none", 0, 0, 0),
    ("stop", D, ["run_shell(python scripts/backfill.py)"], "none", 2, 0, 0),
    ("before you run anything, tell me what the migration will do", D, [], "none", 1, 0, 0),
    ("lol", P, [], "none", 0, 0, 0),
    ("which of those two options is cheaper?", P, [], "none", 0, 0, 0),
    ("i'll be back after lunch", W, [], "none", 0, 0, 0),
    ("what did the last test run say?", P, ["run_shell(uv run pytest)"], "none", 0, 0, 0),
]

if __name__ == "__main__":
    out = Path(__file__).with_name("routing.jsonl")
    with out.open("w") as f:
        for i, (msg, cwd, recent, tool, urg, destr, conf) in enumerate(ROWS):
            f.write(json.dumps({
                "id": i,
                "state": {"user_message": msg, "cwd": cwd, "recent_tool_calls": recent},
                "tool": tool, "urgency": urg, "destructive": destr, "needs_confirmation": conf,
            }) + "\n")
    from collections import Counter
    print(f"wrote {len(ROWS)} rows to {out}")
    print("tool:", dict(Counter(r[3] for r in ROWS)))
    print("urgency:", dict(sorted(Counter(r[4] for r in ROWS).items())))
    print("destructive:", dict(Counter(r[5] for r in ROWS)), " needs_confirmation:", dict(Counter(r[6] for r in ROWS)))
