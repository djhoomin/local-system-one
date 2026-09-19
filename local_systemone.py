"""Local stand-in for TypeSafe's ``system_one`` call.

Same request shape (``state`` + a map of ``Choice`` / ``Score`` / ``Noul`` questions) and the
same response types as ``typesafe_sdk``, backed by a model that runs on this machine.

Backends
--------
``nli``     A zero-shot NLI encoder (DeBERTa-v3). Every option of every question becomes one
            hypothesis; all hypotheses are scored against the state in a single batched forward
            pass and softmaxed per question. No decoding at all -- the closest local analogue to
            a System One model.
``ollama``  Any chat model served by the local Ollama daemon. Each question is rendered as a
            lettered multiple-choice prompt and the answer distribution is read off the logprobs
            of the first generated token, so it costs one prefill per question, not a generation.

Usage::

    from local_systemone import LocalClient
    client = LocalClient(backend="nli")
    resp = client.system_one(state=..., questions={...})
"""

from __future__ import annotations

import json
import math
import os
import string
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

DEFAULT_NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
DEFAULT_OLLAMA_MODEL = "phi3:latest"
OLLAMA_URL = "http://localhost:11434"


# --------------------------------------------------------------------------------------------
# Question normalisation: every primitive becomes an ordered list of (label, description).
# --------------------------------------------------------------------------------------------


def _text(value: Any) -> str:
    """Render string / JSON object / list content the way the API accepts it."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _options(q: Choice | Score | Noul) -> list[tuple[str, str | None]]:
    if isinstance(q, Choice):
        return [(k, _text(v) if v is not None else None) for k, v in q.criteria.items()]
    if isinstance(q, Score):
        return [(str(i), _text(level)) for i, level in enumerate(q.criteria)]
    if isinstance(q, Noul):
        crit = q.criteria or {}
        return [
            ("yes", _text(crit["true"]) if crit.get("true") is not None else None),
            ("no", _text(crit["false"]) if crit.get("false") is not None else None),
        ]
    raise TypeError(f"unsupported question type: {type(q).__name__}")


def _entropy_confidence(probs: list[float]) -> float:
    """1 - normalised entropy: 1.0 when all mass is on one option, 0.0 when uniform."""
    if len(probs) < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, 1.0 - h / math.log(len(probs)))


def _to_answer(q: Choice | Score | Noul, labels: list[str], probs: list[float]):
    if isinstance(q, Choice):
        best = max(range(len(probs)), key=probs.__getitem__)
        return ChoiceAnswer(
            choice=labels[best],
            confidence=probs[best],
            probabilities=dict(zip(labels, probs)),
        )
    if isinstance(q, Score):
        return ScoreAnswer(
            score=sum(i * p for i, p in enumerate(probs)),
            confidence=_entropy_confidence(probs),
            legend={i: level for i, level in enumerate(q.criteria)},
            probabilities={i: p for i, p in enumerate(probs)},
        )
    return NoulAnswer(noul=probs[0])


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [e / z for e in exps]


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------


class NLIBackend:
    """Zero-shot classification via entailment. One forward pass for the whole request."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_idx = labels["entailment"]

    @staticmethod
    def _hypothesis(q: Choice | Score | Noul, label: str, desc: str | None) -> str:
        # Declarative hypotheses score far better than "The answer is <label>"; the question
        # itself goes into the premise (see _premise) so it still conditions every option.
        if isinstance(q, Noul):
            return f"{label.capitalize()}." + (f" {desc}" if desc else "")
        return f"{desc}." if desc else f"{label}."

    @staticmethod
    def _premise(state: str, q: Choice | Score | Noul) -> str:
        instr = _text(q.instructions) if q.instructions else ""
        return f"{state}\n\nQuestion: {instr}" if instr else state

    def run(self, state: str, questions: dict[str, Choice | Score | Noul]):
        import torch

        # Flatten every option of every question into one batch of (premise, hypothesis) pairs.
        spans: list[tuple[str, list[str]]] = []
        premises: list[str] = []
        hyps: list[str] = []
        for name, q in questions.items():
            opts = _options(q)
            spans.append((name, [label for label, _ in opts]))
            premises.extend([self._premise(state, q)] * len(opts))
            hyps.extend(self._hypothesis(q, label, desc) for label, desc in opts)

        enc = self.tok(
            premises, hyps, truncation="only_first", max_length=512,
            padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**enc).logits[:, self.entail_idx].tolist()

        answers, cursor = {}, 0
        for name, labels in spans:
            q = questions[name]
            probs = _softmax(logits[cursor : cursor + len(labels)])
            answers[name] = _to_answer(q, labels, probs)
            cursor += len(labels)
        usage = Usage(input_tokens=int(enc["attention_mask"].sum()), output_tokens=0)
        return answers, usage


class OllamaBackend:
    """Reads the answer distribution from the logprobs of the first generated token.

    ``style="letters"`` renders a lettered multiple-choice prompt with an ``Answer:`` prefill.
    ``style="json"`` lists the options as JSON and prefills ``{"answer": "`` so the next token is
    the option label itself -- the shape of a native tool call, which small tool-tuned models
    handle far better than an A/B/C quiz.
    """

    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, url: str = OLLAMA_URL,
                 style: str | None = None):
        import httpx

        self.name = model_name
        self.url = url
        self.http = httpx.Client(timeout=120)
        self.style = style or os.environ.get("SYSTEMONE_STYLE", "letters")
        try:  # some models (gemma3n) return an empty body here
            caps = self.http.post(f"{url}/api/show", json={"model": model_name}).json().get("capabilities", [])
        except ValueError:
            caps = []
        self.thinking = "thinking" in caps

    @staticmethod
    def _kind(q: Choice | Score | Noul) -> str:
        return {"Choice": "Pick the best option.", "Score": "Pick the level that fits best.",
                "Noul": "Answer yes or no."}[type(q).__name__]

    def _messages(self, state: str, q: Choice | Score | Noul, opts: list[tuple[str, str | None]]):
        if self.style == "json":
            options = json.dumps([{"answer": label, **({"description": desc} if desc else {})}
                                  for label, desc in opts], ensure_ascii=False)
            user = (f"STATE:\n{state}\n\nQUESTION: {_text(q.instructions)}\n{self._kind(q)}\n"
                    f"OPTIONS: {options}\n\nReply with a JSON object {{\"answer\": <one of the option "
                    f"answers>}} and nothing else.")
            return [{"role": "user", "content": user}, {"role": "assistant", "content": '{"answer": "'}]
        lines = [f"{string.ascii_uppercase[i]}) {label}" + (f" - {desc}" if desc else "")
                 for i, (label, desc) in enumerate(opts)]
        user = (f"STATE:\n{state}\n\nQUESTION: {_text(q.instructions)}\n{self._kind(q)}\n"
                + "\n".join(lines)
                + "\n\nReply with the single letter of your answer and nothing else.")
        # The assistant prefill forces the very next token to be the answer; without it some
        # models open with a newline or prose before the letter.
        return [{"role": "user", "content": user}, {"role": "assistant", "content": "Answer:"}]

    def _ask(self, state: str, q: Choice | Score | Noul):
        opts = _options(q)
        if self.style == "letters" and len(opts) > 26:
            raise ValueError("letters style supports at most 26 options per question")
        body = {
            "model": self.name, "stream": False, "logprobs": True, "top_logprobs": 20,
            # num_predict=1 makes some models (gemma3) return no logprobs at all; ask for 2
            # and read the first token's distribution.
            "options": {"temperature": 0, "num_predict": 2},
            "messages": self._messages(state, q, opts),
        }
        if self.thinking:
            body["think"] = False
        r = self.http.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        if not data.get("logprobs"):
            raise RuntimeError(f"{self.name} returned no logprobs (content={data['message']['content']!r})")
        # Credit each option with the mass on its letter or on a prefix of its label (" A",
        # " yes", "web" for web_search ...), summed over every matching token.
        labels = [label for label, _ in opts]
        mass = [0.0] * len(opts)
        for t in data["logprobs"][0]["top_logprobs"]:
            tok = t["token"].strip().strip('"\'')
            if not tok:
                continue
            for i, label in enumerate(labels):
                is_letter = self.style == "letters" and tok == string.ascii_uppercase[i]
                is_prefix = label.lower().startswith(tok.lower()) and (len(tok) >= 2 or len(label) == len(tok))
                if is_letter or is_prefix:
                    mass[i] += math.exp(t["logprob"])
        floor = 1e-6  # options absent from top-k get a small floor rather than zero
        total = sum(mass) + floor * len(opts)
        probs = [(m + floor) / total for m in mass]
        return _to_answer(q, labels, probs), data.get("prompt_eval_count", 0)

    def run(self, state: str, questions: dict[str, Choice | Score | Noul]):
        # Large models can crash the runner under concurrent requests; SYSTEMONE_WORKERS=1 serialises.
        workers = int(os.environ.get("SYSTEMONE_WORKERS", "4"))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda kv: (kv[0], self._ask(state, kv[1])), questions.items()))
        answers = {name: ans for name, (ans, _) in results}
        usage = Usage(input_tokens=sum(n for _, (_, n) in results), output_tokens=len(results))
        return answers, usage


class StudentBackend:
    """A distilled encoder with fixed heads (see distill/train_student.py).

    Unlike the other backends this is task-specific: it answers the questions it was trained on
    (matched by question *name*: tool / urgency / destructive / needs_confirmation) and ignores
    instructions and criteria. That is the trade for ~30 ms latency.
    """

    def __init__(self, model_dir: str | None = None):
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from distill.train_student import DEVICE, Student, render
        from transformers import AutoTokenizer

        default = Path(__file__).resolve().parent / "models" / "student"
        if model_dir is None:
            d, label = default, default.name
        elif Path(model_dir).exists():
            d, label = Path(model_dir), Path(model_dir).name
        else:  # a Hugging Face repo id, e.g. Mannedood/local-system-one-student
            from huggingface_hub import snapshot_download

            d = Path(snapshot_download(model_dir, allow_patterns=["config.json", "model.safetensors", "student.pt"]))
            label = model_dir
        self.cfg = json.loads((d / "config.json").read_text())
        self.name = f"{self.cfg['encoder']}@{label}"
        self.device = DEVICE
        self.render = render
        self.tok = AutoTokenizer.from_pretrained(self.cfg["encoder"])
        self.model = Student(self.cfg["encoder"])
        if (d / "model.safetensors").exists():
            from safetensors.torch import load_file

            self.model.load_state_dict(load_file(d / "model.safetensors"))
        else:
            self.model.load_state_dict(torch.load(d / "student.pt", map_location="cpu"))
        self.model.to(self.device).eval()

    def run(self, state: str, questions: dict[str, Choice | Score | Noul]):
        import torch

        unknown = [k for k in questions if k not in self.cfg["heads"]]
        if unknown:
            raise ValueError(f"student was not trained on question(s) {unknown}; has {list(self.cfg['heads'])}")
        enc = self.tok(state, truncation=True, max_length=self.cfg["max_len"], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**enc)
        answers = {}
        for name, q in questions.items():
            logits = out[name][0]
            if isinstance(q, Choice):
                probs = torch.softmax(logits, -1).tolist()
                answers[name] = _to_answer(q, self.cfg["tool_labels"], probs)
            elif isinstance(q, Score):
                probs = torch.softmax(logits, -1).tolist()
                answers[name] = _to_answer(q, [str(i) for i in range(len(probs))], probs)
            else:
                p = torch.sigmoid(logits)[0].item()
                answers[name] = NoulAnswer(noul=p)
        return answers, Usage(input_tokens=int(enc["attention_mask"].sum()), output_tokens=0)


# --------------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------------


class LocalClient:
    """Drop-in for ``TypeSafeClient`` restricted to ``system_one``."""

    def __init__(self, backend: str = "nli", model: str | None = None):
        if backend == "nli":
            self.backend = NLIBackend(model or DEFAULT_NLI_MODEL)
        elif backend == "student":
            self.backend = StudentBackend(model)
        elif backend == "ollama":
            self.backend = OllamaBackend(model or DEFAULT_OLLAMA_MODEL)
            backend = f"ollama-{self.backend.style}" if self.backend.style != "letters" else backend
        else:
            raise ValueError(f"unknown backend {backend!r}; expected 'nli', 'ollama' or 'student'")
        self.model = f"local/{backend}:{self.backend.name}"

    def system_one(self, state, questions, *, model: str | None = None, **_ignored) -> SystemOneResponse:
        answers, usage = self.backend.run(_text(state), dict(questions))
        return SystemOneResponse(model=self.model, usage=usage, answers=answers)
