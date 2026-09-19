# local-system-one

*"We have Jev at home."*

A thought experiment, not a product. TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
is a "System One" model: state in, typed decisions with calibrated probabilities out, no text
generation. Its architecture is not public; what is observable is the interface and the
behaviour, which is what this repo is about. It was invite-only when this started, so the question
was: how far does the same *interface* get you with models that run on a laptop — and what does it
take to get calibration and latency in the same place? Nothing here is a claim about what Jev *is*
internally, nor of parity with it, and I have no affiliation with TypeSafe;
everything said about Jev is from their public docs and blog, quoted as of Sep 2026.

`local_systemone.py` reproduces the `system_one(state, questions)` interface — same `Choice` /
`Score` / `Noul` question types and the same `typesafe_sdk` answer types — on top of local models,
so anything written against the SDK runs unchanged against either.

## Run

```bash
uv run demo.py                                            # local NLI encoder (DeBERTa-v3 zero-shot)
SYSTEMONE_BACKEND=ollama SYSTEMONE_MODEL=phi3 uv run demo.py   # local LLM, first-token logprobs
TYPESAFE_API_KEY=... uv run demo.py                       # hosted Jev, once you have a key
```

## Backends

| backend   | model                                      | how it decides                                                         | latency (M1 Pro) | quality on demo |
|-----------|--------------------------------------------|------------------------------------------------------------------------|------------------|-----------------|
| `nli`     | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | one batched entailment pass over every option of every question, no decoding | ~130–300 ms / state | guardrails OK, routing weak on structured state |
| `ollama`  | any Ollama chat model (`phi3` default)     | lettered multiple-choice prompt, distribution read from first-token logprobs | ~1 s / state (4 q's) | all 12 demo decisions correct; probabilities overconfident |
| `typesafe`| `jev-latest`                               | the real thing                                                         | 70–500 ms claimed | withheld, see below |

The `nli` backend is the closest match to how Jev *behaves* (every answer scored in one pass,
probabilities from a head rather than from sampling) — whether that is how Jev is built is not
public — but a 180M zero-shot model doesn't read JSON state well. The `ollama`
backend understands the state but its probabilities are near-binary — calibration is the thing
Jev is selling and neither stand-in has it.

## Routing set and calibration eval

`data/routing.jsonl` is 123 hand-labeled agent states (balanced over the 5 tools, plus urgency /
destructive / needs-confirmation labels; regenerate with `uv run data/build_routing_set.py`).

```bash
uv run eval.py --backend nli                        # runs a backend, caches results/<backend>.jsonl
uv run eval.py --backend ollama --model phi3
SYSTEMONE_WORKERS=1 uv run eval.py --backend ollama --model gemma3:12b-it-qat   # big models: serialise
SYSTEMONE_STYLE=json uv run eval.py --backend ollama --model lfm2.5:1.2b        # tool-call idiom prompt
uv run eval.py --report                             # accuracy + ECE/Brier/AUROC + reliability bins
uv run calibrate.py                                 # post-hoc temperature / Platt scaling, 5-fold CV
```

`eval.py` re-joins gold labels from the data file at report time, so relabeling never requires
re-running a model. See the results section below for what we found.

## Results (123 states, 2026-09-18)

Raw backend outputs, then after post-hoc calibration (temperature scaling for `tool`, Platt for
the nouls; 5-fold CV so the numbers are held-out). Ollama 0.34.2; `json` = the option list is
rendered as JSON and the prefill is `{"answer": "` (tool-call idiom) instead of an A–E quiz.

| backend | ms/state | tool acc | tool ECE raw → cal | urgency ±1 | destructive AUROC | needs_confirm AUROC | needs_confirm Brier raw → cal (base 0.18) |
|---|---|---|---|---|---|---|---|
| nli (DeBERTa-v3 180M) | 123 | 0.50 | 0.08 → 0.06 | 0.38 | 0.72 | 0.47 | 0.30 → 0.19 |
| gemma3 270m, letters | 441 | 0.12 | 0.37 → 0.08 | 0.41 | 0.68 | 0.47 | 0.32 → 0.19 |
| gemma3 270m, json | 133 | 0.20 | 0.80 → 0.03 | 0.41 | 0.77 | 0.51 | 0.75 → 0.19 |
| granite 4.2 3b, letters | 875 | 0.53 | 0.35 → 0.13 | 0.91 | 0.93 | 0.42 | 0.65 → 0.19 |
| granite 4.2 3b, json | 1047 | 0.67 | 0.30 → 0.06 (T=4.3) | 0.79 | 0.90 | 0.43 | 0.70 → 0.19 |
| lfm2.5 350m, letters | 144 | 0.32 | 0.48 → 0.08 | 0.26 | 0.75 | 0.53 | 0.20 → 0.19 |
| lfm2.5 350m, json | 170 | 0.31 | 0.41 → 0.11 | 0.41 | 0.64 | 0.57 | 0.76 → 0.19 |
| lfm2.5 1.2b, letters | 379 | 0.35 | 0.46 → 0.10 | 0.33 | 0.88 | 0.52 | 0.22 → 0.19 |
| lfm2.5 1.2b, json | 476 | 0.47 | 0.37 → 0.14 | 0.47 | 0.88 | 0.50 | 0.72 → 0.19 |
| gemma3n e4b, letters | 2062 | 0.72 | 0.28 → 0.06 (T=4.4) | 0.99 | 0.95 | 0.71 | 0.76 → 0.17 |
| gemma3n e4b, json | 2423 | 0.67 | 0.31 → 0.08 | 0.99 | 0.88 | 0.73 | 0.73 → — |
| gemma3 12b, letters | 2800–5500 | 0.83 | 0.15 → 0.06 (T=2.1) | 0.86 | 0.94 | 0.78 | 0.70 → 0.16 |

Native function calling (`uv run eval_toolcall.py`), routing accuracy only — the test the
tool-use post-training is actually aimed at. `json` = whole state as the user turn; `natural` =
message as the user turn with cwd / recent calls in the system prompt:

| model | json state | natural framing |
|---|---|---|
| lfm2.5 350m | 0.29 | 0.24 |
| lfm2.5 1.2b | 0.50 | 0.30 |
| granite 4.2 3b | 0.41 | 0.45 |

What that says:

- **Raw LLM logprobs are useless as probabilities.** Every LLM puts most states at confidence
  ≥0.99 regardless of accuracy. One fitted scalar (T) fixes this: ECE drops to ~0.06, on par
  with the NLI encoder's natural calibration, at much higher accuracy for the larger models.
- **Gemma 3 270M cannot do this task through any prompt.** In the lettered format it always picks
  the last option (reverse the list and its answer flips); in the JSON/tool-call format it always
  picks the first (`web_search` 123/123). Its tool-calling post-training is about emitting the
  call syntax, not about judging which tool fits a JSON state. Below chance on routing.
- **LFM2.5 (350M / 1.2B) tracks the input but under-reads it.** Unlike gemma-270M its answers
  vary with the state, but routing tops out at 0.47 (1.2B, JSON style) — below the NLI encoder —
  with the same `run_shell` pull from the context fields (92/123 states in letters mode). Native
  function calling, the thing it was post-trained for, does no better (0.50), and the "natural"
  chat framing makes it mostly decline to call anything. `destructive` is decent on the 1.2B
  (AUROC 0.88). At 150–500 ms it is the fastest thing here that isn't guessing, but it is not a
  zero-shot router.
- **Granite 4.2 3B reads the state but under-routes**: 0.67 in JSON style, with `send_email → none`
  (14) and `web_search → none` (6) as the dominant errors — it defaults to "no tool" when unsure.
  The JSON idiom is worth +14 points over letters for a tool-tuned model. Excellent on
  `destructive` (AUROC 0.90–0.93).
- **Gemma 3n E4B is the best small model by a wide margin**: 0.72 routing, the best urgency
  (±1: 0.99) and `destructive` (AUROC 0.95) of anything tested, and real `needs_confirmation`
  signal (AUROC 0.71) — the only model besides the 12B with any. Its error profile is the 12B's
  in miniature (`none → web_search`, `edit_file → run_shell`), just more of each. It loads as
  3 GB fully on GPU where the 12B spills to CPU on 16 GB, so it is ~2.7× faster here. The JSON
  idiom does *not* help it (0.67), unlike Granite.
- **Gemma 3 12B is the best local router** at 0.83; errors are mostly `none → web_search` on
  conceptual questions, which is partly a label-definition issue. It is also the only model with
  any signal on `needs_confirmation` (AUROC 0.78), and the only one where Platt scaling beats
  the base rate on it — barely.
- **The NLI encoder is well-calibrated but not accurate** — a single systematic bias toward
  `run_shell` driven by `cwd` / `recent_tool_calls` in the state. That kind of bias is what a
  fine-tune fixes trivially.
- **`destructive` is solved** by any ≥3B LLM once the label means irreversible loss rather than
  "touches a file". Only 9 positives, so treat the number as indicative.
- **`needs_confirmation` is the hard one** for everything below 12B: no signal, and post-hoc
  calibration can only collapse it to the base rate. It is also the most subjective label.

**Bottom line:** calibrated typed decisions are reachable locally with a ≥3B LLM + a 100-example
calibration set + one scalar, at ~1–3 s/state; gemma3n-e4b is the sweet spot on a 16 GB machine.
And with ~2k teacher-labeled synthetic states, a 149M encoder gets 82% of the 12B's routing
accuracy, better calibration than any zero-shot model, at 19 ms — see Distillation below. Nothing in DeBERTa's weight class reads the state
zero-shot — not the NLI encoder, gemma-270M, or LFM2.5-350M — and LFM2.5-1.2B doesn't either, so reaching Jev's 70–500 ms with LLM-level
accuracy means a fine-tuned encoder, and a 123-row set is too small to train one for four tasks
at once. The credible route is to pseudo-label a few thousand synthetic states with gemma-12B and
distil into ModernBERT (or DeBERTa), keeping the human labels purely for evaluation.

## Distillation (the fine-tune)

Everything under 3B failed zero-shot through systematic bias, so the latency-class model has to be
trained. `distill/` does that without touching the human labels:

```bash
uv run distill/gen_states.py                       # gemma3n writes ~3k synthetic states (tool × domain × mood seeds)
uv run distill/label_states.py --phase small       # gemma3n: urgency / destructive / needs_confirmation, calibrated
uv run distill/label_states.py --phase tool        # gemma3 12b: tool, calibrated (serial, ~1.4 s/state)
uv run distill/label_states.py --phase confirm     # gemma3 12b: needs_confirmation (overrides gemma3n's)
uv run distill/train_student.py --max-len 128      # ModernBERT-base + 4 heads on the soft labels
uv run eval.py --backend student                   # scored on the 123 human rows like any other backend
./distill/run_pipeline.sh                          # all of the above, chained
```

Design choices:
- **Targets are the teacher's *calibrated* distributions** (temperature / Platt from
  `results/calibration.json`), so the student is distilled toward calibrated probabilities rather
  than the teacher's raw logprob spikes.
- **`cwd` / `recent_tool_calls` are sampled independently of the message** in the synthetic set, so
  the student cannot learn the context-field shortcut that sank the zero-shot small models.
- **Model selection uses a 5% synthetic dev split only.** The 123 human rows are reported per
  epoch for information but never trained or selected on; they remain the test set.
- The student is **task-specific**: it answers the four questions it was trained on by name and
  ignores instructions/criteria. That is the trade for encoder latency; Jev is general.
- **Fixed schema, one domain.** Inputs must be the JSON shape it was trained on (`user_message`,
  `cwd`, `recent_tool_calls`); rename or add a key and it is out of distribution without any
  error. All training states were a coding agent's — a support or ops message in the same shape
  still gets an answer with plausible-looking probabilities, but nothing here says those are
  calibrated. Encoders degrade quietly, not loudly: inside the box it is a 19 ms calibrated
  function; outside it, a guess with a confident face.

### Student results (ModernBERT-base, 149M, 1,900 synthetic states, 4 epochs, 6 min on MPS)

| model | ms/state | tool acc | tool ECE raw | tool NLL raw | urgency ±1 | destructive AUROC | needs_confirm AUROC |
|---|---|---|---|---|---|---|---|
| NLI DeBERTa zero-shot | 123 | 0.50 | 0.08 | 1.40 | 0.38 | 0.72 | 0.47 |
| gemma3n E4B (teacher: urgency/nouls) | 2062 | 0.72 | 0.28 | 2.45 | 0.99 | 0.95 | 0.71 |
| gemma3 12B (teacher: tool) | 5508 | 0.83 | 0.15 | 0.71 | 0.86 | 0.94 | 0.78 |
| student v1 (needs_confirm from gemma3n) | 19 | 0.68 | 0.07 | 0.82 | 0.99 | 0.98 | 0.42 |
| student v2 (needs_confirm from 12B) | 19 | 0.72 | 0.12 | 0.79 | 0.99 | 0.97 | 0.62 |
| student v3 (+ destructive from 12B) | 19 | 0.67 | 0.08 | 0.79 | 0.99 | 0.97 | 0.64 |
| **student v4 (+ 317 destructive-seeded states)** | **19** | **0.76** | 0.08 | **0.76** | 0.98 | **0.99** | **0.67** |
| DeBERTa-v3-base student, 2 of 4 epochs (see below) | 27 | 0.46 | 0.23 | 1.24 | 1.00 | 0.90 | 0.74 |

- **Calibration transferred.** The student's raw softmax is the best-calibrated routing output of
  any backend (ECE 0.07, reliability 0.93→0.95, 0.46→0.50); fitting a temperature gives T≈0.9,
  i.e. nothing to fix. Distilling from temperature-scaled teacher targets yields a model that is
  calibrated by construction — no post-hoc step. This reproduces the System One *behaviour* in
  miniature; whether it resembles the System One recipe is not knowable from outside.
- **19 ms/state**: 100× faster than gemma3n, ~300× faster than the 12B, in DeBERTa's weight class.
- **Routing 0.68** ≈ 82% of the 12B teacher; student agrees with the teacher on 86% of synthetic
  dev, so there is headroom in fidelity (more data / epochs), not just in the teacher ceiling.
  Epochs swung 0.56–0.73 on the human set, so ±4 pt is noise at n=123.
- **`destructive` AUROC 0.98** and urgency ±1 0.99 — best or tied-best of anything.
- **`needs_confirmation` needed the better teacher.** With gemma3n's labels (v1) it did not
  transfer at all (AUROC 0.42): the Platt fit (a=0.42, b=−6.2) squashed an already weak signal to
  the base rate and the student learned a constant ~0.25. Re-labeling that one head with the 12B
  (`--phase confirm`, ~40 min) took it to 0.62 with a monotone reliability curve — about half the
  gap to the teacher's 0.78. The v1→v2 routing change (0.68→0.72) is seed noise on identical
  tool labels.
- **`destructive` inherited a compressed scale.** Same mechanism, milder: gemma3n's Platt slope for
  it (a=0.31, fitted on 9 human positives) left only 1.7% of synthetic targets above 0.5, so the
  student ranks correctly (AUROC 0.97) but the positive class tops out near 0.5 — threshold at
  ~0.15, or re-label with the 12B. Lesson: Platt on a handful of positives over-regularises;
  a floor on the slope (or temperature-only scaling for nouls) would avoid squashing the teacher.
- Remaining errors: `web_search → run_shell` (×11, residual context over-weighting) and
  `none → web_search` (×9, inherited from the 12B).

### v3 → v4: the destructive head was a data problem, not a teacher problem

Re-labeling `destructive` with the 12B (v3) barely moved it, because only **28 of 2,000 synthetic
states (1.4%)** were destructive at all — gemma3n's `run_shell` seed mostly produced benign
commands — versus 7% in the human set. ~25 positives is not enough to learn a scale from.
`gen_states.py --intent destructive` adds 317 states seeded on irreversible actions (rm -rf,
drop/truncate, force-push, key rotation, plus decoys that only sound destructive). Trained on the
2,300-row set, v4's positives score 0.28–0.93 against a negative median of 0.01 (Brier 0.026,
Platt fit ≈ identity) — the compression is gone — and routing rose to 0.76, 92% of the 12B teacher.

### DeBERTa-v3-base as the student: no

Three separate problems on an M1 Pro. transformers 5 loads it in fp16 (fixed with
`dtype=torch.float32`); its disentangled-attention kernels crash MPS whenever the GPU is shared
with Ollama; and even alone it grew to ~10 GB of Metal allocations and swapped the trainer out
mid-epoch 3, at ~12× ModernBERT's step time. It trailed ModernBERT at every epoch it completed
(0.46 vs 0.69 routing at epoch 2). Its half-trained checkpoint is in the table for the record.

### Jev itself

API access arrived once the local work was done, and Jev was run through the identical harness
(`uv run eval.py --backend typesafe`). Those results are **withheld from this repository**:
TypeSafe's customer agreement restricts publishing benchmarks or performance information about
the service, and permission to include them has been requested. If granted, the Jev row will be
added to the tables above and its predictions to `results/`.

Where that leaves the meme — they aren't the same product. **Jev is general**: any state, any
question schema, one-shot, with no training data, labels or GPU (per its public docs). Nothing
local has that, which is why this repo needed a distillation pipeline at all. **The student is
specific**: four fixed heads, ~2k teacher-labeled states and an afternoon — and in return, 19 ms
on-device, no network, no per-call cost, nothing leaving the machine. Jev is the default; the
student is what you build when you're latency-bound or air-gapped. (Hyperparameters were never
explored — one config, dev loss still falling at every checkpoint — so the student's remaining gap
to its teachers is partly headroom, not ceiling.)

### Two limits, two different levers

The student's limits are not one thing:

- **Domain is a data problem.** More varied states — support tickets, ops runbooks, other key sets —
  under the same four heads is just labeling budget; a few thousand more teacher-labeled states is
  an afternoon. Nothing structural stops the box from getting wider.
- **Schema is an architecture problem.** No amount of data lets four fixed heads answer a fifth
  question, because question text never enters the model. Making it general means making the
  question an *input*: encode state and question together (a cross-encoder), or encode them
  separately and score each option against the state (a bi-encoder — roughly what the zero-shot
  NLI backend was doing, badly). That is a model that reads arbitrary criteria and returns
  calibrated distributions over arbitrary option lists — and *that* is the thing that plausibly
  needs Jev-scale data and whatever RLCD actually is.

Data buys a wider box; a general box is a different model. The student can grow sideways almost
for free, but it can't grow up without becoming the thing it was imitating.

Which is the odd shape of the whole exercise: architecturally this is going *backwards* — BERT
with a classification head, four logits, no generation, the thing everyone abandoned once decoders
could answer anything. What changed isn't the architecture but what feeds it: a decoder big enough
to label any schema, and a training signal that makes a small model's probabilities mean
something. The encoder was never the limitation; the cost of teaching it a new task was. Going
backwards is a step forward once that cost drops to nothing — and a general System One model is
what the old shape looks like when it drops to zero.

Next steps if pushing further: label the remaining 285 balanced states, and an lr/epoch sweep —
the dev loss was still falling at the last checkpoint of every run.

## Demo page

`demo/index.html` is a self-contained interactive demo: pick one of ten real test states, switch
between the student and its teachers, and the response arrives after each model's *measured*
latency — the student is instant, the 12B makes you wait 5.5 s. All outputs are the cached
predictions from `results/`, not illustrations. Serve it locally (`python3 -m http.server`) or
open it from the published copy.

## Files





- `local_systemone.py` — `LocalClient(backend=...)` with `.system_one()`; returns real `SystemOneResponse`
- `demo.py` — tool routing + urgency score + two guardrail checks over three agent states
- `eval.py` / `calibrate.py` — routing-set evaluation and post-hoc calibration
- `eval_toolcall.py` — native function-calling accuracy for tool-tuned models
- `distill/` — synthetic state generation, teacher labeling, student training, pipeline script
- `demo/index.html` — interactive demo of the call, with real cached outputs and measured latencies
- `data/build_routing_set.py` — the labeled set as code
- `.env.example` — `TYPESAFE_API_KEY` goes in `.env`

## Models not in the Ollama library

LFM2.5 350M / 1.2B were imported from Hugging Face GGUFs (`ollama pull hf.co/...` fails on the
xet CDN redirect): download the Q8_0 file, then `ollama create lfm2.5:1.2b -f Modelfile` with
`FROM ./LFM2.5-1.2B-Instruct-Q8_0.gguf`. Ollama picks up the chat template from the GGUF.
Ollama ≥0.34 is needed for Granite 4.2 (thinking + tools template); on 0.15 it ran as a base model.

## Provenance and license

Code and the hand-written `data/routing.jsonl` are MIT. `data/synthetic_states.jsonl` was
generated with Gemma 3n and its labels come from Gemma 3 12B / 3n, so the Gemma terms of use
apply to that data. The 123 human labels are one person's judgement on a made-up task and should
be read as a benchmark of the *method*, not of the models' general ability.
