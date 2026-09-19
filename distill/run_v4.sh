#!/bin/zsh
# Waits for the DeBERTa run, then: 300 targeted destructive states -> label (3n + 12B) -> ModernBERT v4
set -e
cd "$(dirname "$0")/.."
while pgrep -f "distill/run_deberta.sh" > /dev/null; do sleep 20; done
BEFORE=$(wc -l < data/synthetic_states.jsonl)
echo "=== gen destructive supplement ($(date +%H:%M))"
uv run distill/gen_states.py --intent destructive --prompts 16 --per-prompt 20 --seed 7
AFTER=$(wc -l < data/synthetic_states.jsonl); NEW=$((AFTER-BEFORE)); echo "new states: $NEW"
echo "=== label supplement: small ($(date +%H:%M))"
uv run distill/label_states.py --phase small --tail $NEW
ollama stop gemma3n:e4b || true
for ph in tool confirm destructive; do
  echo "=== label supplement: $ph ($(date +%H:%M))"
  uv run distill/label_states.py --phase $ph --tail $NEW
done
ollama stop gemma3:12b-it-qat || true
sleep 5
echo "=== train modernbert v4 ($(date +%H:%M))"
uv run distill/train_student.py --epochs 4 --max-len 128 --out models/student-v4
uv run eval.py --backend student --model models/student-v4
mv results/student-student-v4.jsonl results/student-v4.jsonl 2>/dev/null || true
uv run eval.py --report | grep -E "^\| (backend|---|student|typesafe)"
uv run calibrate.py | grep -A4 "^=== student-v4"
echo "=== done ($(date +%H:%M))"
