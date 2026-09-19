#!/bin/zsh
# 12B relabel of destructive -> retrain ModernBERT on v3 -> eval  (DeBERTa handled separately)
set -e
cd "$(dirname "$0")/.."
echo "=== phase destructive ($(date +%H:%M))"
uv run distill/label_states.py --phase destructive --limit 2000
ollama stop gemma3:12b-it-qat || true
sleep 5
echo "=== train modernbert v3 ($(date +%H:%M))"
uv run distill/train_student.py --epochs 4 --max-len 128 --out models/student-v3
uv run eval.py --backend student --model models/student-v3
echo "=== eval ($(date +%H:%M))"
uv run eval.py --report
uv run calibrate.py | grep -A4 "^=== student"
echo "=== done ($(date +%H:%M))"
