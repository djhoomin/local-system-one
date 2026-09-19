#!/bin/zsh
set -e
cd "$(dirname "$0")/.."
echo "=== phase confirm ($(date +%H:%M))"
uv run distill/label_states.py --phase confirm --limit 2000
ollama stop gemma3:12b-it-qat || true
sleep 5
echo "=== train ($(date +%H:%M))"
uv run distill/train_student.py --epochs 4 --max-len 128
echo "=== eval ($(date +%H:%M))"
uv run eval.py --backend student
uv run calibrate.py | grep -A4 "^=== student"
echo "=== done ($(date +%H:%M))"
