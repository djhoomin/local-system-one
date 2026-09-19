#!/bin/zsh
# Waits for run_v3.sh, then trains DeBERTa-v3-base on the v3 labels with the GPU to itself.
set -e
cd "$(dirname "$0")/.."
while pgrep -f "distill/run_v3.sh" > /dev/null; do sleep 20; done
ollama stop gemma3:12b-it-qat 2>/dev/null || true
echo "=== train deberta v3 ($(date +%H:%M))"
uv run distill/train_student.py --encoder microsoft/deberta-v3-base --epochs 4 --max-len 128 --out models/student-deberta-v3
uv run eval.py --backend student --model models/student-deberta-v3
uv run eval.py --report | grep -E "^\| (backend|---|student)"
uv run calibrate.py | grep -A4 "^=== student"
echo "=== done ($(date +%H:%M))"
