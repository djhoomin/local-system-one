#!/bin/zsh
# Wait for gen_states.py to finish, then label, train and evaluate the student.
set -e
cd "$(dirname "$0")/.."
LIMIT=${LIMIT:-2000}
[[ "$SKIP_GEN_WAIT" == 1 ]] || while pgrep -f "distill/gen_states.py" > /dev/null; do sleep 15; done
echo "=== states: $(wc -l < data/synthetic_states.jsonl)"
echo "=== phase small ($(date +%H:%M))"
uv run distill/label_states.py --phase small --limit $LIMIT
echo "=== phase tool ($(date +%H:%M))"
uv run distill/label_states.py --phase tool --limit $LIMIT
echo "=== train ($(date +%H:%M))"
uv run distill/train_student.py --epochs 4 --max-len 128
echo "=== eval ($(date +%H:%M))"
uv run eval.py --backend student
uv run calibrate.py | grep -A4 "^=== student"
echo "=== done ($(date +%H:%M))"
