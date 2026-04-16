#!/usr/bin/env bash
# train-v12-continue-18hr.sh — Continue V12 training for another 18 hours
#
# Picks up from the latest checkpoint in alphazero_checkpoints_v12/.
# Adds 3600 more generations on top of whatever gen we're at (timeout
# is the real stopping condition).
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v12-continue-18hr.sh > train-v12-continue-18hr.log 2>&1 &
#   tail -f train-v12-continue-18hr.log
#
# Dashboard (run in a separate terminal):
#   python3 ~/AJS_CTS/ClawTheSpire/dashboard.py

set -uo pipefail

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

SAVE_DIR="../alphazero_checkpoints_v12"
PROGRESS_FILE="../training_v12_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v12/boss_fights.jsonl"

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "!! No V12 checkpoints found in $SAVE_DIR"
    exit 1
fi

echo "=== STS2 AlphaZero Training V12 — 18 Hour Continuation ==="
echo "  Duration cap:  18 hours (hard timeout)"
echo "  Gen budget:    3600 (additional — timeout is the real stop)"
echo "  Games/gen:     10"
echo "  MCTS sims:     400 base (progressive: 160→720)"
echo "  Resuming from: $(basename "$LATEST_CKPT")"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+18H 2>/dev/null || date -d '+18 hours' 2>/dev/null || echo '(18 hours from now)')"
echo "-----------------------------------"

TIMEOUT_SECS=$((18 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 3600 \
    --games-per-gen 10 \
    --sims 400 \
    --batch-size 64 \
    --epochs 3 \
    --lr 1e-3 \
    --temperature 1.0 \
    --save-dir "$SAVE_DIR" \
    --progress-file "$PROGRESS_FILE" \
    --boss-log-file "$BOSS_LOG_FILE" &
TRAIN_PID=$!

(
    sleep "$TIMEOUT_SECS"
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo ""
        echo "!! 18-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$TRAIN_PID" 2>/dev/null || exit 0
            sleep 1
        done
        echo "!! Training didn't exit after 30s — sending SIGKILL"
        kill -KILL "$TRAIN_PID" 2>/dev/null || true
    fi
) &
WATCHDOG_PID=$!

cleanup() {
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$TRAIN_PID"
RC=$?

kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true

echo ""
if [ "$RC" -eq 143 ] || [ "$RC" -eq 137 ]; then
    echo "=== V12 18-hour continuation cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V12 gen budget exhausted at $(date) ==="
else
    echo "=== V12 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
