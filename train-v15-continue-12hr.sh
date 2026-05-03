#!/usr/bin/env bash
# train-v15-continue-12hr.sh — Continue V15 training, 12-hour run
#
# Continues from the latest V15 checkpoint (does NOT wipe or re-seed from V14).
# Includes the card pool corruption fix (copy.copy at card_db boundaries).
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v15-continue-12hr.sh > train-v15-continue-12hr.log 2>&1 &
#   tail -f train-v15-continue-12hr.log
#
# Dashboard (run in a separate terminal):
#   bash ~/AJS_CTS/ClawTheSpire/restart-dashboard.sh

set -uo pipefail

cd "$(dirname "$0")/sts2-solver"

# ── Kill any existing training processes ──────────────────────────────
EXISTING_PIDS=$(pgrep -f "sts2_solver.alphazero.self_play train" 2>/dev/null || true)
if [ -n "$EXISTING_PIDS" ]; then
    echo "!! Found existing training process(es): $EXISTING_PIDS"
    echo "   Sending SIGTERM..."
    for pid in $EXISTING_PIDS; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 15); do
        REMAINING=$(pgrep -f "sts2_solver.alphazero.self_play train" 2>/dev/null || true)
        [ -z "$REMAINING" ] && break
        sleep 1
    done
    REMAINING=$(pgrep -f "sts2_solver.alphazero.self_play train" 2>/dev/null || true)
    if [ -n "$REMAINING" ]; then
        echo "   Still alive after 15s — sending SIGKILL to: $REMAINING"
        for pid in $REMAINING; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
    echo "   Previous training stopped."
else
    echo "No existing training processes found."
fi

source .venv/bin/activate

SAVE_DIR="../alphazero_checkpoints_v15"
PROGRESS_FILE="../training_v15_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v15/boss_fights.jsonl"

mkdir -p "$SAVE_DIR"

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No V15 checkpoint found in $SAVE_DIR — use train-v15-18hr.sh to start fresh"
    exit 1
fi

echo "=== STS2 AlphaZero Training V15 — Continue 12 Hours ==="
echo "  Duration cap:  12 hours (hard timeout)"
echo "  Gen budget:    3200"
echo "  Games/gen:     8"
echo "  MCTS sims:     500 base (progressive scaling)"
echo "  Start:         Resuming from $(basename "$LATEST_CKPT")"
echo "  Batch size:    64"
echo "  Epochs:        3"
echo "  LR:            1e-4"
echo "  Bug fix:       Card pool corruption (copy.copy at card_db boundaries)"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+12H 2>/dev/null || date -d '+12 hours' 2>/dev/null || echo '(12 hours from now)')"
echo "-----------------------------------"

TIMEOUT_SECS=$((12 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 3200 \
    --games-per-gen 8 \
    --sims 500 \
    --batch-size 64 \
    --epochs 3 \
    --lr 1e-4 \
    --temperature 1.0 \
    --save-dir "$SAVE_DIR" \
    --progress-file "$PROGRESS_FILE" \
    --boss-log-file "$BOSS_LOG_FILE" &
TRAIN_PID=$!

(
    sleep "$TIMEOUT_SECS"
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo ""
        echo "!! 12-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
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
    echo "=== V15 12-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V15 gen budget exhausted before 12-hour cap at $(date) ==="
else
    echo "=== V15 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
