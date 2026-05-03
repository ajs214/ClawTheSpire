#!/usr/bin/env bash
# train-v13-continue-14hr.sh — 14-hour V13 continuation (3rd run)
#
# Resumes from latest V13 checkpoint (~gen 1645+ total).
# Same hyperparameters, updated simulator with v0.102/v0.103 patch fixes.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v13-continue-14hr.sh > train-v13-continue-14hr.log 2>&1 &
#   tail -f train-v13-continue-14hr.log
#
# Dashboard (run in a separate terminal):
#   bash ~/AJS_CTS/ClawTheSpire/start-dashboard.sh

set -uo pipefail  # no -e: we want to catch timeout's 124 exit code

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

SAVE_DIR="../alphazero_checkpoints_v13"
PROGRESS_FILE="../training_v13_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v13/boss_fights.jsonl"

mkdir -p "$SAVE_DIR"

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)

echo "=== STS2 AlphaZero Training V13 — 14 Hour Continuation ==="
echo "  Duration cap:  14 hours (hard timeout)"
echo "  Gen budget:    2800"
echo "  Games/gen:     10"
echo "  MCTS sims:     600 base (progressive: 240→1080)"
echo "  Start:         ${LATEST_CKPT:+Resuming from $(basename "$LATEST_CKPT")}${LATEST_CKPT:-FRESH (random weights)}"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+14H 2>/dev/null || date -d '+14 hours' 2>/dev/null || echo '(14 hours from now)')"
echo "-----------------------------------"

# Pure-bash 14-hour watchdog — works without coreutils on macOS.
TIMEOUT_SECS=$((14 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 2800 \
    --games-per-gen 10 \
    --sims 600 \
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
        echo "!! 14-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
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
    echo "=== V13 14-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V13 gen budget exhausted before 14-hour cap at $(date) ==="
else
    echo "=== V13 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
