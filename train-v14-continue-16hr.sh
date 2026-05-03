#!/usr/bin/env bash
# train-v14-continue-16hr.sh — V14 continuation, 16-hour run (tuned hyperparams)
#
# V14 changes:
#   - Card ranking loss decoupled from skip (cards only compete with cards)
#   - Pick bonus: bias toward taking a card in early game (fades with deck size)
#   - Deck summary excludes base cards (starter deck filtered out)
#
# Hyperparameter tuning (vs previous runs):
#   - MCTS sims:    500 (was 600) — network is strong enough to need less search
#   - Games/gen:    8   (was 10)  — faster feedback loop, buffer already full
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v14-continue-16hr.sh > train-v14-continue-16hr.log 2>&1 &
#   tail -f train-v14-continue-16hr.log
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

SAVE_DIR="../alphazero_checkpoints_v14"
PROGRESS_FILE="../training_v14_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v14/boss_fights.jsonl"

mkdir -p "$SAVE_DIR"

echo "=== STS2 AlphaZero Training V14 — Continuation, 16 Hours (Tuned) ==="
echo "  Duration cap:  16 hours (hard timeout)"
echo "  Gen budget:    2800"
echo "  Games/gen:     8  (was 10)"
echo "  MCTS sims:     500 base (was 600) — progressive scaling still active"
LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)
echo "  Start:         ${LATEST_CKPT:+Resuming from $(basename "$LATEST_CKPT")}${LATEST_CKPT:-FRESH (random weights)}"
echo "  Batch size:    64"
echo "  Epochs:        3"
echo "  LR:            1e-3 (with decay)"
echo "  Changes:       Skip-decoupled ranking, pick bonus, filtered deck summary"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+16H 2>/dev/null || date -d '+16 hours' 2>/dev/null || echo '(16 hours from now)')"
echo "-----------------------------------"

TIMEOUT_SECS=$((16 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 2800 \
    --games-per-gen 8 \
    --sims 500 \
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
        echo "!! 16-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
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
    echo "=== V14 16-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V14 gen budget exhausted before 16-hour cap at $(date) ==="
else
    echo "=== V14 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
