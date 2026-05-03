#!/usr/bin/env bash
# train-v21-18hr.sh — Launch V21 training (COLD START)
#
# V21 changes (fixing V20 over-skipping):
#   1. No forced skip when deck < 13 cards
#   2. Protect good cards (WR >= 25%) from forced skip
#   3. Halved exploration rates (bloat 5-15%, quality 8-12%)
#   4. Moderate pick_bonus (0.08/10 vs V20's 0.06/6)
#   5. Weakened bootstrap (value<0.2, WR<0.20 vs 0.4/0.28)
#   6. Anneal: full exploration gen 0-200, decay to 50% by gen 600
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v21-18hr.sh > train-v21-18hr.log 2>&1 &
#   tail -f train-v21-18hr.log

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

# Clear stale Python cache
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

SAVE_DIR="../alphazero_checkpoints_v21"
PROGRESS_FILE="../training_v21_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v21/boss_fights.jsonl"

mkdir -p "$SAVE_DIR"

echo ""
echo "=== STS2 AlphaZero Training V21 — COLD START — 18 Hours ==="
echo "  V21 changes (fixing V20 over-skipping):"
echo "    1. No forced skip when deck < 13"
echo "    2. Protect good cards (WR >= 25%) from skip"
echo "    3. Halved exploration rates"
echo "    4. Moderate pick_bonus (0.08/10)"
echo "    5. Weakened bootstrap thresholds"
echo "    6. Exploration anneal (full→50% over 200-600 gens)"
echo "  Duration cap:  18 hours (hard timeout)"
echo "  Gen budget:    4000"
echo "  Games/gen:     8"
echo "  MCTS sims:     500 base (progressive scaling)"
echo "  Start:         COLD (random init)"
echo "  Batch size:    64"
echo "  Epochs:        3"
echo "  LR:            1e-4"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+18H 2>/dev/null || date -d '+18 hours' 2>/dev/null || echo '(18 hours from now)')"
echo "-----------------------------------"

TIMEOUT_SECS=$((18 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 4000 \
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
    echo "=== V21 18-hour training cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V21 gen budget exhausted at $(date) ==="
else
    echo "=== V21 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
