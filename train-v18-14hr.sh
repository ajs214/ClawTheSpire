#!/usr/bin/env bash
# train-v18-14hr.sh — V18 training, 14-hour run (warm start from V17)
#
# V18 changes (over V17):
#   - Boss/elite relic picks now network-driven via option_eval_head
#     with OPTION_RELIC_PICK=17 + OptionSample for training data
#   - Deck select (upgrade/remove/transform) uses card_eval_head
#     instead of deterministic tier-list (live play only, training
#     data already existed from card reward head)
#   - Multi-card deck select scored by network priority ordering
#
# Warm-starts from latest V17 checkpoint (no architecture changes —
# OPTION_RELIC_PICK=17 is a new option type constant that the existing
# option_eval_head handles via its type embedding layer)
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v18-14hr.sh > train-v18-14hr.log 2>&1 &
#   tail -f train-v18-14hr.log
#
# Dashboard:
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

# Clear stale Python cache to ensure latest code is used
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

SAVE_DIR="../alphazero_checkpoints_v18"
PROGRESS_FILE="../training_v18_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v18/boss_fights.jsonl"
V17_DIR="../alphazero_checkpoints_v17"

mkdir -p "$SAVE_DIR"

# Warm start: copy latest V17 checkpoint to V18 dir as gen_0000.pt
LATEST_V17=$(ls -t "$V17_DIR"/gen_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_V17" ] && [ ! -f "$SAVE_DIR/gen_0000.pt" ]; then
    echo "Warm-starting V18 from V17: $(basename "$LATEST_V17")"
    cp "$LATEST_V17" "$SAVE_DIR/gen_0000.pt"
    echo "  Copied → $SAVE_DIR/gen_0000.pt"
elif [ -f "$SAVE_DIR/gen_0000.pt" ]; then
    echo "V18 checkpoint dir already has data — resuming."
else
    echo "ERROR: No V17 checkpoint found in $V17_DIR"
    exit 1
fi

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)

echo "=== STS2 AlphaZero Training V18 — 14 Hours ==="
echo "  Duration cap:  14 hours (hard timeout)"
echo "  Gen budget:    3400"
echo "  Games/gen:     8"
echo "  MCTS sims:     500 base (progressive scaling)"
echo "  Start:         Resuming from $(basename "$LATEST_CKPT")"
echo "  Batch size:    64"
echo "  Epochs:        3"
echo "  LR:            1e-4"
echo "  Changes:       Network relic picks (OPTION_RELIC_PICK), deck select via card_eval_head"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+14H 2>/dev/null || date -d '+14 hours' 2>/dev/null || echo '(14 hours from now)')"
echo "-----------------------------------"

TIMEOUT_SECS=$((14 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 3400 \
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
    echo "=== V18 14-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V18 gen budget exhausted before 14-hour cap at $(date) ==="
else
    echo "=== V18 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
