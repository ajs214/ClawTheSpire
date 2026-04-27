#!/usr/bin/env bash
# train-v19-18hr.sh — V19 training, 18-hour run (warm start from V18)
#
# V19 changes (over V18):
#   - Skip encouragement: deck-size-aware bloat penalty pushes toward skip
#     on losing runs with 18+ card decks (SKIP_BLOAT_ALPHA=0.10)
#   - Relic oversampling: OPTION_RELIC_PICK samples added 3x to option
#     buffer (~5% → ~15% of training batches)
#   - Live runner: pre-combat potions now MCTS-driven (survival-only heuristic)
#   - Live runner: mid-combat discard/exhaust routed through MCTS
#
# Warm-starts from latest V18 checkpoint.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v19-18hr.sh > train-v19-18hr.log 2>&1 &
#   tail -f train-v19-18hr.log
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

SAVE_DIR="../alphazero_checkpoints_v19"
PROGRESS_FILE="../training_v19_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v19/boss_fights.jsonl"
V18_DIR="../alphazero_checkpoints_v18"

mkdir -p "$SAVE_DIR"

# Warm start: copy latest V18 checkpoint to V19 dir as gen_0000.pt
LATEST_V18=$(ls -t "$V18_DIR"/gen_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_V18" ] && [ ! -f "$SAVE_DIR/gen_0000.pt" ]; then
    echo "Warm-starting V19 from V18: $(basename "$LATEST_V18")"
    cp "$LATEST_V18" "$SAVE_DIR/gen_0000.pt"
    echo "  Copied → $SAVE_DIR/gen_0000.pt"
elif [ -f "$SAVE_DIR/gen_0000.pt" ]; then
    echo "V19 checkpoint dir already has data — resuming."
else
    echo "ERROR: No V18 checkpoint found in $V18_DIR"
    exit 1
fi

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)

echo "=== STS2 AlphaZero Training V19 — 18 Hours ==="
echo "  Duration cap:  18 hours (hard timeout)"
echo "  Gen budget:    4000"
echo "  Games/gen:     8"
echo "  MCTS sims:     500 base (progressive scaling)"
echo "  Start:         Resuming from $(basename "$LATEST_CKPT")"
echo "  Batch size:    64"
echo "  Epochs:        3"
echo "  LR:            1e-4"
echo "  Changes:       Skip bloat penalty, relic 3x oversample, MCTS potions/discard (live)"
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
    echo "=== V19 18-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V19 gen budget exhausted before 18-hour cap at $(date) ==="
else
    echo "=== V19 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
