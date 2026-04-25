#!/usr/bin/env bash
# train-v17-15hr.sh — V17 training, 15-hour run (warm start from V16)
#
# V17 changes (over V16):
#   - StatusCard intents now add junk cards to discard pile (deck pollution)
#   - Boss phase transitions: Ceremonial Beast, Vantom, Kin Priest get
#     more aggressive move tables below 50% HP
#   - Combat timeout raised from 30 → 50 turns (enables poison/stall)
#   - Act 1 map corrected to 17 floors (was 15 in simulator)
#   - Unknown enemy intent prediction anchored on observed damage (not hardcoded 12)
#   - Force-play override dropped from 50% → 1% in live play
#   - Pre-MCTS potion forcing (live play only, doesn't affect training)
#
# Warm-starts from latest V16 checkpoint (no architecture changes)
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v17-15hr.sh > train-v17-15hr.log 2>&1 &
#   tail -f train-v17-15hr.log
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

SAVE_DIR="../alphazero_checkpoints_v17"
PROGRESS_FILE="../training_v17_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v17/boss_fights.jsonl"
V16_DIR="../alphazero_checkpoints_v16"

mkdir -p "$SAVE_DIR"

# Warm start: copy latest V16 checkpoint to V17 dir as gen_0000.pt
LATEST_V16=$(ls -t "$V16_DIR"/gen_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_V16" ] && [ ! -f "$SAVE_DIR/gen_0000.pt" ]; then
    echo "Warm-starting V17 from V16: $(basename "$LATEST_V16")"
    cp "$LATEST_V16" "$SAVE_DIR/gen_0000.pt"
    echo "  Copied → $SAVE_DIR/gen_0000.pt"
elif [ -f "$SAVE_DIR/gen_0000.pt" ]; then
    echo "V17 checkpoint dir already has data — resuming."
else
    echo "ERROR: No V16 checkpoint found in $V16_DIR"
    exit 1
fi

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)

echo "=== STS2 AlphaZero Training V17 — 15 Hours ==="
echo "  Duration cap:  15 hours (hard timeout)"
echo "  Gen budget:    3600"
echo "  Games/gen:     8"
echo "  MCTS sims:     500 base (progressive scaling)"
echo "  Start:         Resuming from $(basename "$LATEST_CKPT")"
echo "  Batch size:    64"
echo "  Epochs:        3"
echo "  LR:            1e-4"
echo "  Changes:       StatusCard sim, boss phases, 17-floor maps, 50-turn timeout"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+15H 2>/dev/null || date -d '+15 hours' 2>/dev/null || echo '(15 hours from now)')"
echo "-----------------------------------"

TIMEOUT_SECS=$((15 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 3600 \
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
        echo "!! 15-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
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
    echo "=== V17 15-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V17 gen budget exhausted before 15-hour cap at $(date) ==="
else
    echo "=== V17 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
