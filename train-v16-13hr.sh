#!/usr/bin/env bash
# train-v16-13hr.sh — V16 training, 13-hour run (warm start from V15/V16)
#
# V16 changes (over V15):
#   - 64 enemy damage values corrected against game data
#   - 7 missing Silent card effects implemented
#   - Act 1 room count 17→15 (matches real game)
#   - Neow blessings expanded (7→11, sample 3 per run)
#   - Treasure chest gold (50-75g)
#   - Escalating card removal cost (75→100→125→...)
#   - Empirical relic scoring (354 relics from 22K runs)
#   - HP preservation bonus + HP-scaled rest exploration
#   - Boosted ranking loss + decaying exploration for underpicked cards
#   - Sim-to-live divergence fixes
#
# LR starts at 2e-4 (higher than normal) to help the value head
# recalibrate quickly for the 15-floor runs, then decays via cosine.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v16-13hr.sh > train-v16-13hr.log 2>&1 &
#   tail -f train-v16-13hr.log
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

# Clear stale Python cache
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

SAVE_DIR="../alphazero_checkpoints_v16"
PROGRESS_FILE="../training_v16_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v16/boss_fights.jsonl"
V15_DIR="../alphazero_checkpoints_v15"

mkdir -p "$SAVE_DIR"

# Warm start: copy latest V15 checkpoint if V16 dir is empty
LATEST_V15=$(ls -t "$V15_DIR"/gen_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_V15" ] && [ ! -f "$SAVE_DIR/gen_0000.pt" ]; then
    echo "Warm-starting V16 from V15: $(basename "$LATEST_V15")"
    cp "$LATEST_V15" "$SAVE_DIR/gen_0000.pt"
    echo "  Copied → $SAVE_DIR/gen_0000.pt"
elif [ -f "$SAVE_DIR/gen_0000.pt" ] || ls "$SAVE_DIR"/gen_*.pt >/dev/null 2>&1; then
    echo "V16 checkpoint dir already has data — resuming."
else
    echo "ERROR: No V15 checkpoint found in $V15_DIR"
    exit 1
fi

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)

echo "=== STS2 AlphaZero Training V16 — 13 Hours ==="
echo "  Duration cap:  13 hours (hard timeout)"
echo "  Gen budget:    3200"
echo "  Games/gen:     8"
echo "  MCTS sims:     500 base (progressive scaling)"
echo "  Start:         Resuming from $(basename "$LATEST_CKPT")"
echo "  Batch size:    64"
echo "  Epochs:        3"
echo "  LR:            2e-4 (higher — recalibrating for 15-floor runs)"
echo "  Changes:       Enemy fixes, card effects, 15-floor map, Neow expansion"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+13H 2>/dev/null || date -d '+13 hours' 2>/dev/null || echo '(13 hours from now)')"
echo "-----------------------------------"

TIMEOUT_SECS=$((13 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 3200 \
    --games-per-gen 8 \
    --sims 500 \
    --batch-size 64 \
    --epochs 3 \
    --lr 2e-4 \
    --temperature 1.0 \
    --save-dir "$SAVE_DIR" \
    --progress-file "$PROGRESS_FILE" \
    --boss-log-file "$BOSS_LOG_FILE" &
TRAIN_PID=$!

(
    sleep "$TIMEOUT_SECS"
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo ""
        echo "!! 13-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
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
    echo "=== V16 13-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V16 gen budget exhausted before 13-hour cap at $(date) ==="
else
    echo "=== V16 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
