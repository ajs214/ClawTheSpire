#!/usr/bin/env bash
# train-v21-chain.sh — Run multiple 18-hour V21 training segments back-to-back
#
# Waits for any existing training to finish, then kicks off N segments.
# Each segment resumes from the latest checkpoint.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v21-chain.sh 3 > train-v21-chain.log 2>&1 &
#   tail -f train-v21-chain.log
#
# Default: 2 segments if no argument given.

set -uo pipefail

SEGMENTS=${1:-2}
SEGMENT_HOURS=18
cd "$(dirname "$0")/sts2-solver"

echo "=== V21 Chain Training: $SEGMENTS x ${SEGMENT_HOURS}h segments ==="
echo "Started at $(date)"
echo ""

# ── Wait for any existing training to finish ──────────────────────────
EXISTING_PID=$(pgrep -f "sts2_solver.alphazero.self_play train" 2>/dev/null | head -1 || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Existing training running (pid $EXISTING_PID) — waiting for it to finish..."
    while kill -0 "$EXISTING_PID" 2>/dev/null; do
        sleep 60
    done
    echo "Previous training finished at $(date)"
    echo ""
    sleep 5  # let files flush
fi

source .venv/bin/activate

SAVE_DIR="../alphazero_checkpoints_v21"
PROGRESS_FILE="../training_v21_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v21/boss_fights.jsonl"
TIMEOUT_SECS=$((SEGMENT_HOURS * 3600))

for SEG in $(seq 1 "$SEGMENTS"); do
    # Clear stale Python cache
    find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: No V21 checkpoint found in $SAVE_DIR"
        exit 1
    fi

    echo "=================================================="
    echo "  Segment $SEG/$SEGMENTS — ${SEGMENT_HOURS}h"
    echo "  Resuming from: $(basename "$LATEST_CKPT")"
    echo "  Starting at $(date)"
    echo "  Expected end: $(date -v+${SEGMENT_HOURS}H 2>/dev/null || date -d "+${SEGMENT_HOURS} hours" 2>/dev/null || echo "(${SEGMENT_HOURS}h from now)")"
    echo "=================================================="

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
            echo "!! Segment $SEG: ${SEGMENT_HOURS}h cap hit — sending SIGTERM"
            kill -TERM "$TRAIN_PID" 2>/dev/null || true
            for _ in $(seq 1 30); do
                kill -0 "$TRAIN_PID" 2>/dev/null || exit 0
                sleep 1
            done
            echo "!! Still alive — sending SIGKILL"
            kill -KILL "$TRAIN_PID" 2>/dev/null || true
        fi
    ) &
    WATCHDOG_PID=$!

    wait "$TRAIN_PID"
    RC=$?

    kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true

    echo ""
    echo "Segment $SEG finished at $(date) (exit $RC)"

    if [ "$SEG" -lt "$SEGMENTS" ]; then
        echo "Pausing 10s before next segment..."
        sleep 10
    fi
done

echo ""
echo "=== All $SEGMENTS segments complete at $(date) ==="
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
