#!/usr/bin/env bash
# train-v15-18hr.sh — V15 training, 18-hour run
#
# V15 changes (over V14):
#   - Real relic mechanics: ~60 Silent-eligible relics get actual combat hooks
#     (previously proxy multipliers). Effects in effects.py, combat_engine.py,
#     full_run.py, relic_effects.py.
#   - Relic-aware card_eval_head: 336→357 input dims (8 relic embed + 13 synergy)
#   - Checkpoint migration: pad_card_eval_weights() zero-fills new columns
#   - Out-of-combat relics: egg auto-upgrades, on-pickup transforms, pre-combat
#   - Ghost relic fixes, LIZARD_TAIL death prevention, tea set energy bonus
#
# Hyperparameters (same as V14 tuned):
#   - MCTS sims:    500
#   - Games/gen:    8
#   - Batch size:   64
#   - Epochs:       3
#   - LR:           1e-3 (with decay)
#
# Warm-starts from V14 checkpoint (automatic migration of card_eval_head weights)
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v15-18hr.sh > train-v15-18hr.log 2>&1 &
#   tail -f train-v15-18hr.log
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

# V15 uses its own checkpoint dir but warm-starts from V14
SAVE_DIR="../alphazero_checkpoints_v15"
PROGRESS_FILE="../training_v15_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v15/boss_fights.jsonl"

# Copy latest V14 checkpoint to V15 dir for warm start
V14_DIR="../alphazero_checkpoints_v14"
mkdir -p "$SAVE_DIR"

# Wipe failed V15 checkpoints and re-seed from V14
LATEST_V14=$(ls -t "$V14_DIR"/gen_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_V14" ]; then
    echo "Wiping failed V15 checkpoints and re-seeding from V14..."
    rm -f "$SAVE_DIR"/gen_*.pt
    rm -f "$SAVE_DIR"/run_logs.jsonl
    rm -f "$SAVE_DIR"/boss_fights.jsonl
    cp "$LATEST_V14" "$SAVE_DIR/gen_0000.pt"
    # Also copy vocabs if they exist
    for f in "$V14_DIR"/*.json; do
        [ -f "$f" ] && cp "$f" "$SAVE_DIR/" 2>/dev/null || true
    done
    echo "  V14 checkpoint copied → V15 will auto-migrate card_eval_head weights (336→357)"
    echo "  LR set to 1e-4 to prevent catastrophic forgetting with reset optimizer"
fi

echo "=== STS2 AlphaZero Training V15 — Real Relics, 18 Hours ==="
echo "  Duration cap:  18 hours (hard timeout)"
echo "  Gen budget:    3200"
echo "  Games/gen:     8"
echo "  MCTS sims:     500 base (progressive scaling)"
LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)
echo "  Start:         ${LATEST_CKPT:+Resuming from $(basename "$LATEST_CKPT")}${LATEST_CKPT:-FRESH (random weights)}"
echo "  Batch size:    64"
echo "  Epochs:        3"
echo "  LR:            1e-4 (reduced — optimizer reset needs gentle start)"
echo "  Changes:       Real relic mechanics, relic-aware card_eval_head (357 dims)"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+18H 2>/dev/null || date -d '+18 hours' 2>/dev/null || echo '(18 hours from now)')"
echo "-----------------------------------"

TIMEOUT_SECS=$((18 * 3600))

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
    echo "=== V15 18-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V15 gen budget exhausted before 18-hour cap at $(date) ==="
else
    echo "=== V15 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
