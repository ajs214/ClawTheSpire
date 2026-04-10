#!/usr/bin/env bash
# train-v6-7hr.sh — 7-hour time-boxed V6 training run
#
# Same settings as train-v6.sh (see that file for details on what V6
# changed), but wraps the worker in a hard 7-hour timeout so the
# process stops on the dot regardless of per-generation cost.  The
# self-play loop saves a checkpoint every 10 gens and writes the
# progress file every gen, so being killed mid-gen loses at most a
# single in-flight run.
#
# Pick a generation budget larger than we can realistically finish in
# 7 hours — the timeout is the real stopping condition.  V6 gens run
# ~35-50s with 400 MCTS sims, so 7h ≈ 500-720 gens; we request 900
# to leave headroom.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v6-7hr.sh &   # background
#   bash ~/AJS_CTS/ClawTheSpire/train-v6-7hr.sh            # foreground

set -uo pipefail  # no -e: we want to catch timeout's 124 exit code

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

SAVE_DIR="../alphazero_checkpoints_v6"
PROGRESS_FILE="../training_v6_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v6/boss_fights.jsonl"
V5_SAVE_DIR="../alphazero_checkpoints_v5"

mkdir -p "$SAVE_DIR"

# Seed V6 from the latest V5 checkpoint if V6 is empty
if [ -z "$(ls -A "$SAVE_DIR" 2>/dev/null)" ] && [ -d "$V5_SAVE_DIR" ]; then
    LATEST_V5=$(ls -t "$V5_SAVE_DIR"/*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_V5" ]; then
        cp "$LATEST_V5" "$SAVE_DIR/"
        echo "Seeded V6 from V5 checkpoint: $(basename "$LATEST_V5")"
    fi
fi

echo "=== STS2 AlphaZero Training V6 — 7 Hour Run ==="
echo "  Duration cap:  7 hours (hard timeout)"
echo "  Gen budget:    900 (stop condition is the timeout, not the gen count)"
echo "  Games/gen:     12"
echo "  MCTS sims:     400 base (progressive: 240→560)"
echo "  Card picker:   Organic (XGBoost removed)"
echo "  Deck eval:     Unified score_card everywhere"
echo "  Relic synergy: Threaded through pick_card & shop scoring"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+7H 2>/dev/null || date -d '+7 hours' 2>/dev/null || echo '(7 hours from now)')"
echo "-----------------------------------"

# Pure-bash 7-hour watchdog — works without coreutils on macOS.
# Launch the worker in the background, then a sleep+kill guard alongside
# it.  Whichever finishes first wins; we clean up the loser.
TIMEOUT_SECS=$((7 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 900 \
    --games-per-gen 12 \
    --sims 400 \
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
    # If training is still alive, send SIGTERM so the worker can flush
    # progress + boss log before exiting.
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo ""
        echo "!! 7-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        # Give it 30s to shut down cleanly, then hard-kill.
        for _ in $(seq 1 30); do
            kill -0 "$TRAIN_PID" 2>/dev/null || exit 0
            sleep 1
        done
        echo "!! Training didn't exit after 30s — sending SIGKILL"
        kill -KILL "$TRAIN_PID" 2>/dev/null || true
    fi
) &
WATCHDOG_PID=$!

# Forward SIGINT / SIGTERM from whoever is running this script down to
# both children so Ctrl-C (or a parent kill) tears everything down.
cleanup() {
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$TRAIN_PID"
RC=$?

# Training finished (either on its own or because the watchdog killed
# it) — kill the watchdog if it's still sleeping.
kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true

echo ""
if [ "$RC" -eq 143 ] || [ "$RC" -eq 137 ]; then
    echo "=== V6 7-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V6 gen budget exhausted before 7-hour cap at $(date) ==="
else
    echo "=== V6 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
exit $RC
