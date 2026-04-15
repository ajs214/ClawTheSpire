#!/usr/bin/env bash
# train-v12-12hr.sh — 12-hour time-boxed V12 training run
#
# V12 changes over V11 (no checkpoint shape changes — resumes from V11):
#   - 9 simulator/live-play divergence fixes:
#       * Enemy buff/debuff/status intent resolution
#       * Fixed Silent cards: Infernal Blade, Memento Mori, Tools of the Trade,
#         Burst, Bullet Time
#       * Randomized targeting for multi-target cards
#       * Temperature 0.15 in live play MCTS (was greedy)
#       * Force-play override aligned: 50% in live play (was 0%)
#       * Bridge fix: no double-application of Strength/Dexterity/Weak
#       * Heuristic fallback when MCTS errors
#       * Fuzzy enemy intent matching (±3 damage tolerance)
#       * Infection status card handling
#   - 19 new card effect implementations:
#       * 9 Silent: Murder, Expose, Well-Laid Plans, Knife Trap, Automation,
#         Shadowmeld, Accelerant, Nightmare, Storm of Steel
#       * 10 Colorless: Restlessness, Impatience, Purity, Thinking Ahead,
#         Panic Button, Volley, Dark Shackles, Shockwave, Master of Strategy,
#         Prowess
#       * Power triggers: Automation energy, Shadowmeld block doubling,
#         Accelerant poison multiplier
#   - MCTS training improvements:
#       * Base sims: 100 (was 50) — doubled for better policy/value targets
#       * Progressive scaling: 40%→180% (was 60%→140%)
#       * Late-gen sims reach 180, close to live play's 200
#   - Total custom card effects: 75 (was 56)
#
# No migration needed — V12 uses same network architecture as V11.
# Resumes directly from latest V11 checkpoint.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v12-12hr.sh > train-v12-12hr.log 2>&1 &
#   tail -f train-v12-12hr.log
#
# Dashboard (run in a separate terminal):
#   python3 ~/AJS_CTS/ClawTheSpire/dashboard.py

set -uo pipefail  # no -e: we want to catch timeout's 124 exit code

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

SAVE_DIR="../alphazero_checkpoints_v12"
PROGRESS_FILE="../training_v12_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v12/boss_fights.jsonl"

mkdir -p "$SAVE_DIR"

# V12 uses same network architecture as V11 — copy latest V11 checkpoint
if ! ls "$SAVE_DIR"/gen_*.pt &>/dev/null; then
    V11_DIR="../alphazero_checkpoints_v11"
    LATEST_V11=$(ls -t "$V11_DIR"/gen_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_V11" ]; then
        echo "Copying latest V11 checkpoint to V12: $LATEST_V11"
        cp "$LATEST_V11" "$SAVE_DIR/gen_0000.pt"
    else
        echo "!! No V11 checkpoint found in $V11_DIR"
        echo "!! V12 needs a V11 checkpoint to resume from."
        echo "!! Either run V11 training first or manually place a checkpoint."
        exit 1
    fi
fi

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)

echo "=== STS2 AlphaZero Training V12 — 12 Hour Run ==="
echo "  Duration cap:  12 hours (hard timeout)"
echo "  Gen budget:    1800"
echo "  Games/gen:     10"
echo "  MCTS sims:     400 base (progressive: 160→720)"
echo "  Resuming from: $(basename "$LATEST_CKPT")"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "  V12 improvements:"
echo "    - 9 simulator/live-play gap fixes"
echo "    - 19 new card effect implementations (75 total)"
echo "    - Doubled training MCTS sims (100 base, 40%→180% progressive)"
echo "    - Enemy intent buff/debuff/status resolution"
echo "    - Fuzzy enemy intent matching"
echo "    - Heuristic fallback for MCTS errors"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+12H 2>/dev/null || date -d '+12 hours' 2>/dev/null || echo '(12 hours from now)')"
echo "-----------------------------------"

# Pure-bash 12-hour watchdog — works without coreutils on macOS.
TIMEOUT_SECS=$((12 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 1800 \
    --games-per-gen 10 \
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
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo ""
        echo "!! 12-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
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
    echo "=== V12 12-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V12 gen budget exhausted before 12-hour cap at $(date) ==="
else
    echo "=== V12 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
