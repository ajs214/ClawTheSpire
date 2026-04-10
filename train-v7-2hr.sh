#!/usr/bin/env bash
# train-v7-2hr.sh — 2-hour time-boxed V7 training run
#
# V7 changes vs V6 (see commit ac83aa2 for full details):
#   * Picker retune — archetype commitment gate raised to 3+ cards with
#     separation 2, commitment denominator floor raised from 3 → 5, and
#     an empirical Act-1 premium-neutral bonus (+0.10) for cards with the
#     strongest win-lift from the V6 boss-log data.
#   * Simulator relic leaks fixed — elite kills now drop a relic, the map
#     forces a treasure chest at floor 9, event options that actually
#     grant relics are detected via a bbcode-stripped regex, and boss
#     kills grant an Ancient-pool drop.  Shop relic pool rewritten around
#     the real rarity strings (the old filter was silently including
#     starter relics).  Mocked-win runs now see 8.15 relics/run, up from
#     V6's ~1.2.
#   * Perceived-value heuristic — the flat 0.45 unknown-relic fallback in
#     score_relic_for_deck is replaced with a description-based scorer
#     that combines a rarity prior with positive/negative keyword scans,
#     clamped to [0.15, 0.95].  Score distribution across the 150-relic
#     shop pool now spans 0.38-0.95 with mean 0.627, and the 0.4 shop
#     gate correctly rejects the ~5% of relics that are genuine trap
#     picks.
#
# Seed V7 from the latest V6 checkpoint so we start from the strongest
# weights we have rather than cold-starting.  The self-play loop saves a
# checkpoint every 10 gens and writes the progress file every gen, so
# being killed mid-gen loses at most a single in-flight run.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v7-2hr.sh > train-v7.log 2>&1 &
#   bash ~/AJS_CTS/ClawTheSpire/train-v7-2hr.sh            # foreground
#
# At V6's observed ~35-50s/gen, 2 hours buys ~140-200 gens, so we request
# 300 and rely on the timeout as the real stopping condition.

set -uo pipefail  # no -e: we want to catch timeout's 124 exit code

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

SAVE_DIR="../alphazero_checkpoints_v7"
PROGRESS_FILE="../training_v7_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v7/boss_fights.jsonl"
V6_SAVE_DIR="../alphazero_checkpoints_v6"

mkdir -p "$SAVE_DIR"

# Seed V7 from the latest V6 checkpoint if V7 is empty
if [ -z "$(ls -A "$SAVE_DIR" 2>/dev/null)" ] && [ -d "$V6_SAVE_DIR" ]; then
    LATEST_V6=$(ls -t "$V6_SAVE_DIR"/*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_V6" ]; then
        cp "$LATEST_V6" "$SAVE_DIR/"
        echo "Seeded V7 from V6 checkpoint: $(basename "$LATEST_V6")"
    fi
fi

echo "=== STS2 AlphaZero Training V7 — 2 Hour Run ==="
echo "  Duration cap:  2 hours (hard timeout)"
echo "  Gen budget:    300 (stop condition is the timeout, not the gen count)"
echo "  Games/gen:     12"
echo "  MCTS sims:     400 base (progressive: 240→560)"
echo "  Picker:        V7 organic (raised commitment bar + neutral bonus)"
echo "  Relic synergy: V7 perceived-value heuristic"
echo "  Relic faucets: elite/treasure/event/boss/shop all wired"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+2H 2>/dev/null || date -d '+2 hours' 2>/dev/null || echo '(2 hours from now)')"
echo "-----------------------------------"

# Pure-bash 2-hour watchdog — works without coreutils on macOS.
TIMEOUT_SECS=$((2 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 300 \
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
        echo "!! 2-hour cap hit — sending SIGTERM to training pid $TRAIN_PID"
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
    echo "=== V7 2-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V7 gen budget exhausted before 2-hour cap at $(date) ==="
else
    echo "=== V7 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
