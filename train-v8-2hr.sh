#!/usr/bin/env bash
# train-v8-2hr.sh — 2-hour time-boxed V8 training run
#
# V8 headline change: FULL RELIC POOL.  The simulator now knows about
# ~260 Silent-relevant relics (up from ~17 in V7).  Most have real
# in-combat effects; the rest grant small damage/block multipliers so
# the network is never incentivised to skip them.  Out-of-combat
# pickup effects (max HP, gold, extra potion slots) also flow through
# the self-play loop for the first time.
#
# Other V8 deltas vs V7:
#   * relic_effects.py registry — data-driven start-of-combat /
#     turn-start / play-card / end-of-turn / end-of-combat hooks.
#   * combat_engine.play_card + start_combat + start_turn + end_turn
#     now dispatch through the registry instead of inline if-blocks.
#   * effects.calculate_attack_damage / calculate_block_gain thread
#     the global damage + block multipliers from the relic pool.
#   * Incoming damage reduction (Torii / Tungsten Rod proxies) applied
#     to enemy hits in _enemy_attacks_player.
#   * full_run.py IMPLEMENTED_RELIC_POOL dynamically built from the
#     registry (Silent-only, starter relics excluded).
#   * Every relic pickup site (elite, boss, event, treasure, shop)
#     now routes through _grant_relic which applies pickup-time HP,
#     gold, and potion-slot bumps.
#   * Progress snapshot carries a new "top_relics" / "relic_pool_size"
#     block so the dashboard can show which relics the agent is
#     actually seeing in V8.
#
# Seed V8 from the latest V7 checkpoint so we start from the strongest
# weights we have rather than cold-starting.  The self-play loop saves
# a checkpoint every 10 gens and writes the progress file every gen,
# so being killed mid-gen loses at most a single in-flight run.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v8-2hr.sh > train-v8.log 2>&1 &
#   bash ~/AJS_CTS/ClawTheSpire/train-v8-2hr.sh            # foreground
#
# At V7's observed ~60-70s/gen, 2 hours buys ~100-120 gens, so we
# request 300 and rely on the timeout as the real stopping condition.

set -uo pipefail  # no -e: we want to catch timeout's 124 exit code

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

SAVE_DIR="../alphazero_checkpoints_v8"
PROGRESS_FILE="../training_v8_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v8/boss_fights.jsonl"
V7_SAVE_DIR="../alphazero_checkpoints_v7"

mkdir -p "$SAVE_DIR"

# Seed V8 from the latest V7 checkpoint if V8 is empty
if [ -z "$(ls -A "$SAVE_DIR" 2>/dev/null)" ] && [ -d "$V7_SAVE_DIR" ]; then
    LATEST_V7=$(ls -t "$V7_SAVE_DIR"/*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_V7" ]; then
        cp "$LATEST_V7" "$SAVE_DIR/"
        echo "Seeded V8 from V7 checkpoint: $(basename "$LATEST_V7")"
    fi
fi

echo "=== STS2 AlphaZero Training V8 — 2 Hour Run ==="
echo "  Duration cap:  2 hours (hard timeout)"
echo "  Gen budget:    300 (stop condition is the timeout, not the gen count)"
echo "  Games/gen:     12"
echo "  MCTS sims:     400 base (progressive: 240→560)"
echo "  Relic pool:    FULL (~260 Silent-relevant relics)"
echo "  Relic effects: registry-driven (real effects + proxies)"
echo "  Pickup flow:   HP/gold/potion-slot bumps wired end-to-end"
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
    echo "=== V8 2-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V8 gen budget exhausted before 2-hour cap at $(date) ==="
else
    echo "=== V8 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
