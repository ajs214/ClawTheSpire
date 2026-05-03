#!/usr/bin/env bash
# train-v13-14hr.sh — 14-hour time-boxed V13 training run (FRESH START)
#
# V13 changes over V12:
#   - Dedicated card_eval_head: deck-aware card pick scoring via
#     mean-pooled deck embeddings + 2-layer MLP (336→128→64→1).
#     Replaces the generic option_eval_head for card reward decisions.
#   - Ranking loss on card picks (β=0.10): good runs push chosen card
#     above alternatives, bad runs push alternatives above chosen.
#   - Shadow advisor loss (α=0.15): when the heuristic disagrees with
#     the network on any option decision, also trains the heuristic's
#     preferred option toward the run outcome value.
#   - Rest site healing fix (3-pronged):
#     1. Confidence-gated guard rail in live play (self-retiring)
#     2. Shadow advisor signal in option training loss
#     3. Exploration forcing: 20% chance to force REST when heuristic
#        recommends it during training
#   - Card pick confidence gate in live play: defers to organic picker
#     when card_eval_head's score spread < 0.20 (head still learning)
#
# Cold start — trains from random weights so the new card_eval_head
# learns from scratch alongside the existing heads.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v13-14hr.sh > train-v13-14hr.log 2>&1 &
#   tail -f train-v13-14hr.log
#
# Dashboard (run in a separate terminal):
#   python3 ~/AJS_CTS/ClawTheSpire/dashboard.py

set -uo pipefail  # no -e: we want to catch timeout's 124 exit code

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

SAVE_DIR="../alphazero_checkpoints_v13"
PROGRESS_FILE="../training_v13_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v13/boss_fights.jsonl"

mkdir -p "$SAVE_DIR"

LATEST_CKPT=$(ls -t "$SAVE_DIR"/gen_*.pt 2>/dev/null | head -1)

echo "=== STS2 AlphaZero Training V13 — 14 Hour Run (COLD START) ==="
echo "  Duration cap:  14 hours (hard timeout)"
echo "  Gen budget:    2800"
echo "  Games/gen:     10"
echo "  MCTS sims:     400 base (progressive: 160→720)"
echo "  Start:         ${LATEST_CKPT:+Resuming from $(basename "$LATEST_CKPT")}${LATEST_CKPT:-FRESH (random weights)}"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo "  Boss log:      $BOSS_LOG_FILE"
echo ""
echo "  V13 improvements:"
echo "    - Dedicated card_eval_head (deck-aware card pick scoring)"
echo "    - Ranking loss on card picks (β=0.10)"
echo "    - Shadow advisor loss on all option types (α=0.15)"
echo "    - Rest site healing: exploration forcing + guard rail"
echo "    - Card pick confidence gate (organic picker fallback)"
echo ""
echo "Starting at $(date)"
echo "Expected end:  $(date -v+14H 2>/dev/null || date -d '+14 hours' 2>/dev/null || echo '(14 hours from now)')"
echo "-----------------------------------"

# Pure-bash 14-hour watchdog — works without coreutils on macOS.
TIMEOUT_SECS=$((14 * 3600))

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 2800 \
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
    echo "=== V13 14-hour cap reached at $(date) (exit $RC) ==="
elif [ "$RC" -eq 0 ]; then
    echo "=== V13 gen budget exhausted before 14-hour cap at $(date) ==="
else
    echo "=== V13 training exited with code $RC at $(date) ==="
fi
echo "Checkpoints: $SAVE_DIR"
echo "Boss log:    $BOSS_LOG_FILE"
