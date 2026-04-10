#!/usr/bin/env bash
# collect-boss-data.sh — ~1 hour data-collection run.
#
# What it does:
#   - Warm-starts from the final V5 checkpoint (gen_1080.pt)
#   - Uses a SEPARATE save dir so V5 checkpoints stay untouched
#   - Runs 30 generations × 12 games × 400 base sims
#   - Writes per-boss-fight detail to boss_fights.jsonl (NEW logging)
#
# Expected duration:
#   - XGBoost refresh (200 games in subprocess): ~3-5 min
#   - 30 gens × ~80s/gen (progressive MCTS scaling):  ~40-50 min
#   - Total:                                          ~45-55 min
#
# Boss fight yield: 30 × 12 = 360 games × ~66% boss_reach ≈ 240 boss fights.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/collect-boss-data.sh > ~/AJS_CTS/ClawTheSpire/collect-boss-data.log 2>&1 &
#   # ...or foreground:
#   bash ~/AJS_CTS/ClawTheSpire/collect-boss-data.sh

set -euo pipefail

cd "$(dirname "$0")"

# Archive any existing boss log so this run is clean and easy to analyse.
if [ -s boss_fights.jsonl ]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    mv boss_fights.jsonl "boss_fights_${stamp}.jsonl"
    echo "Archived existing boss log → boss_fights_${stamp}.jsonl"
fi
: > boss_fights.jsonl

# Separate checkpoint dir for this collection run so V5 artefacts stay intact.
SAVE_DIR="./alphazero_checkpoints_collect"
PROGRESS_FILE="./collect_progress.json"
V5_SAVE_DIR="./alphazero_checkpoints_v5"

# ALWAYS start from a clean dir so warm_start picks the right checkpoint.
# (Previous runs may have left stale gen_XXXX.pt files with newer mtimes,
#  causing the trainer to load the wrong checkpoint.)
if [ -d "$SAVE_DIR" ]; then
    rm -f "$SAVE_DIR"/*.pt
fi
rm -f "$PROGRESS_FILE"
mkdir -p "$SAVE_DIR"

# Seed from the final V5 checkpoint so we start with the fully trained network.
LATEST_V5="$V5_SAVE_DIR/gen_1080.pt"
if [ ! -f "$LATEST_V5" ]; then
    LATEST_V5=$(ls -t "$V5_SAVE_DIR"/*.pt 2>/dev/null | head -1)
fi
if [ -n "${LATEST_V5:-}" ] && [ -f "$LATEST_V5" ]; then
    cp "$LATEST_V5" "$SAVE_DIR/"
    echo "Seeded collection run from: $(basename "$LATEST_V5")"
else
    echo "ERROR: No V5 checkpoint found in $V5_SAVE_DIR — aborting." >&2
    exit 1
fi

cd sts2-solver
source .venv/bin/activate

# Sanity check: imports work before committing to the full run.
python3 -c "
from sts2_solver.alphazero.full_run import play_full_run, FullRunResult
from sts2_solver.alphazero.self_play import train_worker
print('[sanity] imports OK — boss_detail field present:', 'boss_detail' in FullRunResult.__dataclass_fields__)
" || { echo "Import sanity check failed — aborting."; exit 1; }

echo ""
echo "=== Boss-Fight Data Collection (~1 hour) ==="
echo "  Save dir:     $SAVE_DIR  (isolated)"
echo "  Boss log:     ../boss_fights.jsonl"
echo "  Generations:  30"
echo "  Games/gen:    12"
echo "  Base sims:    400"
echo ""
echo "Starting at $(date)"
echo "-----------------------------------"

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 30 \
    --games-per-gen 12 \
    --sims 400 \
    --batch-size 64 \
    --epochs 3 \
    --lr 1e-3 \
    --temperature 0.3 \
    --save-dir "$SAVE_DIR" \
    --progress-file "$PROGRESS_FILE"

cd ..
echo ""
echo "=== Collection complete at $(date) ==="
echo ""
echo "Boss fights logged: $(wc -l < boss_fights.jsonl)"
echo ""
echo "Next step: ./analyze-boss-fights.py"
