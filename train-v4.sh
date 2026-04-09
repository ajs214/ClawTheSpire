#!/usr/bin/env bash
# train-v4.sh — V4: Property-based organic card picker + cleaned card data
#
# What's new in V4 (over V3):
#   - Organic card picker: momentum scoring, archetype detection from card
#     properties (vars, powers_applied), no hardcoded name lists
#   - Card data parser: captures vars (PoisonPerTurn, IntangiblePower, etc.),
#     rarity, and description from raw JSON
#   - XGBoost features: property-based archetype counts (replaces stale
#     frozenset name lists with 15 ghost STS1 cards removed)
#   - Archetype tracking: per-game archetype + commitment logged to progress
#   - Alpha-blended ML handoff: starts pure rule-based, ramps to XGBoost
#     as wins accumulate (threshold: 500 wins)
#
# Builds on V3 combat network checkpoint (auto-loads latest).
# Uses its own checkpoint dir + progress file for clean comparison.
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v4.sh &   # background
#   bash ~/AJS_CTS/ClawTheSpire/train-v4.sh            # foreground

set -euo pipefail

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

# V4 gets its own dirs, but seeds from V3 checkpoint
SAVE_DIR="../alphazero_checkpoints_v4"
PROGRESS_FILE="../training_v4_progress.json"
V3_SAVE_DIR="../alphazero_checkpoints_v3"

mkdir -p "$SAVE_DIR"

# Copy latest V3 checkpoint as V4 starting point (if V4 dir is empty)
if [ -z "$(ls -A "$SAVE_DIR" 2>/dev/null)" ] && [ -d "$V3_SAVE_DIR" ]; then
    LATEST_V3=$(ls -t "$V3_SAVE_DIR"/*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_V3" ]; then
        cp "$LATEST_V3" "$SAVE_DIR/"
        echo "Seeded V4 from V3 checkpoint: $(basename "$LATEST_V3")"
    fi
fi

echo "=== STS2 AlphaZero Training V4 ==="
echo "  Card picker:   Organic (property-based, alpha-blended ML)"
echo "  Card data:     vars + rarity + description parsed"
echo "  Archetype:     detected from card mechanics, not name lists"
echo "  Generations:   1,080"
echo "  Games/gen:     12"
echo "  MCTS sims:     200 base (dynamic scaling)"
echo "  Est. duration: ~6 hours"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo ""
echo "Starting at $(date)"
echo "-----------------------------------"

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 1080 \
    --games-per-gen 12 \
    --sims 200 \
    --batch-size 64 \
    --epochs 3 \
    --lr 1e-3 \
    --temperature 1.0 \
    --save-dir "$SAVE_DIR" \
    --progress-file "$PROGRESS_FILE"

echo ""
echo "=== V4 Training complete at $(date) ==="
echo "Checkpoints saved to: $SAVE_DIR"
