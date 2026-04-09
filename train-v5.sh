#!/usr/bin/env bash
# train-v5.sh — V5: Deeper search + LR warm restarts + dynamic map pathing
#
# What's new in V5 (over V4):
#   - MCTS: base 400 sims (was 200), progressive scaling 60%→140% over training
#   - MCTS: complex decisions (10+ actions) now get 2x sims (was 1.5x)
#   - LR schedule: warm restarts every ~120 gens (was single cosine decay)
#     Network gets periodic fresh starts to escape local minima
#   - Value loss weight: 0.5 (was 0.25) — network learns win prediction faster
#   - Temperature: floors at 0.2 late training (was 0.3) — more exploitation
#   - Dynamic map pathing: simulator routes based on HP/gold/deck (matches live)
#   - Improved shop: archetype-aware card buying, smart removal, relic/potion
#   - Rest sites: character-specific thresholds, organic upgrade scoring
#   - XGBoost: auto-refreshes card picker model before training starts
#     Collects 200 games of card pick data, retrains XGBoost, then
#     blended_score() ramps ML weight as wins accumulate (alpha=wins/500)
#
# Builds on V4 combat network checkpoint (auto-loads latest).
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v5.sh &   # background
#   bash ~/AJS_CTS/ClawTheSpire/train-v5.sh            # foreground

set -euo pipefail

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

# V5 gets its own dirs, but seeds from V4 checkpoint
SAVE_DIR="../alphazero_checkpoints_v5"
PROGRESS_FILE="../training_v5_progress.json"
V4_SAVE_DIR="../alphazero_checkpoints_v4"

mkdir -p "$SAVE_DIR"

# Copy latest V4 checkpoint as V5 starting point (if V5 dir is empty)
if [ -z "$(ls -A "$SAVE_DIR" 2>/dev/null)" ] && [ -d "$V4_SAVE_DIR" ]; then
    LATEST_V4=$(ls -t "$V4_SAVE_DIR"/*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_V4" ]; then
        cp "$LATEST_V4" "$SAVE_DIR/"
        echo "Seeded V5 from V4 checkpoint: $(basename "$LATEST_V4")"
    fi
fi

echo "=== STS2 AlphaZero Training V5 ==="
echo "  MCTS sims:     400 base (progressive: 240→560, scaled by complexity)"
echo "  LR schedule:   Warm restarts (T_0=120, eta_min=5e-5)"
echo "  Value weight:  0.5 (was 0.25)"
echo "  Temperature:   1.0 → 0.2 (cosine decay)"
echo "  Map pathing:   Dynamic (HP/gold/deck-aware)"
echo "  Shop logic:    Archetype-aware (organic scorer)"
echo "  Rest sites:    Character-specific thresholds"
echo "  XGBoost:       Auto-refresh before training (200 games → retrain)"
echo "  Generations:   1,080"
echo "  Games/gen:     12"
echo "  Est. duration: ~12-14 hours (deeper search = slower games)"
echo "  Save dir:      $SAVE_DIR"
echo "  Progress:      $PROGRESS_FILE"
echo ""
echo "Starting at $(date)"
echo "-----------------------------------"

python3 -m src.sts2_solver.alphazero.self_play train \
    --generations 1080 \
    --games-per-gen 12 \
    --sims 400 \
    --batch-size 64 \
    --epochs 3 \
    --lr 1e-3 \
    --temperature 1.0 \
    --save-dir "$SAVE_DIR" \
    --progress-file "$PROGRESS_FILE"

echo ""
echo "=== V5 Training complete at $(date) ==="
echo "Checkpoints saved to: $SAVE_DIR"
