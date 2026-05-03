#!/usr/bin/env bash
# train-v6.sh — V6: Relic-aware picker + unified deck eval
#
# What's new in V6 (over V5):
#   - XGBoost card picker is gone. The organic picker is now the single
#     source of truth for card value (score_card / pick_card); no more
#     subprocess refresh, no more alpha-gated blend.
#   - Unified deck eval: shop removals, event removals, rest-site
#     upgrades, shop buys, and advisor deck-selects all share the same
#     score_card / _organic_removal_score / _organic_upgrade_value
#     helpers, so training and live play rank cards identically.
#   - Relaxed deck-size curve: size penalty starts at 17 cards (was
#     tighter), ramps 0.04 → 0.22 at 24+. Early-run card picks see
#     zero pressure so the deck can actually grow.
#   - No duplicate penalty on strong non-Power cards: any non-Power
#     card whose intrinsic power score clears 0.35 can be stacked
#     without any dup penalty. Powers still get hit hard.
#   - New relic_synergy module:
#       * relic_card_bonus bumps individual card scores by up to ±0.25
#         based on owned relics (Wrist Blade → 0-cost attacks, Snecko
#         Skull → poison, Paper Krane → Weak, Dead Branch → Exhaust,
#         Velvet Choker/Ectoplasm penalties, etc.)
#       * score_relic_for_deck grades shop/boss relics by the deck's
#         mechanical fingerprint instead of an archetype-tag match.
#     Threaded through pick_card / score_card, the simulator's shop
#     and card rewards, deterministic_advisor, and the alphazero
#     full_run card-reward + shop-fallback paths.
#   - Per-version boss-fight log: each run's detail goes into
#     <save-dir>/boss_fights.jsonl so V6 doesn't clobber V5's log.
#   - Progress file now tracks boss_fight_win_rate (wins / reached-boss)
#     so the dashboard can show "how often we actually close the run".
#
# Builds on V5 combat network checkpoint (auto-loads latest).
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v6.sh &   # background
#   bash ~/AJS_CTS/ClawTheSpire/train-v6.sh            # foreground

set -euo pipefail

cd "$(dirname "$0")/sts2-solver"

source .venv/bin/activate

# V6 gets its own dirs, but seeds from V5 checkpoint
SAVE_DIR="../alphazero_checkpoints_v6"
PROGRESS_FILE="../training_v6_progress.json"
BOSS_LOG_FILE="../alphazero_checkpoints_v6/boss_fights.jsonl"
V5_SAVE_DIR="../alphazero_checkpoints_v5"

mkdir -p "$SAVE_DIR"

# Copy latest V5 checkpoint as V6 starting point (if V6 dir is empty)
if [ -z "$(ls -A "$SAVE_DIR" 2>/dev/null)" ] && [ -d "$V5_SAVE_DIR" ]; then
    LATEST_V5=$(ls -t "$V5_SAVE_DIR"/*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_V5" ]; then
        cp "$LATEST_V5" "$SAVE_DIR/"
        echo "Seeded V6 from V5 checkpoint: $(basename "$LATEST_V5")"
    fi
fi

echo "=== STS2 AlphaZero Training V6 ==="
echo "  MCTS sims:     400 base (progressive: 240→560, scaled by complexity)"
echo "  LR schedule:   Warm restarts (T_0=120, eta_min=5e-5)"
echo "  Value weight:  0.5"
echo "  Temperature:   1.0 → 0.2 (cosine decay)"
echo "  Card picker:   Organic only (XGBoost removed)"
echo "  Deck eval:     Unified score_card across all surfaces"
echo "  Deck size:     Relaxed curve (no penalty until 17+)"
echo "  Dup penalty:   Strong non-Power cards exempt"
echo "  Relics:        Threaded through pick_card & shop scoring"
echo "  Boss log:      $BOSS_LOG_FILE"
echo "  Generations:   1,080"
echo "  Games/gen:     12"
echo "  Est. duration: ~12-14 hours"
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
    --progress-file "$PROGRESS_FILE" \
    --boss-log-file "$BOSS_LOG_FILE"

echo ""
echo "=== V6 Training complete at $(date) ==="
echo "Checkpoints saved to: $SAVE_DIR"
echo "Boss-fight log:       $BOSS_LOG_FILE"
