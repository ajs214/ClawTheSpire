#!/bin/bash
# Run 10 live games using Config Profile A (Champion / network routing).
#
# Usage:
#   bash run-10-profile-a.sh
#   bash run-10-profile-a.sh 2>&1 | tee profile-a-10.log

GAMES=10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect checkpoint
CKPT_DIR=$(ls -d alphazero_checkpoints_v* 2>/dev/null | sort -V | tail -1)
CKPT_FILE=$(ls -t "$CKPT_DIR"/gen_*.pt 2>/dev/null | head -1)
CKPT_LABEL="${CKPT_DIR##*/}/${CKPT_FILE##*/}"

echo "════════════════════════════════════════════"
echo "  Profile A (Champion): $GAMES games"
echo "  Checkpoint: $CKPT_LABEL"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════"
echo ""

for i in $(seq 1 "$GAMES"); do
    echo "────────────────────────────────────────────"
    echo "  [A] Game $i of $GAMES  ($(date '+%H:%M:%S'))"
    echo "────────────────────────────────────────────"
    STS2_CONFIG_PROFILE=a bash play.sh batch --once
    echo ""
done

echo "════════════════════════════════════════════"
echo "  Done! Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Checkpoint: $CKPT_LABEL"
echo ""
echo "  Analyze results:"
echo "    python3 encounter-report.py"
echo "════════════════════════════════════════════"
