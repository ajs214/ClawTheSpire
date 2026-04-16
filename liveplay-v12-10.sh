#!/bin/bash
# liveplay-v12-10.sh — Run 10 live-play games with Config A using V12's latest checkpoint.
#
# Usage:
#   bash liveplay-v12-10.sh
#   nohup bash liveplay-v12-10.sh > liveplay-v12-10.log 2>&1 &
#   tail -f liveplay-v12-10.log

GAMES=10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CKPT_DIR="alphazero_checkpoints_v12"
CKPT_FILE=$(ls -t "$CKPT_DIR"/gen_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT_FILE" ]; then
    echo "!! No V12 checkpoints found in $CKPT_DIR"
    exit 1
fi

CKPT_LABEL="${CKPT_DIR}/${CKPT_FILE##*/}"

echo "════════════════════════════════════════════"
echo "  Live Play: $GAMES games — Config A + V12"
echo "  Checkpoint: $CKPT_LABEL"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════"
echo ""

WINS=0
LOSSES=0

for i in $(seq 1 "$GAMES"); do
    echo "────────────────────────────────────────────"
    echo "  [A] Game $i of $GAMES  ($(date '+%H:%M:%S'))"
    echo "────────────────────────────────────────────"
    OUTPUT=$(STS2_CONFIG_PROFILE=a bash play.sh batch --once 2>&1)
    echo "$OUTPUT"

    # Tally wins/losses from batch_runner output
    if echo "$OUTPUT" | grep -q "Result: victory"; then
        WINS=$((WINS + 1))
    else
        LOSSES=$((LOSSES + 1))
    fi
    echo ""
done

echo "════════════════════════════════════════════"
echo "  Done! $GAMES games finished"
echo "  Wins: $WINS  Losses: $LOSSES  WR: $(( WINS * 100 / GAMES ))%"
echo "  Checkpoint: $CKPT_LABEL"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════"
