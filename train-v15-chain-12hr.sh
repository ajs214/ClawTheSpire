#!/usr/bin/env bash
# train-v15-chain-12hr.sh — Wait for current training to finish, then run another 12 hours
#
# Usage:
#   nohup bash ~/AJS_CTS/ClawTheSpire/train-v15-chain-12hr.sh > train-v15-chain-12hr.log 2>&1 &

set -uo pipefail

cd "$(dirname "$0")"

echo "=== V15 Chain: Waiting for current training to finish ==="
echo "Started at $(date)"

# Wait for any existing training process to finish
while true; do
    EXISTING=$(pgrep -f "sts2_solver.alphazero.self_play train" 2>/dev/null || true)
    if [ -z "$EXISTING" ]; then
        break
    fi
    echo "  $(date '+%H:%M:%S') — Training still running (pid $EXISTING), waiting..."
    sleep 60
done

echo ""
echo "=== Previous training finished at $(date) — starting 12-hour continuation ==="
echo ""

# Launch the continuation script (it handles everything)
exec bash "$(dirname "$0")/train-v15-continue-12hr.sh"
