#!/usr/bin/env bash
# restart-v4.sh — Kill any running V4 training and restart with latest code

set -euo pipefail

echo "Stopping any running V4 training..."
pkill -f "self_play.*v4" 2>/dev/null && echo "  Killed existing process" || echo "  No running process found"
sleep 2

cd "$(dirname "$0")"

echo "Starting V4 training..."
nohup bash train-v4.sh > train-v4.log 2>&1 &
echo "  PID: $!"
echo ""
echo "Monitor with:"
echo "  tail -f ~/AJS_CTS/ClawTheSpire/train-v4.log"
echo "  bash ~/AJS_CTS/ClawTheSpire/open-dashboard.sh"
