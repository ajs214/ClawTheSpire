#!/usr/bin/env bash
# start-dashboard.sh — Launch the ClawTheSpire training dashboard
#
# Opens a browser to http://localhost:8090 and starts the dashboard
# server in the foreground.
#
# Usage:
#   bash ~/AJS_CTS/ClawTheSpire/start-dashboard.sh
#   bash ~/AJS_CTS/ClawTheSpire/start-dashboard.sh --port 9000

cd "$(dirname "$0")"

PORT="${1:-8090}"
# Strip --port flag if present
if [ "$PORT" = "--port" ]; then
    PORT="${2:-8090}"
fi

echo "=== ClawTheSpire Training Dashboard ==="
echo "  URL: http://localhost:$PORT"
echo "  Press Ctrl+C to stop"
echo ""

# Try to open browser (macOS / Linux)
if command -v open &>/dev/null; then
    open "http://localhost:$PORT" 2>/dev/null &
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:$PORT" 2>/dev/null &
fi

python3 dashboard.py --port "$PORT"
