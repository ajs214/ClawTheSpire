#!/usr/bin/env bash
# Launch the ClawTheSpire training dashboard
# Usage: bash open-dashboard.sh

cd "$(dirname "$0")" || exit 1

PORT=8090

# Kill any existing dashboard process on this port
lsof -ti :"$PORT" 2>/dev/null | xargs kill 2>/dev/null

echo "Starting ClawTheSpire dashboard on http://localhost:$PORT ..."
python3 dashboard.py --port "$PORT" &
DASH_PID=$!

# Wait a moment for the server to start
sleep 1

# Open in default browser
if command -v open &>/dev/null; then
    open "http://localhost:$PORT"
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:$PORT"
else
    echo "Open http://localhost:$PORT in your browser"
fi

echo "Dashboard running (PID $DASH_PID). Press Ctrl+C to stop."
wait $DASH_PID
