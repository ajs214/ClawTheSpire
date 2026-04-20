#!/usr/bin/env bash
# restart-dashboard.sh — Kill any running dashboard and start fresh
#
# Usage:
#   bash ~/AJS_CTS/ClawTheSpire/restart-dashboard.sh

cd "$(dirname "$0")"

# Kill existing dashboard processes
EXISTING=$(pgrep -f "dashboard\.py" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Killing existing dashboard (pid $EXISTING)..."
    kill -TERM $EXISTING 2>/dev/null || true
    sleep 1
    # Force-kill if still alive
    REMAINING=$(pgrep -f "dashboard\.py" 2>/dev/null || true)
    if [ -n "$REMAINING" ]; then
        kill -KILL $REMAINING 2>/dev/null || true
        sleep 1
    fi
    echo "Done."
else
    echo "No existing dashboard found."
fi

echo "Starting dashboard..."
python3 dashboard.py &
DASH_PID=$!
echo "Dashboard running (pid $DASH_PID) — http://localhost:8050"
