#!/usr/bin/env bash
# restart-dashboard.sh — Kill ALL running dashboards and start fresh
#
# Usage:
#   bash ~/AJS_CTS/ClawTheSpire/restart-dashboard.sh

cd "$(dirname "$0")"

# Kill everything listening on 8090 or running dashboard.py
echo "Stopping existing dashboard processes..."
PIDS=$(lsof -ti :8090 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "  Killing processes on port 8090: $PIDS"
    echo "$PIDS" | xargs kill -TERM 2>/dev/null || true
    sleep 1
fi

# Also catch any dashboard.py processes that might be on other ports
MORE=$(pgrep -f "python.*dashboard\.py" 2>/dev/null || true)
if [ -n "$MORE" ]; then
    echo "  Killing dashboard.py processes: $MORE"
    echo "$MORE" | xargs kill -TERM 2>/dev/null || true
    sleep 1
fi

# Force-kill stragglers
REMAINING=$(lsof -ti :8090 2>/dev/null; pgrep -f "python.*dashboard\.py" 2>/dev/null) || true
if [ -n "$REMAINING" ]; then
    echo "  Force-killing stragglers: $REMAINING"
    echo "$REMAINING" | sort -u | xargs kill -KILL 2>/dev/null || true
    sleep 1
fi

echo "Starting dashboard..."
python3 dashboard.py &
DASH_PID=$!
sleep 1

# Verify it's running
if kill -0 "$DASH_PID" 2>/dev/null; then
    echo "Dashboard running (pid $DASH_PID) — http://localhost:8090"
    open "http://localhost:8090" 2>/dev/null || true
else
    echo "ERROR: Dashboard failed to start. Check dashboard.py for errors."
    exit 1
fi
