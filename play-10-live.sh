#!/usr/bin/env bash
# play-10-live.sh — Play 10 live games with full solver TUI visible
#
# Shows the Rich Live dashboard (solver panel, advisor panel, log) for
# each game.  Do NOT pipe through tee — the TUI uses terminal escape
# codes that don't work in a pipe.
#
# Usage:
#   bash ~/AJS_CTS/ClawTheSpire/play-10-live.sh

set -uo pipefail

cd "$(dirname "$0")/sts2-solver"
source .venv/bin/activate

export STS2_CONFIG_PROFILE=a
export STS2_API_BASE_URL=http://127.0.0.1:8080

echo "=== Live Play: 10 Games, Profile A (Network-Driven) ==="
echo "  Config profile: A (USE_NETWORK_ROUTING=True)"
echo "  Character:      Silent"
echo "  Solver TUI:     ENABLED"
echo "  Started at:     $(date)"
echo ""

for i in $(seq 1 10); do
    echo "--- Game $i/10 ---"
    python3 -m sts2_solver.batch_runner --once --character Silent
    RC=$?
    if [ "$RC" -eq 0 ]; then
        echo "  Game $i complete."
    else
        echo "  Game $i exited with code $RC"
    fi
    echo ""
    sleep 5
done

echo "=== 10 Games Complete at $(date) ==="
echo "Check logs/ directory for detailed run logs."
