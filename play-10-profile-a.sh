#!/usr/bin/env bash
# play-10-profile-a.sh — Play 10 live games using config Profile A (network-driven)
#
# Requires: Slay the Spire 2 running with STS2-Agent mod (localhost:8081)
#
# Usage:
#   bash ~/AJS_CTS/ClawTheSpire/play-10-profile-a.sh
#   bash ~/AJS_CTS/ClawTheSpire/play-10-profile-a.sh 2>&1 | tee play-a.log

set -uo pipefail

cd "$(dirname "$0")/sts2-solver"
source .venv/bin/activate

export STS2_CONFIG_PROFILE=a
export STS2_API_BASE_URL=http://127.0.0.1:8080

echo "=== Live Play: 10 Games, Profile A (Network-Driven) ==="
echo "  Config profile: A (USE_NETWORK_ROUTING=True)"
echo "  Character:      Silent"
echo "  Started at:     $(date)"
echo ""

WINS=0
LOSSES=0

for i in $(seq 1 10); do
    echo "--- Game $i/10 ---"
    python3 -m sts2_solver.batch_runner --once --character Silent 2>&1
    RC=$?
    if [ "$RC" -eq 0 ]; then
        echo "  Game $i complete."
    else
        echo "  Game $i exited with code $RC"
    fi
    echo ""
done

echo "=== 10 Games Complete at $(date) ==="
echo "Check logs/ directory for detailed run logs."
