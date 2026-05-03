#!/bin/bash
# Run A/B comparison: batch games with each config profile.
#
# Usage:
#   bash run-ab-test.sh          # 3 games each (default)
#   bash run-ab-test.sh 5        # 5 games each
#
# After it finishes, run the encounter report to compare:
#   python3 encounter-report.py

GAMES="${1:-10}"

# Resolve the repo root from the script's own location so this works
# regardless of the caller's cwd. The previous version `cd`ed into a
# hardcoded `~/AJS_CTS/ClawTheSpire` path, which didn't exist on at
# least one environment and caused every game to fail silently with
# "No such file or directory" (see ab-test-run.log).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect which checkpoint will be used (latest version, latest gen)
CKPT_DIR=$(ls -d alphazero_checkpoints_v* 2>/dev/null | sort -V | tail -1)
CKPT_FILE=$(ls -t "$CKPT_DIR"/gen_*.pt 2>/dev/null | head -1)
CKPT_LABEL="${CKPT_DIR##*/}/${CKPT_FILE##*/}"

echo "════════════════════════════════════════════"
echo "  A/B Test: $GAMES games per profile"
echo "  Checkpoint: $CKPT_LABEL"
echo "  Repo:  $SCRIPT_DIR"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════"
echo ""

echo ">> Profile A (Champion) — $GAMES games"
echo "────────────────────────────────────────────"
for i in $(seq 1 "$GAMES"); do
    echo "  [A] Game $i of $GAMES"
    STS2_CONFIG_PROFILE=a bash play.sh batch --once
done

echo ""
echo ">> Profile B (Challenger) — $GAMES games"
echo "────────────────────────────────────────────"
for i in $(seq 1 "$GAMES"); do
    echo "  [B] Game $i of $GAMES"
    STS2_CONFIG_PROFILE=b bash play.sh batch --once
done

echo ""
echo "════════════════════════════════════════════"
echo "  Done! Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Checkpoint used: $CKPT_LABEL"
echo ""
echo "  Run the encounter report to compare:"
echo "    python3 encounter-report.py --compare"
echo ""
echo "  Check boss fight logs:"
echo "    grep '\"boss_fight\"' logs/run_*.jsonl | python3 -m json.tool"
echo "════════════════════════════════════════════"
