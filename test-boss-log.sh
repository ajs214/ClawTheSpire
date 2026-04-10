#!/usr/bin/env bash
# Quick smoke-test: run 2 generations with small game counts to verify
# that boss_fights.jsonl is produced and looks sane.
set -eu
cd "$(dirname "$0")"
source sts2-solver/.venv/bin/activate

# Scrub any existing log so we can see fresh output from this test
: > boss_fights.jsonl

cd sts2-solver
python -m sts2_solver.alphazero.self_play train \
    --generations 2 \
    --games-per-gen 6 \
    --sims 40 \
    --save-dir ../alphazero_checkpoints_test \
    --progress-file ../test_progress.json

cd ..
echo ""
echo "=== boss_fights.jsonl ==="
wc -l boss_fights.jsonl
echo ""
echo "=== first entry ==="
head -1 boss_fights.jsonl | python3 -m json.tool | head -40
