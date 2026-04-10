#!/usr/bin/env bash
# stop-v7-start-v8.sh — one-shot handoff from V7 to V8 training.
#
# What this does (in order):
#   1. Sends SIGTERM to any running V7 training process (self_play or
#      the train-v7-2hr.sh wrapper), waits a few seconds for a clean
#      flush of the progress file, then SIGKILLs anything still alive.
#   2. Verifies no training is still running.
#   3. Launches ./train-v8-2hr.sh in the background with nohup so the
#      terminal can be closed without killing training.
#   4. Prints the head of train-v8.log so you can confirm the seed
#      copy from V7 and the relic-pool banner actually appeared.
#
# Usage:
#   bash ~/AJS_CTS/ClawTheSpire/stop-v7-start-v8.sh
#
# Requires that train-v8-2hr.sh lives in the same directory and is
# executable (the install step below handles that).

set -u

cd "$(dirname "$0")" || exit 1

echo "=== Step 1: stopping any running V7 training ==="
STOPPED=0
for pat in "sts2_solver.alphazero.self_play" "train-v7-2hr.sh"; do
    pids=$(pgrep -f "$pat" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  SIGTERM $pat ($pids)"
        pkill -TERM -f "$pat" 2>/dev/null || true
        STOPPED=1
    fi
done

if [ "$STOPPED" -eq 1 ]; then
    echo "  Waiting 5s for clean shutdown..."
    sleep 5
fi

# Force-kill anything still alive
for pat in "sts2_solver.alphazero.self_play" "train-v7-2hr.sh"; do
    if pgrep -f "$pat" > /dev/null 2>&1; then
        echo "  SIGKILL $pat (did not exit cleanly)"
        pkill -9 -f "$pat" 2>/dev/null || true
        sleep 1
    fi
done

if pgrep -f "sts2_solver.alphazero.self_play\|train-v7-2hr.sh" > /dev/null 2>&1; then
    echo "  !! Still detecting training processes — aborting so nothing runs twice."
    pgrep -af "sts2_solver.alphazero.self_play\|train-v7-2hr.sh"
    exit 1
fi

echo "  ✓ No training processes running."
echo ""
echo "=== Step 2: launching V8 training ==="

if [ ! -x ./train-v8-2hr.sh ]; then
    chmod +x ./train-v8-2hr.sh 2>/dev/null || true
fi

if [ ! -f ./train-v8-2hr.sh ]; then
    echo "  !! train-v8-2hr.sh not found in $(pwd)"
    exit 1
fi

# Start fresh log file so we don't mix old error noise with the new run.
: > train-v8.log

nohup bash ./train-v8-2hr.sh >> train-v8.log 2>&1 &
V8_PID=$!

# Detach the new background job from this shell so closing the
# terminal doesn't HUP the training.
disown "$V8_PID" 2>/dev/null || true

echo "  V8 training launched (PID $V8_PID)"
echo "  Log: $(pwd)/train-v8.log"
echo ""
echo "  Waiting 6 seconds for the seed copy + banner to print..."
sleep 6

echo ""
echo "=== Step 3: first lines of train-v8.log ==="
tail -25 train-v8.log
echo ""
echo "=== Done ==="
echo "Training is now running in the background."
echo "Follow live with:  tail -f $(pwd)/train-v8.log"
echo "Stop manually with: pkill -f sts2_solver.alphazero.self_play"
