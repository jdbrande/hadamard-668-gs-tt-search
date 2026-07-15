#!/usr/bin/env bash
# Supervised nonstop runner (v2): ./run_forever.sh [workers]
#
# Never relaunches a full batch. Every cycle it:
#   1. exits (printing CSV path + sequences) if SOLUTION.json exists
#   2. quarantines corrupt checkpoints / stale temps   (ctl.py clean-bad)
#   3. fills only MISSING worker ids                   (ctl.py launch-missing)
#   4. merges pools hourly                             (ctl.py merge)
# All process detection happens in Python (flock probe + ps scan) --
# no pgrep, so it behaves identically on macOS and Linux.
set -u
W=${1:-12}
WD=${WORKDIR:-work}
LAST_MERGE=0
while true; do
  if [ -f "$WD/SOLUTION.json" ]; then
    echo "SOLVED:"
    python3 ctl.py watch --workdir "$WD"
    exit 0
  fi
  python3 ctl.py clean-bad --workdir "$WD" >/dev/null 2>&1 || true
  python3 ctl.py launch-missing --workers "$W" --workdir "$WD"
  NOW=$(date +%s)
  if [ $((NOW - LAST_MERGE)) -ge 3600 ]; then
    python3 ctl.py merge --workdir "$WD" || true
    LAST_MERGE=$NOW
  fi
  sleep 30
done
