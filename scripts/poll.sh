#!/bin/bash
# Pull every 3 min until both main and fine runs are done (or VM vanishes).
cd "$(dirname "$0")/.."; source /workspace/tools/setup.sh >/dev/null 2>&1
for i in $(seq 18); do
  sleep 180
  ./scripts/pull.sh >/dev/null 2>&1 || { echo "PULL FAILED"; colab sessions; }
  echo "$(date +%T) files=$(ls results/raw | wc -l) $(tail -1 results/run.log | cut -c1-80) | $(tail -1 results/run2.log 2>/dev/null | cut -c1-60)"
  [ -f results/raw/DONE ] && [ -f results/raw/DONE_main ] && { echo ALLDONE; break; }
done
