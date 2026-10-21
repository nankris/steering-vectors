#!/bin/bash
set -e
source /workspace/tools/setup.sh
cd /workspace/work/tig/095b-steering-sweet-spot
timeout 55s colab exec -s b9-095b -f scripts/poll.py
timeout 55s colab download -s b9-095b /content/095b_snapshot.tar.gz results/snapshot.tar.gz
python - <<'PY'
import tarfile
with tarfile.open('results/snapshot.tar.gz') as t:t.extractall('.',filter='data')
PY
cp run.log figures/colab_run.txt
date -u --iso-8601=seconds >> results/usage_checks.txt
colab usage | tee -a results/usage_checks.txt
