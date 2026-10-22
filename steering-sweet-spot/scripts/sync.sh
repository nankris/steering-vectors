#!/bin/bash
set -e
[ -n "${COLAB_SETUP:-}" ] && source "$COLAB_SETUP"
cd "$(dirname "$0")/.."
timeout 55s colab exec -s sweetspot -f scripts/poll.py
timeout 55s colab download -s sweetspot /content/sweetspot_snapshot.tar.gz results/snapshot.tar.gz
python - <<'PY'
import tarfile
with tarfile.open('results/snapshot.tar.gz') as t:t.extractall('.',filter='data')
PY
cp run.log figures/colab_run.txt
date -u --iso-8601=seconds >> results/usage_checks.txt
colab usage | tee -a results/usage_checks.txt
