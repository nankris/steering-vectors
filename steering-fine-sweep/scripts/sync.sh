#!/bin/bash
[ -n "${COLAB_SETUP:-}" ] && source "$COLAB_SETUP"
cd "$(dirname "$0")/.."

for attempt in 1 2 3; do
    echo "Polling attempt $attempt..."
    if colab exec -s finesweep --timeout 60 -f scripts/poll.py; then
        break
    fi
    sleep 3
done

for attempt in 1 2 3; do
    if colab download -s finesweep /content/finesweep_snapshot.tar.gz results/snapshot.tar.gz; then
        break
    fi
    sleep 3
done

python3 - <<'PY'
import tarfile, os
if os.path.exists("results/snapshot.tar.gz"):
    try:
        with tarfile.open("results/snapshot.tar.gz") as t:
            t.extractall("results/extracted")
        if os.path.exists("results/extracted/results"):
            os.system("cp -ru results/extracted/results/* results/")
        if os.path.exists("results/extracted/run.log"):
            os.system("cp results/extracted/run.log figures/colab_run.txt")
    except Exception as e:
        print("Error extracting snapshot:", e)
PY

echo -n "Usage check: "
colab usage | tee -a results/usage_log.txt
