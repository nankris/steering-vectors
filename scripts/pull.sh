#!/bin/bash
# Pull results from the Colab VM (session b6-095) into results/raw/. Safe to rerun.
cd "$(dirname "$0")/.."
cat > /tmp/pack095.py <<'P'
import subprocess
subprocess.run("cd /content/w && tar czf /content/w/out.tgz out/*.json out/DONE* run.log run2.log 2>/dev/null; nvidia-smi > /content/w/nvsmi.txt", shell=True)
P
colab exec -s b6-095 -f /tmp/pack095.py >/dev/null
colab download -s b6-095 /content/w/out.tgz /tmp/out095.tgz && mkdir -p /tmp/x095 && tar xzf /tmp/out095.tgz -C /tmp/x095 && cp /tmp/x095/out/* results/raw/ && cp /tmp/x095/run*.log results/ 2>/dev/null
colab download -s b6-095 /content/w/nvsmi.txt results/nvidia_smi.txt
ls results/raw | wc -l
