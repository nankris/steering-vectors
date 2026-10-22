#!/bin/bash
# Pull results from the Colab VM (session steering) into results/raw/. Safe to rerun.
cd "$(dirname "$0")/.."
cat > /tmp/packsteer.py <<'P'
import subprocess
subprocess.run("cd /content/w && tar czf /content/w/out.tgz out/*.json out/DONE* run.log run2.log 2>/dev/null; nvidia-smi > /content/w/nvsmi.txt", shell=True)
P
colab exec -s steering -f /tmp/packsteer.py >/dev/null
colab download -s steering /content/w/out.tgz /tmp/outsteer.tgz && mkdir -p /tmp/xsteer && tar xzf /tmp/outsteer.tgz -C /tmp/xsteer && cp /tmp/xsteer/out/* results/raw/ && cp /tmp/xsteer/run*.log results/ 2>/dev/null
colab download -s steering /content/w/nvsmi.txt results/nvidia_smi.txt
ls results/raw | wc -l
