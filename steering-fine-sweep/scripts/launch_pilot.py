import os
import subprocess
import time

os.makedirs("/content/finesweep", exist_ok=True)
log = open("/content/finesweep/run.log", "a")
p = subprocess.Popen(
    ["nohup", "bash", "-c", "cd /content/finesweep && python -u run_gpu.py pilot"],
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print("Detached pilot PID", p.pid, "UTC epoch", time.time())
with open("/content/finesweep/pid", "w") as f:
    f.write(str(p.pid))
