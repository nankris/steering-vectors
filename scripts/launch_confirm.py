import os
import subprocess
import time

os.makedirs("/content/095c", exist_ok=True)
log = open("/content/095c/run.log", "a")
p = subprocess.Popen(
    ["nohup", "bash", "-c", "cd /content/095c && python -u run_gpu.py confirm"],
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print("Detached confirm PID", p.pid, "UTC epoch", time.time())
with open("/content/095c/pid", "w") as f:
    f.write(str(p.pid))
