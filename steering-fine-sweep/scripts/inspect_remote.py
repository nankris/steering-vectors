import os
import subprocess

print("=== CHECKING LOG ===")
if os.path.exists("/content/finesweep/run.log"):
    print("run.log size:", os.path.getsize("/content/finesweep/run.log"))
    with open("/content/finesweep/run.log") as f:
        print(f.read())
else:
    print("No /content/finesweep/run.log")

print("=== PS AUX ===")
print(subprocess.getoutput("ps aux | grep python"))

print("=== GPU ===")
print(subprocess.getoutput("nvidia-smi"))
