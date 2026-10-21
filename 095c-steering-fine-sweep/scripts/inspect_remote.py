import os
import subprocess

print("=== CHECKING LOG ===")
if os.path.exists("/content/095c/run.log"):
    print("run.log size:", os.path.getsize("/content/095c/run.log"))
    with open("/content/095c/run.log") as f:
        print(f.read())
else:
    print("No /content/095c/run.log")

print("=== PS AUX ===")
print(subprocess.getoutput("ps aux | grep python"))

print("=== GPU ===")
print(subprocess.getoutput("nvidia-smi"))
