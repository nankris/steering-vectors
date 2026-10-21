import os
import subprocess
import tarfile

root = "/content/095c"
log_file = os.path.join(root, "run.log")
if os.path.exists(log_file):
    print("=== TAIL LOG ===")
    print(subprocess.getoutput(f"tail -n 50 {log_file}"))

print("=== PROCESSES ===")
print(subprocess.getoutput("ps -eo pid,etime,args | grep -E '[p]ython3? -u run_gpu.py'"))

print("=== GPU STATUS ===")
print(subprocess.getoutput("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv"))

tar_path = "/content/095c_snapshot.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    for item in ["results", "run.log"]:
        full_path = os.path.join(root, item)
        if os.path.exists(full_path):
            tar.add(full_path, arcname=item)
print("Snapshot created successfully.")
