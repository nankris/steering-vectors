import os
import subprocess

print("Setting up /content/finesweep...")
os.makedirs("/content/finesweep/data", exist_ok=True)
os.makedirs("/content/finesweep/results", exist_ok=True)

# Ensure packages are installed
print("Checking packages...")
subprocess.run(["pip", "install", "-q", "transformers>=4.40.0", "datasets", "accelerate"], check=True)
print("Packages installed/verified.")

# Check GPU
print("GPU info:")
subprocess.run(["nvidia-smi"], check=True)
