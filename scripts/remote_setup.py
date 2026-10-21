import os
import subprocess

print("Setting up /content/095c...")
os.makedirs("/content/095c/data", exist_ok=True)
os.makedirs("/content/095c/results", exist_ok=True)

# Ensure packages are installed
print("Checking packages...")
subprocess.run(["pip", "install", "-q", "transformers>=4.40.0", "datasets", "accelerate"], check=True)
print("Packages installed/verified.")

# Check GPU
print("GPU info:")
subprocess.run(["nvidia-smi"], check=True)
