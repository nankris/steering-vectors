import os,subprocess,tarfile,time
os.makedirs('/content/sweetspot',exist_ok=True)
with tarfile.open('/content/recovery.tar.gz') as t:t.extractall('/content/sweetspot',filter='data')
cmd='cd /content/sweetspot && pip install -q transformers==5.18.0 datasets accelerate && python -u run_gpu.py judgeconfirm && python -u baseline_concepts.py'
p=subprocess.Popen(['nohup','bash','-c',cmd],stdout=open('/content/sweetspot/run.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('Detached recovery PID',p.pid,'epoch',time.time())
