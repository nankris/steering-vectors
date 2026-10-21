import os,subprocess,tarfile,time
os.makedirs('/content/095b',exist_ok=True)
with tarfile.open('/content/recovery.tar.gz') as t:t.extractall('/content/095b',filter='data')
cmd='cd /content/095b && pip install -q transformers==5.18.0 datasets accelerate && python -u run_gpu.py judgeconfirm && python -u baseline_concepts.py'
p=subprocess.Popen(['nohup','bash','-c',cmd],stdout=open('/content/095b/run.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('Detached recovery PID',p.pid,'epoch',time.time())
