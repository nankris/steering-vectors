import os,subprocess,time,json
os.makedirs('/content/095b',exist_ok=True)
log=open('/content/095b/run.log','a')
p=subprocess.Popen(['nohup','bash','-c','cd /content/095b && python -u run_gpu.py pilot && python -u run_gpu.py judgepilot'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Detached pilot PID',p.pid,'UTC epoch',time.time())
open('/content/095b/pid','w').write(str(p.pid))
