import os,subprocess,time,json
os.makedirs('/content/sweetspot',exist_ok=True)
log=open('/content/sweetspot/run.log','a')
p=subprocess.Popen(['nohup','bash','-c','cd /content/sweetspot && python -u run_gpu.py pilot && python -u run_gpu.py judgepilot'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Detached pilot PID',p.pid,'UTC epoch',time.time())
open('/content/sweetspot/pid','w').write(str(p.pid))
