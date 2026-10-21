import os,subprocess,time
root='/content/095b'
assert os.path.exists(root+'/selection.json'), 'Upload committed selection first'
assert os.path.exists(root+'/results/judgepilot.done'), 'Pilot judge must complete first'
log=open(root+'/run.log','a')
p=subprocess.Popen(['nohup','bash','-c','cd /content/095b && python -u run_gpu.py confirm4 && python -u run_gpu.py confirm17 && python -u run_gpu.py judgeconfirm'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
open(root+'/pid','w').write(str(p.pid))
print('Detached confirmation PID',p.pid,'UTC epoch',time.time())
