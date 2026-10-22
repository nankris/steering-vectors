import subprocess,os,time,tarfile
root='/content/sweetspot'
print(subprocess.getoutput('tail -8 '+root+'/run.log'))
print(subprocess.getoutput("ps -eo pid,etime,args | grep '[p]ython -u run_gpu'"))
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv'))
with tarfile.open('/content/sweetspot_snapshot.tar.gz','w:gz') as tar:
 for p in ['results','run.log']:
  if os.path.exists(root+'/'+p):tar.add(root+'/'+p,arcname=p)
