cd /content/w
python steer_run.py > run.log 2>&1
mv out/DONE out/DONE_main
STRENGTHS=2.25,2.5,2.75,3.5 python steer_run.py > run2.log 2>&1
