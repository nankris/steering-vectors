"""Rebuild timing/accounting from saved CLI events and persisted run markers."""
import json,re,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'results';root=R.parent
utc=datetime.timezone.utc;now=datetime.datetime.now(utc)
events=[json.loads(x) for x in (R/'recovery_cli.jsonl').read_text().splitlines()]
def dt(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
created=[dt(x['timestamp']) for x in events if x['event_type']=='session_created'];stopped=[dt(x['timestamp']) for x in events if x['event_type']=='session_terminated'];assert len(created)==len(stopped)==2
intervals=[{'created_utc':a.isoformat(),'terminated_utc':b.isoformat(),'minutes':(b-a).total_seconds()/60,'cu_estimated':(b-a).total_seconds()/3600*1.54} for a,b in zip(created,stopped)]
t=json.loads((R/'timing.json').read_text());start=dt(t['start_ist']);t.update(end_ist=now.astimezone(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),actual_minutes=(now-start).total_seconds()/60,GPU='NVIDIA L4',cu_estimated=sum(x['cu_estimated'] for x in intervals),allocations=intervals)
log=(root/'run.log').read_text();t['stage_minutes']={'setup':((dt('2026-10-18T23:30:46Z')-start).total_seconds()+(dt('2026-10-19T03:48:01Z')-created[1]).total_seconds())/60,'gpu_run':sum(float(x) for x in re.findall(r'DONE (?:pilot|confirm4|confirm17) seconds ([\d.]+)',log))/60,'eval':None,'writeup':(now-dt('2026-10-19T03:48:00Z')).total_seconds()/60}
# Original incomplete judge duration ends at the last saved pre-reboot batch.
eval_seconds=173.725+325+sum(float(x) for x in re.findall(r'DONE judgeconfirm seconds ([\d.]+)',log))
# Auxiliary and diagnostic include their model load time via timestamp span.
for before,after in [('LOAD Qwen/Qwen3-8B','DONE baseline_concepts'),('LOAD Qwen/Qwen3-8B','DONE recovery QA')]:
 matches=[(dt(m.group(1)),m.group(2)) for m in re.finditer(r'(?m)^(\S+Z) (.*)$',log)]
 end=next((a for a,b in matches if after in b),None)
 if end:
  begin=max(a for a,b in matches if before in b and a<end);eval_seconds+=(end-begin).total_seconds()
t['stage_minutes']['eval']=eval_seconds/60
t['interruption_minutes']=(dt('2026-10-19T03:47:41.046432Z')-dt('2026-10-18T23:53:46Z')).total_seconds()/60
t['stage_timing_note']='Reconstructed from persisted log timestamps. GPU run = completed generator stage durations. Eval includes partial original judge plus recovered judge, auxiliary and QA. Writeup begins approximately 03:48 UTC and overlaps recovery eval; stage times are not additive. The interruption is the gap without newly persisted local model results, not measured GPU idle time.'
t['cu_method']='Observed 1.54 CU/hour times both recorded allocation lifecycle intervals. First termination reason is pruned, so first allocation time may overestimate actual billed time. Shared account balance is not experiment-specific. Well below 6 CU even assigning all prior account debit (2.35 CU) plus recovery to this experiment.'
(R/'timing.json').write_text(json.dumps(t,indent=2));b=json.loads((R/'budget.json').read_text());b.update(allocations=intervals,total_cu_estimated=t['cu_estimated'],cost_method=t['cu_method']);(R/'budget.json').write_text(json.dumps(b,indent=2));print(json.dumps(t,indent=2))
