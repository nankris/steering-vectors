"""Check experiment completeness and freshness; no model work."""
import json,hashlib,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];r=root/'results'
d=json.loads((r/'data.json').read_text());sets=[]
for part,n in [('construction',16),('pilot',16),('confirmation',200)]:
 assert len(d[part])==n
 for x in d[part]:assert hashlib.sha256(x['prompt'].encode()).hexdigest()==x['sha256']
 sets.append({x['sha256'] for x in d[part]})
assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
report={'freshness':'PASS','counts':{},'token_totals':{}}
for stage,n in [('pilot',592),('confirmation',2800)]:
 original={}
 for typ in ['outputs','judged']:
  data=[json.loads(x) for x in (r/f'{stage}_{typ}.jsonl').read_text().splitlines()]
  assert len(data)==n and len({x['key'] for x in data})==n
  assert all(0<=x['output_tokens']<=80 for x in data)
  if typ=='outputs':original={x['key']:x for x in data}
  else:
   assert set(original)=={x['key'] for x in data}
   for x in data:
    assert all(x[k]==v for k,v in original[x['key']].items()), x['key']
  if stage=='confirmation':
   from collections import Counter
   groups=Counter((x['model'],x['concept'],x['strength']) for x in data)
   assert len(groups)==14 and set(groups.values())=={200}
  report['counts'][stage+'_'+typ]=n
  if typ=='judged':
   assert all(x['coherent']==(x['readable'] and x['relevant']) for x in data)
   report['token_totals'][stage]={k:sum(x[k] for x in data) for k in ['input_tokens','output_tokens','judge_input_tokens','judge_output_tokens']}
assert len(json.loads((r/'manual_validation.json').read_text()))==50
assert (root/'figures/diagram.excalidraw').exists()
report['integrity']='PASS'
(r/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
