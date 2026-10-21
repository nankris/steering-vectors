"""Check that resumed judge reproduces saved pre-interruption labels."""
from run_gpu import *
import platform
old=read('confirmation_judged.jsonl')[:1620]
rng=np.random.default_rng(9595);subset=[old[int(i)] for i in rng.choice(len(old),36,replace=False)]
if not (OUT/'recoveryqa_outputs.jsonl').exists():append('recoveryqa_outputs.jsonl',subset)
judge('recoveryqa')
a={r['key']:r for r in subset};b=read('recoveryqa_judged.jsonl');fields=['concept_present','readable','relevant'];comparison={f:sum(a[r['key']][f]==r[f] for r in b) for f in fields}
(OUT/'recovery_qa.json').write_text(json.dumps({'n':len(b),'matches':comparison,'note':'Random seed 9595, sample from first 1620 labels. Diagnostic only; never changes old labels or hypotheses.'},indent=2))
(OUT/'recovery_environment.json').write_text(json.dumps({'platform':platform.platform(),'python':platform.python_version(),'packages':subprocess.getoutput('pip freeze'),'gpu':subprocess.getoutput('nvidia-smi')},indent=2))
log('DONE recovery QA',comparison)
