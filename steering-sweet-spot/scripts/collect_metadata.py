"""Run briefly on remote VM after inference, before stopping the session."""
import json,subprocess,platform
from pathlib import Path
from transformers import AutoTokenizer
root=Path('/content/sweetspot');r=root/'results';data=json.loads((r/'data.json').read_text());sels=json.loads((root/'selection.json').read_text())
concepts={'ocean':'Include a clear reference to the ocean or marine life in your answer.', 'optimistic':'Use a clearly positive, hopeful, optimistic tone in your answer.', 'archaic':'Write your answer in archaic English, with words such as thou, thee, thy, hath, or doth.'}
ledger={}
for name in ['Qwen/Qwen3-4B','Qwen/Qwen3-1.7B']:
 t=AutoTokenizer.from_pretrained(name);counts={}
 for concept,suffix in concepts.items():
  n=0
  for x in data['construction']:
   for s in ['', ' '+suffix]:
    txt=t.apply_chat_template([{'role':'user','content':x['prompt']+s}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    n+=len(t(txt).input_ids)
  counts[concept]=n*(3 if name.endswith('-4B') else 1)
 ledger[name]={'vector_forward_input_tokens':counts,'note':'4B vectors captured at three layers; confirmation reuses selected vectors; 1.7B one layer per concept.'}
(r/'vector_token_counts.json').write_text(json.dumps(ledger,indent=2))
(r/'environment.json').write_text(json.dumps({'platform':platform.platform(),'python':platform.python_version(),'packages':subprocess.getoutput('pip freeze'),'gpu':subprocess.getoutput('nvidia-smi'),'gpu_query':subprocess.getoutput('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv')},indent=2))
print('Metadata and vector token counts saved')
