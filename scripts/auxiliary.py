import sys,json
from pathlib import Path
import pandas as pd
from analyze import rows,diff,wilson
R=Path(__file__).resolve().parents[1]/'results'
df=pd.DataFrame(rows('confirmation_judged.jsonl'));aux=pd.DataFrame(rows('baseline_concepts.jsonl'));assert len(aux)==800 and aux.key.nunique()==800
out=[]
for (model,concept,layer,strength),g in df[df.concept!='baseline'].groupby(['model','concept','layer','strength']):
 b=df[(df.model==model)&(df.concept=='baseline')] if concept=='ocean' else aux[(aux.model==model)&(aux.judged_concept==concept)]
 m=g.merge(b[['id','concept_present']],on='id',suffixes=('','_base'));assert len(m)==200
 delta=m.concept_present.mean()-m.concept_present_base.mean();ci=diff(m.concept_present,m.concept_present_base);lo,hi=wilson(int(m.concept_present_base.sum()),len(m))
 out.append(dict(model=model,concept=concept,layer=int(layer),strength=int(strength),baseline_rate=m.concept_present_base.mean(),baseline_lo=lo,baseline_hi=hi,steered_rate=m.concept_present.mean(),uplift=delta,uplift_lo=ci[0],uplift_hi=ci[1]))
pd.DataFrame(out).to_csv(R/'concept_uplifts.csv',index=False)
json.dump({'auxiliary_judge_input_tokens':int(aux.judge_input_tokens.sum()),'auxiliary_judge_decision_tokens':int(aux.judge_output_tokens.sum())},open(R/'auxiliary_tokens.json','w'),indent=2)
print(pd.DataFrame(out).to_string(index=False))
