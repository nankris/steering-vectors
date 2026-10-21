"""CPU analysis: freeze pilot selection, then evaluate unchanged registered claims."""
import json,math,argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def rows(name):return [json.loads(x) for x in (R/name).read_text().splitlines() if x.strip()]
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
def diff(a,b):
 x=np.array(a,float)-np.array(b,float);rng=np.random.default_rng(9503);means=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1);return np.quantile(means,[.025,.975]).tolist()
def paired(df,base):
 m=df.merge(base[['id','coherent']],on='id',suffixes=('','_base'));assert len(m)==len(df)
 return m.coherent_base.astype(int).mean()-m.coherent.astype(int).mean(),diff(m.coherent_base,m.coherent)
def summarize(df):
 records=[]
 for (model,concept,layer,strength),g in df.groupby(['model','concept','layer','strength']):
  b=df[(df.model==model)&(df.concept=='baseline')];loss,ci=paired(g,b)
  k=int(g.concept_present.sum());n=len(g);lo,hi=wilson(k,n);cl,ch=wilson(int(g.coherent.sum()),n)
  records.append(dict(model=model,concept=concept,layer=int(layer),strength=int(strength),n=n,concept_rate=k/n,concept_lo=lo,concept_hi=hi,coherence=g.coherent.mean(),coherence_lo=cl,coherence_hi=ch,loss=loss,loss_lo=ci[0],loss_hi=ci[1],readable=g.readable.mean(),relevant=g.relevant.mean(),median_ppl=g.ppl.median(),median_nll=g.nll.median(),truncated=g.truncated.mean(),output_tokens=int(g.output_tokens.sum()),input_tokens=int(g.input_tokens.sum()),judge_input_tokens=int(g.judge_input_tokens.sum()),judge_output_tokens=int(g.judge_output_tokens.sum())))
 return pd.DataFrame(records)
def select():
 df=pd.DataFrame(rows('pilot_judged.jsonl'));assert len(df)==592 and df.key.nunique()==592
 s=summarize(df);s.to_csv(R/'pilot_metrics.csv',index=False);sel={}
 for concept,g in s[s.concept!='baseline'].groupby('concept'):
  qualified=g[g.loss<=.125]
  if len(qualified):v=qualified.sort_values(['concept_rate','coherence','strength','layer'],ascending=[False,False,True,False]).iloc[0]
  else:
   g=g.assign(score=g.concept_rate-g.loss.clip(lower=0));v=g.sort_values(['score','coherence','strength','layer'],ascending=[False,False,True,False]).iloc[0]
  half=(v.concept_hi-v.concept_lo)/2;lh=(v.loss_hi-v.loss_lo)/2
  sel[concept]={'layer':int(v.layer),'strength':int(v.strength),'X':max(0,float(v.concept_rate-half)),'Y':max(0,min(1,float(v.loss+lh))),'pilot_concept':float(v.concept_rate),'pilot_coherence':float(v.coherence),'pilot_loss':float(v.loss),'H2_margin':.20}
 (ROOT/'selection.json').write_text(json.dumps(sel,indent=2));print(json.dumps(sel,indent=2))
 rng=np.random.default_rng(9510);ix=rng.choice(len(df),50,replace=False);val=df.iloc[ix].copy();val['manual_concept']='';val['manual_readable']='';val['manual_relevant']='';val[['key','prompt','response','concept','layer','strength','manual_concept','manual_readable','manual_relevant']].to_json(R/'validation_to_label.json',orient='records',indent=2)
def confirm():
 df=pd.DataFrame(rows('confirmation_judged.jsonl'));assert len(df)==2800 and df.key.nunique()==2800
 s=summarize(df);s.to_csv(R/'confirmation_metrics.csv',index=False);sel=json.loads((ROOT/'selection.json').read_text());claims=[]
 for model in df.model.unique():
  for c,v in sel.items():
   ss=s[(s.model==model)&(s.concept==c)];a=ss[ss.strength==v['strength']].iloc[0]
   h1='PASS' if a.concept_lo>=v['X'] and a.loss_hi<=v['Y'] else 'FAIL' if a.concept_hi<v['X'] or a.loss_lo>v['Y'] else 'UNRESOLVED'
   small=df[(df.model==model)&(df.concept==c)&(df.strength==v['strength'])];large=df[(df.model==model)&(df.concept==c)&(df.strength==2*v['strength'])];m=small.merge(large,on='id',suffixes=('_a','_b'));ci=diff(m.coherent_a,m.coherent_b);delta=float(m.coherent_a.mean()-m.coherent_b.mean());h2='PASS' if ci[0]>=.2 else 'FAIL' if ci[1]<.2 else 'UNRESOLVED'
   claims.append(dict(model=model,concept=c,layer=int(a.layer),strength=v['strength'],X=v['X'],Y=v['Y'],H1=h1,H2=h2,cliff=delta,cliff_lo=ci[0],cliff_hi=ci[1]))
 val=pd.DataFrame(json.loads((R/'manual_validation.json').read_text()));jd=pd.DataFrame(rows('pilot_judged.jsonl'));m=val.merge(jd,on='key');checks={}
 for manual,pred in [('manual_concept','concept_present'),('manual_readable','readable'),('manual_relevant','relevant')]:
  a=(m[manual].astype(bool)==m[pred]);lo,hi=wilson(int(a.sum()),len(a));checks[pred]={'correct':int(a.sum()),'n':len(a),'accuracy':float(a.mean()),'lo':lo,'hi':hi,'disagreements':m.loc[~a,'key'].tolist()}
 trueuse=m.manual_readable.astype(bool)&m.manual_relevant.astype(bool);a=trueuse==m.coherent;lo,hi=wilson(int(a.sum()),len(a));checks['useful_coherence']={'correct':int(a.sum()),'n':len(a),'accuracy':float(a.mean()),'lo':lo,'hi':hi,'disagreements':m.loc[~a,'key'].tolist()}
 validated=checks['concept_present']['accuracy']>=.8 and checks['useful_coherence']['accuracy']>=.8
 allstatuses=[c[k] for c in claims for k in ['H1','H2']]
 verdict='INCONCLUSIVE' if not validated else 'DISPROVEN' if 'FAIL' in allstatuses else 'PROVEN' if set(allstatuses)=={'PASS'} else 'INCONCLUSIVE'
 result={'verdict':verdict,'judge_validation_pass':validated,'validation':checks,'claims':claims,'generations':len(df),'tokens':{k:int(df[k].sum()) for k in ['input_tokens','output_tokens','judge_input_tokens','judge_output_tokens']}}
 (R/'verdict.json').write_text(json.dumps(result,indent=2));pd.DataFrame(claims).to_csv(R/'claims.csv',index=False);print(json.dumps(result,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['select','confirm']);a=ap.parse_args();select() if a.stage=='select' else confirm()
