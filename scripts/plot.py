import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'figure.dpi':150,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
p=pd.read_csv(R/'pilot_metrics.csv');c=pd.read_csv(R/'confirmation_metrics.csv');cl=pd.read_csv(R/'claims.csv');sel=json.loads((ROOT/'selection.json').read_text());colors={'ocean':'#157a9c','optimistic':'#b15e16','archaic':'#7954a1'}
fig,axs=plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
for ax,(concept,v) in zip(axs,sel.items()):
 g=p[p.concept==concept];z=g.pivot(index='layer',columns='strength',values='concept_rate');co=g.pivot(index='layer',columns='strength',values='coherence');im=ax.imshow(z*100,vmin=0,vmax=100,cmap='Blues',aspect='auto');ax.set_xticks(range(4),[2,4,6,8]);ax.set_yticks(range(3),[12,24,30]);ax.set_xlabel('Strength (0.1 x reference norm per unit)');ax.set_ylabel('Layer (of 36)');ax.set_title(concept.capitalize())
 for i in range(3):
  for j in range(4):
   chosen=z.index[i]==v['layer'] and z.columns[j]==v['strength'];txt=f'{z.iloc[i,j]*100:.0f}%\n({co.iloc[i,j]*100:.0f}%)'+ (' *' if chosen else '')
   ax.text(j,i,txt,ha='center',va='center',fontsize=10,color='white' if z.iloc[i,j]>.55 else '#1a2836',fontweight='bold' if chosen else 'normal')
fig.colorbar(im,ax=axs,shrink=.7,label='Concept present (%)');fig.suptitle('Pilot: concept activation and useful answers can diverge\nEach cell: concept % (useful coherence %); * = frozen choice; n=16/cell',fontsize=14);fig.savefig(F/'pilot_heatmap.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
for ax,(concept,v) in zip(axs,sel.items()):
 g=p[p.concept==concept]
 for layer,gg in g.groupby('layer'):
  gg=gg.sort_values('strength');ax.plot(gg.coherence*100,gg.concept_rate*100,'o-',color='#afb8c0',alpha=.65,lw=1,ms=4)
 for model,marker,col in [('Qwen/Qwen3-4B','o',colors[concept]),('Qwen/Qwen3-1.7B','s','#24374b')]:
  gg=c[(c.concept==concept)&(c.model==model)].sort_values('strength');ax.plot(gg.coherence*100,gg.concept_rate*100,color=col,lw=1.5,alpha=.7)
  for _,r in gg.iterrows():
   ax.errorbar(r.coherence*100,r.concept_rate*100,xerr=[[100*max(0,r.coherence-r.coherence_lo)],[100*max(0,r.coherence_hi-r.coherence)]],yerr=[[100*max(0,r.concept_rate-r.concept_lo)],[100*max(0,r.concept_hi-r.concept_rate)]],fmt=marker,color=col,ms=7,mfc=col if r.strength==v['strength'] else 'white',capsize=2,label=model.split('/')[-1] if r.strength==v['strength'] else None)
 ax.set(xlim=(-3,103),ylim=(-3,103),title=concept.capitalize(),xlabel='Useful coherence (%)',ylabel='Concept present (%)');ax.grid(alpha=.2);ax.legend(fontsize=9,loc='upper right' if concept=='ocean' else 'lower left')
fig.suptitle('Fresh data: activation versus useful coherence\nFilled = selected strength; open = doubled; gray = pilot; Wilson 95% intervals',fontsize=14);fig.savefig(F/'frontier.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
labels=[]
for i,(_,r) in enumerate(cl.iterrows()):
 labels.append(r['model'].split('/')[-1]+' / '+r['concept']);ax.errorbar(r.cliff*100,i,xerr=[[100*max(0,r.cliff-r.cliff_lo)],[100*max(0,r.cliff_hi-r.cliff)]],fmt='o',color=colors[r.concept],capsize=4,ms=7);ax.text(104,i,r.H2,va='center',fontsize=10)
ax.axvline(20,color='#bd3b38',ls='--',label='Registered minimum cliff: 20 pp');ax.axvline(0,color='#999999',lw=1);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(-15,115);ax.set_xlabel('Useful-coherence decrease when strength doubles (percentage points)');ax.set_title('Does coherence fall sharply beyond the selected setting?\nPaired bootstrap 95% intervals; 200 fresh prompts per row');ax.grid(axis='x',alpha=.2);ax.legend(loc='upper left');fig.savefig(F/'coherence_cliff.png');plt.close(fig)
def pct(x,lo,hi):return f'{100*x:.1f} [{100*lo:.1f}, {100*hi:.1f}]'
table=[]
for _,r in cl.iterrows():
 m=c[(c.model==r.model)&(c.concept==r.concept)&(c.strength==r.strength)].iloc[0]
 table.append([r.model.split('/')[-1].replace('Qwen3-',''),r.concept,f'{r.layer} / {r.strength}',pct(m.concept_rate,m.concept_lo,m.concept_hi),pct(m.coherence,m.coherence_lo,m.coherence_hi),pct(m.loss,m.loss_lo,m.loss_hi),f'{m.median_ppl:.1f}',r.H1+' / '+r.H2])
columns=['Model','Concept','Layer / s*','Concept % [95% CI]','Useful % [95% CI]','Loss pp [95% CI]','Median PPL','H1 / H2']
pd.DataFrame(table,columns=columns).to_csv(R/'main_table.csv',index=False)
fig,ax=plt.subplots(figsize=(16,4.8));ax.axis('off');t=ax.table(cellText=table,colLabels=columns,cellLoc='center',loc='center',colWidths=[.05,.07,.07,.17,.17,.17,.08,.22]);t.auto_set_font_size(False);t.set_fontsize(10.5);t.scale(1,2.35)
for (i,j),cell in t.get_celld().items():
 cell.set_edgecolor('white');cell.set_facecolor('#24374b' if i==0 else ('#edf3f7' if i%2 else '#ffffff'))
 if i==0:cell.set_text_props(color='white',weight='bold')
ax.set_title('Frozen settings on fresh data: 200 prompts per concept and model',fontsize=16,pad=24)
bs=[]
for _,r in c[c.concept=='baseline'].iterrows():bs.append(f'{r.model.split("/")[-1]} baseline useful {r.coherence*100:.1f}%; median PPL {r.median_ppl:.1f}')
fig.text(.5,.06,' | '.join(bs)+'\nUseful = readable AND relevant. Positive loss = worse than baseline. Pointwise 95% CIs. H1/H2 follow the registered margins.',ha='center',fontsize=10)
fig.savefig(F/'table.png',bbox_inches='tight');plt.close(fig)
