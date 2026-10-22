import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'figures';els=[]
def base(kind,x,y,w,h,**kw):
 i=len(els);e=dict(id=f'e{i}',type=kind,x=x,y=y,width=w,height=h,angle=0,strokeColor='#24374b',backgroundColor='transparent',fillStyle='solid',strokeWidth=2,strokeStyle='solid',roughness=1,opacity=100,groupIds=[],frameId=None,index=f'a{i:03}',roundness={'type':3},seed=9502+i,version=1,versionNonce=1234+i,isDeleted=False,boundElements=None,updated=1,link=None,locked=False);e.update(kw);els.append(e);return e
base('text',40,15,1100,40,text='Steering test: discovery first, fresh confirmation second',fontSize=20,fontFamily=1,textAlign='left',verticalAlign='top',containerId=None,originalText='Steering test: discovery first, fresh confirmation second',autoResize=True,lineHeight=1.25,baseline=26)
labels=[('1. Build three vectors\n16 Alpaca instructions\nOcean / optimism / archaic',40,100,'#e1effa'),('2. Pilot on Qwen3-4B\n16 different Alpaca prompts\n3 layers x 4 strengths',430,100,'#e1effa'),('3. Check + freeze choices\nQwen3-8B judge\n50 AI operator spot checks',820,100,'#fff0ca'),('6. Paired uncertainty\nConcept + useful coherence\nCompare s* against 2s*',40,390,'#e2f2e9'),('5. Independent model\nRepeat on Qwen3-1.7B\nLayer scaled by depth',430,390,'#e2f2e9'),('4. Fresh confirmation\n200 Dolly prompts / concept\nBaseline, s*, and 2s*',820,390,'#e2f2e9')]
for txt,x,y,col in labels:
 base('rectangle',x,y,340,150,backgroundColor=col)
 base('text',x+14,y+25,315,100,text=txt,fontSize=20,fontFamily=1,textAlign='left',verticalAlign='top',containerId=None,originalText=txt,autoResize=True,lineHeight=1.5,baseline=18)
for x,y,dx,dy in [(385,175,38,0),(775,175,38,0),(990,260,0,115),(810,465,-32,0),(420,465,-32,0)]:
 base('arrow',x,y,abs(dx),abs(dy),points=[[0,0],[dx,dy]],startBinding=None,endBinding=None,startArrowhead=None,endArrowhead='arrow',elbowed=False)
txt='One L4 GPU, sequential model loads. Every batch saved; no confirmation item used for selection.'
base('text',45,585,1100,30,text=txt,fontSize=20,fontFamily=1,textAlign='left',verticalAlign='top',containerId=None,originalText=txt,autoResize=True,lineHeight=1.25,baseline=18)
for e in els:
 if e['type']=='text':
  e['height']=e['fontSize']*e['lineHeight']*len(e['text'].split('\n'));e['baseline']=e['height']-e['fontSize']*.2
(p/'diagram.excalidraw').write_text(json.dumps({'type':'excalidraw','version':2,'source':'https://excalidraw.com','elements':els,'appState':{'viewBackgroundColor':'#ffffff','gridSize':None},'files':{}},indent=2))
