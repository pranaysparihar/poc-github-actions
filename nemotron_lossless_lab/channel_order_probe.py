#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,mmap,re,struct,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
MID='nvidia/Nemotron-3-Embed-1B-BF16';FN='model.safetensors';SHA='f959c3b04e66b42de280bfb97c140cb7e0bfe25e3ecb0b4464c68a8436b2d04f'
@dataclass(frozen=True)
class T:name:str;dtype:str;shape:tuple[int,...];start:int;end:int

def dg(p):
 h=hashlib.sha256();f=open(p,'rb')
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 f.close();return h.hexdigest()
def parse(p):
 f=open(p,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));f.close();base=8+n;z=[]
 for k,v in h.items():
  if k=='__metadata__':continue
  a,b=v['data_offsets'];z.append(T(k,v['dtype'],tuple(map(int,v['shape'])),base+a,base+b))
 return z
def H(x):
 c=np.bincount(x.astype(np.int64),minlength=65536).astype(float);c=c[c>0];p=c/c.sum();return float(-(p*np.log2(p)).sum())
def fl(w):return (w.astype(np.uint32)<<16).view(np.float32)
def ok(w):
 s=(w>>15).astype(bool);return np.where(s,np.bitwise_not(w),w^np.uint16(0x8000)).astype(np.uint16)
def feat(v):
 n,d=v.shape;ids=np.linspace(0,d-1,min(48,d),dtype=int);x=fl(v[:,ids]).astype(float);mu=x.mean(1);sd=x.std(1);x=(x-mu[:,None])/(sd[:,None]+1e-12);x-=x.mean(0)
 _,_,vt=np.linalg.svd(x,full_matrices=False);pc=x@vt[:min(4,len(vt))].T
 return mu,sd,pc
def morton(pc):
 lo=np.percentile(pc,1,0);hi=np.percentile(pc,99,0);q=np.rint(np.clip((pc-lo)/(hi-lo+1e-12),0,1)*255).astype(np.uint32);key=np.zeros(len(q),np.uint64)
 for b in range(8):
  for j in range(q.shape[1]):key|=((q[:,j]>>b)&1).astype(np.uint64)<<(b*q.shape[1]+j)
 return np.argsort(key,kind='stable')
def probe(v,axis):
 mu,sd,pc=feat(v);orders={'mean':np.argsort(mu,kind='stable'),'std':np.argsort(sd,kind='stable'),'morton':morton(pc),'pc0':np.argsort(pc[:,0],kind='stable')};raw=H(v.ravel());best=None
 for oname,o in orders.items():
  x=v[o];prev=np.zeros_like(x);prev[1:]=x[:-1];ko=ok(x);kp=np.zeros_like(ko);kp[1:]=ko[:-1]
  cand={
   'xor':x^prev,
   'delta':(x.astype(np.uint32)-prev.astype(np.uint32)).astype(np.uint16),
   'odelta':(ko.astype(np.uint32)-kp.astype(np.uint32)).astype(np.uint16),
  }
  side=(math.ceil(math.log2(max(2,len(v))))+2)*len(v)/v.size
  for m,r in cand.items():
   rate=H(r.ravel())+side;z={'axis':axis,'order':oname,'mode':m,'rate':rate,'payload':rate-side,'side':side,'gain':raw-rate,'raw':raw}
   if best is None or rate<best['rate']:best=z
 return best
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='channel-order.json');ap.add_argument('--model-path',default='');ap.add_argument('--max-tensors',type=int,default=8);ap.add_argument('--sample-mib',type=int,default=8);a=ap.parse_args();p=Path(a.model_path) if a.model_path else Path(hf_hub_download(MID,filename=FN));d=dg(p);print('sha',d,flush=True)
 if d!=SHA:raise SystemExit('bad sha')
 ts=[t for t in parse(p) if t.dtype=='BF16' and len(t.shape)==2 and (t.end-t.start)>=1<<20];ts=sorted(ts,key=lambda t:(0 if 'layers.0.' in t.name else 1,-(t.end-t.start)))[:a.max_tensors];out=[]
 with open(p,'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  for i,t in enumerate(ts,1):
   rows,cols=t.shape;rb=cols*2;nr=max(1,min(rows,(a.sample_mib<<20)//rb));s=t.start+((rows-nr)//2)*rb;w=np.frombuffer(mm[s:s+nr*rb],dtype='<u2').copy().reshape(nr,cols);print(i,t.name,w.shape,flush=True)
   cs=[]
   if nr>=8:cs.append(probe(w,'rows'))
   if cols>=8:cs.append(probe(w.T,'cols'))
   b=min(cs,key=lambda q:q['rate']);out.append({'name':t.name,'shape':t.shape,'sample_shape':w.shape,'best':b,'all':cs});print(b,flush=True)
 wt=np.array([np.prod(x['sample_shape']) for x in out]);rate=float(np.average([x['best']['rate'] for x in out],weights=wt));r={'sha':d,'rate':rate,'saving':1-rate/16,'target':8.0,'tensors':out};Path(a.output).write_text(json.dumps(r,indent=2,sort_keys=True));print('ACCEPTANCE',rate,1-rate/16,flush=True)
if __name__=='__main__':main()
