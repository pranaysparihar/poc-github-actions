#!/usr/bin/env python3
"""Exact coupled-neuron representation for SwiGLU layers.

A hidden MLP neuron is the triplet (gate row, up row, down column).  The same
permutation acts on all three tensors, so we search canonical orderings in the
joint space and measure exact reversible residuals, charging the permutation.
"""
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
 f=open(p,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));f.close();base=8+n;z={}
 for k,v in h.items():
  if k=='__metadata__':continue
  a,b=v['data_offsets'];z[k]=T(k,v['dtype'],tuple(map(int,v['shape'])),base+a,base+b)
 return z
def H(x):
 c=np.bincount(x.astype(np.int64),minlength=65536).astype(float);c=c[c>0];p=c/c.sum();return float(-(p*np.log2(p)).sum())
def fl(w):return (w.astype(np.uint32)<<16).view(np.float32)
def ok(w):
 s=(w>>15).astype(bool);return np.where(s,np.bitwise_not(w),w^np.uint16(0x8000)).astype(np.uint16)
def signatures(g,u,d,samp=64):
 n,dim=g.shape;ids=np.linspace(0,dim-1,min(dim,samp),dtype=int)
 x=np.concatenate([fl(g[:,ids]),fl(u[:,ids]),fl(d[ids,:].T)],axis=1).astype(np.float64)
 mu=x.mean(1,keepdims=True);sd=x.std(1,keepdims=True)+1e-12;z=(x-mu)/sd;z-=z.mean(0)
 _,_,vt=np.linalg.svd(z,full_matrices=False);pc=z@vt[:min(8,len(vt))].T
 return np.concatenate([mu,sd,pc],axis=1)
def morton(f,bits=8):
 x=f[:,:min(6,f.shape[1])];lo=np.percentile(x,1,0);hi=np.percentile(x,99,0);q=np.rint(np.clip((x-lo)/(hi-lo+1e-12),0,1)*((1<<bits)-1)).astype(np.uint32);key=np.zeros(len(q),np.uint64)
 for b in range(bits):
  for j in range(q.shape[1]):key|=((q[:,j]>>b)&1).astype(np.uint64)<<(b*q.shape[1]+j)
 return np.argsort(key,kind='stable')
def stream(g,u,d,order,mode):
 gg=g[order];uu=u[order];dd=d[:,order].T # neuron-major
 if mode=='raw':return np.concatenate([gg.ravel(),uu.ravel(),dd.ravel()])
 def r(x):
  p=np.zeros_like(x);p[1:]=x[:-1]
  if mode=='xor':return x^p
  if mode=='delta':return (x.astype(np.uint32)-p.astype(np.uint32)).astype(np.uint16)
  xo=ok(x);po=np.zeros_like(xo);po[1:]=xo[:-1];return (xo.astype(np.uint32)-po.astype(np.uint32)).astype(np.uint16)
 return np.concatenate([r(gg).ravel(),r(uu).ravel(),r(dd).ravel()])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-path',default='');ap.add_argument('--output',default='mlp-neurons.json');ap.add_argument('--layers',type=int,default=4);a=ap.parse_args();p=Path(a.model_path) if a.model_path else Path(hf_hub_download(MID,filename=FN));sha=dg(p);print('sha',sha,flush=True)
 if sha!=SHA:raise SystemExit('bad sha')
 ts=parse(p);out=[]
 with open(p,'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  for l in range(a.layers):
   ns=[f'layers.{l}.mlp.{x}_proj.weight' for x in ('gate','up','down')]
   if not all(n in ts for n in ns):continue
   def rd(n):
    t=ts[n];return np.frombuffer(mm[t.start:t.end],dtype='<u2').copy().reshape(t.shape)
   g,u,d=map(rd,ns);assert g.shape==u.shape and d.shape==(g.shape[1],g.shape[0])
   print('layer',l,g.shape,flush=True);ftr=signatures(g,u,d)
   orders={'identity':np.arange(len(g)),'mean':np.argsort(ftr[:,0],kind='stable'),'std':np.argsort(ftr[:,1],kind='stable'),'morton':morton(ftr),'pc0':np.argsort(ftr[:,2],kind='stable')}
   raw=H(stream(g,u,d,orders['identity'],'raw'));best=None;allr=[];side=math.ceil(math.log2(math.factorial(len(g))))/(g.size+u.size+d.size)
   # lgamma avoids enormous integer in report, but factorial above may be too huge; override exact Stirling/gamma.
   side=math.lgamma(len(g)+1)/math.log(2)/(g.size+u.size+d.size)
   for on,o in orders.items():
    for m in ('xor','delta','odelta'):
     h=H(stream(g,u,d,o,m));rate=h+side;z={'order':on,'mode':m,'payload':h,'permutation_side':side,'rate':rate,'gain':raw-rate};allr.append(z)
     if best is None or rate<best['rate']:best=z
   print('best',best,'raw',raw,flush=True);out.append({'layer':l,'raw':raw,'best':best,'all':allr})
 rate=float(np.mean([x['best']['rate'] for x in out]));r={'sha':sha,'layers':out,'weighted_rate':rate,'saving':1-rate/16,'target':8.0};Path(a.output).write_text(json.dumps(r,indent=2,sort_keys=True));print('ACCEPTANCE',rate,1-rate/16,flush=True)
if __name__=='__main__':main()
