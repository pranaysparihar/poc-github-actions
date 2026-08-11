#!/usr/bin/env python3
"""Exact cross-layer matching of coupled SwiGLU neurons.

For each transition, a neuron is the coupled object (gate row, up row, down
column). Both layers are canonically ordered from compact signatures, then the
target is represented by an exact reversible residual from the matched parent.
Permutation and per-neuron offset costs are charged.
"""
from __future__ import annotations
import argparse,hashlib,json,math,mmap,struct,time
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
def read_layer(mm,ts,l):
 ns=[f'layers.{l}.mlp.{x}_proj.weight' for x in ('gate','up','down')]
 def rd(n):
  t=ts[n];return np.frombuffer(mm[t.start:t.end],dtype='<u2').copy().reshape(t.shape)
 g,u,d=map(rd,ns);assert g.shape==u.shape and d.shape==(g.shape[1],g.shape[0]);return g,u,d
def sig(g,u,d,s=64):
 n,h=g.shape;ids=np.linspace(0,h-1,min(s,h),dtype=int)
 x=np.concatenate([fl(g[:,ids]),fl(u[:,ids]),fl(d[ids,:].T)],1).astype(np.float64)
 mu=x.mean(1,keepdims=True);sd=x.std(1,keepdims=True)+1e-12;z=(x-mu)/sd;z-=z.mean(0)
 _,_,vt=np.linalg.svd(z,full_matrices=False);pc=z@vt[:min(8,len(vt))].T
 return np.concatenate([mu,sd,pc],1)
def morton(f,bits=8):
 x=f[:,2:2+min(6,f.shape[1]-2)];lo=np.percentile(x,1,0);hi=np.percentile(x,99,0);q=np.rint(np.clip((x-lo)/(hi-lo+1e-12),0,1)*255).astype(np.uint32);k=np.zeros(len(q),np.uint64)
 for b in range(bits):
  for j in range(q.shape[1]):k|=((q[:,j]>>b)&1).astype(np.uint64)<<(b*q.shape[1]+j)
 return np.argsort(k,kind='stable')
def orders(f):
 return {'identity':np.arange(len(f)),'mean':np.argsort(f[:,0],kind='stable'),'std':np.argsort(f[:,1],kind='stable'),'pc0':np.argsort(f[:,2],kind='stable'),'morton':morton(f)}
def triplet(g,u,d,o):return np.concatenate([g[o],u[o],d[:,o].T],1)
def rate(parent,target,mode):
 p=parent;t=target;side=0
 if mode=='xor':r=t^p
 elif mode=='delta':r=(t.astype(np.uint32)-p.astype(np.uint32)).astype(np.uint16)
 else:
  po=ok(p);to=ok(t)
  if mode=='odelta':r=(to.astype(np.uint32)-po.astype(np.uint32)).astype(np.uint16)
  elif mode=='affine':
   dd=((to.astype(np.uint32)-po.astype(np.uint32))&65535).astype(np.uint16);off=np.median(dd,axis=1).astype(np.uint16);pred=(po.astype(np.uint32)+off[:,None].astype(np.uint32)).astype(np.uint16);r=(to.astype(np.uint32)-pred.astype(np.uint32)).astype(np.uint16);side=16*len(t)
  elif mode=='affine3':
   po3=np.split(po,3,axis=1);to3=np.split(to,3,axis=1);rs=[]
   for a,b in zip(po3,to3):
    dd=((b.astype(np.uint32)-a.astype(np.uint32))&65535).astype(np.uint16);off=np.median(dd,axis=1).astype(np.uint16);pred=(a.astype(np.uint32)+off[:,None].astype(np.uint32)).astype(np.uint16);rs.append((b.astype(np.uint32)-pred.astype(np.uint32)).astype(np.uint16))
   r=np.concatenate(rs,1);side=48*len(t)
  else:raise ValueError(mode)
 return H(r.ravel()),side

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-path',default='');ap.add_argument('--output',default='alignment.json');ap.add_argument('--layers',type=int,default=8);a=ap.parse_args();p=Path(a.model_path) if a.model_path else Path(hf_hub_download(MID,filename=FN));sha=dg(p);print('sha',sha,flush=True)
 if sha!=SHA:raise SystemExit('bad sha')
 ts=parse(p);out=[]
 with open(p,'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  prev=read_layer(mm,ts,0);prevf=sig(*prev);prevo=orders(prevf)
  for l in range(1,a.layers):
   cur=read_layer(mm,ts,l);curf=sig(*cur);curo=orders(curf);n=cur[0].shape[0];vals=sum(x.size for x in cur);perm=math.lgamma(n+1)/math.log(2)
   raw=H(np.concatenate([x.ravel() for x in cur]));best=None;allr=[]
   # Both sides use the same canonicalization family, so position pairs define a bijection.
   for on in prevo.keys() & curo.keys():
    pp=triplet(*prev,prevo[on]);tt=triplet(*cur,curo[on])
    for mode in ('xor','delta','odelta','affine','affine3'):
     h,extra=rate(pp,tt,mode);total=h+(perm+extra+3*n)/vals;z={'order':on,'mode':mode,'payload':h,'side':(perm+extra+3*n)/vals,'rate':total,'gain':raw-total}
     allr.append(z)
     if best is None or total<best['rate']:best=z
   print('layer',l,'raw',raw,'best',best,flush=True);out.append({'layer':l,'raw':raw,'best':best,'all':allr});prev,prevf,prevo=cur,curf,curo
 ratev=float(np.mean([x['best']['rate'] for x in out]));r={'sha':sha,'transitions':out,'rate':ratev,'saving':1-ratev/16,'target':8.0};Path(a.output).write_text(json.dumps(r,indent=2,sort_keys=True));print('ACCEPTANCE',ratev,1-ratev/16,flush=True)
if __name__=='__main__':main()
