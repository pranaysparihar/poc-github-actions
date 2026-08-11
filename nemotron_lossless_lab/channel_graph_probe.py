#!/usr/bin/env python3
"""Exact nonlocal channel-graph representation probe for BF16 matrices.

Rows or columns are reordered by a compact data-derived signature. Each channel
is represented exactly from an earlier parent with one of several reversible
integer predictors (XOR, modulo delta, monotone-word delta, dyadic affine
prediction). The reported rate includes a conservative permutation, parent,
and method cost. No values are approximated.
"""
from __future__ import annotations
import argparse, hashlib, json, math, mmap, os, re, struct, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download

MID='nvidia/Nemotron-3-Embed-1B-BF16'; FN='model.safetensors'; SHA='f959c3b04e66b42de280bfb97c140cb7e0bfe25e3ecb0b4464c68a8436b2d04f'
@dataclass(frozen=True)
class T:
    name:str; dtype:str; shape:tuple[int,...]; start:int; end:int
    @property
    def nbytes(self):return self.end-self.start

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def parse(p):
    with p.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n))
    base=8+n; z=[]
    for k,v in h.items():
        if k=='__metadata__':continue
        a,b=v['data_offsets'];z.append(T(k,v['dtype'],tuple(map(int,v['shape'])),base+a,base+b))
    return z
def H(x):
    c=np.bincount(x.astype(np.int64),minlength=65536).astype(float);c=c[c>0];p=c/c.sum();return float(-(p*np.log2(p)).sum())
def bf16_float(w):return (w.astype(np.uint32)<<16).view(np.float32)
def ordkey(w):
    s=(w>>15).astype(bool);return np.where(s,np.bitwise_not(w),w^np.uint16(0x8000)).astype(np.uint16)
def invord(k):
    # Ordered keys >= 0x8000 originated from positive words.
    pos=(k&0x8000)!=0;return np.where(pos,k^np.uint16(0x8000),np.bitwise_not(k)).astype(np.uint16)
def morton_order(feat,bits=8):
    """Interleave four quantized PCA coordinates into a deterministic key."""
    f=feat[:,:min(4,feat.shape[1])]
    lo=np.percentile(f,1,axis=0);hi=np.percentile(f,99,axis=0);q=np.clip((f-lo)/(hi-lo+1e-12),0,1)
    q=np.rint(q*((1<<bits)-1)).astype(np.uint32)
    key=np.zeros(len(q),np.uint64)
    for b in range(bits):
        for d in range(q.shape[1]):key|=((q[:,d]>>b)&1).astype(np.uint64)<<(b*q.shape[1]+d)
    return np.argsort(key,kind='stable')
def features(x,max_samples=64):
    """Features for vectors in rows of x, with deterministic PCA whitening."""
    n,d=x.shape;ids=np.linspace(0,d-1,min(d,max_samples),dtype=int);f=x[:,ids].astype(np.float64)
    mean=f.mean(1,keepdims=True);std=f.std(1,keepdims=True)+1e-9
    z=(f-mean)/std
    z-=z.mean(0,keepdims=True)
    # SVD only on <=64 columns.
    _,_,vt=np.linalg.svd(z,full_matrices=False)
    p=z@vt[:min(8,len(vt))].T
    return np.concatenate([mean,std,p],axis=1)
def residual(parent,target,mode):
    if mode=='xor':return target^parent
    if mode=='delta':return (target.astype(np.uint32)-parent.astype(np.uint32)).astype(np.uint16)
    po=ordkey(parent);to=ordkey(target)
    if mode=='odelta':return (to.astype(np.uint32)-po.astype(np.uint32)).astype(np.uint16)
    # dyadic affine modes are exact predictors in ordered-word space.
    p,q=mode
    pred=((po.astype(np.uint32)*p)>>q).astype(np.uint16)
    return (to.astype(np.uint32)-pred.astype(np.uint32)).astype(np.uint16)
MODES=['xor','delta','odelta',(1,1),(3,1),(1,2),(3,2),(5,2)]
def pick_graph(mat,axis,window=12,select_samples=192):
    # vectors become rows.
    v=mat if axis=='rows' else mat.T
    nv,dim=v.shape
    f=features(bf16_float(v))
    orders={
      'mean':np.argsort(f[:,0],kind='stable'),
      'std':np.argsort(f[:,1],kind='stable'),
      'morton':morton_order(f[:,2:]),
      'lex':np.lexsort(tuple(f[:,j] for j in range(min(f.shape[1]-1,7),1,-1))),
    }
    sample_idx=np.linspace(0,dim-1,min(dim,select_samples),dtype=int)
    best=None
    for oname,order in orders.items():
        roots=[];res=[];parents=np.full(nv,-1,np.int32);methods=[]
        # First vector raw; each later vector searches a bounded earlier window.
        roots.append(v[order[0]].copy());methods.append('raw')
        for pos in range(1,nv):
            j=int(order[pos]);y=v[j];best_local=None
            for pp in range(max(0,pos-window),pos):
                i=int(order[pp]);x=v[i]
                for m in MODES:
                    rr=residual(x[sample_idx],y[sample_idx],m)
                    score=H(rr)
                    if best_local is None or score<best_local[0]:best_local=(score,pp,m)
            _,pp,m=best_local;i=int(order[pp]);parents[j]=i;methods.append(str(m))
            res.append(residual(v[i],y,m))
        stream=np.concatenate(roots+res)
        payload=H(stream)
        # Conservative side information: arbitrary permutation + parent + 4-bit method.
        side=(2*math.ceil(math.log2(max(2,nv)))+4)*nv/(nv*dim)
        total=payload+side
        cur={'axis':axis,'order':oname,'vectors':nv,'dim':dim,'payload_entropy':payload,'side_bits_per_value':side,'total_bits_per_value':total,'raw_entropy':H(v.ravel()),'gain':H(v.ravel())-total}
        if best is None or total<best['total_bits_per_value']:best=cur
    return best

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-path',default='');ap.add_argument('--output',default='channel-graph.json');ap.add_argument('--max-tensors',type=int,default=8);ap.add_argument('--sample-mib',type=int,default=8)
    a=ap.parse_args();t0=time.time();p=Path(a.model_path) if a.model_path else Path(hf_hub_download(MID,filename=FN));d=sha(p);print('sha',d,flush=True)
    if d!=SHA:raise SystemExit('bad hash')
    ts=[t for t in parse(p) if t.dtype=='BF16' and len(t.shape)==2 and t.nbytes>=1<<20]
    # One layer-0 tensor per major role, then next largest.
    def priority(t):
        n=t.name
        first=0 if re.search(r'layers\.0\.',n) else 1
        return (first,-t.nbytes,n)
    ts=sorted(ts,key=priority)[:a.max_tensors]
    out=[]
    with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for ix,t in enumerate(ts,1):
            rows,cols=t.shape;rowb=cols*2;cap=min(t.nbytes,a.sample_mib<<20);nr=max(1,min(rows,cap//rowb));startrow=max(0,(rows-nr)//2);s=t.start+startrow*rowb;e=s+nr*rowb
            w=np.frombuffer(mm[s:e],dtype='<u2').copy().reshape(nr,cols)
            print(f'[{ix}/{len(ts)}] {t.name} sample={w.shape}',flush=True)
            candidates=[]
            if w.shape[0]>=8:candidates.append(pick_graph(w,'rows'))
            if w.shape[1]>=8:candidates.append(pick_graph(w,'cols'))
            best=min(candidates,key=lambda z:z['total_bits_per_value'])
            out.append({'name':t.name,'shape':t.shape,'sample_shape':w.shape,'best':best,'all':candidates})
            print(json.dumps(best,sort_keys=True),flush=True)
    weights=np.array([np.prod(x['sample_shape']) for x in out],float)
    rate=float(np.average([x['best']['total_bits_per_value'] for x in out],weights=weights))
    report={'sha256':d,'target':8.0,'weighted_rate':rate,'projected_saving':1-rate/16,'tensors':out,'seconds':time.time()-t0}
    Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True));print('ACCEPTANCE',json.dumps({k:report[k] for k in ('weighted_rate','projected_saving','target')},indent=2),flush=True)
if __name__=='__main__':main()
