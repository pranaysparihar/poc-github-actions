#!/usr/bin/env python3
"""Bounded structural probe on the real Nemotron BF16 checkpoint.

The program never modifies a weight. It measures exact code-length proxies for
representation families that could plausibly beat the ordinary ~10.5 b/BF16
field entropy: layer-axis lifting, consensus templates, same-role residuals,
and bounded held-out contexts. It verifies the pinned checkpoint SHA first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import re
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

MODEL_ID = "nvidia/Nemotron-3-Embed-1B-BF16"
FILE = "model.safetensors"
SHA = "f959c3b04e66b42de280bfb97c140cb7e0bfe25e3ecb0b4464c68a8436b2d04f"

@dataclass(frozen=True)
class T:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int
    @property
    def nbytes(self): return self.end-self.start
    @property
    def n(self): return self.nbytes//2
    @property
    def width(self): return int(self.shape[-1]) if len(self.shape)>=2 else max(1,self.n)

def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def parse(p: Path):
    with p.open('rb') as f:
        hlen=struct.unpack('<Q',f.read(8))[0]
        h=json.loads(f.read(hlen))
    base=8+hlen; out=[]
    for n,m in h.items():
        if n=='__metadata__': continue
        a,b=m['data_offsets']
        out.append(T(n,m['dtype'],tuple(map(int,m['shape'])),base+int(a),base+int(b)))
    return sorted(out,key=lambda x:x.start)

def H(x: np.ndarray, alphabet=65536) -> float:
    c=np.bincount(x.astype(np.int64),minlength=alphabet).astype(np.float64)
    c=c[c>0]; p=c/c.sum()
    return float(-(p*np.log2(p)).sum())

def layer(n: str):
    m=re.search(r'(?:layers|layer|blocks|blk)[._](\d+)',n)
    return int(m.group(1)) if m else None

def key(n: str):
    return re.sub(r'(?:(?<=\.)|(?<=_))\d+(?=\.|_)','{L}',n)

def role(n: str):
    s=n.lower()
    for r,ns in [
      ('embed',('embed_tokens','embedding','word_embeddings')),
      ('q',('q_proj','query')),('k',('k_proj','key')),('v',('v_proj','value')),
      ('o',('o_proj','out_proj','attention.dense')),
      ('gate',('gate_proj','w1')),('up',('up_proj','w3')),('down',('down_proj','w2')),
      ('norm',('norm',))]:
        if any(z in s for z in ns): return r
    return 'other'

def read_sample(mm, t:T, mib:int):
    cap=min(t.nbytes,mib<<20); cap-=cap%2
    if cap==t.nbytes: return np.frombuffer(mm[t.start:t.end],dtype='<u2').copy()
    rowb=max(2,t.width*2)
    each=max(rowb,(cap//3//rowb)*rowb)
    each=min(each,t.nbytes); starts=[0,(t.nbytes-each)//2,t.nbytes-each]
    xs=[]
    for s in starts:
        s=(s//rowb)*rowb; e=min(t.nbytes,s+each); e-=((e-s)%2)
        xs.append(np.frombuffer(mm[t.start+s:t.start+e],dtype='<u2').copy())
    return np.concatenate(xs)

def field_stats(w):
    hi=(w>>8).astype(np.uint8); lo=(w&255).astype(np.uint8)
    hw=H(w); hh=H(hi,256); hl=hw-hh
    return {'word':hw,'high':hh,'low':H(lo,256),'low_given_high':hl}

def context_nll(train_keys, train_syms, test_keys, test_syms, alphabet=256, min_count=32):
    """Bounded held-out categorical model with global backoff.

    Context alphabet is deliberately limited to at most 16 bits so metadata can
    be fully charged by a future coder. This is a predictor probe, not an
    in-sample entropy estimate.
    """
    tk=np.asarray(train_keys,dtype=np.uint32); ts=np.asarray(train_syms,dtype=np.uint16)
    qk=np.asarray(test_keys,dtype=np.uint32); qs=np.asarray(test_syms,dtype=np.uint16)
    base=np.bincount(ts.astype(np.int64),minlength=alphabet).astype(np.float64)+0.5
    base/=base.sum(); probs=base[qs.astype(np.int64)].copy()
    # Dense table only if context range is manageable.
    maxk=int(max(tk.max(initial=0),qk.max(initial=0)))+1
    if maxk>65536: return float((-np.log2(probs)).mean()),0.0
    table=np.zeros((maxk,alphabet),dtype=np.uint32)
    np.add.at(table,(tk.astype(np.int64),ts.astype(np.int64)),1)
    totals=table.sum(1); good=np.where(totals>=min_count)[0]
    use=(qk<maxk)&(totals[np.minimum(qk,maxk-1)]>=min_count)
    if np.any(use):
        k=qk[use].astype(np.int64); s=qs[use].astype(np.int64)
        p=(table[k,s].astype(np.float64)+0.5)/(totals[k]+0.5*alphabet)
        p=0.95*p+0.05*base[s]
        probs[use]=p
    return float((-np.log2(np.clip(probs,2**-80,1))).mean()),float(use.mean())

def heldout(w,width):
    n=len(w); cut=int(n*.7); gap=min(max(width,16),4096); q0=min(n,cut+gap)
    hi=(w>>8).astype(np.uint8); lo=(w&255).astype(np.uint8); idx=np.arange(n)
    lhi=np.zeros(n,np.uint8); llo=np.zeros(n,np.uint8); uhi=np.zeros(n,np.uint8); ulo=np.zeros(n,np.uint8)
    lhi[1:]=hi[:-1]; llo[1:]=lo[:-1]; rowstart=(idx%max(width,1))==0; lhi[rowstart]=0; llo[rowstart]=0
    if width>0 and n>width: uhi[width:]=hi[:-width]; ulo[width:]=lo[:-width]
    col=(idx%max(width,1)%16).astype(np.uint16)
    contexts={
      'hi_prev':(lhi.astype(np.uint16),hi),
      'hi_prev_col16':((lhi.astype(np.uint16)<<4)|col,hi),
      'lo_hi':(hi.astype(np.uint16),lo),
      'lo_hi_prevlo':((hi.astype(np.uint16)<<8)|llo.astype(np.uint16),lo),
      'lo_hi_uplo':((hi.astype(np.uint16)<<8)|ulo.astype(np.uint16),lo),
    }
    out={}
    for name,(k,s) in contexts.items():
        nll,cov=context_nll(k[:cut],s[:cut],k[q0:],s[q0:])
        out[name]={'nll':nll,'coverage':cov}
    best_hi=min(v['nll'] for k,v in out.items() if k.startswith('hi_'))
    best_lo=min(v['nll'] for k,v in out.items() if k.startswith('lo_'))
    out['best_split_total']=best_hi+best_lo
    return out

def transform_entropies(w,width):
    n=len(w); idx=np.arange(n); left=np.zeros(n,np.uint16); up=np.zeros(n,np.uint16); diag=np.zeros(n,np.uint16)
    left[1:]=w[:-1]; left[(idx%max(width,1))==0]=0
    if width>0 and n>width:
        up[width:]=w[:-width]
        if n>width+1: diag[width+1:]=w[:-(width+1)]
        diag[(idx%width)==0]=0
    pred=((left.astype(np.uint32)+up.astype(np.uint32)-diag.astype(np.uint32))&65535).astype(np.uint16)
    cs={
      'raw':w,'xor_left':w^left,'xor_up':w^up,'xor_lu':w^left^up,
      'delta_left':(w.astype(np.uint32)-left.astype(np.uint32)).astype(np.uint16),
      'delta_up':(w.astype(np.uint32)-up.astype(np.uint32)).astype(np.uint16),
      'lorenzo':(w.astype(np.uint32)-pred.astype(np.uint32)).astype(np.uint16),
    }
    return {k:H(v) for k,v in cs.items()}

def duplicate_fraction(w,bw):
    nb=len(w)//bw
    if nb<1:return 0.0
    a=np.ascontiguousarray(w[:nb*bw].reshape(nb,bw))
    v=a.view(np.dtype((np.void,a.dtype.itemsize*bw))).ravel()
    _,c=np.unique(v,return_counts=True)
    return float(np.maximum(c-1,0).sum()/nb)

def layer_axis(group_words):
    """Compare exact reversible transforms along transformer depth."""
    a=np.stack(group_words,axis=0).astype(np.uint16); L,N=a.shape
    raw=H(a.ravel())
    diff=a.copy(); diff[1:]=(a[1:].astype(np.uint32)-a[:-1].astype(np.uint32)).astype(np.uint16)
    xor=a.copy(); xor[1:]=a[1:]^a[:-1]
    # Integer reversible Haar lifting along depth; odd tail copied.
    haar=a.copy(); active=L; level=0
    while active>=2:
        pairs=active//2
        x=haar[:2*pairs:2].astype(np.uint32); y=haar[1:2*pairs:2].astype(np.uint32)
        d=((y-x)&65535).astype(np.uint16)
        s=((x+((d.astype(np.uint32))>>1))&65535).astype(np.uint16)
        tail=haar[2*pairs:active].copy()
        haar[:pairs]=s; haar[pairs:2*pairs]=d
        if len(tail): haar[2*pairs:2*pairs+len(tail)]=tail
        active=pairs; level+=1
    # Shared exact template: median word in monotone BF16 ordering; cost one layer.
    # Use modal word per coordinate as an optimistic exact consensus template.
    template=np.zeros(N,np.uint16)
    # L is only 16, so coordinate-wise mode in chunks is bounded.
    chunk=1<<18
    res=[]
    for s0 in range(0,N,chunk):
        x=a[:,s0:s0+chunk]
        # sort depth values; choose median (not mode) as robust template.
        t=np.sort(x,axis=0)[L//2]
        template[s0:s0+len(t)]=t
        res.append((x.astype(np.uint32)-t[None,:].astype(np.uint32)).astype(np.uint16))
    rr=np.concatenate(res,axis=1)
    template_rate=H(template)/L + H(rr.ravel())
    return {'layers':L,'values_per_layer':N,'raw':raw,'delta':H(diff.ravel()),'xor':H(xor.ravel()),'haar':H(haar.ravel()),'template_amortized':template_rate}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-path',default=''); p.add_argument('--output',default='quick.json'); p.add_argument('--sample-mib',type=int,default=6); p.add_argument('--max-tensors',type=int,default=12)
    a=p.parse_args(); t0=time.time()
    path=Path(a.model_path) if a.model_path else Path(hf_hub_download(MODEL_ID,filename=FILE))
    sha=digest(path); print('path',path); print('sha256',sha,flush=True)
    if sha!=SHA: raise SystemExit('wrong model hash')
    ts=parse(path); bf=[t for t in ts if t.dtype=='BF16' and t.nbytes>=4096]
    largest=sorted(bf,key=lambda x:x.nbytes,reverse=True); sel=[]; seen=set()
    for t in largest:
        if role(t.name) not in seen: sel.append(t); seen.add(role(t.name))
    for t in largest:
        if t not in sel:sel.append(t)
        if len(sel)>=a.max_tensors:break
    rows=[]; samples={}
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for i,t in enumerate(sel,1):
            w=read_sample(mm,t,a.sample_mib); samples[t.name]=w
            print(f'[{i}/{len(sel)}] {t.name} {t.shape} {w.nbytes/(1<<20):.1f}MiB',flush=True)
            rows.append({'name':t.name,'role':role(t.name),'layer':layer(t.name),'shape':t.shape,'n':len(w),'fields':field_stats(w),'heldout':heldout(w,t.width),'transforms':transform_entropies(w,t.width),'duplicates':{str(b):duplicate_fraction(w,b) for b in (8,16,32,64,128)}})
        # Same-role consecutive-layer exact residuals and depth-axis transforms.
        groups=defaultdict(list)
        for t in bf:
            if layer(t.name) is not None: groups[(key(t.name),t.shape)].append(t)
        depth=[]; pairs=[]; budget=16
        for (gk,sh),g in sorted(groups.items(),key=lambda kv:-kv[1][0].nbytes):
            g.sort(key=lambda x:layer(x.name))
            if len(g)<4:continue
            nbytes=min(4<<20,*(x.nbytes for x in g)); nbytes-=nbytes%2
            rel=(min(x.nbytes for x in g)-nbytes)//2; rel-=rel%2
            ws=[np.frombuffer(mm[x.start+rel:x.start+rel+nbytes],dtype='<u2').copy() for x in g]
            depth.append({'group':gk,'shape':sh,**layer_axis(ws)})
            for x,y,wx,wy in zip(g,g[1:],ws,ws[1:]):
                pairs.append({'a':x.name,'b':y.name,'match':float(np.mean(wx==wy)),'xor':H(wx^wy),'delta':H((wy.astype(np.uint32)-wx.astype(np.uint32)).astype(np.uint16))})
            budget-=1
            if budget<=0:break
    weights=np.array([r['n'] for r in rows],float)
    agg={
      'sampled_values':int(weights.sum()),
      'joint_entropy':float(np.average([r['fields']['word'] for r in rows],weights=weights)),
      'best_heldout_split':float(np.average([r['heldout']['best_split_total'] for r in rows],weights=weights)),
      'best_spatial_transform_entropy':float(np.average([min(r['transforms'].values()) for r in rows],weights=weights)),
      'best_depth_axis_entropy':min((min(d['delta'],d['xor'],d['haar'],d['template_amortized']) for d in depth),default=16.0),
      'required':8.0,
    }
    out={'sha256':sha,'model_bytes':path.stat().st_size,'bf16_tensors':len(bf),'bf16_values':sum(t.n for t in bf),'aggregate':agg,'tensors':rows,'depth_groups':depth,'pairs':pairs,'seconds':time.time()-t0}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True))
    print('ACCEPTANCE',json.dumps(agg,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':main()
