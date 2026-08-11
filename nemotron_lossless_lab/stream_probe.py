#!/usr/bin/env python3
"""Generate exact reversible BF16 stream layouts for strong context compressors."""
from __future__ import annotations
import argparse,hashlib,json,mmap,struct
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
MID='nvidia/Nemotron-3-Embed-1B-BF16';FN='model.safetensors';SHA='f959c3b04e66b42de280bfb97c140cb7e0bfe25e3ecb0b4464c68a8436b2d04f'
@dataclass(frozen=True)
class T:name:str;dtype:str;shape:tuple[int,...];start:int;end:int

def digest(p):
 h=hashlib.sha256();f=open(p,'rb')
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 f.close();return h.hexdigest()
def parse(p):
 f=open(p,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));f.close();base=8+n;z=[]
 for k,v in h.items():
  if k=='__metadata__':continue
  a,b=v['data_offsets'];z.append(T(k,v['dtype'],tuple(map(int,v['shape'])),base+a,base+b))
 return sorted(z,key=lambda t:t.start)
def pack_global_bitplanes(w):
 return b''.join(np.packbits(((w>>b)&1).astype(np.uint8),bitorder='little').tobytes() for b in range(16))
def pack_tiled_bitplanes(w,tile=256):
 out=bytearray();n=len(w)
 for s in range(0,n,tile):
  x=w[s:s+tile]
  for b in range(16):out+=np.packbits(((x>>b)&1).astype(np.uint8),bitorder='little').tobytes()
 return bytes(out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-path',default='');ap.add_argument('--outdir',default='streams');ap.add_argument('--total-mib',type=int,default=32);a=ap.parse_args()
 p=Path(a.model_path) if a.model_path else Path(hf_hub_download(MID,filename=FN));d=digest(p);print('sha',d,flush=True)
 if d!=SHA:raise SystemExit('bad hash')
 ts=[t for t in parse(p) if t.dtype=='BF16' and len(t.shape)==2 and (t.end-t.start)>=1<<20]
 # Deterministic role-diverse prefix, bounded total bytes.
 names=['embed_tokens.weight','layers.0.mlp.down_proj.weight','layers.0.mlp.gate_proj.weight','layers.0.mlp.up_proj.weight','layers.0.self_attn.q_proj.weight','layers.0.self_attn.k_proj.weight','layers.0.self_attn.v_proj.weight','layers.0.self_attn.o_proj.weight']
 by={t.name:t for t in ts};sel=[by[n] for n in names if n in by]
 budget=a.total_mib<<20;per=max(2,(budget//max(1,len(sel)))//2*2);chunks=[];shapes=[]
 with open(p,'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  for t in sel:
   rowb=t.shape[-1]*2;take=min(t.end-t.start,per);rows=max(1,take//rowb);take=rows*rowb;off=max(0,((t.end-t.start)-take)//2);off=(off//rowb)*rowb
   w=np.frombuffer(mm[t.start+off:t.start+off+take],dtype='<u2').copy();chunks.append(w);shapes.append((t.name,rows,t.shape[-1],len(w)))
 w=np.concatenate(chunks);hi=(w>>8).astype(np.uint8);lo=(w&255).astype(np.uint8)
 o=Path(a.outdir);o.mkdir(parents=True,exist_ok=True)
 streams={
  'raw.bin':w.astype('<u2').tobytes(),
  'fields.bin':hi.tobytes()+lo.tobytes(),
  'bitplanes-global.bin':pack_global_bitplanes(w),
  'bitplanes-tile256.bin':pack_tiled_bitplanes(w),
 }
 # Tensor/row-local exponent centering, exact modulo 256. Store one mode byte per row.
 rm=bytearray();rh=bytearray();rl=bytearray();cm=bytearray();ch=bytearray();cl=bytearray();cursor=0
 for name,rows,cols,n in shapes:
  x=w[cursor:cursor+n].reshape(rows,cols);cursor+=n;xh=(x>>8).astype(np.uint8);xl=(x&255).astype(np.uint8)
  # Median is deterministic and compact; residual is modulo 256.
  rmed=np.median(xh,axis=1).astype(np.uint8);rm+=rmed.tobytes();rh+=((xh.astype(np.uint16)-rmed[:,None].astype(np.uint16))&255).astype(np.uint8).tobytes();rl+=xl.tobytes()
  cmed=np.median(xh,axis=0).astype(np.uint8);cm+=cmed.tobytes();ch+=((xh.astype(np.uint16)-cmed[None,:].astype(np.uint16))&255).astype(np.uint8).tobytes();cl+=xl.tobytes()
 streams['row-expnorm.bin']=bytes(rm+rh+rl);streams['col-expnorm.bin']=bytes(cm+ch+cl)
 manifest={'sha256':d,'words':len(w),'shapes':shapes,'streams':{}}
 for n,b in streams.items():
  q=o/n;q.write_bytes(b);manifest['streams'][n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};print(n,len(b),flush=True)
 (o/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__':main()
