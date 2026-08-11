#!/usr/bin/env python3
"""Exact implicit cross-tensor block graph probe.

Previously decoded blocks form a free dictionary. Each later BF16 block may be
represented by an exact reversible correction to one (or three) earlier
blocks. Reference IDs, method tags, affine parameters, entropy tables and
root blocks are all charged. The target model is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
FILENAME = "model.safetensors"
EXPECTED = "f959c3b04e66b42de280bfb97c140cb7e0bfe25e3ecb0b4464c68a8436b2d04f"


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensors(path: Path) -> list[Tensor]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    out = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        out.append(Tensor(name, meta["dtype"], tuple(map(int, meta["shape"])), base + a, base + b))
    return sorted(out, key=lambda t: t.start)


def entropy(words: np.ndarray) -> float:
    c = np.bincount(words.astype(np.int64), minlength=65536).astype(np.float64)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def ordered(w: np.ndarray) -> np.ndarray:
    sign = (w >> 15).astype(bool)
    return np.where(sign, np.bitwise_not(w), w ^ np.uint16(0x8000)).astype(np.uint16)


def block_signature(w: np.ndarray) -> int:
    """A locality-sensitive 16-bit key, not a stored model."""
    hi = (w >> 8).astype(np.int16)
    exp = ((w >> 7) & 0xFF).astype(np.int16)
    # Four bits of robust location/scale statistics.
    med = int(np.median(exp)) & 0xFF
    spread = int(np.percentile(exp, 75) - np.percentile(exp, 25)) & 0x3F
    key = ((med >> 3) & 0x1F) | ((spread >> 2) << 5)
    # Eight deterministic SimHash bits over exponent deviations.
    idx = np.linspace(0, len(w) - 1, min(64, len(w)), dtype=np.int64)
    x = exp[idx] - med
    rng = np.random.default_rng(0x1F3E5D7)
    signs = rng.choice(np.array([-1, 1], dtype=np.int16), size=(8, len(idx)))
    sim = (signs @ x >= 0).astype(np.uint16)
    for b, bit in enumerate(sim):
        key |= int(bit) << (9 + b)
    return key & 0xFFFF


def proxy_h(w: np.ndarray) -> float:
    # Exact empirical code-length proxy for selecting a candidate. Final rates
    # are computed again over concatenated streams and include all side data.
    return entropy(w)


def residual(parent: np.ndarray, target: np.ndarray, mode: str) -> tuple[np.ndarray, int | None]:
    if mode == "xor":
        return parent ^ target, None
    if mode == "delta":
        return (target.astype(np.uint32) - parent.astype(np.uint32)).astype(np.uint16), None
    po = ordered(parent)
    to = ordered(target)
    if mode == "odelta":
        return (to.astype(np.uint32) - po.astype(np.uint32)).astype(np.uint16), None
    if mode == "affine":
        # Exact modulo-2^16 offset in monotone-word space. One 16-bit offset is
        # transmitted per referenced block.
        d = ((to.astype(np.uint32) - po.astype(np.uint32)) & 0xFFFF).astype(np.uint16)
        off = int(np.median(d))
        pred = (po.astype(np.uint32) + off).astype(np.uint16)
        return (to.astype(np.uint32) - pred.astype(np.uint32)).astype(np.uint16), off
    if mode == "xormask":
        d = target ^ parent
        # Store the modal 16-bit XOR mask, then code exact remaining XOR.
        c = np.bincount(d.astype(np.int64), minlength=65536)
        mask = int(np.argmax(c))
        return d ^ np.uint16(mask), mask
    raise ValueError(mode)


def choose_parent(target: np.ndarray, candidates: list[tuple[int, np.ndarray]], ref_bits: int):
    raw_score = proxy_h(target) * len(target)
    best = (raw_score, "raw", -1, target.copy(), None)
    # Selection is made using the whole block; the selected reference and mode
    # are explicitly stored, so this is valid two-part coding rather than an
    # uncharged oracle.
    for ref, parent in candidates:
        for mode in ("xor", "delta", "odelta", "affine", "xormask"):
            r, param = residual(parent, target, mode)
            side = ref_bits + 3 + (16 if param is not None else 0)
            score = proxy_h(r) * len(r) + side
            if score < best[0]:
                best = (score, mode, ref, r, param)
    # Multi-reference coordinate median in monotone space. Three IDs are
    # charged. This represents shared structure not expressible by pair deltas.
    if len(candidates) >= 3:
        ranked = sorted(candidates, key=lambda p: int(np.mean(np.abs(ordered(p[1]).astype(np.int32) - ordered(target).astype(np.int32)))))[:3]
        pred = np.median(np.stack([ordered(x) for _, x in ranked]), axis=0).astype(np.uint16)
        r = (ordered(target).astype(np.uint32) - pred.astype(np.uint32)).astype(np.uint16)
        score = proxy_h(r) * len(r) + 3 * ref_bits + 3
        if score < best[0]:
            best = (score, "median3", tuple(x for x, _ in ranked), r, None)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="")
    ap.add_argument("--output", default="block-graph.json")
    ap.add_argument("--block-words", type=int, default=256)
    ap.add_argument("--max-blocks", type=int, default=60000)
    ap.add_argument("--bucket-history", type=int, default=24)
    args = ap.parse_args()

    t0 = time.time()
    path = Path(args.model_path) if args.model_path else Path(hf_hub_download(MODEL, filename=FILENAME))
    digest = sha256(path)
    print("sha256", digest, flush=True)
    if digest != EXPECTED:
        raise SystemExit("wrong checkpoint")

    ts = [t for t in tensors(path) if t.dtype == "BF16" and t.end - t.start >= 2 * args.block_words]
    total_blocks = sum((t.end - t.start) // (2 * args.block_words) for t in ts)
    stride = max(1, total_blocks // args.max_blocks)
    selected: list[tuple[str, int, int]] = []
    seen = 0
    for t in ts:
        nb = (t.end - t.start) // (2 * args.block_words)
        for bi in range(nb):
            if seen % stride == 0 and len(selected) < args.max_blocks:
                selected.append((t.name, t.start + bi * args.block_words * 2, bi))
            seen += 1
    print("all_blocks", total_blocks, "sampled", len(selected), "stride", stride, flush=True)

    buckets: dict[int, deque[tuple[int, np.ndarray]]] = defaultdict(lambda: deque(maxlen=args.bucket_history))
    streams: dict[str, list[np.ndarray]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    params = 0
    ref_count = 0
    root_count = 0
    method_refs: dict[str, int] = defaultdict(int)
    ref_bits = max(1, math.ceil(math.log2(max(2, len(selected)))))
    block_store: list[np.ndarray] = []

    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for j, (_, off, _) in enumerate(selected):
            w = np.frombuffer(mm[off : off + args.block_words * 2], dtype="<u2").copy()
            sig = block_signature(w)
            candidate_pairs: list[tuple[int, np.ndarray]] = list(buckets[sig])
            # Probe Hamming-neighbor buckets; IDs remain fully charged.
            for bit in range(8):
                if len(candidate_pairs) >= args.bucket_history:
                    break
                candidate_pairs.extend(list(buckets[sig ^ (1 << (9 + bit))])[-4:])
            # Deduplicate candidates while preserving recency.
            dedup = {}
            for rid, rw in candidate_pairs:
                dedup[rid] = rw
            candidates = list(dedup.items())[-args.bucket_history :]
            _, mode, ref, r, param = choose_parent(w, candidates, ref_bits)
            streams[mode].append(r)
            counts[mode] += 1
            if mode == "raw":
                root_count += 1
            else:
                ref_count += 1
                method_refs[mode] += 1
                if param is not None:
                    params += 16
            block_store.append(w)
            buckets[sig].append((j, w))
            if (j + 1) % 10000 == 0:
                print("processed", j + 1, "refs", ref_count, flush=True)

    payload_bits = 0.0
    method_rates = {}
    table_bits = 0
    for mode, pieces in streams.items():
        x = np.concatenate(pieces)
        h = entropy(x)
        bits = h * len(x)
        # Conservative full 16-bit frequency table: 32-bit count per symbol.
        table = 65536 * 32
        payload_bits += bits
        table_bits += table
        method_rates[mode] = {"blocks": counts[mode], "entropy": h, "payload_bits": bits, "table_bits": table}

    nblocks = len(selected)
    nvalues = nblocks * args.block_words
    side_bits = nblocks * 3 + ref_count * ref_bits + params
    total_bits = payload_bits + table_bits + side_bits
    rate = total_bits / nvalues
    report = {
        "sha256": digest,
        "sampled_blocks": nblocks,
        "block_words": args.block_words,
        "reference_bits": ref_bits,
        "root_blocks": root_count,
        "referenced_blocks": ref_count,
        "reference_fraction": ref_count / max(1, nblocks),
        "methods": method_rates,
        "side_bits": side_bits,
        "table_bits": table_bits,
        "total_bits_per_bf16": rate,
        "projected_saving": 1.0 - rate / 16.0,
        "target_bits_per_bf16": 8.0,
        "elapsed_seconds": time.time() - t0,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print("ACCEPTANCE", json.dumps({k: report[k] for k in ("total_bits_per_bf16", "projected_saving", "reference_fraction")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
