#!/usr/bin/env python3
"""Exact, held-out structural audit for Nemotron BF16 weights.

This is an acceptance gate, not a compressor.  Every reported conditional rate
is evaluated on held-out symbols.  High-cardinality contexts back off to a
lower-order distribution so that table memorisation cannot masquerade as
compression.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import math
import mmap
import os
import re
import struct
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from huggingface_hub import hf_hub_download

MODEL_ID = "nvidia/Nemotron-3-Embed-1B-BF16"
MODEL_FILE = "model.safetensors"
EXPECTED_SHA256 = "f959c3b04e66b42de280bfb97c140cb7e0bfe25e3ecb0b4464c68a8436b2d04f"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def nvalues(self) -> int:
        return self.nbytes // 2

    @property
    def row_width(self) -> int:
        if len(self.shape) >= 2:
            return int(self.shape[-1])
        return max(1, self.nvalues)


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_safetensors(path: Path) -> tuple[int, dict, list[TensorInfo]]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors header")
        hlen = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(hlen))
    data_base = 8 + hlen
    tensors: list[TensorInfo] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        offs = meta["data_offsets"]
        tensors.append(
            TensorInfo(
                name=name,
                dtype=meta["dtype"],
                shape=tuple(int(x) for x in meta["shape"]),
                start=data_base + int(offs[0]),
                end=data_base + int(offs[1]),
            )
        )
    tensors.sort(key=lambda t: t.start)
    return data_base, header, tensors


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def entropy_symbols(x: np.ndarray, alphabet: int) -> float:
    return entropy_from_counts(np.bincount(x.astype(np.int64), minlength=alphabet))


def compressor_ratio(data: bytes, method: str) -> float:
    if not data:
        return 1.0
    if method == "zlib9":
        out = zlib.compress(data, 9)
    elif method == "bz2":
        out = bz2.compress(data, 9)
    elif method == "xz":
        out = lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    else:
        raise ValueError(method)
    return len(out) / len(data)


def order_key(u: np.ndarray) -> np.ndarray:
    """Reversible BF16 bits -> monotone uint16 ordering key."""
    u = u.astype(np.uint16, copy=False)
    sign = (u >> 15).astype(bool)
    return np.where(sign, np.bitwise_not(u), u ^ np.uint16(0x8000)).astype(np.uint16)


def role_of(name: str) -> str:
    s = name.lower()
    rules = [
        ("embed", ("embed_tokens", "embedding", "word_embeddings")),
        ("q_proj", ("q_proj", "query")),
        ("k_proj", ("k_proj", "key")),
        ("v_proj", ("v_proj", "value")),
        ("o_proj", ("o_proj", "out_proj", "attention.dense")),
        ("gate_proj", ("gate_proj", "w1")),
        ("up_proj", ("up_proj", "w3")),
        ("down_proj", ("down_proj", "w2")),
        ("norm", ("norm",)),
    ]
    for role, needles in rules:
        if any(n in s for n in needles):
            return role
    return "other"


def layer_number(name: str) -> int | None:
    pats = [r"(?:layers|layer|blocks|blk)\.(\d+)", r"(?:layers|layer|blocks|blk)_(\d+)"]
    for p in pats:
        m = re.search(p, name)
        if m:
            return int(m.group(1))
    return None


def role_key(name: str) -> str:
    return re.sub(r"(?:(?<=\.)|(?<=_))\d+(?=\.|_)", "{L}", name)


def sample_ranges(t: TensorInfo, max_bytes: int) -> list[tuple[int, int]]:
    """Deterministic, row-aligned start/middle/end windows."""
    if t.nbytes <= max_bytes:
        return [(t.start, t.end)]
    each = max_bytes // 3
    each -= each % 2
    row_bytes = max(2, t.row_width * 2)
    each = max(row_bytes, (each // row_bytes) * row_bytes)
    each = min(each, t.nbytes)
    starts = [0, max(0, (t.nbytes - each) // 2), max(0, t.nbytes - each)]
    out = []
    for s in starts:
        s = (s // row_bytes) * row_bytes
        e = min(t.nbytes, s + each)
        e -= (e - s) % 2
        out.append((t.start + s, t.start + e))
    return out


def make_examples(words: np.ndarray, width: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Decoder-available contexts and symbols for held-out NLL probes."""
    n = len(words)
    if n < 4:
        return {}
    hi = (words >> 8).astype(np.uint8)
    lo = (words & 255).astype(np.uint8)
    idx = np.arange(n, dtype=np.uint64)
    left_w = np.roll(words, 1)
    left_hi = np.roll(hi, 1)
    left_lo = np.roll(lo, 1)
    left2_lo = np.roll(lo, 2)
    left4_lo = np.roll(lo, 4)
    valid_left = (idx % max(width, 1)) > 0
    left_w[~valid_left] = 0
    left_hi[~valid_left] = 0
    left_lo[~valid_left] = 0
    left2_lo[(idx % max(width, 1)) < 2] = 0
    left4_lo[(idx % max(width, 1)) < 4] = 0
    if width > 0 and n > width:
        up_w = np.zeros(n, dtype=np.uint16)
        up_w[width:] = words[:-width]
        up_hi = (up_w >> 8).astype(np.uint8)
        up_lo = (up_w & 255).astype(np.uint8)
    else:
        up_w = np.zeros(n, dtype=np.uint16)
        up_hi = np.zeros(n, dtype=np.uint8)
        up_lo = np.zeros(n, dtype=np.uint8)
    col16 = (idx % max(width, 1) % 16).astype(np.uint8)
    row16 = ((idx // max(width, 1)) % 16).astype(np.uint8)

    # uint64 keys, deliberately bounded-cardinality where possible.
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["hi_prev"] = (left_hi.astype(np.uint64), hi)
    out["hi_left_up"] = ((left_hi.astype(np.uint64) << 8) | up_hi.astype(np.uint64), hi)
    out["hi_left_up_pos"] = (
        (left_hi.astype(np.uint64) << 24)
        | (up_hi.astype(np.uint64) << 16)
        | (row16.astype(np.uint64) << 8)
        | col16.astype(np.uint64),
        hi,
    )
    out["lo_given_hi"] = (hi.astype(np.uint64), lo)
    out["lo_hi_left"] = ((hi.astype(np.uint64) << 8) | left_lo.astype(np.uint64), lo)
    out["lo_hi_left_up"] = (
        (hi.astype(np.uint64) << 16)
        | (left_lo.astype(np.uint64) << 8)
        | up_lo.astype(np.uint64),
        lo,
    )
    out["lo_hi_lags"] = (
        (hi.astype(np.uint64) << 24)
        | (left_lo.astype(np.uint64) << 16)
        | (left2_lo.astype(np.uint64) << 8)
        | left4_lo.astype(np.uint64),
        lo,
    )
    out["word_left"] = (left_w.astype(np.uint64), words.astype(np.uint16))
    out["word_left_up"] = (
        (left_w.astype(np.uint64) << 16) | up_w.astype(np.uint64),
        words.astype(np.uint16),
    )
    return out


class HeldoutCategorical:
    """Static categorical model trained on one split, evaluated on another.

    Contexts with insufficient support back off to the supplied base model.
    This avoids the uncharged-table/memorisation error that plagued earlier
    exploratory work.
    """

    def __init__(self, alphabet: int, alpha: float = 0.5, min_count: int = 8):
        self.alphabet = int(alphabet)
        self.alpha = float(alpha)
        self.min_count = int(min_count)
        self.base_counts: np.ndarray | None = None
        self.keys: np.ndarray | None = None
        self.offsets: np.ndarray | None = None
        self.symbols: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        self.totals: np.ndarray | None = None

    def fit(self, keys: np.ndarray, symbols: np.ndarray) -> "HeldoutCategorical":
        keys = np.asarray(keys, dtype=np.uint64)
        symbols64 = np.asarray(symbols, dtype=np.uint64)
        self.base_counts = np.bincount(symbols64.astype(np.int64), minlength=self.alphabet).astype(np.int64)
        # Pair key and symbol in a structured array so it works for 16-bit alphabet.
        rec = np.empty(len(keys), dtype=[("k", "<u8"), ("s", "<u4")])
        rec["k"] = keys
        rec["s"] = symbols64.astype(np.uint32)
        uniq, cnt = np.unique(rec, return_counts=True)
        order = np.argsort(uniq["k"], kind="stable")
        uniq = uniq[order]
        cnt = cnt[order].astype(np.int64)
        ks, first, totals = np.unique(uniq["k"], return_index=True, return_counts=True)
        # totals here are number of distinct symbols; actual observation totals below.
        obs_totals = np.add.reduceat(cnt, first)
        self.keys = ks
        self.offsets = np.append(first, len(uniq)).astype(np.int64)
        self.symbols = uniq["s"].astype(np.uint32)
        self.counts = cnt
        self.totals = obs_totals
        return self

    def nll(self, keys: np.ndarray, symbols: np.ndarray) -> tuple[float, float, float]:
        assert self.base_counts is not None
        assert self.keys is not None and self.offsets is not None
        assert self.symbols is not None and self.counts is not None and self.totals is not None
        keys = np.asarray(keys, dtype=np.uint64)
        syms = np.asarray(symbols, dtype=np.uint32)
        base_total = int(self.base_counts.sum())
        base_prob = (self.base_counts + self.alpha) / (base_total + self.alpha * self.alphabet)
        pos = np.searchsorted(self.keys, keys)
        seen_ctx = (pos < len(self.keys)) & (self.keys[np.minimum(pos, len(self.keys) - 1)] == keys)
        probs = base_prob[syms.astype(np.int64)].astype(np.float64)
        used = 0
        matched = 0
        # Group held-out examples by context to avoid Python per-symbol loops.
        for p in np.unique(pos[seen_ctx]):
            mask = seen_ctx & (pos == p)
            total = int(self.totals[p])
            if total < self.min_count:
                continue
            used += int(mask.sum())
            a, b = int(self.offsets[p]), int(self.offsets[p + 1])
            train_syms = self.symbols[a:b]
            train_counts = self.counts[a:b]
            q = syms[mask]
            loc = np.searchsorted(train_syms, q)
            hit = (loc < len(train_syms)) & (train_syms[np.minimum(loc, len(train_syms) - 1)] == q)
            c = np.zeros(len(q), dtype=np.float64)
            c[hit] = train_counts[loc[hit]]
            matched += int(hit.sum())
            # Context distribution with Jeffreys smoothing. For unseen symbol in
            # a seen context, interpolate 5% of the global model rather than
            # pretending an impossible zero probability.
            ctx_p = (c + self.alpha) / (total + self.alpha * self.alphabet)
            ctx_p = 0.95 * ctx_p + 0.05 * base_prob[q.astype(np.int64)]
            probs[mask] = ctx_p
        nll = float((-np.log2(np.clip(probs, 2.0 ** -80, 1.0))).mean())
        return nll, used / max(1, len(keys)), matched / max(1, used)


def heldout_context_rates(words: np.ndarray, width: int) -> dict[str, dict[str, float]]:
    examples = make_examples(words, width)
    n = len(words)
    cut = int(n * 0.7)
    # Leave a small gap so a context does not straddle the train/test boundary.
    test0 = min(n, cut + max(16, min(width, 4096)))
    out: dict[str, dict[str, float]] = {}
    for name, (keys, syms) in examples.items():
        alphabet = 65536 if syms.dtype == np.uint16 else 256
        # Full 16-bit context tables can become enormous; use stricter support.
        min_count = 16 if alphabet == 65536 else 8
        model = HeldoutCategorical(alphabet=alphabet, alpha=0.5, min_count=min_count)
        model.fit(keys[:cut], syms[:cut])
        nll, coverage, match = model.nll(keys[test0:], syms[test0:])
        out[name] = {"heldout_nll_bits": nll, "context_coverage": coverage, "symbol_match_in_used_context": match}
    return out


def residual_candidates(words: np.ndarray, width: int) -> dict[str, float]:
    if len(words) < 2:
        return {}
    n = len(words)
    idx = np.arange(n)
    left = np.zeros(n, dtype=np.uint16)
    left[1:] = words[:-1]
    left[(idx % max(width, 1)) == 0] = 0
    up = np.zeros(n, dtype=np.uint16)
    diag = np.zeros(n, dtype=np.uint16)
    if width > 0 and n > width:
        up[width:] = words[:-width]
        if n > width + 1:
            diag[width + 1 :] = words[: -(width + 1)]
        diag[(idx % width) == 0] = 0
    pred_lorenzo = (left.astype(np.uint32) + up.astype(np.uint32) - diag.astype(np.uint32)) & 0xFFFF
    ok = order_key(words)
    ok_left = order_key(left)
    ok_up = order_key(up)
    cands = {
        "raw_word": words,
        "xor_left": words ^ left,
        "xor_up": words ^ up,
        "xor_left_up": words ^ left ^ up,
        "delta_left": (words.astype(np.uint32) - left.astype(np.uint32)).astype(np.uint16),
        "delta_up": (words.astype(np.uint32) - up.astype(np.uint32)).astype(np.uint16),
        "lorenzo_mod16": (words.astype(np.uint32) - pred_lorenzo).astype(np.uint16),
        "ordered_delta_left": (ok.astype(np.uint32) - ok_left.astype(np.uint32)).astype(np.uint16),
        "ordered_delta_up": (ok.astype(np.uint32) - ok_up.astype(np.uint32)).astype(np.uint16),
    }
    return {k: entropy_symbols(v, 65536) for k, v in cands.items()}


def duplicate_block_stats(words: np.ndarray, block_words: int) -> dict[str, float]:
    nblocks = len(words) // block_words
    if nblocks <= 0:
        return {"blocks": 0, "duplicate_fraction": 0.0, "ideal_ref_bits_per_value": 16.0}
    arr = words[: nblocks * block_words].reshape(nblocks, block_words)
    raw = arr.view(np.dtype((np.void, arr.dtype.itemsize * block_words))).ravel()
    _, counts = np.unique(raw, return_counts=True)
    duplicate_blocks = int(np.maximum(counts - 1, 0).sum())
    # Optimistic: first block raw, duplicate refs cost ceil(log2 nblocks).
    ref_bits = max(1, math.ceil(math.log2(max(2, nblocks))))
    bits = (nblocks - duplicate_blocks) * block_words * 16 + duplicate_blocks * ref_bits
    return {
        "blocks": int(nblocks),
        "duplicate_fraction": duplicate_blocks / nblocks,
        "ideal_ref_bits_per_value": bits / (nblocks * block_words),
    }


def berlekamp_massey_binary(bits: np.ndarray) -> int:
    """Linear complexity over GF(2), O(n*L); bounded caller samples only."""
    s = np.asarray(bits, dtype=np.uint8)
    n = len(s)
    c = np.zeros(n, dtype=np.uint8)
    b = np.zeros(n, dtype=np.uint8)
    c[0] = b[0] = 1
    L = 0
    m = -1
    for N in range(n):
        d = int(s[N])
        if L:
            d ^= int(np.bitwise_and(c[1 : L + 1], s[N - np.arange(1, L + 1)]).sum() & 1)
        if d:
            t = c.copy()
            shift = N - m
            if shift < n:
                c[shift:] ^= b[: n - shift]
            if 2 * L <= N:
                L = N + 1 - L
                m = N
                b = t
    return int(L)


def bitplane_linear_complexity(words: np.ndarray, max_bits: int = 8192) -> dict[str, float]:
    out = {}
    take = min(len(words), max_bits)
    w = words[:take]
    for bit in range(16):
        seq = ((w >> bit) & 1).astype(np.uint8)
        L = berlekamp_massey_binary(seq)
        out[str(bit)] = L / max(1, len(seq))
    return out


def per_tensor_probe(words: np.ndarray, t: TensorInfo) -> dict:
    hi = (words >> 8).astype(np.uint8)
    lo = (words & 255).astype(np.uint8)
    joint_h = entropy_symbols(words, 65536)
    hi_h = entropy_symbols(hi, 256)
    lo_h = entropy_symbols(lo, 256)
    # H(low|high) from exact joint histogram.
    conditional_low = joint_h - hi_h
    resid = residual_candidates(words, t.row_width)
    dup = {str(b): duplicate_block_stats(words, b) for b in (8, 16, 32, 64, 128)}
    contexts = heldout_context_rates(words, t.row_width)
    # Compose best high/low held-out rate only from compatible split models.
    hi_names = [n for n in contexts if n.startswith("hi_")]
    lo_names = [n for n in contexts if n.startswith("lo_")]
    best_hi = min((contexts[n]["heldout_nll_bits"] for n in hi_names), default=8.0)
    best_lo = min((contexts[n]["heldout_nll_bits"] for n in lo_names), default=8.0)
    # Compression probes capped at 32 MiB per stream to keep action bounded.
    cap = min(len(words), 16 << 20)
    wb = words[:cap].astype("<u2", copy=False).tobytes()
    hb = hi[:cap].tobytes()
    lb = lo[:cap].tobytes()
    comp = {}
    for label, data in (("word", wb), ("high", hb), ("low", lb)):
        comp[label] = {m: compressor_ratio(data, m) for m in ("zlib9", "bz2", "xz")}
    return {
        "name": t.name,
        "role": role_of(t.name),
        "layer": layer_number(t.name),
        "shape": list(t.shape),
        "row_width": t.row_width,
        "sample_values": int(len(words)),
        "entropy": {
            "joint_word": joint_h,
            "high_byte": hi_h,
            "low_byte": lo_h,
            "low_given_high": conditional_low,
            "sum_high_plus_cond_low": hi_h + conditional_low,
        },
        "heldout_contexts": contexts,
        "best_heldout_field_rate": best_hi + best_lo,
        "residual_entropies": resid,
        "duplicate_blocks": dup,
        "linear_complexity_fraction": bitplane_linear_complexity(words),
        "classical_compressors": comp,
    }


def pair_probe(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    xor = a ^ b
    delta = (b.astype(np.uint32) - a.astype(np.uint32)).astype(np.uint16)
    oa = order_key(a)
    ob = order_key(b)
    odelta = (ob.astype(np.uint32) - oa.astype(np.uint32)).astype(np.uint16)
    return {
        "source": name_a,
        "target": name_b,
        "values": int(n),
        "exact_match_fraction": float(np.mean(a == b)),
        "xor_entropy": entropy_symbols(xor, 65536),
        "delta_entropy": entropy_symbols(delta, 65536),
        "ordered_delta_entropy": entropy_symbols(odelta, 65536),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="nemotron_probe.json")
    ap.add_argument("--max-tensors", type=int, default=28)
    ap.add_argument("--sample-mib-per-tensor", type=int, default=24)
    ap.add_argument("--model-path", default="")
    args = ap.parse_args()

    start_time = time.time()
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        print(f"Downloading {MODEL_ID}/{MODEL_FILE}", flush=True)
        model_path = Path(
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=MODEL_FILE,
                local_dir=os.environ.get("HF_LOCAL_DIR") or None,
            )
        )
    print(f"Model path: {model_path}", flush=True)
    digest = sha256_file(model_path)
    print(f"SHA256: {digest}", flush=True)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"unexpected checkpoint SHA256: {digest}")

    data_base, header, tensors = parse_safetensors(model_path)
    bf16 = [t for t in tensors if t.dtype == "BF16" and t.nbytes >= 4096]
    total_values = sum(t.nvalues for t in bf16)
    print(f"BF16 tensors={len(bf16)} values={total_values:,}", flush=True)

    # Cover the largest tensors while ensuring every major tensor role appears.
    largest = sorted(bf16, key=lambda t: t.nbytes, reverse=True)
    selected: list[TensorInfo] = []
    seen = set()
    for t in largest:
        r = role_of(t.name)
        if r not in seen:
            selected.append(t)
            seen.add(r)
    for t in largest:
        if t not in selected:
            selected.append(t)
        if len(selected) >= args.max_tensors:
            break

    sample_bytes = args.sample_mib_per_tensor << 20
    results = []
    sampled_by_name: dict[str, np.ndarray] = {}
    with model_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for i, t in enumerate(selected, 1):
            chunks = []
            for s, e in sample_ranges(t, sample_bytes):
                chunks.append(np.frombuffer(mm[s:e], dtype="<u2").copy())
            words = np.concatenate(chunks)
            sampled_by_name[t.name] = words
            print(f"[{i}/{len(selected)}] {t.name} shape={t.shape} sample={words.nbytes / (1<<20):.1f} MiB", flush=True)
            results.append(per_tensor_probe(words, t))

        # Cross-layer pairs: same canonical role/shape, consecutive layer number.
        groups: dict[tuple[str, tuple[int, ...]], list[TensorInfo]] = defaultdict(list)
        for t in bf16:
            ln = layer_number(t.name)
            if ln is not None:
                groups[(role_key(t.name), t.shape)].append(t)
        pairs = []
        pair_budget = 24
        for _, group in groups.items():
            group.sort(key=lambda t: layer_number(t.name) or -1)
            for ta, tb in zip(group, group[1:]):
                if pair_budget <= 0:
                    break
                # 8 MiB aligned sample at same tensor-relative location.
                nbytes = min(8 << 20, ta.nbytes, tb.nbytes)
                nbytes -= nbytes % 2
                rel = max(0, (min(ta.nbytes, tb.nbytes) - nbytes) // 2)
                rel -= rel % 2
                a = np.frombuffer(mm[ta.start + rel : ta.start + rel + nbytes], dtype="<u2").copy()
                b = np.frombuffer(mm[tb.start + rel : tb.start + rel + nbytes], dtype="<u2").copy()
                pairs.append(pair_probe(a, b, ta.name, tb.name))
                pair_budget -= 1
            if pair_budget <= 0:
                break

    # Aggregate rates weighted by sampled values.
    w = np.array([r["sample_values"] for r in results], dtype=np.float64)
    def weighted(path: tuple[str, ...]) -> float:
        vals = []
        for r in results:
            x = r
            for p in path:
                x = x[p]
            vals.append(float(x))
        return float(np.average(np.array(vals), weights=w))

    aggregate = {
        "sampled_values": int(w.sum()),
        "zero_order_joint_bits_per_value": weighted(("entropy", "joint_word")),
        "best_heldout_field_bits_per_value": weighted(("best_heldout_field_rate",)),
        "target_50pct_bits_per_value": 8.0,
    }
    aggregate["gap_to_50pct"] = aggregate["best_heldout_field_bits_per_value"] - 8.0
    aggregate["projected_saving_from_best_heldout"] = 1.0 - aggregate["best_heldout_field_bits_per_value"] / 16.0

    report = {
        "contract": {
            "model_id": MODEL_ID,
            "file": MODEL_FILE,
            "expected_sha256": EXPECTED_SHA256,
            "required_max_bits_per_bf16": 8.0,
            "byte_exact": True,
            "self_contained": True,
            "external_base_allowed": False,
        },
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": digest,
            "safetensors_data_base": data_base,
            "tensor_count": len(tensors),
            "bf16_tensor_count": len(bf16),
            "bf16_values": total_values,
        },
        "aggregate": aggregate,
        "tensors": results,
        "cross_layer_pairs": pairs,
        "elapsed_seconds": time.time() - start_time,
    }
    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("\n=== ACCEPTANCE PROBE ===")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"Report: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
