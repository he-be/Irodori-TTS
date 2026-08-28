#!/usr/bin/env python3
"""Codec decode on MPS: chunking, autocast dtype and torch.compile (experiment 12).

Chunked decode was introduced on the CUDA box to bound the transient VRAM; on unified
memory the question is only whether the 2*overlap/window recompute costs more than the
Metal graph-cache benefit of repeating shapes.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import DACVAECodec  # noqa: E402


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000.0


CONFIGS = [
    ("chunk0/fp32", {"chunk_frames": None, "autocast_dtype": None}),
    ("chunk0/fp16", {"chunk_frames": None, "autocast_dtype": torch.float16}),
    ("chunk0/bf16", {"chunk_frames": None, "autocast_dtype": torch.bfloat16}),
    ("chunk96-16/fp16", {"chunk_frames": 96, "overlap_frames": 16, "autocast_dtype": torch.float16}),
    ("chunk256-16/fp16", {"chunk_frames": 256, "overlap_frames": 16, "autocast_dtype": torch.float16}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", type=int, default=[180, 300, 720])
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    codec = DACVAECodec.load(device="mps", dtype=torch.float32)
    g = torch.Generator(device="mps").manual_seed(0)
    for frames in args.frames:
        z = torch.randn(1, frames, codec.latent_dim, device="mps", generator=g) * 2.0
        ref = codec.decode_latent(z, chunk_frames=None, autocast_dtype=None).float()
        print(f"== frames={frames} ({frames * codec.model.hop_length / codec.sample_rate:.2f}s)")
        for name, kw in CONFIGS:
            out = codec.decode_latent(z, **kw).float()
            n = min(out.shape[-1], ref.shape[-1])
            diff = (out[..., :n] - ref[..., :n]).abs().max().item()
            ms = timeit(lambda z=z, kw=kw: codec.decode_latent(z, **kw))
            print(f"  {name:18s} {ms:7.1f} ms  maxdiff_vs_fp32={diff:.2e}")
    if not args.compile:
        return
    codec.model.decoder = torch.compile(codec.model.decoder, dynamic=True)
    for frames in args.frames:
        z = torch.randn(1, frames, codec.latent_dim, device="mps", generator=g) * 2.0
        t0 = time.perf_counter()
        codec.decode_latent(z, autocast_dtype=torch.float16)
        torch.mps.synchronize()
        first = time.perf_counter() - t0
        ms = timeit(lambda z=z: codec.decode_latent(z, autocast_dtype=torch.float16))
        print(f"== compiled frames={frames}: chunk0/fp16 {ms:.1f} ms (first {first:.1f}s)")


if __name__ == "__main__":
    main()
