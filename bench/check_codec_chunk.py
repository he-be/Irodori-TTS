#!/usr/bin/env python3
"""Chunked decode vs full decode: equality, time, transient VRAM."""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irodori_tts.codec import DACVAECodec

def run(codec, z, **kw):
    for _ in range(2): codec.decode_latent(z, **kw)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(8): out = codec.decode_latent(z, **kw)
    torch.cuda.synchronize()
    return out, (time.perf_counter()-t0)/8*1000, (torch.cuda.max_memory_allocated()-base)/2**20

dtype = torch.bfloat16 if (len(sys.argv) > 1 and sys.argv[1] == "bf16") else torch.float32
codec = DACVAECodec.load(device="cuda", dtype=dtype)
torch.manual_seed(0)
for frames in (162, 275, 720, 183, 101, 66, 40):
    z = (torch.randn(1, frames, codec.latent_dim, device="cuda") * 2.0).to(dtype)
    full, t_full, m_full = run(codec, z)
    for chunk, ovl in ((64, 16), (64, 12), (48, 16), (96, 16)):
        out, t, m = run(codec, z, chunk_frames=chunk, overlap_frames=ovl)
        n = min(out.shape[-1], full.shape[-1])
        d = (out[..., :n].float() - full[..., :n].float()).abs().max().item()
        print(f"[frames={frames}] full={t_full:.1f}ms/+{m_full:.0f}MiB chunk{chunk}/ovl{ovl}={t:.1f}ms/+{m:.0f}MiB len_eq={out.shape[-1]==full.shape[-1]} maxdiff={d:.2e} bitwise={torch.equal(out, full)}")
