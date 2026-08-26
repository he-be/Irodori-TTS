#!/usr/bin/env python3
"""Transient VRAM and time of a single codec decode vs latent length."""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irodori_tts.codec import DACVAECodec
dtype = torch.bfloat16 if (len(sys.argv) > 1 and sys.argv[1] == "bf16") else torch.float32
if len(sys.argv) > 2 and sys.argv[2] == "benchmark":
    torch.backends.cudnn.benchmark = True
codec = DACVAECodec.load(device="cuda", dtype=dtype)
torch.manual_seed(0)
for frames in (40, 64, 80, 96, 128, 162, 275, 720):
    z = torch.randn(1, frames, codec.latent_dim, device="cuda").to(dtype)
    codec.decode_latent(z); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(5): out = codec.decode_latent(z)
    torch.cuda.synchronize()
    print(f"frames={frames:5d} ({frames*1920/48000:5.2f}s) transient=+{(torch.cuda.max_memory_allocated()-base)/2**20:6.0f}MiB time={(time.perf_counter()-t0)/5*1000:6.1f}ms")
