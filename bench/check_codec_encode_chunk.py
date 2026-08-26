#!/usr/bin/env python3
"""Chunked reference encode vs full encode: equality, time, transient VRAM."""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irodori_tts.codec import DACVAECodec

def run(codec, wav, sr, **kw):
    for _ in range(1):
        codec.encode_waveform(wav, sample_rate=sr, normalize_db=None, **kw)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(3):
        out = codec.encode_waveform(wav, sample_rate=sr, normalize_db=None, **kw)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / 3 * 1000, (torch.cuda.max_memory_allocated() - base) / 2**20

codec = DACVAECodec.load(device="cuda", dtype=torch.float32)
sr = codec.sample_rate
torch.manual_seed(0)
for seconds in (7.28, 15.0, 30.0, 60.0, 120.0):
    wav = (torch.randn(1, 1, int(seconds * sr)) * 0.1).clamp(-1, 1)
    full, t_full, m_full = run(codec, wav, sr)
    print(f"[{seconds:6.2f}s frames={full.shape[1]}] full={t_full:7.1f}ms +{m_full:7.1f}MiB")
    for chunk, ovl in ((96, 16), (96, 8), (96, 4), (192, 16), (48, 16)):
        out, t, m = run(codec, wav, sr, chunk_frames=chunk, overlap_frames=ovl)
        n = min(out.shape[1], full.shape[1])
        d = (out[:, :n].float() - full[:, :n].float()).abs().max().item()
        rel = d / full.float().abs().max().item()
        print(f"    chunk{chunk}/ovl{ovl}: {t:7.1f}ms +{m:7.1f}MiB frames_eq={out.shape[1]==full.shape[1]} "
              f"maxdiff={d:.2e} rel={rel:.2e} bitwise={torch.equal(out, full)}")
