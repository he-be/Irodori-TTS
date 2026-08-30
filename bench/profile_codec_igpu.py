#!/usr/bin/env python3
"""Per-op profile of the codec decoder on the current CUDA/ROCm device (docs/experiments/12).

  HSA_OVERRIDE_GFX_VERSION=9.0.0 .venv-rocm/bin/python bench/profile_codec_igpu.py --precision fp16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import DACVAECodec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--frames", type=int, default=163)
    ap.add_argument("--chunk", type=int, default=96)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--rows", type=int, default=25)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = bool(args.benchmark)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    codec = DACVAECodec.load("Aratako/Semantic-DACVAE-Japanese-32dim", device="cuda", dtype=dtype)
    z = torch.randn(1, args.frames, 32, device="cuda")
    with torch.no_grad():
        for i in range(2):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = codec.decode_latent(z, chunk_frames=args.chunk, overlap_frames=args.overlap)
            torch.cuda.synchronize()
            print(f"warm{i}: {time.perf_counter()-t0:.3f}s  audio {out.shape[-1]/codec.sample_rate:.2f}s")
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            codec.decode_latent(z, chunk_frames=args.chunk, overlap_frames=args.overlap)
            torch.cuda.synchronize()
            print(f"profiled: {time.perf_counter()-t0:.3f}s")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows, max_name_column_width=60))
    print("peak alloc MiB", torch.cuda.max_memory_allocated() // 2**20)


if __name__ == "__main__":
    main()
