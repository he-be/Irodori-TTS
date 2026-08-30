#!/usr/bin/env python3
"""Probe the AMD iGPU under ROCm torch: device identity, bundled kernel libraries for the
(overridden) gfx target, and raw GEMM / conv1d throughput at fp32 / fp16 / bf16.

Run from the ROCm venv (see docs/experiments/12-igpu-offload.md):
  HSA_OVERRIDE_GFX_VERSION=9.0.0 .venv-rocm/bin/python bench/probe_igpu.py
"""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import torch


def _bench(fn, iters: int = 20, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
    print("HSA_OVERRIDE_GFX_VERSION", os.environ.get("HSA_OVERRIDE_GFX_VERSION"))
    print("cuda.is_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "arch", getattr(p, "gcnArchName", "?"), "SM/CU", p.multi_processor_count,
          "total_mem MiB", p.total_memory // 2**20)
    lib = Path(torch.__file__).parent / "lib"
    for pat in ("rocblas/library/*gfx900*", "rocblas/library/*gfx90c*", "hipblaslt/library/*gfx900*"):
        hits = glob.glob(str(lib / pat))
        print(f"{pat}: {len(hits)} files")
    archs = sorted({Path(f).name.split("_")[-1].split(".")[0] for f in glob.glob(str(lib / "rocblas/library/*.dat"))})
    print("rocblas tensile archs:", archs)

    free, total = torch.cuda.mem_get_info()
    print(f"mem_get_info free={free//2**20} MiB total={total//2**20} MiB")

    # DiT-shaped GEMM: (B*T=486, 1280) x (1280, 3680) -> MLP up-proj at CFG batch 3, 162 frames
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        try:
            a = torch.randn(486, 1280, device="cuda", dtype=dt)
            w = torch.randn(1280, 3680, device="cuda", dtype=dt)
            s = _bench(lambda: a @ w)
            print(f"GEMM 486x1280x3680 {dt}: {s*1e3:.2f} ms  {2*486*1280*3680/s/1e12:.2f} TFLOPS")
            a = torch.randn(2048, 2048, device="cuda", dtype=dt)
            s = _bench(lambda: a @ a)
            print(f"GEMM 2048^3 {dt}: {s*1e3:.2f} ms  {2*2048**3/s/1e12:.2f} TFLOPS")
        except Exception as e:  # noqa: BLE001
            print(f"GEMM {dt}: FAILED {type(e).__name__}: {str(e)[:120]}")

    # codec-shaped conv1d: 512 ch, k=7, length 4096 (mid decoder stage)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        try:
            x = torch.randn(1, 512, 4096, device="cuda", dtype=dt)
            conv = torch.nn.Conv1d(512, 512, 7, padding=3).to("cuda", dt)
            with torch.no_grad():
                s = _bench(lambda: conv(x))
            print(f"conv1d 512->512 k7 L4096 {dt}: {s*1e3:.2f} ms  {2*512*512*7*4096/s/1e12:.2f} TFLOPS")
        except Exception as e:  # noqa: BLE001
            print(f"conv1d {dt}: FAILED {type(e).__name__}: {str(e)[:120]}")

    # sdpa
    for dt in (torch.float32, torch.float16):
        try:
            q = torch.randn(3, 20, 162, 64, device="cuda", dtype=dt)
            s = _bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q))
            print(f"sdpa B3 H20 T162 D64 {dt}: {s*1e3:.2f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"sdpa {dt}: FAILED {type(e).__name__}: {str(e)[:120]}")
    print("peak alloc MiB", torch.cuda.max_memory_allocated() // 2**20)


if __name__ == "__main__":
    main()
