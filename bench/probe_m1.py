#!/usr/bin/env python3
"""Peak matmul / bandwidth probe for this Mac's GPU (mirrors 12-metal-port.md table)."""
import json, platform, subprocess, time
import torch


def bench_matmul(dtype, n=4096, iters=10):
    a = torch.randn(n, n, device="mps", dtype=dtype)
    b = torch.randn(n, n, device="mps", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * n ** 3 / dt / 1e12, dt * 1e3


def bench_bandwidth(mb=512, iters=20):
    n = mb * 1024 * 1024 // 2  # fp16 elements
    a = torch.empty(n, device="mps", dtype=torch.float16).fill_(1.0)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * n * 2 / dt / 1e9, dt * 1e3  # read+write GB/s


def bench_reduce(mb=512, iters=20):
    n = mb * 1024 * 1024 // 2
    a = torch.empty(n, device="mps", dtype=torch.float16).fill_(1.0)
    for _ in range(3):
        a.sum()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a.sum()
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return n * 2 / dt / 1e9, dt * 1e3  # read-only GB/s


out = {
    "platform": platform.platform(),
    "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
    "torch": torch.__version__,
    "recommended_max_memory_mib": torch.mps.recommended_max_memory() / 1024 / 1024,
}
for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
    tflops, ms = bench_matmul(dt)
    out[f"matmul4096_{name}_tflops"] = round(tflops, 3)
    out[f"matmul4096_{name}_ms"] = round(ms, 3)
gbs, ms = bench_bandwidth()
out["copy_rw_gbs"] = round(gbs, 1)
gbs, ms = bench_reduce()
out["read_gbs"] = round(gbs, 1)
print(json.dumps(out, indent=2))
