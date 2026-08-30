#!/usr/bin/env python3
"""Sweep M for fused vs separate DiT Linear GEMMs (docs/experiments/13).

The DiT's wq/wk/wv/gate (4 x 1280->1280) and w1/w3 (2 x 1280->3680) can be evaluated as one
GEMM each; whether that is faster depends erratically on M (rocBLAS kernel selection on gfx900).
This prints the gain per M so the no-fuse ranges in irodori_tts/model.py (LINEAR_FUSION_SKIP_RANGES)
can be regenerated.

  HSA_OVERRIDE_GFX_VERSION=9.0.0 .venv-rocm/bin/python bench/sweep_linear_fusion.py --start 64 --stop 2400 --step 32
  HSA_OVERRIDE_GFX_VERSION=9.0.0 .venv-rocm/bin/python bench/sweep_linear_fusion.py --ms 1216-1264:8 1592-1640:8
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

GROUPS = {"qkvg": (1280, 1280, 4), "w1w3": (1280, 3680, 2)}  # K, N per member, members


def timeit(f, iters: int, rounds: int) -> float:
    for _ in range(5):
        f()
    best = float("inf")
    for _ in range(rounds):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1e3)
    return best


def parse_ms(args: argparse.Namespace) -> list[int]:
    if args.ms:
        out: list[int] = []
        for spec in args.ms:
            rng, _, step = spec.partition(":")
            lo, _, hi = rng.partition("-")
            out.extend(range(int(lo), int(hi) + 1, int(step or 1)))
        return out
    return list(range(args.start, args.stop, args.step))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=64)
    ap.add_argument("--stop", type=int, default=2400)
    ap.add_argument("--step", type=int, default=32)
    ap.add_argument("--ms", nargs="*", help="explicit ranges lo-hi:step")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    args = ap.parse_args()
    dt = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    dev = "cuda"
    weights = {g: [torch.randn(n, k, device=dev, dtype=dt) for _ in range(p)] for g, (k, n, p) in GROUPS.items()}
    fused = {g: torch.cat(ws, 0) for g, ws in weights.items()}
    print(f"{'M':>5} | " + " | ".join(f"{g}: sep / fused ms (gain)" for g in GROUPS))
    for m in parse_ms(args):
        x = torch.randn(m, 1280, device=dev, dtype=dt)
        cols = []
        for g in GROUPS:
            ws, wf = weights[g], fused[g]
            sep = timeit(lambda: [F.linear(x, w) for w in ws], args.iters, args.rounds)
            fus = timeit(lambda: F.linear(x, wf), args.iters, args.rounds)
            cols.append(f"{sep:6.2f} / {fus:6.2f} ({100 * (1 - fus / sep):+5.1f}%)")
        print(f"{m:5d} | " + " | ".join(cols), flush=True)


if __name__ == "__main__":
    main()
