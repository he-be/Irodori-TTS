#!/usr/bin/env python3
"""Measure DACVAE decode variants: eager fp32 / bf16, torch.compile (dynamic) fp32 / bf16."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import DACVAECodec  # noqa: E402


def bench(fn, z: torch.Tensor, n: int = 15) -> float:
    for _ in range(3):
        fn(z)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(z)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> None:
    torch.manual_seed(0)
    codecs = {
        "fp32": DACVAECodec.load(device="cuda", dtype=torch.float32),
        "bf16": DACVAECodec.load(device="cuda", dtype=torch.bfloat16),
    }
    compiled = {}
    for name, codec in codecs.items():
        dec = codec.model.decoder
        compiled[name] = torch.compile(dec, dynamic=True)
    ref = codecs["fp32"]
    lengths = [162, 275, 720, 400, 190]  # repeated / new shapes to see recompiles
    for frames in lengths:
        z = torch.randn(1, frames, ref.latent_dim, device="cuda") * 2.0
        zt = z.transpose(1, 2).contiguous()
        outs = {}
        rows = []
        for name, codec in codecs.items():
            zc = zt.to(codec.dtype)
            emb = codec.model.quantizer.out_proj(zc)
            eager = lambda _z, m=codec.model.decoder, e=emb: m(e)  # noqa: E731
            comp = lambda _z, m=compiled[name], e=emb: m(e)  # noqa: E731
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs[f"{name}_compiled"] = comp(z)
            torch.cuda.synchronize()
            first = (time.perf_counter() - t0) * 1000.0
            outs[f"{name}_eager"] = eager(z)
            rows.append(
                f"{name}: eager={bench(eager, z):.1f}ms compiled={bench(comp, z):.1f}ms (first call {first:.0f}ms)"
            )
        base = outs["fp32_eager"].float()
        diffs = {k: (v.float() - base).abs().max().item() for k, v in outs.items()}
        print(f"[frames={frames} ~{frames*512/48000:.1f}s] " + " | ".join(rows))
        print("    maxdiff vs fp32 eager: " + ", ".join(f"{k}={v:.2e}" for k, v in diffs.items()))


if __name__ == "__main__":
    main()
