#!/usr/bin/env python3
"""Verify that folding weight_norm in the DACVAE codec is output-preserving and measure decode time."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import DACVAECodec  # noqa: E402


def bench(codec: DACVAECodec, z: torch.Tensor, n: int = 20) -> float:
    for _ in range(3):
        codec.decode_latent(z)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        codec.decode_latent(z)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> None:
    dtype_arg = sys.argv[1] if len(sys.argv) > 1 else "fp32"
    dtype = torch.bfloat16 if dtype_arg == "bf16" else torch.float32
    torch.manual_seed(0)
    plain = DACVAECodec.load(device="cuda", dtype=torch.float32, fold_weight_norm=False)
    folded = DACVAECodec.load(device="cuda", dtype=dtype, fold_weight_norm=True)
    for seconds in (6.5, 11.0, 28.8):
        frames = int(seconds * plain.sample_rate / plain.model.hop_length)
        z = torch.randn(1, frames, plain.latent_dim, device="cuda") * 2.0
        a = plain.decode_latent(z)
        b = folded.decode_latent(z)
        diff = (a.float() - b.float()).abs().max().item()
        same = torch.equal(a, b) if dtype == torch.float32 else False
        torch.cuda.reset_peak_memory_stats()
        t_plain = bench(plain, z)
        peak_plain = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        t_fold = bench(folded, z)
        peak_fold = torch.cuda.max_memory_allocated() / 2**20
        print(
            f"[{seconds:.1f}s frames={frames}] bitwise_equal={same} maxdiff={diff:.3e} "
            f"decode plain_fp32={t_plain:.1f}ms folded_{dtype_arg}={t_fold:.1f}ms "
            f"peak plain={peak_plain:.0f}MiB folded={peak_fold:.0f}MiB"
        )


if __name__ == "__main__":
    main()
