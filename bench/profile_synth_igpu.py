#!/usr/bin/env python3
"""Kernel-level profile of one warm synthesize() on the current CUDA/ROCm device (docs/experiments/12).

  HSA_OVERRIDE_GFX_VERSION=9.0.0 IRODORI_OPT_CUDA_GRAPH=0 IRODORI_OPT_VRAM_LIMIT_MB=0 \
    .venv-rocm/bin/python bench/profile_synth_igpu.py --precision fp16 --codec-precision fp16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "bench"))

from bench_runtime import DEFAULT_REF, INPUTS  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--codec-precision", default="fp16")
    ap.add_argument("--input", default="short")
    ap.add_argument("--num-steps", type=int, default=12)
    ap.add_argument("--rows", type=int, default=30)
    args = ap.parse_args()
    ck = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
    rt = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ck, model_device="cuda", model_precision=args.precision,
                   codec_device="cuda", codec_precision=args.codec_precision)
    )
    spec = INPUTS[args.input]
    req = SamplingRequest(text=str(spec["text"]), caption=spec["caption"], ref_wav=DEFAULT_REF,
                          num_steps=args.num_steps, t_schedule_mode="sway", seed=1234)
    for i in range(2):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = rt.synthesize(req)
        torch.cuda.synchronize()
        print(f"warm{i}: {time.perf_counter()-t0:.3f}s", {k: round(v * 1e3) for k, v in r.stage_timings})
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = rt.synthesize(req)
        torch.cuda.synchronize()
    print(f"profiled: {time.perf_counter()-t0:.3f}s", {k: round(v * 1e3) for k, v in r.stage_timings})
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows, max_name_column_width=70))
    print("peak alloc MiB", torch.cuda.max_memory_allocated() // 2**20,
          "reserved", torch.cuda.memory_reserved() // 2**20)


if __name__ == "__main__":
    main()
