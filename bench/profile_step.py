#!/usr/bin/env python3
"""Where does one RF Euler step go on MPS? (experiment 12)

Times ``forward_with_encoded_conditions`` at several latent lengths / CFG batch sizes
(compute-bound: time grows with tokens; dispatch-bound: it does not) and breaks a step
down per module class with forward hooks (each hook syncs, so the breakdown is an
upper bound). Optionally compares the torch.compile'd forward.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    download_hf_checkpoint,
)

HOOKED_TYPES = {"Linear", "RMSNorm", "LowRankAdaLN", "SwiGLU", "JointAttention"}


def _timeit(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000.0


def _sync_time() -> float:
    torch.mps.synchronize()
    return time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--text-len", type=int, default=32)
    ap.add_argument("--speaker-len", type=int, default=64)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
            model_device="mps",
            model_precision=args.precision,
            codec_device="mps",
            codec_precision="fp32",
        )
    )
    model = runtime.model
    dev, dt = runtime.model_device, runtime._model_dtype
    cfg = model.cfg
    print(
        f"[cfg] model_dim={cfg.model_dim} layers={cfg.num_layers} heads={cfg.num_heads} "
        f"patched_latent_dim={cfg.patched_latent_dim} text_dim={cfg.text_dim} "
        f"speaker_dim={cfg.speaker_dim}"
    )

    def make(batch: int, latent_len: int) -> dict:
        text_state = torch.randn(batch, args.text_len, cfg.text_dim, device=dev, dtype=dt)
        text_mask = torch.ones(batch, args.text_len, dtype=torch.bool, device=dev)
        speaker_state = torch.randn(batch, args.speaker_len, cfg.speaker_dim, device=dev, dtype=dt)
        speaker_mask = torch.ones(batch, args.speaker_len, dtype=torch.bool, device=dev)
        caption_state = None
        caption_mask = None
        if cfg.use_caption_condition:
            caption_state = torch.randn(
                batch, args.text_len, cfg.caption_dim_resolved, device=dev, dtype=dt
            )
            caption_mask = torch.ones(batch, args.text_len, dtype=torch.bool, device=dev)
        kv = model.build_context_kv_cache(
            text_state=text_state, speaker_state=speaker_state, caption_state=caption_state
        )
        mask = model.build_combined_attn_mask(
            latent_len=latent_len,
            text_mask=text_mask,
            speaker_mask=speaker_mask,
            caption_mask=caption_mask,
        )
        return {
            "x_t": torch.randn(batch, latent_len, cfg.patched_latent_dim, device=dev, dtype=dt),
            "t": torch.full((batch,), 0.5, device=dev, dtype=dt),
            "text_state": text_state,
            "text_mask": text_mask,
            "speaker_state": speaker_state,
            "speaker_mask": speaker_mask,
            "caption_state": caption_state,
            "caption_mask": caption_mask,
            "context_kv_cache": kv,
            "attn_mask": mask,
        }

    fwd = model.forward_with_encoded_conditions
    with torch.inference_mode():
        print("== scaling (ms per forward, fast path with KV cache + precombined mask)")
        for batch in (1, 3, 4):
            for latent_len in (45, 90, 180, 360):
                kw = make(batch, latent_len)
                ms = _timeit(lambda kw=kw: fwd(**kw))
                tokens = batch * latent_len
                print(
                    f"  batch={batch} latent={latent_len:4d} tokens={tokens:5d}: {ms:7.1f} ms  "
                    f"({ms / tokens * 1000:.0f} us/token)"
                )

        print("== per-module breakdown (batch=3, latent=90; hooks sync -> upper bound)")
        acc: dict[str, float] = defaultdict(float)
        cnt: dict[str, int] = defaultdict(int)
        stack: list[float] = []

        def pre(module, _inputs):
            stack.append(_sync_time())

        def post(module, _inputs, _output):
            name = type(module).__name__
            acc[name] += _sync_time() - stack.pop()
            cnt[name] += 1

        handles = []
        for module in model.modules():
            if type(module).__name__ in HOOKED_TYPES:
                handles.append(module.register_forward_pre_hook(pre))
                handles.append(module.register_forward_hook(post))
        kw = make(3, 90)
        fwd(**kw)
        torch.mps.synchronize()
        acc.clear()
        cnt.clear()
        t0 = _sync_time()
        fwd(**kw)
        total = _sync_time() - t0
        for handle in handles:
            handle.remove()
        # Linear inside JointAttention/SwiGLU/LowRankAdaLN is counted at both levels.
        for name, sec in sorted(acc.items(), key=lambda item: -item[1]):
            print(f"  {name:16s} n={cnt[name]:4d} {sec * 1000:7.1f} ms")
        print(f"  total (hooked) {total * 1000:.1f} ms")

        if args.compile:
            print("== torch.compile (inductor/mps, dynamic=True)")
            cfwd = torch.compile(model.forward_with_encoded_conditions, dynamic=True)
            for batch, latent_len in ((3, 90), (3, 180), (4, 360)):
                kw = make(batch, latent_len)
                t0 = time.perf_counter()
                cfwd(**kw)
                torch.mps.synchronize()
                first = time.perf_counter() - t0
                ms = _timeit(lambda kw=kw: cfwd(**kw))
                eager = _timeit(lambda kw=kw: fwd(**kw))
                print(
                    f"  batch={batch} latent={latent_len}: compiled {ms:.1f} ms vs eager "
                    f"{eager:.1f} ms (first call {first:.1f} s)"
                )


if __name__ == "__main__":
    main()
