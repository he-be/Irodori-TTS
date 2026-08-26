#!/usr/bin/env python3
"""Worst-case VRAM stress test for the hard cap (IRODORI_OPT_VRAM_LIMIT_MB).

Unlike bench_runtime.py (representative inputs, speed focus) this script drives the
runtime with the *documented maxima* of the checkpoint -- max_text_len=256,
max_caption_len=512, ref_max_seconds=120 -- plus a session that fills the CUDA Graph
LRU, and reports what the caching allocator actually had to reserve. The number that
matters for the cap is ``max_memory_reserved`` (the cap limits reserved bytes: weights
+ transients + CUDA Graph private pool + fragmentation).

Example:
  uv run --no-sync python bench/stress_vram.py --tag cap3072 \
      --env IRODORI_OPT_VRAM_LIMIT_MB=3072 \
      --output docs/experiments/results/09_cap3072.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MiB = 2**20
DEFAULT_REF = str(REPO_ROOT / "outputs" / "sample.wav")

# A paragraph that tokenizes well past max_text_len=256 for the ModernBERT-ja tokenizer;
# the runtime truncates to the cap, which is exactly what we want to measure.
_SENTENCE = (
    "音声合成の処理時間とメモリ使用量を正確に把握するためには、"
    "実際の運用で起こりうる最大の入力を与えて、どの段階でどれだけの領域が確保されるのかを"
    "一つずつ測定していく必要があります。"
)
LONG_TEXT = _SENTENCE * 12
LONG_CAPTION = (
    "落ち着いた大人の女性の声で、近い距離感を保ちながら、やわらかく、"
    "ゆっくりと、聞き手を安心させるような口調で、抑揚は控えめに、"
    "語尾はやさしく丁寧に読み上げてください。"
) * 12


def build_long_ref(seconds: float, out_dir: Path) -> str:
    """Tile outputs/sample.wav up to `seconds` (worst-case reference encode)."""
    import numpy as np
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"ref_{int(seconds)}s.wav"
    if dst.exists():
        return str(dst)
    data, sr = sf.read(DEFAULT_REF, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    need = int(seconds * sr)
    reps = int(need // len(data)) + 1
    sf.write(str(dst), np.tile(data, reps)[:need], sr)
    return str(dst)


def _private_pool_bytes() -> int:
    """Bytes held by CUDA Graph private pools (inside `allocated`, never released)."""
    total = 0
    for seg in torch.cuda.memory_snapshot():
        pool_id = seg.get("segment_pool_id") or (0, 0)
        if tuple(pool_id) != (0, 0):
            total += int(seg.get("total_size", 0))
    return total


def _smi_used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        return int(out.splitlines()[0])
    except (ValueError, IndexError):
        return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--tag", default="stress")
    parser.add_argument("--output", default=None)
    parser.add_argument("--repeats", type=int, default=2, help="repeats per stress case")
    parser.add_argument(
        "--graph-fill",
        type=int,
        default=14,
        help="number of distinct-length warmup requests (fills the CUDA Graph LRU)",
    )
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument(
        "--env", action="append", default=[], help="KEY=VALUE applied before runtime load"
    )
    args = parser.parse_args()

    for item in args.env:
        k, _, v = item.partition("=")
        os.environ[k] = v

    from irodori_tts.inference_runtime import (  # noqa: E402
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        download_hf_checkpoint,
    )
    from irodori_tts.opt_config import get_opt_config  # noqa: E402

    scratch = Path(
        os.environ.get("IRODORI_STRESS_DIR", REPO_ROOT / "outputs" / "stress_refs")
    )
    ref_15 = build_long_ref(15, scratch)
    ref_30 = build_long_ref(30, scratch)
    ref_60 = build_long_ref(60, scratch)
    ref_120 = build_long_ref(120, scratch)

    cases: dict[str, dict] = {
        # documented maxima, one dimension at a time
        "text_max": {"text": LONG_TEXT, "ref_wav": args.ref},
        "caption_max": {
            "text": LONG_TEXT,
            "caption": LONG_CAPTION,
            "ref_wav": args.ref,
        },
        "caption_max_noref": {"text": LONG_TEXT, "caption": LONG_CAPTION, "no_ref": True},
        "ref15": {"text": LONG_TEXT, "ref_wav": ref_15},
        "ref30": {"text": LONG_TEXT, "ref_wav": ref_30},
        # `worst` is the operating-policy worst case: every documented maximum except the
        # reference, which is capped at 30 s for local use (ref_max_seconds=120 in the
        # checkpoint is out of policy but measured below for headroom).
        "worst": {"text": LONG_TEXT, "caption": LONG_CAPTION, "ref_wav": ref_30},
        "ref60": {"text": LONG_TEXT, "ref_wav": ref_60},
        "ref120": {"text": LONG_TEXT, "ref_wav": ref_120},
        "worst_ref120": {"text": LONG_TEXT, "caption": LONG_CAPTION, "ref_wav": ref_120},
    }
    names = args.cases or list(cases)

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint(args.hf_checkpoint),
            model_device="cuda",
            model_precision=args.precision,
            codec_device="cuda",
            codec_precision=args.codec_precision,
        )
    )
    torch.cuda.synchronize(dev)
    load_sec = time.perf_counter() - t0
    after_load = {
        "allocated": int(torch.cuda.memory_allocated(dev)),
        "reserved": int(torch.cuda.memory_reserved(dev)),
    }
    print(
        f"[load] {load_sec:.1f}s alloc={after_load['allocated']/MiB:.0f} "
        f"reserved={after_load['reserved']/MiB:.0f} MiB "
        f"limit={get_opt_config().vram_limit_mb} MB "
        f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','')!r}",
        flush=True,
    )

    def make(spec: dict) -> SamplingRequest:
        return SamplingRequest(
            text=str(spec["text"]),
            caption=spec.get("caption"),
            ref_wav=spec.get("ref_wav"),
            no_ref=bool(spec.get("no_ref", False)),
            num_steps=40,
            seed=1234,
        )

    # Fill the CUDA Graph LRU with distinct latent-length buckets, as a long-lived
    # server process would. Text lengths ramp so each request lands in its own bucket.
    fill_errors: list[str] = []
    for i in range(int(args.graph_fill)):
        text = _SENTENCE[: 20 + i * 9] or _SENTENCE[:20]
        try:
            runtime.synthesize(make({"text": text, "ref_wav": args.ref}))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # pragma: no cover
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc):
                raise
            fill_errors.append(f"fill[{i}]: {str(exc).splitlines()[0]}")
            torch.cuda.empty_cache()
            break
    graph_stats_fill = (
        runtime._graph_runner.stats() if runtime._graph_runner is not None else None
    )
    print(
        f"[fill] graph={graph_stats_fill} pool={_private_pool_bytes()/MiB:.0f}MiB "
        f"alloc={torch.cuda.memory_allocated(dev)/MiB:.0f}MiB errors={len(fill_errors)}",
        flush=True,
    )

    results: dict[str, dict] = {}
    for name in names:
        spec = cases[name]
        req = make(spec)
        rec: dict[str, object] = {}
        try:
            # first call may capture a new graph / grow the pool: measured too
            torch.cuda.reset_peak_memory_stats(dev)
            walls = []
            audio_sec = 0.0
            for _ in range(max(1, int(args.repeats))):
                torch.cuda.synchronize(dev)
                t = time.perf_counter()
                out = runtime.synthesize(req)
                torch.cuda.synchronize(dev)
                walls.append(time.perf_counter() - t)
                audio_sec = float(out.audio.shape[-1]) / float(out.sample_rate)
            rec.update(
                ok=True,
                audio_seconds=audio_sec,
                wall_first=walls[0],
                wall_last=walls[-1],
                max_allocated=int(torch.cuda.max_memory_allocated(dev)),
                max_reserved=int(torch.cuda.max_memory_reserved(dev)),
                private_pool=_private_pool_bytes(),
                smi_used_after=_smi_used(),
            )
            print(
                f"[{name}] ok audio={audio_sec:.1f}s wall={walls[0]*1000:.0f}/"
                f"{walls[-1]*1000:.0f}ms peak_alloc={rec['max_allocated']/MiB:.0f} "
                f"peak_reserved={rec['max_reserved']/MiB:.0f} MiB "
                f"pool={rec['private_pool']/MiB:.0f} smi={rec['smi_used_after']}MiB",
                flush=True,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc):
                raise
            oom_lines = [ln for ln in str(exc).splitlines() if "out of memory" in ln]
            rec.update(
                ok=False,
                error=(oom_lines or str(exc).splitlines())[0],
                max_allocated=int(torch.cuda.max_memory_allocated(dev)),
                max_reserved=int(torch.cuda.max_memory_reserved(dev)),
                private_pool=_private_pool_bytes(),
                smi_used_after=_smi_used(),
            )
            print(f"[{name}] OOM {rec['error']}", flush=True)
            torch.cuda.empty_cache()
        if runtime._graph_runner is not None:
            rec["graph"] = runtime._graph_runner.stats()
        results[name] = rec

    record = {
        "tag": args.tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip(),
        "env": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(dev),
            "env_overrides": list(args.env),
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        },
        "config": {
            "precision": args.precision,
            "codec_precision": args.codec_precision,
            "repeats": args.repeats,
            "graph_fill": args.graph_fill,
            "opt": get_opt_config().describe(),
        },
        "load_sec": load_sec,
        "cuda_mem_after_load": after_load,
        "graph_after_fill": graph_stats_fill,
        "fill_errors": fill_errors,
        "results": results,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[stress] wrote {out}")

    worst = max(
        (int(r.get("max_reserved", 0)) for r in results.values()),
        default=0,
    )
    print(f"[stress] worst peak_reserved={worst/MiB:.0f} MiB over {len(results)} cases")


if __name__ == "__main__":
    main()
