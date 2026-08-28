#!/usr/bin/env python3
"""Warm benchmark harness for Irodori-TTS inference on a single machine.

Loads the runtime once, runs warmup + timed repeats for a set of representative
inputs, and writes a JSON record (stage timings, wall median/p95, RTF, audio
hashes, MPS memory sampled in a background thread).

Example:
  uv run python bench/bench_runtime.py --precision fp16 --tag fp16 \
      --output docs/experiments/results/metal_fp16.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)

DEFAULT_REF = str(REPO_ROOT / "outputs" / "sample.wav")

# Representative inputs: short / medium / long Japanese sentences.
INPUTS: dict[str, dict[str, str | None]] = {
    "short": {
        "text": "こんにちは、私はAIです。これは音声合成のテストです。",
        "caption": None,
    },
    "medium": {
        "text": (
            "今日は朝から雨が降っていましたが、午後になると雲の切れ間から日差しが差し込み、"
            "公園の木々がきらきらと輝いて見えました。"
        ),
        "caption": None,
    },
    "long": {
        "text": (
            "音声合成の速度を改善するためには、まず現状の処理時間を正確に測定し、"
            "どの段階に時間がかかっているのかを把握することが重要です。"
            "その上で、無駄な計算や同期を取り除き、GPUを効率的に使う工夫を一つずつ積み重ねていきます。"
            "最終的には、品質を保ったまま待ち時間を短くすることが目標です。"
        ),
        "caption": None,
    },
    "caption_noref": {
        "text": "こんにちは、私はAIです。これは音声合成のテストです。",
        "caption": "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。",
    },
}


class MpsMemSampler:
    """Sample torch.mps driver/current allocation in a background thread (MPS has no
    peak counters, so the peak is the max over samples)."""

    def __init__(self, interval_ms: int = 20) -> None:
        self.interval_ms = int(interval_ms)
        self.samples: list[tuple[float, int, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def _loop() -> None:
            while not self._stop.is_set():
                self.samples.append(
                    (
                        time.perf_counter(),
                        int(torch.mps.current_allocated_memory()),
                        int(torch.mps.driver_allocated_memory()),
                    )
                )
                self._stop.wait(self.interval_ms / 1000.0)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self, t_start: float, t_end: float) -> dict[str, float] | None:
        window = [(a, d) for (t, a, d) in self.samples if t_start <= t <= t_end]
        if not window:
            return None
        return {
            "max_current_allocated_mib": max(a for a, _ in window) / 2**20,
            "max_driver_allocated_mib": max(d for _, d in window) / 2**20,
            "n": float(len(window)),
        }


def _audio_hash(audio: torch.Tensor) -> str:
    return hashlib.sha256(audio.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[idx])


def _mps_mem() -> dict[str, int]:
    return {
        "allocated": int(torch.mps.current_allocated_memory()),
        "driver": int(torch.mps.driver_allocated_memory()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--codec-precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--inputs", nargs="+", default=["short", "medium", "long"])
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--cfg-guidance-mode", default="independent")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-wav-dir", default=None)
    parser.add_argument("--no-util", action="store_true", help="Skip MPS memory sampling.")
    parser.add_argument(
        "--cooldown", type=float, default=0.0, help="Seconds to sleep between inputs (thermal)."
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="KEY=VALUE environment overrides applied before runtime load (recorded in JSON).",
    )
    args = parser.parse_args()

    for item in args.env:
        k, _, v = item.partition("=")
        os.environ[k] = v

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        checkpoint = download_hf_checkpoint(args.hf_checkpoint)

    t_load0 = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint,
            model_device="mps",
            model_precision=args.precision,
            codec_device="mps",
            codec_precision=args.codec_precision,
            compile_model=bool(args.compile),
            compile_dynamic=bool(args.compile_dynamic),
        )
    )
    torch.mps.synchronize()
    load_sec = time.perf_counter() - t_load0
    mem_after_load = _mps_mem()

    def make_request(name: str) -> SamplingRequest:
        spec = INPUTS[name]
        caption = spec["caption"]
        no_ref = name.endswith("noref")
        return SamplingRequest(
            text=str(spec["text"]),
            caption=caption,
            ref_wav=None if no_ref else args.ref,
            no_ref=no_ref,
            num_steps=int(args.num_steps),
            cfg_guidance_mode=str(args.cfg_guidance_mode),
            seed=int(args.seed),
        )

    results: dict[str, object] = {}
    sampler = None if args.no_util else MpsMemSampler()
    if sampler is not None:
        sampler.start()

    # Warmup across all inputs first so that compile/graph capture is excluded.
    t_warm0 = time.perf_counter()
    for name in args.inputs:
        for _ in range(int(args.warmup)):
            runtime.synthesize(make_request(name))
    torch.mps.synchronize()
    warm_sec = time.perf_counter() - t_warm0
    mem_after_warm = _mps_mem()

    for name in args.inputs:
        if args.cooldown > 0:
            time.sleep(float(args.cooldown))
        req = make_request(name)
        walls: list[float] = []
        stages: dict[str, list[float]] = {}
        hashes: set[str] = set()
        audio_seconds = 0.0
        t_start = time.perf_counter()
        for _ in range(int(args.repeats)):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            result = runtime.synthesize(req)
            torch.mps.synchronize()
            walls.append(time.perf_counter() - t0)
            for sname, sec in result.stage_timings:
                stages.setdefault(sname, []).append(float(sec))
            hashes.add(_audio_hash(result.audio))
            audio_seconds = float(result.audio.shape[-1]) / float(result.sample_rate)
        t_end = time.perf_counter()
        mem_now = _mps_mem()
        mem_window = sampler.summary(t_start, t_end) if sampler is not None else None
        if args.save_wav_dir:
            from irodori_tts.inference_runtime import save_wav

            out_dir = Path(args.save_wav_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_wav(out_dir / f"{args.tag}_{name}.wav", result.audio, result.sample_rate)
        med = statistics.median(walls)
        results[name] = {
            "audio_seconds": audio_seconds,
            "wall_median": med,
            "wall_p95": _percentile(walls, 0.95),
            "wall_min": min(walls),
            "rtf_median": med / audio_seconds if audio_seconds > 0 else None,
            "stages_median_ms": {k: statistics.median(v) * 1000.0 for k, v in stages.items()},
            "audio_hashes": sorted(hashes),
            "deterministic": len(hashes) == 1,
            "mps_mem_window": mem_window,
            "mps_mem_after": mem_now,
            "messages": list(result.messages),
        }
        peak = (mem_window or {}).get("max_current_allocated_mib", mem_now["allocated"] / 2**20)
        print(
            f"[{args.tag}] {name}: audio={audio_seconds:.2f}s wall_med={med*1000:.0f}ms "
            f"p95={results[name]['wall_p95']*1000:.0f}ms rtf={results[name]['rtf_median']:.3f} "
            f"peak_alloc={peak:.0f}MiB det={len(hashes)==1}",
            flush=True,
        )
        print("   stages(ms): " + ", ".join(f"{k}={v:.1f}" for k, v in results[name]["stages_median_ms"].items()), flush=True)

    if sampler is not None:
        sampler.stop()

    record = {
        "tag": args.tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip(),
        "env": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "mps",
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "mps_recommended_max_memory": int(torch.mps.recommended_max_memory()),
            "env_overrides": list(args.env),
        },
        "config": {
            "checkpoint": checkpoint,
            "precision": args.precision,
            "codec_precision": args.codec_precision,
            "compile": bool(args.compile),
            "compile_dynamic": bool(args.compile_dynamic),
            "num_steps": args.num_steps,
            "cfg_guidance_mode": args.cfg_guidance_mode,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "ref": args.ref,
        },
        "load_sec": load_sec,
        "warmup_sec_total": warm_sec,
        "mps_mem_after_load": mem_after_load,
        "mps_mem_after_warmup": mem_after_warm,
        "results": results,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[bench] wrote {out}")
    print(
        f"[bench] load={load_sec:.2f}s after_load alloc={mem_after_load['allocated']/2**20:.0f}MiB "
        f"driver={mem_after_load['driver']/2**20:.0f}MiB "
        f"after_warm alloc={mem_after_warm['allocated']/2**20:.0f}MiB "
        f"driver={mem_after_warm['driver']/2**20:.0f}MiB"
    )


if __name__ == "__main__":
    main()
