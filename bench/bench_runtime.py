#!/usr/bin/env python3
"""Warm benchmark harness for Irodori-TTS inference on a single machine.

Loads the runtime once, runs warmup + timed repeats for a set of representative
inputs, and writes a JSON record (stage timings, wall median/p95, RTF, audio
hashes, CUDA memory, GPU utilization sampled via nvidia-smi).

Example:
  uv run --no-sync python bench/bench_runtime.py --precision bf16 --tag bf16 \
      --output docs/experiments/results/02_bf16.json
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


class GpuUtilSampler:
    """Sample nvidia-smi utilization in a background thread."""

    def __init__(self, interval_ms: int = 50) -> None:
        self.interval_ms = int(interval_ms)
        self.samples: list[tuple[float, int, int]] = []
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    f"-lms",
                    str(self.interval_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            self._proc = None
            return

        def _reader() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) != 2:
                    continue
                try:
                    self.samples.append((time.perf_counter(), int(parts[0]), int(parts[1])))
                except ValueError:
                    continue

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self, t_start: float, t_end: float) -> dict[str, float] | None:
        window = [u for (t, u, _m) in self.samples if t_start <= t <= t_end]
        mem = [m for (t, _u, m) in self.samples if t_start <= t <= t_end]
        if not window:
            return None
        return {
            "smi_mem_used_max_mib": float(max(mem)) if mem else None,
            "mean": float(statistics.mean(window)),
            "median": float(statistics.median(window)),
            "p10": float(sorted(window)[int(0.1 * (len(window) - 1))]),
            "max": float(max(window)),
            "n": float(len(window)),
        }


class AmdSysfsSampler(GpuUtilSampler):
    """Sample amdgpu utilization / VRAM from sysfs (ROCm has no nvidia-smi)."""

    def __init__(self, card: str, interval_ms: int = 50) -> None:
        super().__init__(interval_ms)
        self.card = card
        self._stop = threading.Event()

    def start(self) -> None:
        busy = Path(f"/sys/class/drm/{self.card}/device/gpu_busy_percent")
        vram = Path(f"/sys/class/drm/{self.card}/device/mem_info_vram_used")
        if not busy.exists():
            return

        def _reader() -> None:
            while not self._stop.is_set():
                try:
                    u = int(busy.read_text().strip())
                    m = int(vram.read_text().strip()) // (1024 * 1024)
                    self.samples.append((time.perf_counter(), u, m))
                except (OSError, ValueError):
                    pass
                self._stop.wait(self.interval_ms / 1000.0)

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _amd_card_for_torch_device(device: torch.device) -> str | None:
    """Best-effort: find the drm card whose driver is amdgpu (single-iGPU machine)."""
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        drv = card / "device" / "driver"
        try:
            if drv.resolve().name == "amdgpu":
                return card.name
        except OSError:
            continue
    return None


def _make_sampler(device: torch.device) -> GpuUtilSampler | None:
    if device.type != "cuda":
        return None
    if getattr(torch.version, "hip", None):
        card = _amd_card_for_torch_device(device)
        return AmdSysfsSampler(card) if card else None
    return GpuUtilSampler()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _audio_hash(audio: torch.Tensor) -> str:
    return hashlib.sha256(audio.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[idx])


def _cuda_mem(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda", help="Model device (cuda, cpu). ROCm torch also reports cuda.")
    parser.add_argument("--codec-device", default=None, help="Codec device (default: same as --device).")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--threads", type=int, default=0, help="torch.set_num_threads for CPU runs (0 = default).")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="torch.backends.cudnn.benchmark=True (MIOpen: exhaustive find).")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--inputs", nargs="+", default=["short", "medium", "long"])
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--t-schedule-mode", choices=["linear", "sway"], default="linear")
    parser.add_argument("--sway-coeff", type=float, default=-1.0)
    parser.add_argument("--cfg-guidance-mode", default="independent")
    parser.add_argument("--cfg-scale", type=float, default=None, help="Single scale for all conditions (needed for joint).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-wav-dir", default=None)
    parser.add_argument("--no-util", action="store_true", help="Skip nvidia-smi sampling.")
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

    device = torch.device(args.device)
    codec_device_str = args.codec_device or args.device
    if args.threads > 0:
        torch.set_num_threads(int(args.threads))
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    _reset_peak(device)
    t_load0 = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint,
            model_device=args.device,
            model_precision=args.precision,
            codec_device=codec_device_str,
            codec_precision=args.codec_precision,
            compile_model=bool(args.compile),
            compile_dynamic=bool(args.compile_dynamic),
        )
    )
    _sync(device)
    load_sec = time.perf_counter() - t_load0
    mem_after_load = _cuda_mem(device)

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
            t_schedule_mode=str(args.t_schedule_mode),
            sway_coeff=float(args.sway_coeff),
            cfg_guidance_mode=str(args.cfg_guidance_mode),
            cfg_scale=args.cfg_scale,
            seed=int(args.seed),
        )

    results: dict[str, object] = {}
    sampler = None if args.no_util else _make_sampler(device)
    if sampler is not None:
        sampler.start()

    # Warmup across all inputs first so that compile/graph capture is excluded.
    t_warm0 = time.perf_counter()
    for name in args.inputs:
        for _ in range(int(args.warmup)):
            runtime.synthesize(make_request(name))
    _sync(device)
    warm_sec = time.perf_counter() - t_warm0
    mem_after_warm = _cuda_mem(device)

    for name in args.inputs:
        req = make_request(name)
        _reset_peak(device)
        walls: list[float] = []
        stages: dict[str, list[float]] = {}
        hashes: set[str] = set()
        audio_seconds = 0.0
        t_start = time.perf_counter()
        for _ in range(int(args.repeats)):
            _sync(device)
            t0 = time.perf_counter()
            result = runtime.synthesize(req)
            _sync(device)
            walls.append(time.perf_counter() - t0)
            for sname, sec in result.stage_timings:
                stages.setdefault(sname, []).append(float(sec))
            hashes.add(_audio_hash(result.audio))
            audio_seconds = float(result.audio.shape[-1]) / float(result.sample_rate)
        t_end = time.perf_counter()
        mem_peak = _cuda_mem(device)
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
            "gpu_util": sampler.summary(t_start, t_end) if sampler is not None else None,
            "cuda_mem_peak": mem_peak,
            "messages": list(result.messages),
        }
        print(
            f"[{args.tag}] {name}: audio={audio_seconds:.2f}s wall_med={med*1000:.0f}ms "
            f"p95={results[name]['wall_p95']*1000:.0f}ms rtf={results[name]['rtf_median']:.3f} "
            f"util={results[name]['gpu_util']} peak_alloc={mem_peak.get('max_allocated',0)/2**20:.0f}MiB "
            f"det={len(hashes)==1}",
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
            "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "total_memory": int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else None,
            "threads": torch.get_num_threads(),
            "env_overrides": list(args.env),
        },
        "config": {
            "checkpoint": checkpoint,
            "device": args.device,
            "codec_device": codec_device_str,
            "cudnn_benchmark": bool(args.cudnn_benchmark),
            "precision": args.precision,
            "codec_precision": args.codec_precision,
            "compile": bool(args.compile),
            "compile_dynamic": bool(args.compile_dynamic),
            "num_steps": args.num_steps,
            "t_schedule_mode": args.t_schedule_mode,
            "sway_coeff": args.sway_coeff,
            "cfg_guidance_mode": args.cfg_guidance_mode,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "ref": args.ref,
        },
        "load_sec": load_sec,
        "warmup_sec_total": warm_sec,
        "cuda_mem_after_load": mem_after_load,
        "cuda_mem_after_warmup": mem_after_warm,
        "results": results,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[bench] wrote {out}")
    print(
        f"[bench] load={load_sec:.2f}s after_load alloc={mem_after_load.get('allocated',0)/2**20:.0f}MiB "
        f"reserved={mem_after_load.get('reserved',0)/2**20:.0f}MiB "
        f"after_warm alloc={mem_after_warm.get('allocated',0)/2**20:.0f}MiB "
        f"reserved={mem_after_warm.get('reserved',0)/2**20:.0f}MiB"
    )


if __name__ == "__main__":
    main()
