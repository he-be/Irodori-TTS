#!/usr/bin/env python3
"""TTS side of the VLM/TTS co-existence stress (experiment 10).

Runs as a subprocess so that "load" / "unload" really means a CUDA context
appearing and disappearing on the GPU.  Speaks line-delimited JSON on
stdin/stdout:

    {"cmd": "load"}                      -> {"event": "loaded", ...}
    {"cmd": "synth", "text": "...", "id": 3}
                                         -> {"event": "synth", "audio_seconds": ..}
    {"cmd": "stats"}                     -> {"event": "stats", ...}
    {"cmd": "exit"}                      -> process exits

Every reply carries "ok"; on failure it carries "error" plus "oom" so that the
orchestrator can tell a VRAM failure from anything else.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REF = str(REPO_ROOT / "outputs" / "sample.wav")


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    import torch  # imported after env is inherited

    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        download_hf_checkpoint,
    )

    device = torch.device("cuda")
    runtime = None
    ref = os.environ.get("COEXIST_REF", DEFAULT_REF)
    precision = os.environ.get("COEXIST_PRECISION", "bf16")
    checkpoint_src = os.environ.get("COEXIST_CHECKPOINT", "Aratako/Irodori-TTS-v4.1-Small")

    emit({"event": "ready", "pid": os.getpid()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = msg.get("cmd")
        t0 = time.perf_counter()
        try:
            if cmd == "exit":
                break

            if cmd == "load":
                torch.cuda.reset_peak_memory_stats(device)
                checkpoint = download_hf_checkpoint(checkpoint_src)
                runtime = InferenceRuntime.from_key(
                    RuntimeKey(
                        checkpoint=checkpoint,
                        model_device="cuda",
                        model_precision=precision,
                        codec_device="cuda",
                        codec_precision="fp32",
                    )
                )
                torch.cuda.synchronize(device)
                emit({
                    "event": "loaded", "ok": True, "wall": time.perf_counter() - t0,
                    "alloc_mib": torch.cuda.memory_allocated(device) / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
                })
                continue

            if cmd == "unload":
                runtime = None
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                emit({"event": "unloaded", "ok": True, "wall": time.perf_counter() - t0,
                      "reserved_mib": torch.cuda.memory_reserved(device) / 2**20})
                continue

            if cmd == "synth":
                if runtime is None:
                    emit({"event": "synth", "ok": False, "error": "not loaded", "id": msg.get("id")})
                    continue
                req = SamplingRequest(
                    text=str(msg["text"]),
                    caption=msg.get("caption"),
                    ref_wav=None if msg.get("no_ref") else msg.get("ref", ref),
                    no_ref=bool(msg.get("no_ref", False)),
                    num_steps=int(msg.get("num_steps", 40)),
                    cfg_guidance_mode="independent",
                    seed=int(msg.get("seed", 1234)),
                )
                torch.cuda.synchronize(device)
                ts = time.perf_counter()
                result = runtime.synthesize(req)
                torch.cuda.synchronize(device)
                wall = time.perf_counter() - ts
                audio_seconds = float(result.audio.shape[-1]) / float(result.sample_rate)
                emit({
                    "event": "synth", "ok": True, "id": msg.get("id"), "wall": wall,
                    "audio_seconds": audio_seconds, "rtf": wall / audio_seconds,
                    "stages_ms": {k: v * 1000.0 for k, v in result.stage_timings},
                    "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                })
                continue

            if cmd == "reset_peak":
                torch.cuda.reset_peak_memory_stats(device)
                emit({"event": "reset_peak", "ok": True})
                continue

            if cmd == "stats":
                emit({
                    "event": "stats", "ok": True,
                    "alloc_mib": torch.cuda.memory_allocated(device) / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
                    "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                })
                continue

            emit({"event": "error", "ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:  # noqa: BLE001 - the point is to survive anything
            text = f"{type(exc).__name__}: {exc}"
            emit({
                "event": msg.get("cmd", "error"), "ok": False, "id": msg.get("id"),
                "error": text[:600],
                "oom": isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text.lower(),
                "wall": time.perf_counter() - t0,
                "traceback": traceback.format_exc()[-800:],
            })

    emit({"event": "exiting", "ok": True})


if __name__ == "__main__":
    main()
