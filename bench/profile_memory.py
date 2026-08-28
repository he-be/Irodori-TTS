#!/usr/bin/env python3
"""Per-stage memory profile for one synthesize() call per input (MPS).

MPS has no peak counter, so each wrapped stage is bracketed by a sampling thread
and its peak is the max over samples (20 ms) of ``current_allocated_memory``."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.bench_runtime import DEFAULT_REF, INPUTS  # noqa: E402
from irodori_tts import inference_runtime as ir  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)

MiB = 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--codec-precision", default="fp32")
    parser.add_argument("--inputs", nargs="+", default=["short", "long"])
    args = parser.parse_args()

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
            model_device="mps",
            model_precision=args.precision,
            codec_device="mps",
            codec_precision=args.codec_precision,
        )
    )
    print(
        f"[load] alloc={torch.mps.current_allocated_memory()/MiB:.0f}MiB "
        f"driver={torch.mps.driver_allocated_memory()/MiB:.0f}MiB"
    )

    stage_peaks: dict[str, float] = {}

    class _Peak:
        def __init__(self) -> None:
            self.peak = 0
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._loop, daemon=True)

        def _loop(self) -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, torch.mps.current_allocated_memory())
                self._stop.wait(0.02)

        def __enter__(self):
            self._t.start()
            return self

        def __exit__(self, *exc):
            self._stop.set()
            self._t.join(timeout=1)
            self.peak = max(self.peak, torch.mps.current_allocated_memory())

    def wrap(obj, name, label):
        fn = getattr(obj, name)

        def inner(*a, **k):
            torch.mps.synchronize()
            base = torch.mps.current_allocated_memory()
            with _Peak() as pk:
                out = fn(*a, **k)
                torch.mps.synchronize()
            stage_peaks[label] = max(stage_peaks.get(label, 0.0), (pk.peak - base) / MiB)
            return out

        setattr(obj, name, inner)

    wrap(ir, "sample_euler_rf_cfg", "sample_rf(transient)")
    wrap(runtime.codec, "decode_latent", "decode_latent(transient)")
    wrap(runtime.codec, "encode_waveform", "encode_reference(transient)")
    wrap(runtime.model, "encode_conditions", "encode_conditions(transient)")

    for name in args.inputs:
        spec = INPUTS[name]
        no_ref = name.endswith("noref")
        req = SamplingRequest(
            text=str(spec["text"]),
            caption=spec["caption"],
            ref_wav=None if no_ref else DEFAULT_REF,
            no_ref=no_ref,
            seed=1234,
        )
        runtime.synthesize(req)  # warm (Metal pipeline compilation etc.)
        stage_peaks.clear()
        torch.mps.synchronize()
        base = torch.mps.current_allocated_memory()
        with _Peak() as pk:
            result = runtime.synthesize(req)
            torch.mps.synchronize()
        print(
            f"[{name}] audio={result.audio.shape[-1]/result.sample_rate:.2f}s "
            f"resident={base/MiB:.0f}MiB request_peak_alloc={pk.peak/MiB:.0f}MiB "
            f"driver={torch.mps.driver_allocated_memory()/MiB:.0f}MiB"
        )
        for k, v in stage_peaks.items():
            print(f"    {k}: +{v:.0f}MiB")


if __name__ == "__main__":
    main()
