#!/usr/bin/env python3
"""Per-stage peak VRAM profile for one synthesize() call per input."""

from __future__ import annotations

import argparse
import subprocess
import sys
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
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--codec-precision", default="fp32")
    parser.add_argument("--inputs", nargs="+", default=["short", "long"])
    args = parser.parse_args()

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
            model_device="cuda",
            model_precision=args.precision,
            codec_device="cuda",
            codec_precision=args.codec_precision,
        )
    )
    print(
        f"[load] peak_alloc={torch.cuda.max_memory_allocated(dev)/MiB:.0f}MiB "
        f"alloc={torch.cuda.memory_allocated(dev)/MiB:.0f}MiB reserved={torch.cuda.memory_reserved(dev)/MiB:.0f}MiB"
    )

    stage_peaks: dict[str, float] = {}

    def wrap(obj, name, label):
        fn = getattr(obj, name)

        def inner(*a, **k):
            torch.cuda.synchronize(dev)
            base = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            out = fn(*a, **k)
            torch.cuda.synchronize(dev)
            stage_peaks[label] = max(
                stage_peaks.get(label, 0.0), (torch.cuda.max_memory_allocated(dev) - base) / MiB
            )
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
        runtime.synthesize(req)  # warm (graph capture etc.)
        stage_peaks.clear()
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
        result = runtime.synthesize(req)
        torch.cuda.synchronize(dev)
        print(
            f"[{name}] audio={result.audio.shape[-1]/result.sample_rate:.2f}s "
            f"resident={base/MiB:.0f}MiB request_peak_alloc={torch.cuda.max_memory_allocated(dev)/MiB:.0f}MiB "
            f"reserved={torch.cuda.memory_reserved(dev)/MiB:.0f}MiB"
        )
        for k, v in stage_peaks.items():
            print(f"    {k}: +{v:.0f}MiB")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"[smi] memory.used now={smi}MiB (after all requests; includes CUDA context)")
    if runtime._graph_runner is not None:
        st = runtime._graph_runner.stats()
        print(f"[graph] entries={st['entries']} const_sets={st['const_sets']} static={st['static_bytes']/MiB:.0f}MiB")


if __name__ == "__main__":
    main()
