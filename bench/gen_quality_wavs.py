#!/usr/bin/env python3
"""Generate WAV sets for quality comparison (no watermark in all sets).

Sets (each: short / medium / long / caption_noref):
  fp32_legacy_s1234 : FP32, legacy path, seed 1234   (reference)
  fp32_legacy_s4321 : FP32, legacy path, seed 4321   (what a *different sample* looks like)
  bf16_opt_s1234    : BF16 model, all optimizations, codec FP32
  bf16_opt_codecbf16_s1234 : same + codec BF16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.bench_runtime import DEFAULT_REF, INPUTS  # noqa: E402
from bench.check_equivalence import LEGACY  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    save_wav,
)
from irodori_tts.opt_config import OptConfig, set_opt_config  # noqa: E402


def run_set(out_dir: Path, name: str, precision: str, codec_precision: str, cfg: OptConfig, seed: int) -> None:
    set_opt_config(cfg)
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
            model_device="cuda",
            model_precision=precision,
            codec_device="cuda",
            codec_precision=codec_precision,
        )
    )
    for input_name, spec in INPUTS.items():
        no_ref = input_name.endswith("noref")
        req = SamplingRequest(
            text=str(spec["text"]),
            caption=spec["caption"],
            ref_wav=None if no_ref else DEFAULT_REF,
            no_ref=no_ref,
            seed=seed,
        )
        result = runtime.synthesize(req)
        path = out_dir / name / f"{input_name}.wav"
        save_wav(path, result.audio, result.sample_rate)
        print(f"[{name}] {input_name}: {result.audio.shape[-1]/result.sample_rate:.2f}s -> {path}")
    runtime.unload()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/quality")
    parser.add_argument("--sets", nargs="+", default=["fp32_legacy_s1234", "fp32_legacy_s4321", "bf16_opt_s1234", "bf16_opt_codecbf16_s1234"])
    args = parser.parse_args()
    out = Path(args.out)
    # FP32 reference sets do not fit under the local VRAM cap; disable it for this script.
    legacy_nowm = OptConfig(**{**LEGACY.__dict__, "watermark": False, "vram_limit_mb": 0})
    opt = OptConfig(**{**OptConfig.from_env().__dict__, "vram_limit_mb": 0})
    specs = {
        "fp32_legacy_s1234": ("fp32", "fp32", legacy_nowm, 1234),
        "fp32_legacy_s4321": ("fp32", "fp32", legacy_nowm, 4321),
        "bf16_opt_s1234": ("bf16", "fp32", opt, 1234),
        "bf16_opt_codecbf16_s1234": ("bf16", "bf16", opt, 1234),
    }
    for name in args.sets:
        run_set(out, name, *specs[name])


if __name__ == "__main__":
    main()
