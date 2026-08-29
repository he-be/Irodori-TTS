"""Build (export + compile) and warm the Neural Engine packages for the RF step (13-ane.md).

    uv run python bench/build_ane.py --shapes full [--units ne]

Exports every package of the shape set into ~/.cache/irodori-tts/ane (or
IRODORI_OPT_ANE_CACHE_DIR), then loads each one in the worker once so the OS caches the
ANE-compiled program; later processes load in well under a second.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irodori_tts.ane_dit import AneStepRunner  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    download_hf_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--shapes", default="full", choices=["dev", "full"])
    parser.add_argument("--units", default="ne", choices=["ne", "all", "gpu", "cpu"])
    args = parser.parse_args()
    ck = download_hf_checkpoint(args.hf_checkpoint)
    rt = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=ck,
            model_device="mps",
            model_precision="fp16",
            codec_device="mps",
            codec_precision="fp32",
            compile_model=False,
            compile_dynamic=False,
        )
    )
    t0 = time.perf_counter()
    runner = AneStepRunner(rt.model, shapes_name=args.shapes, compute_units=args.units)
    runner.preload()
    print(f"[ane] built and warmed {len(runner.packages)} packages in {time.perf_counter() - t0:.0f} s")
    runner.shutdown()


if __name__ == "__main__":
    main()
