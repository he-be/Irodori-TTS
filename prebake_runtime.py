#!/usr/bin/env python3
"""Precompute a ready-to-load runtime bundle (weights already cast, codec already folded).

A cold load spends most of its time redoing work that only depends on the
checkpoint: building the module, copying 714 FP32 tensors in, casting them to
the runtime dtype, unpickling the codec and folding its weight_norm hooks.  This
tool does that once and writes the finished tensors next to a manifest; later
loads mmap them straight onto the MPS device.  See docs/experiments/11-load-time.md.

    uv run --no-sync python prebake_runtime.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small

The runtime finds the bundle on its own; ``IRODORI_OPT_PREBAKE=0`` ignores it and
``IRODORI_OPT_PREBAKE_DIR`` moves the cache (default ~/.cache/irodori-tts/prebake).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from irodori_tts import prebake as prebake_mod  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    RuntimeKey,
    build_prebake_bundle,
    default_runtime_device,
    download_hf_checkpoint,
)


def _dir_size_mib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", default=None, help="Local model.safetensors path.")
    src.add_argument(
        "--hf-checkpoint", default=None, help="Hugging Face repo id (or repo/subfolder)."
    )
    ap.add_argument("--model-device", default=default_runtime_device())
    ap.add_argument("--model-precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--codec-device", default=None, help="Defaults to --model-device.")
    ap.add_argument("--codec-precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    ap.add_argument(
        "--root", default=None, help="Bundle cache root (default: ~/.cache/irodori-tts/prebake)."
    )
    ap.add_argument("--list", action="store_true", help="List existing bundles and exit.")
    ap.add_argument(
        "--prune", action="store_true", help="Delete every bundle under the root and exit."
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser() if args.root else prebake_mod.default_root()

    if args.list or args.prune:
        if not root.is_dir():
            print(f"no bundles under {root}")
            return
        for entry in sorted(root.iterdir()):
            manifest = entry / prebake_mod.MANIFEST_NAME
            if not manifest.is_file():
                continue
            ident = json.loads(manifest.read_text()).get("identity", {})
            print(
                f"{entry.name}  {_dir_size_mib(entry):7.0f} MiB  "
                f"{ident.get('model_precision')}/{ident.get('codec_precision')}  "
                f"{ident.get('checkpoint', {}).get('path')}"
            )
        if args.prune:
            shutil.rmtree(root)
            print(f"removed {root}")
        return

    if not args.checkpoint and not args.hf_checkpoint:
        ap.error("one of --checkpoint / --hf-checkpoint is required (or use --list / --prune)")
    checkpoint = args.checkpoint or download_hf_checkpoint(args.hf_checkpoint)
    key = RuntimeKey(
        checkpoint=str(checkpoint),
        model_device=args.model_device,
        codec_repo=args.codec_repo,
        model_precision=args.model_precision,
        codec_device=args.codec_device or args.model_device,
        codec_precision=args.codec_precision,
    )
    directory = build_prebake_bundle(key, root=root)
    print(f"prebake bundle: {directory}  ({_dir_size_mib(directory):.0f} MiB)")


if __name__ == "__main__":
    main()
