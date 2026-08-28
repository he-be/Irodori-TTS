#!/usr/bin/env python3
"""Compare legacy vs optimized inference paths on the same runtime instance.

Reports audio hash equality and max abs diff for several inputs / CFG modes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.bench_runtime import DEFAULT_REF, INPUTS  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)
from irodori_tts.opt_config import OptConfig, set_opt_config  # noqa: E402

LEGACY = OptConfig(
    reuse_conditions=False,
    crop_text=False,
    fast_sampler=False,
    codec_fold_weight_norm=False,
    reference_cache=False,
    cpu_cast=False,
    rope_real=False,
)


def _hash(audio: torch.Tensor) -> str:
    return hashlib.sha256(audio.float().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--inputs", nargs="+", default=["short", "caption_noref"])
    parser.add_argument("--modes", nargs="+", default=["independent", "joint", "alternating"])
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--speaker-kv-scale", type=float, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["fast", "rope_complex"],
        help="Which optimized variants to compare against legacy.",
    )
    args = parser.parse_args()

    variants: dict[str, OptConfig] = {
        "fast": OptConfig.from_env(),
        "rope_complex": OptConfig(rope_real=False),
        "crop_only": OptConfig(fast_sampler=False, reuse_conditions=False),
        "reuse_only": OptConfig(fast_sampler=False, crop_text=False),
        "sampler_only": OptConfig(crop_text=False, reuse_conditions=False),
    }

    set_opt_config(OptConfig.from_env())  # honors IRODORI_OPT_* (e.g. COMPILE_DIT=1)
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
            model_device="mps",
            model_precision=args.precision,
            codec_device="mps",
            codec_precision="fp32",
        )
    )

    compiled_forward = runtime.model.forward_with_encoded_conditions
    eager_forward = getattr(runtime, "_eager_forward_with_encoded_conditions", compiled_forward)

    def run(name: str, mode: str, cfg: OptConfig) -> torch.Tensor:
        set_opt_config(cfg)
        # RoPE tables are cached per module; drop them so the variant's layout is rebuilt.
        for mod in runtime.model.modules():
            if "_freqs_cis_cache" in mod._buffers:
                mod._buffers["_freqs_cis_cache"] = torch.empty(0, 0, device=runtime.model_device)
        # Legacy reference always uses the uncompiled forward.
        runtime.model.forward_with_encoded_conditions = (
            eager_forward if cfg is LEGACY else compiled_forward
        )
        spec = INPUTS[name]
        no_ref = name.endswith("noref")
        scales = {}
        if mode == "joint":
            scales = {"cfg_scale": 3.0}
        req = SamplingRequest(
            text=str(spec["text"]),
            caption=spec["caption"],
            ref_wav=None if no_ref else DEFAULT_REF,
            no_ref=no_ref,
            num_steps=args.num_steps,
            cfg_guidance_mode=mode,
            seed=1234,
            speaker_kv_scale=args.speaker_kv_scale,
            **scales,
        )
        return runtime.synthesize(req).audio

    all_ok = True
    for name in args.inputs:
        for mode in args.modes:
            ref = run(name, mode, LEGACY)
            h_ref = _hash(ref)
            for vname in args.variants:
                cfg = variants[vname]
                # run twice: the second run has every Metal pipeline cached
                t0 = time.perf_counter()
                out = run(name, mode, cfg)
                t1 = time.perf_counter()
                out2 = run(name, mode, cfg)
                t2 = time.perf_counter()
                n = min(ref.shape[-1], out.shape[-1])
                diff = (ref[..., :n].float() - out[..., :n].float()).abs().max().item()
                diff2 = (out[..., :n].float() - out2[..., :n].float()).abs().max().item()
                same_len = ref.shape[-1] == out.shape[-1]
                ok = _hash(out) == h_ref
                status = "HASH_EQ" if ok else f"maxdiff={diff:.3e}"
                if not ok and (diff > 5e-3 or not same_len):
                    all_ok = False
                print(
                    f"[{name}/{mode}/{vname}] {status} len_eq={same_len} "
                    f"repeat_maxdiff={diff2:.1e} t1={1000*(t1-t0):.0f}ms t2={1000*(t2-t1):.0f}ms"
                )
    print("RESULT:", "OK" if all_ok else "MISMATCH")


if __name__ == "__main__":
    main()
