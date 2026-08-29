"""Step-level check of the Neural Engine RF step (13-ane.md).

Captures the real inputs of one CFG forward (batch 3) and one plain forward (batch 1) from a
bench request, then compares three implementations on identical inputs:

  * MPS fp16 (the production path, context KV cache + combined mask)
  * ANE (Core ML, fp16-only, norms on x/64)
  * CPU fp32 reference (legacy path: KV projected inside, boolean masks)

and prints max |diff| / relative error, plus per-call timings.

    uv run python bench/check_ane.py --input short [--shapes dev] [--units ne]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_runtime import DEFAULT_REF, INPUTS  # noqa: E402

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="short", choices=sorted(INPUTS))
    parser.add_argument("--shapes", default="dev", choices=["dev", "full"])
    parser.add_argument("--units", default="ne", choices=["ne", "all", "gpu", "cpu"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--skip-cpu", action="store_true", help="skip the CPU fp32 reference")
    args = parser.parse_args()
    os.environ.setdefault("IRODORI_OPT_ANE", "0")  # capture from the plain MPS path

    ck = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
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
    model = rt.model

    captured: dict[int, dict] = {}
    orig = model.forward_with_encoded_conditions

    def capture(**kw):
        bsz = int(kw["x_t"].shape[0])
        if bsz not in captured:
            captured[bsz] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
            captured[bsz]["context_kv_cache"] = [
                tuple(t.clone() for t in layer) for layer in kw["context_kv_cache"]
            ]
        return orig(**kw)

    model.forward_with_encoded_conditions = capture
    spec = INPUTS[args.input]
    no_ref = args.input.endswith("noref")
    rt.synthesize(
        SamplingRequest(
            text=str(spec["text"]),
            caption=spec["caption"],
            ref_wav=None if no_ref else DEFAULT_REF,
            no_ref=no_ref,
            num_steps=4,
            cfg_guidance_mode="independent",
            seed=1234,
        )
    )
    model.forward_with_encoded_conditions = orig
    print(f"captured batches: {sorted(captured)}")

    from irodori_tts.ane_dit import AneStepRunner

    runner = AneStepRunner(model, shapes_name=args.shapes, compute_units=args.units)

    cpu_model = None
    if not args.skip_cpu:
        t0 = time.perf_counter()
        cpu_model = copy.deepcopy(model).to("cpu", torch.float32).eval()
        print(f"cpu fp32 copy: {time.perf_counter() - t0:.1f} s")

    def timeit(fn, n):
        fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.mps.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    for bsz in sorted(captured):
        kw = captured[bsz]
        x_t, t = kw["x_t"], kw["t"]
        latent_len = int(x_t.shape[1])
        with torch.inference_mode():
            v_mps = orig(**kw).float().cpu()
            t_mps = timeit(lambda kw=kw: orig(**kw), args.repeats)

            ctx = runner.make_context(
                latent_len=latent_len,
                text_state=kw["text_state"],
                text_mask=kw["text_mask"],
                speaker_state=kw["speaker_state"],
                speaker_mask=kw["speaker_mask"],
                caption_state=kw["caption_state"],
                caption_mask=kw["caption_mask"],
            )
            if ctx is None:
                print(f"B={bsz}: no ANE shape fits (S={latent_len}); skipping")
                continue
            t_value = float(t[0].float())
            v_ane = runner.step(ctx, x_t, t_value).float()
            t_ane = timeit(
                lambda ctx=ctx, x_t=x_t, t_value=t_value: runner.step(ctx, x_t, t_value),
                args.repeats,
            )
            stats = runner.reset_stats()

            ref = None
            if cpu_model is not None:
                ckw = {k: (v.to("cpu", torch.float32) if torch.is_tensor(v) else v) for k, v in kw.items()}
                ckw["context_kv_cache"] = None
                ckw["attn_mask"] = None
                t0 = time.perf_counter()
                ref = cpu_model.forward_with_encoded_conditions(**ckw).float()
                t_cpu = time.perf_counter() - t0

        def report(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
            d = (a - b).abs()
            return f"{name}: max|d| {d.max():.3e} mean|d| {d.mean():.3e} rel {d.norm() / b.norm():.3e}"

        print(
            f"B={bsz} S={latent_len} bucket={ctx.shape.latent}/{ctx.shape.ctx}: "
            f"MPS fp16 {t_mps:.1f} ms | ANE {t_ane:.1f} ms "
            f"(predict {stats['predict_sec'] / max(1, stats['steps']) * 1000:.1f} ms, "
            f"wait {stats['wait_sec'] / max(1, stats['steps']) * 1000:.1f} ms)"
            + (f" | CPU fp32 {t_cpu * 1000:.0f} ms" if ref is not None else "")
        )
        print("  " + report("ANE vs MPS16", v_ane, v_mps))
        if ref is not None:
            print("  " + report("ANE vs CPU32", v_ane, ref))
            print("  " + report("MPS16 vs CPU32", v_mps, ref))
        print(f"  |v| ref scale: max {v_mps.abs().max():.3f} mean {v_mps.abs().mean():.3f}")
        runner.drop_context(ctx)

    runner.shutdown()


if __name__ == "__main__":
    main()
