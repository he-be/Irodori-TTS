#!/usr/bin/env python3
"""How close is one DiT step to the GPU's matmul peak? (19-m6-mini.md)

Counts the FLOPs of one ``forward_with_encoded_conditions`` (FlopCounterMode), times the eager
and compiled forward, then replays every distinct Linear shape and the attention call alone at
the same sizes, so the gap between "sum of the GEMMs alone" and "the whole step" is visible.

    uv run --no-sync python bench/probe_dit_ops.py [--batch 3 --latent 180] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, download_hf_checkpoint  # noqa: E402


def timeit(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--latent", type=int, default=180)
    ap.add_argument("--text-len", type=int, default=32)
    ap.add_argument("--speaker-len", type=int, default=64)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
            model_device="mps",
            model_precision="fp16",
            codec_device="mps",
            codec_precision="fp32",
        )
    )
    model = runtime.model
    dev, dt = runtime.model_device, runtime._model_dtype
    cfg = model.cfg
    b = args.batch
    text_state = torch.randn(b, args.text_len, cfg.text_dim, device=dev, dtype=dt)
    text_mask = torch.ones(b, args.text_len, dtype=torch.bool, device=dev)
    speaker_state = torch.randn(b, args.speaker_len, cfg.speaker_dim, device=dev, dtype=dt)
    speaker_mask = torch.ones(b, args.speaker_len, dtype=torch.bool, device=dev)
    caption_state = caption_mask = None
    if cfg.use_caption_condition:
        caption_state = torch.randn(b, args.text_len, cfg.caption_dim_resolved, device=dev, dtype=dt)
        caption_mask = torch.ones(b, args.text_len, dtype=torch.bool, device=dev)
    with torch.inference_mode():
        kv = model.build_context_kv_cache(text_state=text_state, speaker_state=speaker_state, caption_state=caption_state)
        mask = model.build_combined_attn_mask(
            latent_len=args.latent, text_mask=text_mask, speaker_mask=speaker_mask, caption_mask=caption_mask
        )
    kw = dict(
        x_t=torch.randn(b, args.latent, cfg.patched_latent_dim, device=dev, dtype=dt),
        t=torch.full((b,), 0.5, device=dev, dtype=dt),
        text_state=text_state, text_mask=text_mask, speaker_state=speaker_state, speaker_mask=speaker_mask,
        caption_state=caption_state, caption_mask=caption_mask, context_kv_cache=kv, attn_mask=mask,
    )
    fwd = model.forward_with_encoded_conditions
    out: dict = {"batch": b, "latent": args.latent}

    # shapes of every Linear and SDPA call in one step
    lin: dict[tuple, int] = defaultdict(int)
    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            def cap(mod, inp, _o):
                x = inp[0]
                lin[(x.numel() // x.shape[-1], mod.in_features, mod.out_features, mod.bias is not None)] += 1
            hooks.append(m.register_forward_hook(cap))
    sdpa_shapes: list[tuple] = []
    orig_sdpa = F.scaled_dot_product_attention

    def spy(q, k, v, *a, **k2):
        sdpa_shapes.append((tuple(q.shape), tuple(k.shape), k2.get("attn_mask") is not None))
        return orig_sdpa(q, k, v, *a, **k2)

    F.scaled_dot_product_attention = spy
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        fwd(**kw)
    F.scaled_dot_product_attention = orig_sdpa
    for h in hooks:
        h.remove()
    flops = fc.get_total_flops()
    out["gflop_step"] = flops / 1e9
    by_op = {str(k): v / 1e9 for k, v in fc.get_flop_counts().get("Global", {}).items()}
    out["gflop_by_op"] = by_op
    print(f"one step B={b} S={args.latent}: {flops/1e9:.1f} GFLOP  " + ", ".join(f"{k}={v:.1f}" for k, v in by_op.items()))

    with torch.inference_mode():
        out["eager_ms"] = timeit(lambda: fwd(**kw))
        cf = torch.compile(fwd, dynamic=True)
        out["compiled_ms"] = timeit(lambda: cf(**kw))
    for k in ("eager_ms", "compiled_ms"):
        print(f"{k}: {out[k]:.2f} ms = {flops / out[k] / 1e9:.2f} TFLOPS")

    # Upper bound for precomputing the AdaLN modulation (it depends on t only): replay the step
    # with every LowRankAdaLN returning its cached shift/scale/gate-derived output inputs.
    from irodori_tts import model as model_mod

    ada = [m for m in model.modules() if isinstance(m, model_mod.LowRankAdaLN)]
    cache = {}
    orig_fwd = model_mod.LowRankAdaLN.forward

    def record(self, x, cond_embed):
        shift, scale, gate = cond_embed.chunk(3, dim=-1)
        cache[id(self)] = (
            self.shift_up(self.shift_down(F.silu(shift))) + shift,
            self.scale_up(self.scale_down(F.silu(scale))) + scale,
            torch.tanh(self.gate_up(self.gate_down(F.silu(gate))) + gate),
        )
        return orig_fwd(self, x, cond_embed)

    def replay(self, x, cond_embed):
        shift, scale, gate = cache[id(self)]
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return (x * (1.0 + scale) + shift).to(x_dtype), gate

    with torch.inference_mode():
        model_mod.LowRankAdaLN.forward = record
        fwd(**kw)
        model_mod.LowRankAdaLN.forward = replay
        out["adaln_cached_eager_ms"] = timeit(lambda: fwd(**kw))
        cf2 = torch.compile(fwd, dynamic=True)
        out["adaln_cached_compiled_ms"] = timeit(lambda: cf2(**kw))
        model_mod.LowRankAdaLN.forward = orig_fwd
    print(f"AdaLN precomputed ({len(ada)} modules): eager {out['adaln_cached_eager_ms']:.2f} ms, "
          f"compiled {out['adaln_cached_compiled_ms']:.2f} ms")
    torch._dynamo.reset()

    rows = []
    total_alone = 0.0
    with torch.inference_mode():
        for (m_, k_, n_, bias), cnt in sorted(lin.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[0][2] * kv[1]):
            x = torch.randn(m_, k_, device=dev, dtype=dt)
            w = torch.randn(n_, k_, device=dev, dtype=dt)
            bb = torch.randn(n_, device=dev, dtype=dt) if bias else None
            ms = timeit(lambda: F.linear(x, w, bb))
            fl = 2 * m_ * k_ * n_
            total_alone += ms * cnt
            rows.append({"M": m_, "K": k_, "N": n_, "count": cnt, "ms": ms, "tflops": fl / ms / 1e9, "ms_total": ms * cnt})
            print(f"  linear M={m_:5d} K={k_:5d} N={n_:5d} x{cnt:3d}: {ms:6.3f} ms {fl/ms/1e9:5.2f} TF  (sum {ms*cnt:6.2f} ms)")
        out["linears"] = rows
        sd = {}
        for qs, ks, has_mask in sdpa_shapes:
            sd[(qs, ks, has_mask)] = sd.get((qs, ks, has_mask), 0) + 1
        out["sdpa"] = []
        for (qs, ks, has_mask), cnt in sd.items():
            q = torch.randn(*qs, device=dev, dtype=dt)
            k = torch.randn(*ks, device=dev, dtype=dt)
            v = torch.randn(*ks, device=dev, dtype=dt)
            am = mask if has_mask else None
            ms = timeit(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=am))
            fl = 4 * qs[0] * qs[1] * qs[2] * ks[2] * qs[3]
            # same math as two matmuls + softmax, to see whether the fused kernel wins
            def manual():
                s = torch.matmul(q, k.transpose(-1, -2)) * (qs[3] ** -0.5)
                if am is not None:
                    s = s.masked_fill(~am, float("-inf")) if am.dtype == torch.bool else s + am
                return torch.matmul(torch.softmax(s, dim=-1), v)
            ms2 = timeit(manual)
            total_alone += ms * cnt
            out["sdpa"].append({"q": qs, "k": ks, "count": cnt, "ms": ms, "manual_ms": ms2, "tflops": fl / ms / 1e9})
            print(f"  sdpa q={qs} k={ks} x{cnt}: {ms:6.3f} ms {fl/ms/1e9:5.2f} TF | matmul+softmax {ms2:6.3f} ms (sum {ms*cnt:6.2f} ms)")
    out["gemm_sdpa_alone_ms"] = total_alone
    print(f"sum of linears + sdpa alone: {total_alone:.2f} ms (compiled step {out['compiled_ms']:.2f} ms)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
