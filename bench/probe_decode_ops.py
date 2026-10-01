#!/usr/bin/env python3
"""Where does codec decode go on this GPU? (19-m6-mini.md)

1. Whole decode (fp16 autocast, the runtime default) eager and torch.compile'd.
2. Per module class (Conv1d / ConvTranspose1d / Snake1d) with syncing forward hooks: an upper
   bound per class, enough to see which side dominates.
3. Every distinct conv of the decoder at its real input length, alone, fp16: ms and effective
   TFLOPS, next to the same contraction written as a matmul (k=1) or unfold+matmul (k>1), to see
   whether MPS's conv kernels reach the GPU's matmul units.

    uv run --no-sync python bench/probe_decode_ops.py [FRAMES] [--json out.json]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irodori_tts.codec import DACVAECodec  # noqa: E402


def timeit(fn, n=10, warm=3):
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
    ap.add_argument("frames", type=int, nargs="?", default=180)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    codec = DACVAECodec.load(device="mps", dtype=torch.float32)
    model = codec.model
    dec = model.decoder
    z = torch.randn(1, 32, args.frames, device="mps")
    out: dict = {"frames": args.frames, "chip": None}

    def decode():
        with torch.inference_mode(), torch.autocast("mps", dtype=torch.float16):
            return model.decode(z)

    out["decode_eager_ms"] = timeit(decode, n=5)
    print(f"decode eager fp16-autocast: {out['decode_eager_ms']:.1f} ms")

    # 2. per class, syncing hooks
    acc: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    starts: dict[int, float] = {}
    handles = []
    for mod in model.modules():
        name = type(mod).__name__
        if name not in {"NormConv1d", "NormConvTranspose1d", "Snake1d"}:
            continue

        def pre(m, _inp):
            torch.mps.synchronize()
            starts[id(m)] = time.perf_counter()

        def post(m, _inp, _out, name=name):
            torch.mps.synchronize()
            acc[name] += time.perf_counter() - starts[id(m)]
            cnt[name] += 1

        handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    decode()
    acc.clear(), cnt.clear()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    decode()
    torch.mps.synchronize()
    hooked_total = (time.perf_counter() - t0) * 1000
    for h in handles:
        h.remove()
    out["per_class_ms"] = {k: v * 1000 for k, v in acc.items()}
    out["per_class_count"] = dict(cnt)
    out["hooked_total_ms"] = hooked_total
    print(f"hooked decode {hooked_total:.1f} ms; " + ", ".join(f"{k}={v*1000:.1f} ms ({cnt[k]})" for k, v in acc.items()))

    # 3. every conv alone at its real input
    shapes: dict[tuple, dict] = {}
    hs = []
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):

            def cap(m, inp, _out):
                x = inp[0]
                key = (type(m).__name__, m.in_channels, m.out_channels, m.kernel_size[0], m.stride[0],
                       m.dilation[0], tuple(x.shape))
                e = shapes.setdefault(key, {"mod": m, "count": 0})
                e["count"] += 1

            hs.append(mod.register_forward_hook(cap))
    decode()
    for h in hs:
        h.remove()

    rows = []
    for key, e in sorted(shapes.items(), key=lambda kv: -kv[1]["count"] * kv[0][6][-1] * kv[0][1] * kv[0][2] * kv[0][3]):
        kind, cin, cout, k, s, d, xshape = key
        m = e["mod"]
        # inputs as the conv module sees them (after the module's own padding)
        x = torch.randn(*xshape, device="mps", dtype=torch.float16)
        w = m.weight.detach().to(torch.float16)
        b = m.bias.detach().to(torch.float16) if m.bias is not None else None
        if kind == "NormConv1d":
            xp = m.pad(x) if hasattr(m, "pad") else x
            fn = lambda xp=xp, w=w, b=b, m=m: F.conv1d(xp, w, b, m.stride, 0, m.dilation, m.groups)
            y = fn()
            flops = 2 * cin * cout * k * y.shape[-1]
            if k == 1:
                w2 = w[:, :, 0]
                alt = lambda xp=xp, w2=w2: torch.matmul(w2, xp)
                alt_name = "matmul"
            else:
                # (B, C*k, T_out) unfolded, then one GEMM
                w2 = w.reshape(cout, cin * k)

                def alt(xp=xp, w2=w2, k=k, d=d, s=s):
                    cols = F.unfold(xp.unsqueeze(2), (1, k), dilation=(1, d), stride=(1, s))
                    return torch.matmul(w2, cols)

                alt_name = "unfold+matmul"
        else:
            fn = lambda x=x, w=w, b=b, m=m: F.conv_transpose1d(x, w, b, m.stride, m.padding, m.output_padding, m.groups, m.dilation)
            y = fn()
            flops = 2 * cin * cout * k * x.shape[-1]
            # transposed conv = GEMM (C_out*k, C_in) x (C_in, T) then overlap-add (fold)
            w2 = w.permute(1, 2, 0).reshape(cout * k, cin)

            def alt(x=x, w2=w2, k=k, s=s, cout=cout, m=m):
                cols = torch.matmul(w2, x)  # (B, cout*k, T)
                L = (x.shape[-1] - 1) * s + k
                return F.fold(cols, (1, L), (1, k), stride=(1, s))

            alt_name = "matmul+fold"
        ms = timeit(fn)
        try:
            ms_alt = timeit(alt)
        except RuntimeError as exc:  # noqa: PERF203
            ms_alt = float("nan")
            alt_name += f" ({str(exc)[:60]})"
        row = {
            "kind": kind, "cin": cin, "cout": cout, "k": k, "stride": s, "dilation": d,
            "in_shape": list(xshape), "count": e["count"], "gflop": flops / 1e9,
            "ms": ms, "tflops": flops / ms / 1e9, "alt": alt_name, "alt_ms": ms_alt,
            "alt_tflops": flops / ms_alt / 1e9,
        }
        rows.append(row)
        print(f"{kind[4:]:>15} {cin:>4}->{cout:<4} k{k:<2} s{s:<2} d{d} T{xshape[-1]:>6} x{e['count']}: "
              f"{ms:7.2f} ms {row['tflops']:5.2f} TF | {alt_name} {ms_alt:7.2f} ms {row['alt_tflops']:5.2f} TF")
        del x, y
        torch.mps.empty_cache()
    out["convs"] = rows
    out["conv_total_ms"] = sum(r["ms"] * r["count"] for r in rows)
    out["conv_alt_total_ms"] = sum(min(r["ms"], r["alt_ms"]) * r["count"] for r in rows)
    print(f"sum of convs alone: {out['conv_total_ms']:.1f} ms (best-of-two per conv: {out['conv_alt_total_ms']:.1f} ms)")

    # Snake alone at the largest size
    x = torch.randn(1, 96, args.frames * 1920, device="mps", dtype=torch.float16)
    alpha = torch.rand(1, 96, 1, device="mps", dtype=torch.float16) + 0.5
    from dacvae.nn.layers import snake

    out["snake_96ch_eager_ms"] = timeit(lambda: snake(x, alpha))
    out["snake_96ch_bytes_ms_at_140GBs"] = x.numel() * 2 * 2 / 140e9 * 1000
    print(f"snake 96ch x {x.shape[-1]}: eager {out['snake_96ch_eager_ms']:.2f} ms "
          f"(one read+write at 140 GB/s = {out['snake_96ch_bytes_ms_at_140GBs']:.2f} ms)")
    if not args.skip_compile:
        cs = torch.compile(snake)
        out["snake_96ch_compiled_ms"] = timeit(lambda: cs(x, alpha))
        print(f"snake compiled: {out['snake_96ch_compiled_ms']:.2f} ms")
        cdec = torch.compile(model.decode)

        def cdecode():
            with torch.inference_mode(), torch.autocast("mps", dtype=torch.float16):
                return cdec(z)

        out["decode_compiled_ms"] = timeit(cdecode, n=5)
        print(f"decode compiled: {out['decode_compiled_ms']:.1f} ms")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
