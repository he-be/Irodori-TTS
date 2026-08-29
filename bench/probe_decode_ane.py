"""Probe: does the DACVAE codec decoder run on the Neural Engine, and how fast vs MPS? (15-decode-ane.md)

Exports ``codec.decode`` (out_proj + conv stack + watermark passthrough, alpha=0) for a fixed
latent length with ``torch.export``, converts it to a Core ML fp16 mlprogram, prints the MIL op
histogram and the compute plan (which device runs each op under CPU_AND_NE), then times
``predict`` under CPU_AND_NE / ALL / CPU_AND_GPU and compares the audio with the torch fp32
decode (max |diff|, SNR). Finishes with the MPS fp16-autocast eager decode for reference.

    uv run python bench/probe_decode_ane.py [FRAMES ...]      # default 180 (= 7.2 s at hop 1920)

Packages go to ``outputs/decode_ane/``. The first CPU_AND_NE load triggers the OS-side ANE
compile (about 2 min for 180 frames); renaming the .mlmodelc invalidates that cache (13-ane.md).
"""
from __future__ import annotations

import collections
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import coremltools as ct
from irodori_tts.codec import DACVAECodec

FRAMES = [int(a) for a in sys.argv[1:]] or [180]
OUT = Path(__file__).resolve().parent.parent / "outputs" / "decode_ane"
OUT.mkdir(parents=True, exist_ok=True)

m = DACVAECodec.prepare_cpu().eval()
dec = m.decoder
dec.alpha = 0.0
enc_block = dec.wm_model.encoder_block

class DecodeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = m.quantizer.out_proj
        self.layers = dec.model
        self.tail = enc_block
    def forward(self, z):
        x = self.out_proj(z)
        for l in self.layers:
            x = l(x)
        return self.tail.forward_no_conv(x)

mod = DecodeModule().eval()
torch.manual_seed(0)

def mps_time(z, n=5):
    mm = DACVAECodec.prepare_cpu().eval().to("mps")
    mm.decoder.alpha = 0.0
    eb = mm.decoder.wm_model.encoder_block
    mm.decoder.watermark = lambda x, message=None: eb.forward_no_conv(x)
    zz = z.to("mps")
    with torch.inference_mode(), torch.autocast("mps", dtype=torch.float16):
        for _ in range(2): mm.decode(zz); torch.mps.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); y = mm.decode(zz); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000, y.float().cpu()

for S in FRAMES:
    z = torch.randn(1, 32, S) * 1.0
    with torch.inference_mode():
        ref = mod(z)
    print(f"[S={S}] torch fp32 out {tuple(ref.shape)} |ref| max {ref.abs().max():.3f}", flush=True)
    t0 = time.perf_counter()
    ex = torch.export.export(mod, (z,)).run_decompositions({})
    print(f"[S={S}] exported in {time.perf_counter()-t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    ml = ct.convert(ex, inputs=[ct.TensorType(name="z", shape=(1, 32, S), dtype=np.float16)],
                    outputs=[ct.TensorType(name="audio", dtype=np.float16)],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                    convert_to="mlprogram")
    pkg = OUT / f"dec_{S}.mlpackage"; mlc = OUT / f"dec_{S}.mlmodelc"
    for p in (pkg, mlc):
        if p.exists(): shutil.rmtree(p)
    ml.save(str(pkg))
    print(f"[S={S}] converted in {time.perf_counter()-t0:.1f}s", flush=True)
    ops = collections.Counter(op.op_type for op in ml._mil_program.functions["main"].operations)
    print(f"[S={S}] MIL ops: {dict(ops)}", flush=True)
    t0 = time.perf_counter()
    ct.models.utils.compile_model(str(pkg), destination_path=str(mlc))
    print(f"[S={S}] compiled in {time.perf_counter()-t0:.1f}s", flush=True)
    # compute plan: which device runs each op
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
        prog = plan.model_structure.program
        dev = collections.Counter()
        offane = collections.Counter()
        for op in prog.functions["main"].block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is None: continue
            name = type(u.preferred_compute_device).__name__
            dev[name] += 1
            if "NeuralEngine" not in name: offane[(op.operator_name, name)] += 1
        print(f"[S={S}] compute plan devices: {dict(dev)}", flush=True)
        if offane: print(f"[S={S}] ops NOT on ANE: {dict(offane)}", flush=True)
    except Exception as e:
        print(f"[S={S}] compute plan failed: {e!r}", flush=True)
    for units_name, units in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("ALL", ct.ComputeUnit.ALL), ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU)):
        t0 = time.perf_counter()
        cm = ct.models.CompiledMLModel(str(mlc), compute_units=units)
        print(f"[S={S}] {units_name}: load {time.perf_counter()-t0:.1f}s", flush=True)
        inp = {"z": z.numpy().astype(np.float16)}
        for _ in range(2): cm.predict(inp)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); out = cm.predict(inp)["audio"]; ts.append(time.perf_counter() - t0)
        out = torch.from_numpy(np.asarray(out).astype(np.float32))
        d = (out - ref)
        snr = 10 * torch.log10(ref.pow(2).mean() / d.pow(2).mean().clamp_min(1e-12))
        print(f"[S={S}] {units_name}: predict median {np.median(ts)*1000:.1f} ms (min {min(ts)*1000:.1f}), max|diff| {d.abs().max():.4f}, SNR {snr:.1f} dB", flush=True)
        del cm
    ms, ymps = mps_time(z)
    d = ymps - ref
    snr = 10 * torch.log10(ref.pow(2).mean() / d.pow(2).mean().clamp_min(1e-12))
    print(f"[S={S}] MPS fp16 autocast eager: {ms:.1f} ms, max|diff| {d.abs().max():.4f}, SNR {snr:.1f} dB", flush=True)
