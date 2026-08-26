#!/usr/bin/env python3
"""Load / unload profiler for the inference runtime (experiment 11).

Runs the load in a *fresh subprocess* so that "load" means what it means in
practice: python import, CUDA context creation, checkpoint read, model
construction and the move to the GPU.  Per-phase timings come from
``IRODORI_OPT_LOAD_TRACE=1`` inside ``inference_runtime``.

    uv run --no-sync python bench/bench_load.py --repeats 3 --tag baseline \
        --output docs/experiments/results/11_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

WORKER = """
import json, os, sys, time
t_proc = float(os.environ["BENCH_LOAD_T0"])  # time.monotonic() in the parent
t0 = time.perf_counter()
import torch
t_import_torch = time.perf_counter() - t0

t0 = time.perf_counter()
from irodori_tts.inference_runtime import (
    InferenceRuntime, RuntimeKey, download_hf_checkpoint, get_load_trace,
)
t_import_pkg = time.perf_counter() - t0

t0 = time.perf_counter()
torch.cuda.init()
torch.cuda.synchronize()
t_cuda_init = time.perf_counter() - t0

checkpoint = os.environ.get("BENCH_LOAD_CKPT") or download_hf_checkpoint(
    os.environ.get("BENCH_LOAD_HF", "Aratako/Irodori-TTS-v4.1-Small")
)

t0 = time.perf_counter()
runtime = InferenceRuntime.from_key(
    RuntimeKey(
        checkpoint=checkpoint,
        model_device="cuda",
        model_precision=os.environ.get("BENCH_LOAD_PRECISION", "bf16"),
        codec_device="cuda",
        codec_precision="fp32",
    )
)
torch.cuda.synchronize()
t_from_key = time.perf_counter() - t0

def smi():
    out = subprocess_check(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"])
    mine = os.getpid()
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and int(parts[0]) == mine:
            return float(parts[1])
    return 0.0

import subprocess as _sp
def subprocess_check(cmd):
    return _sp.run(cmd, capture_output=True, text=True).stdout

rss = float(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20

result = {
    "t_import_torch": t_import_torch,
    "t_import_pkg": t_import_pkg,
    "t_cuda_init": t_cuda_init,
    "t_from_key": t_from_key,
    "t_process_to_loaded": time.monotonic() - t_proc,
    "phases": [{"name": n, "seconds": v} for n, v in get_load_trace()],
    "alloc_mib": torch.cuda.memory_allocated() / 2**20,
    "reserved_mib": torch.cuda.memory_reserved() / 2**20,
    "smi_loaded_mib": smi(),
    "rss_mib": rss,
}

if os.environ.get("BENCH_LOAD_SYNTH", "0") == "1":
    from irodori_tts.inference_runtime import SamplingRequest
    t0 = time.perf_counter()
    out = runtime.synthesize(SamplingRequest(
        text="こんにちは、ロード時間の計測をしています。",
        ref_wav=str(os.environ.get("BENCH_LOAD_REF")),
        num_steps=40, seed=0,
    ))
    torch.cuda.synchronize()
    result["t_first_synth"] = time.perf_counter() - t0
    result["smi_after_synth_mib"] = smi()

t0 = time.perf_counter()
runtime.unload()
del runtime
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
result["t_unload"] = time.perf_counter() - t0
result["reserved_after_unload_mib"] = torch.cuda.memory_reserved() / 2**20
result["smi_after_unload_mib"] = smi()

if os.environ.get("BENCH_LOAD_RELOAD", "0") == "1":
    # Second load in the same process: every import is already paid, so this is
    # the floor a long-running server would see if it reloaded a checkpoint.
    from irodori_tts.inference_runtime import reset_load_trace
    reset_load_trace()
    t0 = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint,
            model_device="cuda",
            model_precision=os.environ.get("BENCH_LOAD_PRECISION", "bf16"),
            codec_device="cuda",
            codec_precision="fp32",
        )
    )
    torch.cuda.synchronize()
    result["t_second_load"] = time.perf_counter() - t0
    result["phases_second"] = [{"name": n, "seconds": v} for n, v in get_load_trace()]
    runtime.unload()
    del runtime

sys.stdout.write("BENCH_LOAD_JSON " + json.dumps(result) + "\\n")
"""


def run_once(env_extra: dict[str, str], *, synth: bool, ref: str, reload_twice: bool) -> dict:
    env = dict(os.environ)
    env["IRODORI_OPT_LOAD_TRACE"] = "1"
    env["BENCH_LOAD_SYNTH"] = "1" if synth else "0"
    env["BENCH_LOAD_RELOAD"] = "1" if reload_twice else "0"
    env["BENCH_LOAD_REF"] = ref
    env.update(env_extra)
    env["BENCH_LOAD_T0"] = str(time.monotonic())
    proc = subprocess.run(
        [sys.executable, "-c", WORKER],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_LOAD_JSON "):
            return json.loads(line[len("BENCH_LOAD_JSON ") :])
    sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-8000:] + "\n")
    raise RuntimeError("worker did not report a result")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="load")
    ap.add_argument("--checkpoint", default=None, help="local .safetensors (overrides --hf)")
    ap.add_argument("--hf", default="Aratako/Irodori-TTS-v4.1-Small")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--synth", action="store_true", help="also time the first synthesize()")
    ap.add_argument("--reload", action="store_true", help="also time a second in-process load")
    ap.add_argument("--ref", default=str(REPO_ROOT / "outputs" / "sample.wav"))
    ap.add_argument(
        "--drop-caches",
        action="store_true",
        help="echo 3 > /proc/sys/vm/drop_caches before each run (needs sudo)",
    )
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    env_extra = {}
    if args.checkpoint:
        env_extra["BENCH_LOAD_CKPT"] = str(Path(args.checkpoint).resolve())
    else:
        env_extra["BENCH_LOAD_HF"] = args.hf
    env_extra["BENCH_LOAD_PRECISION"] = args.precision
    for item in args.env:
        k, _, v = item.partition("=")
        env_extra[k] = v

    runs = []
    for i in range(args.repeats):
        if args.drop_caches:
            subprocess.run(
                ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=False
            )
        run = run_once(env_extra, synth=args.synth, ref=args.ref, reload_twice=args.reload)
        runs.append(run)
        phases = " ".join(f"{p['name']}={p['seconds']:.2f}" for p in run["phases"])
        print(
            f"[{i}] proc->loaded {run['t_process_to_loaded']:.2f}s "
            f"(torch {run['t_import_torch']:.2f} pkg {run['t_import_pkg']:.2f} "
            f"cuda {run['t_cuda_init']:.2f} from_key {run['t_from_key']:.2f}) | {phases}",
            flush=True,
        )
        print(
            f"     smi {run['smi_loaded_mib']:.0f} MiB | rss {run['rss_mib']:.0f} MiB | "
            f"unload {run['t_unload']:.2f}s -> smi {run['smi_after_unload_mib']:.0f} MiB",
            flush=True,
        )

    def med(key: str) -> float:
        return statistics.median(r[key] for r in runs if key in r)

    summary = {
        "tag": args.tag,
        "repeats": args.repeats,
        "checkpoint": args.checkpoint or args.hf,
        "precision": args.precision,
        "env": env_extra,
        "median": {
            k: med(k)
            for k in (
                "t_import_torch",
                "t_import_pkg",
                "t_cuda_init",
                "t_from_key",
                "t_process_to_loaded",
                "t_unload",
                "rss_mib",
                "smi_loaded_mib",
                "alloc_mib",
                "reserved_mib",
            )
        },
        "phases_median": {
            name: statistics.median(
                next(p["seconds"] for p in r["phases"] if p["name"] == name) for r in runs
            )
            for name in [p["name"] for p in runs[0]["phases"]]
        },
        "runs": runs,
    }
    if args.synth:
        summary["median"]["t_first_synth"] = med("t_first_synth")
    if args.reload:
        summary["median"]["t_second_load"] = med("t_second_load")

    print("\n== median ==")
    for k, v in summary["median"].items():
        print(f"  {k:24s} {v:8.2f}")
    print("  -- phases --")
    for k, v in summary["phases_median"].items():
        print(f"  {k:24s} {v:8.2f}")

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
