"""Which enumerated-shape packages does this machine's ANE compiler accept? (16-m1-mini.md 5-2)

One run = one candidate package: convert -> compile_model -> load in a child process (that is
where the OS compiles for the ANE) -> load again in a second child (cache hit?) -> verdict.

The verdict comes from the unified log, not from speed: a failed ANE compile is silent in
Core ML (the program then runs on the CPU) but aned logs "Compilation failed" / "Model load
failed". A load that exceeds --timeout is killed together with the orphaned ANECompilerService
(root-owned; needs the sudoers entry `NOPASSWD: /usr/bin/pkill -9 -x ANECompilerService`).

    uv run --no-sync python bench/probe_ane_shapes.py --tag b1_full23_1blk --batch 1 \
        --latents full --blocks 1 --timeout 1800
    uv run --no-sync python bench/probe_ane_shapes.py --tag b3_small17 --batch 3 \
        --latents 32,48,64,80,96,112,128,144,160,176,192,208,224,256,272,288,320

--blocks N keeps the first N DiT blocks (a proxy that compiles ~12/N times faster; whether it
fails the same way as the full stack is itself one of the questions).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PKILL = ["sudo", "-n", "/usr/bin/pkill", "-9", "-x", "ANECompilerService"]
LOG_PREDICATE = 'process == "ANECompilerService" OR process == "aned"'
LOG_PATTERNS = {
    "compilation_failed": "Compilation failed: error=Error Domain=com.apple.appleneuralengine.compiler",
    "model_load_failed": "Model load failed",
    "spiller_failure": "Register spiller failure",
    "regalloc_l2_miss": "RegAlloc: failed to allocate tensors",
}


# --------------------------------------------------------------------------------------
# Child: load one compiled model on the requested units and time every shape
# --------------------------------------------------------------------------------------


def child(mlmodelc: str, feeds_json: str, units_name: str, repeats: int) -> None:
    import coremltools as ct
    import numpy as np

    units = {
        "ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }[units_name]
    t0 = time.perf_counter()
    model = ct.models.CompiledMLModel(mlmodelc, compute_units=units)
    load_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    steps = []
    for shapes in json.loads(Path(feeds_json).read_text()):
        feed = {}
        for name, shape in shapes.items():
            if name.startswith("mask_"):
                feed[name] = np.zeros(shape, dtype=np.float16)
            else:
                feed[name] = (rng.standard_normal(shape) * 0.1).astype(np.float16)
        t0 = time.perf_counter()
        model.predict(feed)
        first = time.perf_counter() - t0
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.predict(feed)
            times.append(time.perf_counter() - t0)
        steps.append(
            {
                "x_t": shapes["x_t"],
                "first_ms": round(first * 1e3, 1),
                "median_ms": round(sorted(times)[len(times) // 2] * 1e3, 2),
            }
        )
    print("PROBE_RESULT " + json.dumps({"load_s": round(load_s, 2), "steps": steps}), flush=True)


# --------------------------------------------------------------------------------------
# Parent
# --------------------------------------------------------------------------------------


def compiler_procs() -> list[dict]:
    out = subprocess.run(
        ["ps", "-axo", "pid=,pcpu=,etime=,comm="], capture_output=True, text=True
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[3].endswith("ANECompilerService"):
            rows.append({"pid": int(parts[0]), "pcpu": float(parts[1]), "etime": parts[2]})
    return rows


def kill_compiler(watch_s: float) -> dict:
    before = compiler_procs()
    rc = subprocess.run(PKILL, capture_output=True, text=True)
    time.sleep(2.0)
    after = compiler_procs()
    time.sleep(watch_s)
    later = compiler_procs()
    return {
        "before": before,
        "pkill_rc": rc.returncode,
        "pkill_err": rc.stderr.strip(),
        "after_2s": after,
        f"after_{int(watch_s) + 2}s": later,
    }


def host_state() -> dict:
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return {
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "swap": swap.strip(),
        "compiler": compiler_procs(),
    }


def log_counts(start: datetime, end: datetime) -> dict:
    fmt = "%Y-%m-%d %H:%M:%S"
    out = subprocess.run(
        [
            "/usr/bin/log",
            "show",
            "--start",
            start.strftime(fmt),
            "--end",
            end.strftime(fmt),
            "--predicate",
            LOG_PREDICATE,
            "--style",
            "compact",
        ],
        capture_output=True,
        text=True,
    ).stdout
    lines = out.splitlines()
    counts = {k: sum(p in ln for ln in lines) for k, p in LOG_PATTERNS.items()}
    counts["lines"] = len(lines)
    return counts


def run_child(mlmodelc: Path, feeds: Path, units: str, repeats: int, timeout: float) -> dict:
    cmd = [
        sys.executable,
        __file__,
        "--child",
        str(mlmodelc),
        "--feeds",
        str(feeds),
        "--units",
        units,
        "--repeats",
        str(repeats),
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return {"status": "timeout", "wall_s": round(time.perf_counter() - t0, 1)}
    res: dict = {
        "status": "ok" if proc.returncode == 0 else "error",
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    for line in out.splitlines():
        if line.startswith("PROBE_RESULT "):
            res.update(json.loads(line[len("PROBE_RESULT ") :]))
    if proc.returncode != 0:
        res["stderr_tail"] = err[-1500:]
    return res


def parent(args: argparse.Namespace) -> None:
    import coremltools as ct
    from torch import nn

    from irodori_tts import ane_dit
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, download_hf_checkpoint

    latents = (
        ane_dit.S_BUCKETS_FULL
        if args.latents == "full"
        else tuple(int(x) for x in args.latents.split(","))
    )
    prof = ane_dit.PROFILES[args.profile]
    shapes = [
        ane_dit.Shape(
            args.batch,
            s,
            ane_dit.CtxProfile(
                *(v + args.ctx_pad_start + i for v in (prof.text, prof.speaker, prof.caption))
            ),
        )
        for i, s in enumerate(latents)
    ]

    pre = host_state()
    if pre["compiler"] and args.kill_stale:
        pre["killed_stale"] = kill_compiler(3.0)

    rt = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=download_hf_checkpoint(args.hf_checkpoint),
            model_device="mps",
            model_precision="fp16",
            codec_device="mps",
            codec_precision="fp32",
            compile_model=False,
            compile_dynamic=False,
        )
    )
    model = rt.model
    if args.blocks:

        class _Truncated:  # the attributes export_package / cache_key read
            cfg = model.cfg
            cond_module = model.cond_module
            in_proj = model.in_proj
            blocks = nn.ModuleList(list(model.blocks[: args.blocks]))
            out_norm = model.out_norm
            out_proj = model.out_proj

        model = _Truncated()  # type: ignore[assignment]

    cfg = rt.model.cfg
    heads = int(cfg.num_heads)
    score_mib = [s.batch * heads * s.latent * s.keys() * 2 / 2**20 for s in shapes]
    result: dict = {
        "tag": args.tag,
        "host": {
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip(),
            "macos": platform.mac_ver()[0],
            "coremltools": ct.__version__,
        },
        "package": {
            "batch": args.batch,
            "profile": args.profile,
            "latents": list(latents),
            "ctx_pad_start": args.ctx_pad_start,
            "default_index": args.default_index,
            "skip_model_load": args.skip_model_load,
            "blocks": args.blocks or len(rt.model.blocks),
            "units": args.units,
            # derived: fp16 attention score tensor B*H*S*K per layer
            "score_mib_max": round(max(score_mib), 1),
            "score_mib_sum": round(sum(score_mib), 1),
        },
        "pre": pre,
    }

    cache = Path(args.cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"probe_b{args.batch}_{ane_dit.cache_key(model, shapes)}_d{args.default_index}"
    mlpackage, mlmodelc = cache / f"{stem}.mlpackage", cache / f"{stem}.mlmodelc"
    if not mlpackage.exists():
        t0 = time.perf_counter()
        ane_dit.export_package(
            model,
            shapes,
            mlpackage,
            default_index=args.default_index,
            skip_model_load=args.skip_model_load,
        )
        result["convert_s"] = round(time.perf_counter() - t0, 1)
        result["convert_peak_rss_mib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        )
    if not mlmodelc.exists():
        t0 = time.perf_counter()
        ct.models.utils.compile_model(str(mlpackage), destination_path=str(mlmodelc))
        result["compile_model_s"] = round(time.perf_counter() - t0, 1)
    result["disk_mib"] = round(
        sum(f.stat().st_size for f in mlmodelc.rglob("*") if f.is_file()) / 2**20
    )
    del rt, model

    dims = ane_dit.Dims.from_model(type("M", (), {"cfg": cfg}))  # type: ignore[arg-type]
    feeds = cache / f"{stem}.feeds.json"
    feeds.write_text(json.dumps([ane_dit.input_shapes(s, dims) for s in shapes]))

    start = datetime.now()
    result["load"] = run_child(mlmodelc, feeds, args.units, args.repeats, args.timeout)
    if result["load"]["status"] == "timeout":
        result["timeout_kill"] = kill_compiler(args.watch)
    else:
        result["reload"] = run_child(mlmodelc, feeds, args.units, args.repeats, args.timeout)
        if args.cpu_ref:
            result["cpu_ref"] = run_child(mlmodelc, feeds, "cpu", 2, args.timeout)
    time.sleep(2.0)  # let the log catch up
    result["log"] = log_counts(start, datetime.now())

    if result["load"]["status"] == "timeout":
        verdict = "timeout"
    elif result["load"]["status"] != "ok":
        verdict = "error"
    elif result["log"]["model_load_failed"] or result["log"]["compilation_failed"]:
        verdict = "compile_failed"  # Core ML ran it on the CPU
    else:
        verdict = "ane_ok"
    result["verdict"] = verdict
    result["post"] = host_state()

    if not args.keep:
        for path in (mlpackage, mlmodelc):
            shutil.rmtree(path, ignore_errors=True)
        feeds.unlink(missing_ok=True)

    out = REPO / "docs/experiments/results/ane_probe" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    load = result["load"]
    print(
        f"[probe] {args.tag}: {verdict}  load {load.get('load_s', load['wall_s'])} s  "
        f"reload {result.get('reload', {}).get('load_s')} s  log {result['log']}  -> {out}"
    )
    for st in load.get("steps", []):
        print(f"[probe]   {st['x_t']}: {st['median_ms']} ms (first {st['first_ms']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--feeds", help=argparse.SUPPRESS)
    parser.add_argument("--tag")
    parser.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--latents", default="192,320,768", help="comma list, or 'full' (the 23 production buckets)"
    )
    parser.add_argument("--profile", default="a", choices=["a", "b"])
    parser.add_argument(
        "--ctx-pad-start",
        type=int,
        default=0,
        help="context padding of the first bucket (then +1 each)",
    )
    parser.add_argument(
        "--blocks", type=int, default=0, help="keep only the first N DiT blocks (0 = all)"
    )
    parser.add_argument(
        "--default-index",
        type=int,
        default=0,
        help="which enumerated shape is the default (-1 = largest)",
    )
    parser.add_argument(
        "--skip-model-load", action="store_true", help="ct.convert(skip_model_load=True)"
    )
    parser.add_argument("--units", default="ne", choices=["ne", "all", "gpu", "cpu"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per child load")
    parser.add_argument(
        "--watch", type=float, default=20.0, help="seconds to watch for a respawn after a kill"
    )
    parser.add_argument(
        "--cpu-ref", action="store_true", help="also time CPU_ONLY on the same package"
    )
    parser.add_argument(
        "--kill-stale", action="store_true", help="kill a leftover ANECompilerService first"
    )
    parser.add_argument("--keep", action="store_true", help="keep the package on disk")
    parser.add_argument("--cache-dir", default="~/.cache/irodori-tts/ane-probe")
    args = parser.parse_args()
    if args.child:
        child(args.child, args.feeds, args.units, args.repeats)
        return
    if not args.tag:
        parser.error("--tag is required")
    parent(args)


if __name__ == "__main__":
    main()
