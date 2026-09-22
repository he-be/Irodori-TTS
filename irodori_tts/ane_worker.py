"""Child process that owns the Core ML (Neural Engine) RF-step models.

Kept free of torch imports so ``spawn`` start-up stays cheap. The parent talks to it over a
pipe; tensors travel through ``multiprocessing.shared_memory`` blocks that are sized once
for the largest enumerated shape. Protocol (parent -> child), one tuple per message:

    ("register", {package_key: mlmodelc_path})           compile-cache paths; packages are loaded
                                                         lazily on first use (or eagerly via "load")
    ("load", package_key)                                load one package now
    ("ctx", ctx_key, package_key, {name: shape})         per-request constants: text_state,
                                                         speaker_state, caption_state, the four
                                                         mask parts and rope (read from the shm blocks)
    ("step", ctx_key, x_shape, t_shape)                  x_t / t_embed are in shm; output v goes
                                                         to the "v" block
    ("drop", ctx_key)
    ("quit",)

Replies: ("ok", info) or ("err", message). A "load" / "ctx" reply carrying ``ane_failed`` means
the ANE compiler failed for that package (unified log); the model was discarded and the parent
must stop routing to it. ``coremltools.predict`` holds the GIL, which is the
whole reason this lives in another process (see docs/experiments/13-ane.md, 2-2).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import shared_memory
from multiprocessing.connection import Connection

import numpy as np

OUTPUT_NAME = "v"

# A load that ends in "Model load failed" in the unified log means the ANE compiler gave up and
# Core ML silently fell back to the CPU (3x slower than MPS, and the failure is not cached, so
# every process would pay the 60-4500 s compile again; 16-m1-mini.md 5-2, 17-m1-ane-factors.md).
# The check only runs when the load took long enough to have compiled (a cached load is <= 1.2 s).
ANE_CHECK_MIN_S = 2.0
_LOG_PREDICATE = 'process == "aned"'
_LOG_FAILURE_MARKS = ("Model load failed", "Compilation failed")


def ane_compile_failed(start: datetime, end: datetime) -> dict | None:
    """Counts of aned failure lines logged between ``start`` and ``end`` (None when none, or when
    the unified log cannot be read). Window-based: two workers compiling at the same moment
    cannot be told apart, so a failure in one may be attributed to both."""
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        out = subprocess.run(
            [
                "/usr/bin/log", "show", "--start", start.strftime(fmt), "--end", end.strftime(fmt),
                "--predicate", _LOG_PREDICATE, "--style", "compact",
            ],
            capture_output=True, text=True, timeout=120.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    counts = {m: sum(m in ln for ln in out.splitlines()) for m in _LOG_FAILURE_MARKS}
    return counts if any(counts.values()) else None


def _attach(blocks: dict[str, str]) -> dict[str, shared_memory.SharedMemory]:
    return {name: shared_memory.SharedMemory(name=shm_name) for name, shm_name in blocks.items()}


def _view(block: shared_memory.SharedMemory, shape: tuple[int, ...]) -> np.ndarray:
    return np.ndarray(shape, dtype=np.float16, buffer=block.buf)


def serve(conn: Connection, blocks: dict[str, str], compute_units: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import coremltools as ct

    units = {
        "ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }[compute_units]
    shm = _attach(blocks)
    paths: dict[str, str] = {}
    models: dict[str, object] = {}
    contexts: dict[str, tuple[str, dict[str, np.ndarray]]] = {}

    check_log = compute_units in ("ne", "all")

    def _load(key: str) -> dict:
        start = datetime.now()
        t0 = time.perf_counter()
        models[key] = ct.models.CompiledMLModel(paths[key], compute_units=units)
        sec = time.perf_counter() - t0
        info: dict = {"load_sec": sec}
        if check_log and sec > ANE_CHECK_MIN_S:
            time.sleep(1.0)  # let the log catch up
            failed = ane_compile_failed(start, datetime.now())
            if failed is not None:
                del models[key]  # never run it on the CPU; the parent drops the package
                info["ane_failed"] = failed
        return info

    try:
        while True:
            msg = conn.recv()
            kind = msg[0]
            try:
                if kind == "quit":
                    conn.send(("ok", None))
                    break
                if kind == "register":
                    paths.update(msg[1])
                    conn.send(("ok", None))
                elif kind == "load":
                    key = msg[1]
                    info = _load(key) if key not in models else {"load_sec": 0.0}
                    conn.send(("ok", info))
                elif kind == "ctx":
                    _, ctx_key, package_key, shapes = msg
                    info = _load(package_key) if package_key not in models else {"load_sec": 0.0}
                    if "ane_failed" not in info:
                        # Copy out of the shared blocks: the parent reuses them for the next context.
                        arrays = {
                            name: _view(shm[name], shape).copy() for name, shape in shapes.items()
                        }
                        contexts[ctx_key] = (package_key, arrays)
                    conn.send(("ok", info))
                elif kind == "step":
                    _, ctx_key, x_shape, t_shape = msg
                    package_key, arrays = contexts[ctx_key]
                    feed = dict(arrays)
                    feed["x_t"] = _view(shm["x_t"], x_shape)
                    feed["t_embed"] = _view(shm["t_embed"], t_shape)
                    t0 = time.perf_counter()
                    out = models[package_key].predict(feed)[OUTPUT_NAME]
                    sec = time.perf_counter() - t0
                    v = _view(shm[OUTPUT_NAME], out.shape)
                    np.copyto(v, out, casting="same_kind")
                    conn.send(("ok", {"predict_sec": sec, "shape": tuple(out.shape)}))
                elif kind == "drop":
                    contexts.pop(msg[1], None)
                    conn.send(("ok", None))
                else:
                    conn.send(("err", f"unknown message {kind!r}"))
            except Exception:  # noqa: BLE001 - report to the parent, keep serving
                conn.send(("err", traceback.format_exc()))
    finally:
        for block in shm.values():
            block.close()


def main(conn: Connection, blocks: dict[str, str], compute_units: str) -> None:
    try:
        serve(conn, blocks, compute_units)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        sys.stdout.flush()
