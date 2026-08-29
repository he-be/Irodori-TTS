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

Replies: ("ok", info) or ("err", message). ``coremltools.predict`` holds the GIL, which is the
whole reason this lives in another process (see docs/experiments/13-ane.md, 2-2).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import Connection

import numpy as np

OUTPUT_NAME = "v"


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

    def _load(key: str) -> float:
        t0 = time.perf_counter()
        models[key] = ct.models.CompiledMLModel(paths[key], compute_units=units)
        return time.perf_counter() - t0

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
                    sec = _load(key) if key not in models else 0.0
                    conn.send(("ok", {"load_sec": sec}))
                elif kind == "ctx":
                    _, ctx_key, package_key, shapes = msg
                    load_sec = _load(package_key) if package_key not in models else 0.0
                    # Copy out of the shared blocks: the parent reuses them for the next context.
                    arrays = {name: _view(shm[name], shape).copy() for name, shape in shapes.items()}
                    contexts[ctx_key] = (package_key, arrays)
                    conn.send(("ok", {"load_sec": load_sec}))
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
