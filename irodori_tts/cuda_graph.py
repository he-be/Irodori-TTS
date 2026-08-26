"""CUDA Graph replay for the RF Euler step.

Design (single machine, batch size 1, sequential requests):

* One *const set* per shape signature of the per-request constant tensors
  (condition states/masks, context K/V caches, precombined attention masks,
  latent mask). Static copies live in the const set and are refreshed by
  ``copy_`` when the source tensor object changes (or is marked dirty).
* One *graph entry* per (const set, dynamic signature) where the dynamic
  signature covers the ``x_t``/``t``/``dt`` shapes plus the Python-level
  constants baked into the step (CFG mode, scales, ``use_cfg`` ...).
* All graphs share one memory pool; outputs are cloned on replay so the
  static output buffer can be reused immediately.
* Any capture failure disables the runner for the rest of the process and
  falls back to eager execution.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import torch

logger = logging.getLogger(__name__)

_TreeLeaf = torch.Tensor | None
_Tree = Any


def tree_flatten(tree: _Tree, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten nested tuple/list/dict/Tensor/None into {path: tensor}."""
    out: dict[str, torch.Tensor] = {}
    if tree is None:
        return out
    if isinstance(tree, torch.Tensor):
        out[prefix] = tree
        return out
    if isinstance(tree, (tuple, list)):
        for i, item in enumerate(tree):
            out.update(tree_flatten(item, f"{prefix}/{i}"))
        return out
    if isinstance(tree, dict):
        for k, item in tree.items():
            out.update(tree_flatten(item, f"{prefix}/{k}"))
        return out
    raise TypeError(f"Unsupported tree node type: {type(tree)!r}")


def tree_map(tree: _Tree, fn: Any, prefix: str = "") -> _Tree:
    """Rebuild ``tree`` replacing every tensor leaf by ``fn(path, tensor)``."""
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        return fn(prefix, tree)
    if isinstance(tree, tuple):
        return tuple(tree_map(item, fn, f"{prefix}/{i}") for i, item in enumerate(tree))
    if isinstance(tree, list):
        return [tree_map(item, fn, f"{prefix}/{i}") for i, item in enumerate(tree)]
    if isinstance(tree, dict):
        return {k: tree_map(item, fn, f"{prefix}/{k}") for k, item in tree.items()}
    raise TypeError(f"Unsupported tree node type: {type(tree)!r}")


def tree_signature(tree: _Tree) -> tuple:
    return tuple(
        (path, tuple(t.shape), str(t.dtype)) for path, t in sorted(tree_flatten(tree).items())
    )


class _ConstSet:
    def __init__(self, signature: tuple, source_tree: _Tree) -> None:
        self.signature = signature
        self.static_tree = tree_map(source_tree, lambda _p, t: t.detach().clone())
        self.static_flat = tree_flatten(self.static_tree)
        self.last_ids: dict[str, int] = {}
        self.epoch = -1
        self.refs = 0
        self.bytes = sum(t.numel() * t.element_size() for t in self.static_flat.values())

    def refresh(self, source_tree: _Tree, *, epoch: int, dirty: set[str]) -> int:
        """Copy changed source tensors into the static buffers. Returns bytes copied."""
        copied = 0
        if self.epoch != epoch:
            # New request: never trust object ids from a previous request.
            self.last_ids.clear()
            self.epoch = epoch
        for path, src in tree_flatten(source_tree).items():
            dst = self.static_flat[path]
            top = path.split("/", 2)[1] if path.startswith("/") else path.split("/", 1)[0]
            if self.last_ids.get(path) == id(src) and top not in dirty:
                continue
            dst.copy_(src, non_blocking=True)
            self.last_ids[path] = id(src)
            copied += dst.numel() * dst.element_size()
        return copied


class _GraphEntry:
    def __init__(
        self,
        *,
        const_set: _ConstSet,
        static_x: torch.Tensor,
        static_tt: torch.Tensor,
        static_dt: torch.Tensor,
        static_out: torch.Tensor,
        graph: torch.cuda.CUDAGraph,
    ) -> None:
        self.const_set = const_set
        self.static_x = static_x
        self.static_tt = static_tt
        self.static_dt = static_dt
        self.static_out = static_out
        self.graph = graph
        self.bytes = sum(
            t.numel() * t.element_size() for t in (static_x, static_tt, static_dt, static_out)
        )

    def release(self) -> None:
        """Free the static buffers, then the graph's blocks in the private pool."""
        self.static_x = None
        self.static_tt = None
        self.static_dt = None
        self.static_out = None
        self.graph.reset()
        self.bytes = 0


class RFStepGraphRunner:
    def __init__(
        self,
        *,
        device: torch.device,
        max_entries: int = 12,
        capture_after: int = 1,
        max_static_bytes: int = 0,
        release_pool_on_evict: bool = True,
        shared_pool: bool = True,
        max_latent_frames: int = 0,
    ) -> None:
        self.device = device
        self.max_entries = max(1, int(max_entries))
        self.capture_after = max(0, int(capture_after))
        # Byte budget for the static buffers (const sets are copies of the context K/V
        # caches, so they scale with reference/caption length -- a count-only LRU lets
        # them reach GBs). 0 disables the budget.
        self.max_static_bytes = max(0, int(max_static_bytes))
        self.release_pool_on_evict = bool(release_pool_on_evict)
        # One shared pool lets graphs reuse each other's freed blocks but the pool's
        # segments are only returned to the driver once *every* graph in it is reset.
        # Per-entry pools trade that reuse for eviction actually shrinking the footprint.
        self.shared_pool = bool(shared_pool)
        # Capture cost (private-pool workspace + static const copy) grows with the latent
        # length while the replay gain shrinks (long utterances are compute-bound), so
        # above this many latent frames the step runs eager. 0 = no limit.
        self.max_latent_frames = max(0, int(max_latent_frames))
        self.pool = torch.cuda.graph_pool_handle()
        self.entries: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self.const_sets: dict[tuple, _ConstSet] = {}
        self.counts: dict[tuple, int] = {}
        self.variant = "base"
        self.disabled = False
        self._epoch = 0
        self._dirty: set[str] = set()
        self._stats = {
            "replay": 0,
            "capture": 0,
            "eager": 0,
            "evict": 0,
            "fallback": 0,
            "skip_oversize": 0,
            "skip_long": 0,
            "copied_bytes": 0,
        }

    # -- request lifecycle -------------------------------------------------
    def begin_request(self) -> None:
        self._epoch += 1
        self._dirty.clear()

    def mark_dirty(self, name: str) -> None:
        self._dirty.add(name)

    def set_variant(self, variant: str) -> None:
        if variant != self.variant:
            self.clear()
            self.variant = variant

    def clear(self) -> None:
        for entry in self.entries.values():
            entry.release()
        self.entries.clear()
        self.const_sets.clear()
        self.counts.clear()
        # Keep the pool handle: graphs are freed with their entries.
        torch.cuda.synchronize(self.device)
        if self.release_pool_on_evict:
            torch.cuda.empty_cache()

    def static_bytes(self) -> int:
        return sum(e.bytes for e in self.entries.values()) + sum(
            c.bytes for c in self.const_sets.values()
        )

    def stats(self) -> dict[str, Any]:
        static_bytes = self.static_bytes()
        return {
            **self._stats,
            "entries": len(self.entries),
            "const_sets": len(self.const_sets),
            "static_bytes": static_bytes,
            "disabled": self.disabled,
        }

    # -- main entry point --------------------------------------------------
    def run_step(
        self,
        state: Any,
        *,
        x_t: torch.Tensor,
        tt: torch.Tensor,
        dt: torch.Tensor,
        use_cfg: bool,
        alt_index: int,
    ) -> torch.Tensor:
        if self.disabled or x_t.device.type != "cuda":
            self._stats["eager"] += 1
            return state.step(x_t, tt, dt, use_cfg=use_cfg, step_index=alt_index)

        state.prepare_masks(use_cfg=use_cfg, alt_index=alt_index)
        const_tree = state.const_tree()
        const_sig = (self.variant, tree_signature(const_tree))
        dyn_sig = (
            const_sig,
            tuple(x_t.shape),
            str(x_t.dtype),
            tuple(tt.shape),
            str(tt.dtype),
            tuple(dt.shape),
            str(dt.dtype),
            bool(use_cfg),
            int(alt_index),
            state.python_signature(),
        )

        entry = self.entries.get(dyn_sig)
        if entry is None:
            count = self.counts.get(dyn_sig, 0)
            self.counts[dyn_sig] = count + 1
            if count < self.capture_after:
                self._stats["eager"] += 1
                return state.step(x_t, tt, dt, use_cfg=use_cfg, step_index=alt_index)
            if self.max_latent_frames and int(x_t.shape[1]) > self.max_latent_frames:
                self._stats["skip_long"] += 1
                self._stats["eager"] += 1
                return state.step(x_t, tt, dt, use_cfg=use_cfg, step_index=alt_index)
            if self._const_set_too_large(const_sig, const_tree):
                # A single const set larger than the whole budget (very long reference
                # and/or caption): replaying would double the context K/V in VRAM for
                # little gain, so run this shape eagerly.
                self._stats["skip_oversize"] += 1
                self._stats["eager"] += 1
                return state.step(x_t, tt, dt, use_cfg=use_cfg, step_index=alt_index)
            try:
                entry = self._capture(
                    state,
                    const_sig=const_sig,
                    const_tree=const_tree,
                    dyn_sig=dyn_sig,
                    x_t=x_t,
                    tt=tt,
                    dt=dt,
                    use_cfg=use_cfg,
                    alt_index=alt_index,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("CUDA Graph capture failed; falling back to eager: %s", exc)
                self.disabled = True
                self._stats["fallback"] += 1
                self.clear()
                return state.step(x_t, tt, dt, use_cfg=use_cfg, step_index=alt_index)
        else:
            self.entries.move_to_end(dyn_sig)

        copied = entry.const_set.refresh(const_tree, epoch=self._epoch, dirty=self._dirty)
        self._dirty.clear()
        self._stats["copied_bytes"] += copied
        entry.static_x.copy_(x_t, non_blocking=True)
        entry.static_tt.copy_(tt, non_blocking=True)
        entry.static_dt.copy_(dt, non_blocking=True)
        entry.graph.replay()
        self._stats["replay"] += 1
        return entry.static_out.clone()

    # -- capture -----------------------------------------------------------
    def _capture(
        self,
        state: Any,
        *,
        const_sig: tuple,
        const_tree: _Tree,
        dyn_sig: tuple,
        x_t: torch.Tensor,
        tt: torch.Tensor,
        dt: torch.Tensor,
        use_cfg: bool,
        alt_index: int,
    ) -> _GraphEntry:
        const_set = self.const_sets.get(const_sig)
        if const_set is None:
            const_set = _ConstSet(const_sig, const_tree)
            self.const_sets[const_sig] = const_set
        # Make sure the static copies hold this request's values before capture.
        const_set.refresh(const_tree, epoch=self._epoch, dirty=set(const_set.static_flat.keys()))
        static_state = state.rebind(const_set.static_tree)

        static_x = x_t.detach().clone()
        static_tt = tt.detach().clone()
        static_dt = dt.detach().clone()

        # Warmup on a side stream (required before capture), then capture.
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                static_state.step(static_x, static_tt, static_dt, use_cfg=use_cfg, step_index=alt_index)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool if self.shared_pool else None, stream=side):
            static_out = static_state.step(
                static_x, static_tt, static_dt, use_cfg=use_cfg, step_index=alt_index
            )
        torch.cuda.synchronize(self.device)

        entry = _GraphEntry(
            const_set=const_set,
            static_x=static_x,
            static_tt=static_tt,
            static_dt=static_dt,
            static_out=static_out,
            graph=graph,
        )
        const_set.refs += 1
        self.entries[dyn_sig] = entry
        self._stats["capture"] += 1
        evicted = 0
        while len(self.entries) > 1 and (
            len(self.entries) > self.max_entries
            or (self.max_static_bytes > 0 and self.static_bytes() > self.max_static_bytes)
        ):
            _old_sig, old = self.entries.popitem(last=False)
            old.const_set.refs -= 1
            if old.const_set.refs <= 0:
                self.const_sets.pop(old.const_set.signature, None)
            # Without reset() the graph keeps its blocks in the shared private pool, which
            # never shrinks: the pool would grow with every shape the process ever sees.
            # Drop the pool-allocated output first so reset() can release its blocks.
            old.release()
            self._stats["evict"] += 1
            evicted += 1
        if evicted and self.release_pool_on_evict:
            # Private-pool segments are only returned to the driver by empty_cache().
            torch.cuda.empty_cache()
        return entry

    def _const_set_too_large(self, const_sig: tuple, const_tree: _Tree) -> bool:
        """True when this const set alone would blow the static byte budget."""
        if self.max_static_bytes <= 0 or const_sig in self.const_sets:
            return False
        need = sum(t.numel() * t.element_size() for t in tree_flatten(const_tree).values())
        return need > self.max_static_bytes
