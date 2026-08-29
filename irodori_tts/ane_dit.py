"""RF step (DiT forward with encoded conditions) on the Apple Neural Engine via Core ML.

Why this exists: on this Mac the DiT is compute-bound on the GPU (12-metal-port.md) while the
Neural Engine runs the same matmuls 1.5-2.5x faster (13-ane.md). PyTorch cannot address the
ANE, so the step is exported once, compiled, cached on disk and served by a child process
(``ane_worker.py``; ``predict`` holds the GIL, so a thread could not overlap with MPS work).
The parent keeps the Euler loop, the CFG combination and - optionally - one CFG branch on the
GPU (``IRODORI_OPT_ANE_GPU_BRANCHES``).

What the ANE compiler accepts (all found by bisection, see 13-ane.md section 3):
  * export with ``torch.export`` + ``dynamic_shapes`` (the TorchScript trace makes coremltools
    insert symbolic batch broadcasts into SDPA, after which the whole program runs on the CPU);
  * every input flexible (one static input among enumerated ones -> CPU), and every input's
    enumerated shape list unique per combination (Core ML ties the lists by index and
    de-duplicates them, so a repeated shape mis-aligns the combinations) -> the batch size and
    the context profile are fixed per package, only the latent length is enumerated, and the
    context padding grows by one token per latent bucket so no shape repeats;
  * no broadcast of a (B, 1, C) tensor over the latent axis -> the timestep embedding is fed
    per token (B, S, 512) and the AdaLN low-rank modulation is computed per token (+~9% FLOPs);
  * context keys concatenated onto the self keys as rank-3 (B, L, H*hd), not rank-4;
  * attention as matmul + softmax + matmul with an additive key mask (B, 1, 1, K) assembled
    inside from four per-context parts (SDPA with attn_mask leaves the ANE);
  * RoPE as ``x * cos + (x @ P) * sin`` with a constant pair-permutation P (no rank-5 tensors).

Numerics: fp16 only. The residual stream reaches |h| ~ 2300 in the last block, so every RMS
normalisation of the stream is computed on ``x / 64`` (identical math, keeps ``x*x`` far below
the fp16 limit). Timestep embeddings are computed in fp32 on the CPU.
"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import (
    JointAttention,
    LowRankAdaLN,
    TextToLatentRFDiT,
    get_timestep_embedding,
    precompute_freqs_cis,
)

WRAPPER_VERSION = "5"
MASK_NEG = -1.0e4
_PRESCALE = 1.0 / 64.0
INPUT_NAMES = (
    "x_t", "t_embed", "text_state", "speaker_state", "caption_state",
    "mask_self", "mask_text", "mask_speaker", "mask_caption", "rope",
)
CTX_INPUT_NAMES = (
    "text_state", "speaker_state", "caption_state",
    "mask_self", "mask_text", "mask_speaker", "mask_caption", "rope",
)


# --------------------------------------------------------------------------------------
# Shape sets
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CtxProfile:
    text: int
    speaker: int
    caption: int


@dataclass(frozen=True)
class Shape:
    batch: int
    latent: int
    ctx: CtxProfile  # padded context lengths of this exact combination

    def keys(self) -> int:
        return self.latent + self.ctx.text + self.ctx.speaker + self.ctx.caption


PROFILES: dict[str, CtxProfile] = {
    # text tokens / speaker patches (ref seconds * 25 / 4) / caption tokens
    "a": CtxProfile(64, 64, 16),  # covers the bench inputs (13-51 / 46 / 1-13)
    "b": CtxProfile(256, 256, 64),  # long text, ~40 s reference
}
S_BUCKETS_FULL = (
    32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 576, 640, 704, 768,
    896, 1024, 1152, 1280, 1408, 1536,
)
S_BUCKETS_DEV = (192, 320, 768)
BATCHES = (1, 2, 3)
# Largest latent bucket per batch size. A batch-3 package enumerated up to 1536 frames takes
# the ANE compiler 315 s and then runs on the CPU (13-ane.md 5-1); longer requests at batch 3
# fall back to MPS instead.
S_MAX_BY_BATCH = {1: 1536, 2: 1536, 3: 768}


def package_key(profile: str, batch: int) -> str:
    return f"{profile}_b{batch}"


def _shapes_for(batch: int, prof: CtxProfile, buckets: tuple[int, ...]) -> list[Shape]:
    # +i keeps every input's enumerated shape unique per combination (see module docstring).
    return [
        Shape(batch, s, CtxProfile(prof.text + i, prof.speaker + i, prof.caption + i))
        for i, s in enumerate(buckets)
        if s <= S_MAX_BY_BATCH.get(batch, max(buckets))
    ]


def shape_packages(name: str) -> dict[str, list[Shape]]:
    """Package key -> shape list (batch and profile fixed, latent length enumerated)."""
    if name == "dev":
        buckets, profiles = S_BUCKETS_DEV, ("a",)
    elif name == "full":
        buckets, profiles = S_BUCKETS_FULL, ("a", "b")
    else:
        raise ValueError(f"unknown ANE shape set {name!r} (expected dev|full)")
    return {
        package_key(p, b): _shapes_for(b, PROFILES[p], buckets) for p in profiles for b in BATCHES
    }


# --------------------------------------------------------------------------------------
# Export wrapper
# --------------------------------------------------------------------------------------


def _rms_stream(x: torch.Tensor, eps: float, weight: torch.Tensor | None = None) -> torch.Tensor:
    """RMS normalisation of the residual stream, computed on x/64 so fp16 x*x cannot overflow."""
    xs = x * _PRESCALE
    r = torch.rsqrt((xs * xs).mean(dim=-1, keepdim=True) + eps * (_PRESCALE * _PRESCALE))
    out = xs * r
    return out if weight is None else out * weight


def _rms_heads(x: torch.Tensor, eps: float, weight: torch.Tensor) -> torch.Tensor:
    """q/k RMSNorm over head_dim (values are O(1); no prescale needed)."""
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * weight


# The ANE evaluates silu / sigmoid / tanh through lookup tables (relative error 1.5e-3 /
# 2.3e-3 / 6e-4 vs 3e-4 / 2e-4 / 2e-4 on the GPU, 13-ane.md 3-3). The bias compounds over
# 12 blocks into a 1e-1 error on the velocity, so the activations are written with exp
# (accurate to 4.5e-4 on the ANE). The clamp keeps exp(-x) below the fp16 limit.
_ACT_MODE = os.environ.get("IRODORI_ANE_ACT", "exp")  # exp | lut


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    if _ACT_MODE == "lut":
        return torch.sigmoid(x)
    return 1.0 / (1.0 + torch.exp(-torch.clamp(x, -11.0, 11.0)))


def _silu(x: torch.Tensor) -> torch.Tensor:
    if _ACT_MODE == "lut":
        return F.silu(x)
    return x * _sigmoid(x)


def _tanh(x: torch.Tensor) -> torch.Tensor:
    if _ACT_MODE == "lut":
        return torch.tanh(x)
    return 2.0 * _sigmoid(2.0 * x) - 1.0


class AneStepModule(nn.Module):
    """Same math as ``TextToLatentRFDiT.forward_with_encoded_conditions`` with the context KV
    projections done inside (per step) and the layout rules from the module docstring."""

    def __init__(self, model: TextToLatentRFDiT):
        super().__init__()
        cfg = model.cfg
        if not (cfg.use_speaker_condition_resolved and cfg.use_caption_condition):
            raise RuntimeError("AneStepModule expects speaker + caption conditioning (this checkpoint).")
        self.cond_module = model.cond_module
        self.in_proj = model.in_proj
        self.blocks = model.blocks
        self.out_norm = model.out_norm
        self.out_proj = model.out_proj
        self.heads = int(cfg.num_heads)
        self.head_dim = int(cfg.model_dim) // self.heads
        self.eps = float(cfg.norm_eps)
        self.attn_scale = 1.0 / float(self.head_dim) ** 0.5
        rot = torch.zeros(self.head_dim, self.head_dim)
        for i in range(self.head_dim // 2):
            rot[2 * i + 1, 2 * i] = -1.0  # out[2i]   = -x[2i+1]
            rot[2 * i, 2 * i + 1] = 1.0  # out[2i+1] =  x[2i]
        self.register_buffer("rot_pairs", rot, persistent=False)

    def _rope_half(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        # x: (B, S, H, hd); the first H/2 heads are rotated (JointAttention._apply_rotary_half).
        half = self.heads // 2
        x_rot = x[:, :, :half]
        x_pass = x[:, :, half:]
        cos = rope[None, :, None, :, 0]
        sin = rope[None, :, None, :, 1]
        out = x_rot * cos + torch.matmul(x_rot, self.rot_pairs) * sin
        return torch.cat([out, x_pass], dim=2)

    def _adaln(
        self, m: LowRankAdaLN, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cond is per token (B, S, 3*C).
        shift, scale, gate = cond.chunk(3, dim=-1)
        shift = m.shift_up(m.shift_down(_silu(shift))) + shift
        scale = m.scale_up(m.scale_down(_silu(scale))) + scale
        gate = m.gate_up(m.gate_down(_silu(gate))) + gate
        h = _rms_stream(x, m.eps) * (1.0 + scale) + shift
        return h, _tanh(gate)

    def _attention(
        self,
        att: JointAttention,
        h: torch.Tensor,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor,
        caption_state: torch.Tensor,
        attn_mask: torch.Tensor,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        hh = (self.heads, self.head_dim)
        kw = att.k_norm.weight
        q = _rms_heads(att.wq(h).unflatten(-1, hh), self.eps, att.q_norm.weight)
        k = _rms_heads(att.wk(h).unflatten(-1, hh), self.eps, kw)
        q = self._rope_half(q, rope)
        k_self = self._rope_half(k, rope).flatten(2)
        v_self = att.wv(h)
        k_text = _rms_heads(att.wk_text(text_state).unflatten(-1, hh), self.eps, kw).flatten(2)
        v_text = att.wv_text(text_state)
        k_spk = _rms_heads(att.wk_speaker(speaker_state).unflatten(-1, hh), self.eps, kw).flatten(2)
        v_spk = att.wv_speaker(speaker_state)
        k_cap = _rms_heads(att.wk_caption(caption_state).unflatten(-1, hh), self.eps, kw).flatten(2)
        v_cap = att.wv_caption(caption_state)
        keys = torch.cat([k_self, k_text, k_spk, k_cap], dim=1).unflatten(-1, hh).transpose(1, 2)
        vals = torch.cat([v_self, v_text, v_spk, v_cap], dim=1).unflatten(-1, hh).transpose(1, 2)
        qh = q.transpose(1, 2) * self.attn_scale
        scores = torch.matmul(qh, keys.transpose(-1, -2)) + attn_mask
        y = torch.matmul(torch.softmax(scores, dim=-1), vals)
        y = y.transpose(1, 2).flatten(2)
        y = y * _sigmoid(att.gate(h))
        return att.wo(y)

    def forward(
        self,
        x_t: torch.Tensor,
        t_embed: torch.Tensor,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor,
        caption_state: torch.Tensor,
        mask_self: torch.Tensor,
        mask_text: torch.Tensor,
        mask_speaker: torch.Tensor,
        mask_caption: torch.Tensor,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = torch.cat([mask_self, mask_text, mask_speaker, mask_caption], dim=-1)
        cm = self.cond_module  # Sequential(Linear, SiLU, Linear, SiLU, Linear)
        cond = cm[4](_silu(cm[2](_silu(cm[0](t_embed)))))
        x = self.in_proj(x_t)
        for blk in self.blocks:
            h, gate = self._adaln(blk.attention_adaln, x, cond)
            x = x + gate * self._attention(
                blk.attention, h, text_state, speaker_state, caption_state, attn_mask, rope
            )
            h, gate = self._adaln(blk.mlp_adaln, x, cond)
            mlp = blk.mlp
            x = x + gate * mlp.w2(_silu(mlp.w1(h)) * mlp.w3(h))
        x = _rms_stream(x, self.out_norm.eps, self.out_norm.weight)
        return self.out_proj(x)


# --------------------------------------------------------------------------------------
# Export + on-disk cache
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dims:
    latent_in: int
    t_embed: int
    text: int
    speaker: int
    caption: int
    head_dim: int

    @classmethod
    def from_model(cls, model: TextToLatentRFDiT) -> Dims:
        cfg = model.cfg
        return cls(
            latent_in=int(cfg.patched_latent_dim),
            t_embed=int(cfg.timestep_embed_dim),
            text=int(cfg.text_dim),
            speaker=int(cfg.speaker_dim),
            caption=int(cfg.caption_dim_resolved),
            head_dim=int(cfg.model_dim) // int(cfg.num_heads),
        )


def input_shapes(shape: Shape, dims: Dims) -> dict[str, tuple[int, ...]]:
    b, s, c = shape.batch, shape.latent, shape.ctx
    return {
        "x_t": (b, s, dims.latent_in),
        "t_embed": (b, s, dims.t_embed),
        "text_state": (b, c.text, dims.text),
        "speaker_state": (b, c.speaker, dims.speaker),
        "caption_state": (b, c.caption, dims.caption),
        "mask_self": (b, 1, 1, s),
        "mask_text": (b, 1, 1, c.text),
        "mask_speaker": (b, 1, 1, c.speaker),
        "mask_caption": (b, 1, 1, c.caption),
        "rope": (s, dims.head_dim, 2),
    }


def _weight_fingerprint(model: TextToLatentRFDiT) -> str:
    h = hashlib.sha256()
    for t in (
        model.in_proj.weight,
        model.out_proj.weight,
        model.blocks[0].attention.wq.weight,
        model.blocks[-1].mlp.w2.weight,
    ):
        h.update(t.detach().to("cpu", torch.float16).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _shape_json(shapes: list[Shape]) -> list:
    return [(s.batch, s.latent, s.ctx.__dict__) for s in shapes]


def cache_key(model: TextToLatentRFDiT, shapes: list[Shape]) -> str:
    """Per-package key: a change in one package's shape list leaves the others cached."""
    payload = {
        "wrapper": WRAPPER_VERSION,
        "dims": Dims.from_model(model).__dict__,
        "layers": len(model.blocks),
        "shapes": _shape_json(shapes),
        "weights": _weight_fingerprint(model),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def default_cache_dir() -> Path:
    return Path(os.environ.get("IRODORI_OPT_ANE_CACHE_DIR", "~/.cache/irodori-tts/ane")).expanduser()


def _cpu_wrapper(model: TextToLatentRFDiT) -> AneStepModule:
    parts = nn.ModuleDict(
        {
            "cond_module": model.cond_module,
            "in_proj": model.in_proj,
            "blocks": model.blocks,
            "out_norm": model.out_norm,
            "out_proj": model.out_proj,
        }
    )
    cpu_parts = copy.deepcopy(parts).to("cpu", torch.float32).eval()
    for p in cpu_parts.parameters():
        p.requires_grad_(False)

    class _Shim:  # minimal stand-in so AneStepModule can borrow the CPU copies
        cfg = model.cfg
        cond_module = cpu_parts["cond_module"]
        in_proj = cpu_parts["in_proj"]
        blocks = cpu_parts["blocks"]
        out_norm = cpu_parts["out_norm"]
        out_proj = cpu_parts["out_proj"]

    return AneStepModule(_Shim()).eval()  # type: ignore[arg-type]


def export_package(
    model: TextToLatentRFDiT, shapes: list[Shape], mlpackage: Path, log: bool = True
) -> None:
    """torch.export the step (batch fixed, latent/context lengths symbolic) on a CPU fp32 copy and
    convert it with one enumerated shape per latent bucket."""
    import coremltools as ct

    dims = Dims.from_model(model)
    batches = {s.batch for s in shapes}
    if len(batches) != 1:
        raise ValueError("a package must have a single batch size")
    t0 = time.perf_counter()
    wrapper = _cpu_wrapper(model)
    per_input = {name: [input_shapes(s, dims)[name] for s in shapes] for name in INPUT_NAMES}
    example = {name: torch.randn(per_input[name][0]) * 0.1 for name in INPUT_NAMES}
    for name in ("mask_self", "mask_text", "mask_speaker", "mask_caption"):
        example[name].zero_()

    dyn: dict[str, dict[int, object]] = {name: {} for name in INPUT_NAMES}
    if len(shapes) > 1:
        lat = sorted({s.latent for s in shapes})
        d_s = torch.export.Dim("S", min=min(lat), max=max(lat))
        dyn["x_t"][1] = d_s
        dyn["t_embed"][1] = d_s
        dyn["mask_self"][3] = d_s
        dyn["rope"][0] = d_s
        for state, mask, attr in (
            ("text_state", "mask_text", "text"),
            ("speaker_state", "mask_speaker", "speaker"),
            ("caption_state", "mask_caption", "caption"),
        ):
            lens = sorted({getattr(s.ctx, attr) for s in shapes})
            d = torch.export.Dim(f"L_{attr}", min=min(lens), max=max(lens))
            dyn[state][1] = d
            dyn[mask][3] = d
    with torch.no_grad():
        exported = torch.export.export(
            wrapper, (), example, dynamic_shapes=dict(dyn)
        ).run_decompositions({})
    if log:
        print(f"[ane] exported step on CPU in {time.perf_counter() - t0:.1f} s", flush=True)

    inputs = []
    for name in INPUT_NAMES:
        lst = per_input[name]
        shape = lst[0] if len(lst) == 1 else ct.EnumeratedShapes(shapes=lst, default=lst[0])
        inputs.append(ct.TensorType(name=name, shape=shape, dtype=np.float16))
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        exported,
        inputs=inputs,
        outputs=[ct.TensorType(name="v", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        shutil.rmtree(mlpackage)
    mlmodel.save(str(mlpackage))
    if log:
        print(
            f"[ane] converted {len(shapes)} shapes -> {mlpackage.name} in "
            f"{time.perf_counter() - t0:.1f} s",
            flush=True,
        )
    del wrapper, exported, mlmodel


def ensure_packages(
    model: TextToLatentRFDiT,
    shapes_name: str,
    cache_dir: Path,
    log: bool = True,
    only: set[str] | None = None,
) -> tuple[dict[str, Path], dict[str, list[Shape]]]:
    """Return {package_key: compiled .mlmodelc path}, exporting/compiling on a cache miss."""
    import coremltools as ct

    packages = shape_packages(shapes_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    compiled: dict[str, Path] = {}
    for pkg_key, shapes in packages.items():
        if only is not None and pkg_key not in only:
            continue
        stem = f"{pkg_key}_{cache_key(model, shapes)}"
        mlpackage = cache_dir / f"{stem}.mlpackage"
        mlmodelc = cache_dir / f"{stem}.mlmodelc"
        if not mlmodelc.exists():
            if not mlpackage.exists():
                export_package(model, shapes, mlpackage, log=log)
            t0 = time.perf_counter()
            ct.models.utils.compile_model(str(mlpackage), destination_path=str(mlmodelc))
            if log:
                print(f"[ane] compiled {mlmodelc.name} in {time.perf_counter() - t0:.1f} s", flush=True)
            (cache_dir / f"{stem}.json").write_text(
                json.dumps({"wrapper": WRAPPER_VERSION, "package": pkg_key, "shapes": _shape_json(shapes)}, indent=1)
            )
        compiled[pkg_key] = mlmodelc
    return compiled, packages


# --------------------------------------------------------------------------------------
# Parent-side runner
# --------------------------------------------------------------------------------------


@dataclass
class AneContext:
    key: str
    shape: Shape
    latent_len: int  # real S
    package: str


class AneStepRunner:
    """Owns the worker process and the shared blocks. One instance per loaded model."""

    def __init__(
        self,
        model: TextToLatentRFDiT,
        *,
        shapes_name: str = "dev",
        compute_units: str = "ne",
        cache_dir: Path | None = None,
        log: bool = True,
    ) -> None:
        self.dims = Dims.from_model(model)
        self.log = log
        t0 = time.perf_counter()
        compiled, packages = ensure_packages(
            model, shapes_name, cache_dir or default_cache_dir(), log=log
        )
        self.packages = packages
        self._profile_keys = [k for k in PROFILES if any(pk.startswith(k + "_") for pk in packages)]
        self._profile_keys.sort(key=lambda k: PROFILES[k].text + PROFILES[k].speaker + PROFILES[k].caption)

        all_shapes = [s for v in packages.values() for s in v]
        big = Shape(
            max(BATCHES),
            max(s.latent for s in all_shapes),
            CtxProfile(
                max(s.ctx.text for s in all_shapes),
                max(s.ctx.speaker for s in all_shapes),
                max(s.ctx.caption for s in all_shapes),
            ),
        )
        block_shape = input_shapes(big, self.dims)
        block_shape["v"] = block_shape["x_t"]
        self._blocks: dict[str, shared_memory.SharedMemory] = {}
        for name, shape in block_shape.items():
            self._blocks[name] = shared_memory.SharedMemory(
                create=True, size=int(np.prod(shape)) * 2
            )

        ctx = mp.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        from . import ane_worker  # noqa: F401 - target module for the spawned process

        self._proc = ctx.Process(
            target=ane_worker.main,
            args=(child_conn, {n: b.name for n, b in self._blocks.items()}, compute_units),
            daemon=True,
            name="irodori-ane",
        )
        self._proc.start()
        child_conn.close()
        self._call(("register", {k: str(p) for k, p in compiled.items()}))
        self._rope_cache: dict[int, np.ndarray] = {}
        self._pending: AneContext | None = None
        self._ctx_seq = 0
        self.stats = {"steps": 0, "predict_sec": 0.0, "wait_sec": 0.0, "ctx": 0}
        if log:
            print(
                f"[ane] runner ready ({shapes_name}, {compute_units}, {len(compiled)} packages) "
                f"in {time.perf_counter() - t0:.1f} s",
                flush=True,
            )

    def preload(self, keys: list[str] | None = None) -> None:
        for key in keys or list(self.packages):
            info = self._call(("load", key))
            if self.log and info and info.get("load_sec", 0.0) > 0.0:
                print(f"[ane] worker loaded {key} in {info['load_sec']:.1f} s", flush=True)

    # -- protocol helpers
    def _call(self, msg: tuple) -> dict | None:
        self._conn.send(msg)
        return self._recv()

    def _recv(self) -> dict | None:
        status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"ANE worker error:\n{payload}")
        return payload

    def _view(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        return np.ndarray(shape, dtype=np.float16, buffer=self._blocks[name].buf)

    # -- shape selection
    def find_shape(
        self, batch: int, latent_len: int, lt: int, ls: int, lc: int
    ) -> tuple[Shape, str] | None:
        for prof_key in self._profile_keys:
            prof = PROFILES[prof_key]
            if lt > prof.text or ls > prof.speaker or lc > prof.caption:
                continue
            pkg = package_key(prof_key, batch)
            shapes = self.packages.get(pkg)
            if not shapes:
                continue
            for s in shapes:  # ordered by latent bucket
                if s.latent >= latent_len:
                    return s, pkg
        return None

    def _rope(self, latent_bucket: int) -> np.ndarray:
        rope = self._rope_cache.get(latent_bucket)
        if rope is None:
            table = precompute_freqs_cis(self.dims.head_dim, latent_bucket)
            if table.is_complex():
                table = torch.stack([table.real, table.imag], dim=-1)
            # (S, hd/2, 2) -> (S, hd, 2): every pair (2i, 2i+1) shares cos_i / sin_i.
            rope = table.repeat_interleave(2, dim=1).to(torch.float16).numpy()
            self._rope_cache[latent_bucket] = rope
        return rope

    # -- per-request context
    def make_context(
        self,
        *,
        latent_len: int,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> AneContext | None:
        """Pad the encoded conditions of one CFG bundle into a bucket and ship them to the worker.
        Returns None when no enumerated shape fits (caller falls back to the GPU path)."""
        batch = int(text_state.shape[0])
        found = self.find_shape(
            batch, int(latent_len), int(text_state.shape[1]), int(speaker_state.shape[1]),
            int(caption_state.shape[1]),
        )
        if found is None:
            return None
        shape, pkg = found
        want = input_shapes(shape, self.dims)

        def _pad_state(name: str, state: torch.Tensor) -> None:
            arr = self._view(name, want[name])
            arr.fill(0)
            arr[:, : state.shape[1]] = state.detach().to("cpu", torch.float16).numpy()

        def _mask(name: str, m: torch.Tensor | None, valid: int | None = None) -> None:
            arr = self._view(name, want[name])
            arr.fill(MASK_NEG)
            if m is None:
                arr[..., :valid] = 0.0
            else:
                mb = m.detach().to("cpu").numpy().astype(bool)
                flat = arr.reshape(batch, -1)
                flat[:, : mb.shape[1]][mb] = 0.0

        _pad_state("text_state", text_state)
        _pad_state("speaker_state", speaker_state)
        _pad_state("caption_state", caption_state)
        _mask("mask_self", None, int(latent_len))
        _mask("mask_text", text_mask)
        _mask("mask_speaker", speaker_mask)
        _mask("mask_caption", caption_mask)
        np.copyto(self._view("rope", want["rope"]), self._rope(shape.latent))

        self._ctx_seq += 1
        key = f"ctx{self._ctx_seq}"
        shapes = {n: want[n] for n in CTX_INPUT_NAMES}
        info = self._call(("ctx", key, pkg, shapes))
        if self.log and info and info.get("load_sec", 0.0) > 0.0:
            print(f"[ane] worker loaded {pkg} in {info['load_sec']:.1f} s", flush=True)
        self.stats["ctx"] += 1
        return AneContext(key=key, shape=shape, latent_len=int(latent_len), package=pkg)

    def drop_context(self, ctx: AneContext) -> None:
        self._call(("drop", ctx.key))

    # -- one step
    def submit(self, ctx: AneContext, x_t: torch.Tensor, t_value: float) -> None:
        """Queue one forward. ``x_t`` is (B, S, latent_in) on any device; ``t_value`` the scalar t
        (rounded to fp16 like the GPU path). Non-blocking apart from the x_t host copy."""
        if self._pending is not None:
            raise RuntimeError("ANE step already pending")
        shape = ctx.shape
        want = input_shapes(shape, self.dims)
        x_arr = self._view("x_t", want["x_t"])
        x_arr.fill(0)
        x_arr[:, : ctx.latent_len] = x_t.detach().to("cpu", torch.float16).numpy()
        t_fp16 = torch.full((shape.batch,), float(t_value), dtype=torch.float16)
        t_embed = get_timestep_embedding(t_fp16.float(), self.dims.t_embed).to(torch.float16).numpy()
        np.copyto(self._view("t_embed", want["t_embed"]), t_embed[:, None, :])
        self._conn.send(("step", ctx.key, want["x_t"], want["t_embed"]))
        self._pending = ctx
        self._t_submit = time.perf_counter()

    def wait(self) -> torch.Tensor:
        """Block for the pending step; returns v (B, S, latent_in) fp16 on the CPU."""
        ctx = self._pending
        if ctx is None:
            raise RuntimeError("no ANE step pending")
        info = self._recv()
        self._pending = None
        self.stats["steps"] += 1
        self.stats["predict_sec"] += float(info["predict_sec"])
        self.stats["wait_sec"] += time.perf_counter() - self._t_submit
        out = self._view("v", tuple(info["shape"]))[:, : ctx.latent_len]
        return torch.from_numpy(np.array(out, copy=True))

    def step(self, ctx: AneContext, x_t: torch.Tensor, t_value: float) -> torch.Tensor:
        self.submit(ctx, x_t, t_value)
        return self.wait()

    def reset_stats(self) -> dict:
        out = dict(self.stats)
        self.stats = {"steps": 0, "predict_sec": 0.0, "wait_sec": 0.0, "ctx": 0}
        return out

    # -- lifecycle
    def shutdown(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.is_alive():
            try:
                self._conn.send(("quit",))
                self._conn.recv()
            except (OSError, EOFError):
                pass
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
        for block in getattr(self, "_blocks", {}).values():
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
                pass
        self._blocks = {}

    def __del__(self) -> None:  # best effort
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------
# Singleton per model (created lazily by the sampler or eagerly by the runtime)
# --------------------------------------------------------------------------------------

_RUNNERS: dict[int, AneStepRunner] = {}


def get_ane_runner(model: TextToLatentRFDiT, opt) -> AneStepRunner | None:  # noqa: ANN001
    """Return the runner for ``model`` (building it on first use) or None when ANE is off."""
    if not getattr(opt, "ane", False):
        return None
    runner = _RUNNERS.get(id(model))
    if runner is None:
        runner = AneStepRunner(
            model,
            shapes_name=str(opt.ane_shapes),
            compute_units=str(opt.ane_units),
            log=bool(opt.ane_log),
        )
        _RUNNERS[id(model)] = runner
    return runner


def shutdown_ane_runner(model: TextToLatentRFDiT | None = None) -> None:
    keys = list(_RUNNERS) if model is None else [id(model)]
    for k in keys:
        runner = _RUNNERS.pop(k, None)
        if runner is not None:
            runner.shutdown()
