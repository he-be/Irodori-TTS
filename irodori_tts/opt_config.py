"""Local inference optimization switches (Apple Silicon / Metal build).

This module is intentionally environment-specific: one Mac (M3 Pro, 18 GB unified
memory), the MPS backend, batch size 1. Every switch can be toggled through an
``IRODORI_OPT_*`` environment variable so A/B benchmarks need no code changes:

    IRODORI_OPT_REUSE_COND=0       disable duration->sampling condition reuse
    IRODORI_OPT_CROP_TEXT=0        disable text/caption padding crop
    IRODORI_OPT_FAST_SAMPLER=0     disable sync-free sampler + precomputed masks
    IRODORI_OPT_CODEC_FOLD_WN=0    disable codec weight-norm folding
    IRODORI_OPT_REF_CACHE=0        disable reference latent / speaker state cache
    IRODORI_OPT_CPU_CAST=0         disable cast-on-CPU model loading
    IRODORI_OPT_WATERMARK=1        re-enable SilentCipher watermarking (default: off; the
                                   package is not installed in this build)
    IRODORI_OPT_COMPILE_DIT=1      torch.compile the DiT forward (inductor MPS backend, -17% wall,
                                   ~20 s first call; the Gradio apps turn it on)
    IRODORI_OPT_COMPILE_CODEC=1    torch.compile the codec decoder (-30% decode, ~4 s first call;
                                   the Gradio apps turn it on)
    IRODORI_OPT_DECODE_CHUNK=96    codec decode window in latent frames, 25 fps (default 0 = whole
                                   utterance: on unified memory the overlap recompute only costs time)
    IRODORI_OPT_DECODE_OVERLAP=16  overlap per side in latent frames (receptive field ~10)
    IRODORI_OPT_DECODE_AUTOCAST=0  disable reduced-precision autocast for codec *decode*
                                   (default on; encode stays fp32)
    IRODORI_OPT_DECODE_AUTOCAST_DTYPE=bf16|fp16  autocast dtype for codec decode (default fp16)
    IRODORI_OPT_ENCODE_CHUNK=96    reference encode window in latent frames (0 = whole clip)
    IRODORI_OPT_ENCODE_OVERLAP=32  overlap per side for reference encode
    IRODORI_OPT_MPS_LIMIT_MB=0     cap for the MPS allocator via
                                   torch.mps.set_per_process_memory_fraction (0 = torch default)
    IRODORI_OPT_EMPTY_CACHE=1      torch.mps.empty_cache() after every request (default: off)
    IRODORI_OPT_SKIP_INIT=0        keep the random weight init that the checkpoint overwrites
    IRODORI_OPT_PREBAKE=0          ignore the prebaked runtime bundle (default: use it when present)
    IRODORI_OPT_PREBAKE_DIR=path   where prebaked bundles live (default: ~/.cache/irodori-tts/prebake)
    IRODORI_OPT_LOAD_PARALLEL=0    load weights/tokenizer serially instead of overlapping them with
                                   the transformers import (default: overlapped)
    IRODORI_OPT_ROPE_REAL=0        use the complex-number RoPE instead of the real-valued one

The values are read once at first access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "off", "no", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    return value if value in choices else default


@dataclass(frozen=True)
class OptConfig:
    reuse_conditions: bool = True
    crop_text: bool = True
    fast_sampler: bool = True
    codec_fold_weight_norm: bool = True
    reference_cache: bool = True
    reference_cache_entries: int = 8
    cpu_cast: bool = True
    watermark: bool = False
    compile_dit: bool = False
    compile_codec: bool = False
    decode_chunk_frames: int = 0
    decode_overlap_frames: int = 16
    decode_autocast: bool = True
    decode_autocast_dtype: str = "fp16"
    encode_chunk_frames: int = 96
    encode_overlap_frames: int = 32
    mps_limit_mb: int = 0
    empty_cache_after_request: bool = False
    skip_init: bool = True
    prebake: bool = True
    load_parallel: bool = True
    rope_real: bool = True

    @classmethod
    def from_env(cls) -> OptConfig:
        return cls(
            reuse_conditions=_env_bool("IRODORI_OPT_REUSE_COND", True),
            crop_text=_env_bool("IRODORI_OPT_CROP_TEXT", True),
            fast_sampler=_env_bool("IRODORI_OPT_FAST_SAMPLER", True),
            codec_fold_weight_norm=_env_bool("IRODORI_OPT_CODEC_FOLD_WN", True),
            reference_cache=_env_bool("IRODORI_OPT_REF_CACHE", True),
            reference_cache_entries=max(1, _env_int("IRODORI_OPT_REF_CACHE_ENTRIES", 8)),
            cpu_cast=_env_bool("IRODORI_OPT_CPU_CAST", True),
            watermark=_env_bool("IRODORI_OPT_WATERMARK", False),
            compile_dit=_env_bool("IRODORI_OPT_COMPILE_DIT", False),
            compile_codec=_env_bool("IRODORI_OPT_COMPILE_CODEC", False),
            decode_chunk_frames=max(0, _env_int("IRODORI_OPT_DECODE_CHUNK", 0)),
            decode_overlap_frames=max(0, _env_int("IRODORI_OPT_DECODE_OVERLAP", 16)),
            decode_autocast=_env_bool("IRODORI_OPT_DECODE_AUTOCAST", True),
            decode_autocast_dtype=_env_str(
                "IRODORI_OPT_DECODE_AUTOCAST_DTYPE", "fp16", ("fp16", "bf16")
            ),
            encode_chunk_frames=max(0, _env_int("IRODORI_OPT_ENCODE_CHUNK", 96)),
            encode_overlap_frames=max(0, _env_int("IRODORI_OPT_ENCODE_OVERLAP", 32)),
            mps_limit_mb=max(0, _env_int("IRODORI_OPT_MPS_LIMIT_MB", 0)),
            empty_cache_after_request=_env_bool("IRODORI_OPT_EMPTY_CACHE", False),
            skip_init=_env_bool("IRODORI_OPT_SKIP_INIT", True),
            prebake=_env_bool("IRODORI_OPT_PREBAKE", True),
            load_parallel=_env_bool("IRODORI_OPT_LOAD_PARALLEL", True),
            rope_real=_env_bool("IRODORI_OPT_ROPE_REAL", True),
        )

    def describe(self) -> str:
        return " ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


_CONFIG: OptConfig | None = None


def get_opt_config() -> OptConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = OptConfig.from_env()
    return _CONFIG


def set_opt_config(config: OptConfig) -> None:
    global _CONFIG
    _CONFIG = config
