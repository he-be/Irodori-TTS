"""Local inference optimization switches.

This module is intentionally environment-specific (single machine, batch size 1).
Every switch can be toggled through an ``IRODORI_OPT_*`` environment variable so
that A/B benchmarks can be run without code changes:

    IRODORI_OPT_REUSE_COND=0       disable duration->sampling condition reuse
    IRODORI_OPT_CROP_TEXT=0        disable text/caption padding crop
    IRODORI_OPT_FAST_SAMPLER=0     disable sync-free sampler + precomputed masks
    IRODORI_OPT_CUDA_GRAPH=0       disable CUDA Graph replay of the RF step
    IRODORI_OPT_GRAPH_BUCKET=32    latent length bucket for graph signatures
    IRODORI_OPT_GRAPH_MAX_ENTRIES=6
    IRODORI_OPT_GRAPH_MAX_STATIC_MB=256  byte budget for graph static buffers (0 = count-only LRU)
    IRODORI_OPT_GRAPH_SHARED_POOL=1      share one private pool across graphs (default: one pool per graph,
                                         so an eviction actually returns its VRAM)
    IRODORI_OPT_GRAPH_MAX_LATENT=384     run the step eager above this many latent frames (0 = no limit)
    IRODORI_OPT_GRAPH_RELEASE_POOL=0     skip empty_cache() after a graph eviction
    IRODORI_OPT_CODEC_FOLD_WN=0    disable codec weight-norm folding
    IRODORI_OPT_REF_CACHE=0        disable reference latent / speaker state cache
    IRODORI_OPT_CPU_CAST=0         disable cast-on-CPU model loading
    IRODORI_OPT_WATERMARK=1        re-enable SilentCipher watermarking (default: off)
    IRODORI_OPT_TEXT_BUCKET=16     text/caption token bucket used with CUDA Graph
    IRODORI_OPT_SPEAKER_BUCKET=64  speaker (patched) token bucket used with CUDA Graph
    IRODORI_OPT_COMPILE_DIT=1      torch.compile the DiT forward (inside the CUDA Graph)
    IRODORI_OPT_COMPILE_CODEC=1    torch.compile the codec decoder
    IRODORI_OPT_DECODE_CHUNK=96    codec decode window in latent frames, 25 fps (0 = whole utterance)
    IRODORI_OPT_DECODE_OVERLAP=16  overlap per side in latent frames (receptive field ~10)
    IRODORI_OPT_DECODE_AUTOCAST=0  disable bf16 autocast for codec *decode* (default on; encode stays fp32)
    IRODORI_OPT_TE_DEVICE=cpu      keep the pretrained text/caption backbone (ModernBERT, ~0.6 GB fp16) on
                                   the CPU in fp32; only the projected states go to the model device
                                   (default: model device). Frees VRAM at ~100-300 ms per request.
    IRODORI_OPT_CODEC_CUDNN=auto   cuDNN/MIOpen for codec convs: 1 = always, 0 = never (torch im2col+GEMM),
                                   auto = off on ROCm (MIOpen has only a naive kernel for dilated
                                   conv1d on gfx900, see docs/experiments/12)
    IRODORI_OPT_ENCODE_CHUNK=96    reference encode window in latent frames (0 = whole clip)
    IRODORI_OPT_ENCODE_OVERLAP=32  overlap per side for reference encode
    IRODORI_OPT_VRAM_LIMIT_MB=3840 hard cap for the torch caching allocator (0 = none); CUDA context (~0.5 GB) is extra
                                   3840 covers CUDA Graph pools; with IRODORI_OPT_CUDA_GRAPH=0 the
                                   steady state fits in 3072 (see docs/experiments/10)
    IRODORI_OPT_EMPTY_CACHE=1      release cached blocks after every request (default: off)
    IRODORI_OPT_SKIP_INIT=0        keep the random weight init that the checkpoint overwrites (default: skipped)
    IRODORI_OPT_PREBAKE=0          ignore the prebaked runtime bundle (default: use it when present)
    IRODORI_OPT_PREBAKE_DIR=path   where prebaked bundles live (default: ~/.cache/irodori-tts/prebake)
    IRODORI_OPT_LOAD_PARALLEL=0    load weights/tokenizer serially instead of overlapping them with
                                   the transformers import (default: overlapped)

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


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in {"true", "on", "yes"}:
        val = "1"
    elif val in {"false", "off", "no"}:
        val = "0"
    return val if val in allowed else default


@dataclass(frozen=True)
class OptConfig:
    reuse_conditions: bool = True
    crop_text: bool = True
    fast_sampler: bool = True
    cuda_graph: bool = True
    graph_bucket: int = 32
    graph_max_entries: int = 6
    graph_max_static_mb: int = 256
    graph_shared_pool: bool = False
    graph_max_latent_frames: int = 384
    graph_release_pool_on_evict: bool = True
    graph_capture_after: int = 1
    codec_fold_weight_norm: bool = True
    reference_cache: bool = True
    reference_cache_entries: int = 8
    cpu_cast: bool = True
    watermark: bool = False
    text_bucket: int = 16
    speaker_bucket: int = 64
    compile_dit: bool = False
    compile_codec: bool = False
    decode_chunk_frames: int = 96
    decode_overlap_frames: int = 16
    decode_autocast_bf16: bool = True
    codec_cudnn: str = "auto"
    text_encoder_device: str = "model"
    encode_chunk_frames: int = 96
    encode_overlap_frames: int = 32
    vram_limit_mb: int = 3840
    empty_cache_after_request: bool = False
    skip_init: bool = True
    prebake: bool = True
    load_parallel: bool = True

    @classmethod
    def from_env(cls) -> OptConfig:
        return cls(
            reuse_conditions=_env_bool("IRODORI_OPT_REUSE_COND", True),
            crop_text=_env_bool("IRODORI_OPT_CROP_TEXT", True),
            fast_sampler=_env_bool("IRODORI_OPT_FAST_SAMPLER", True),
            cuda_graph=_env_bool("IRODORI_OPT_CUDA_GRAPH", True),
            graph_bucket=max(1, _env_int("IRODORI_OPT_GRAPH_BUCKET", 32)),
            graph_max_entries=max(1, _env_int("IRODORI_OPT_GRAPH_MAX_ENTRIES", 6)),
            graph_max_static_mb=max(0, _env_int("IRODORI_OPT_GRAPH_MAX_STATIC_MB", 256)),
            graph_shared_pool=_env_bool("IRODORI_OPT_GRAPH_SHARED_POOL", False),
            graph_max_latent_frames=max(0, _env_int("IRODORI_OPT_GRAPH_MAX_LATENT", 384)),
            graph_release_pool_on_evict=_env_bool("IRODORI_OPT_GRAPH_RELEASE_POOL", True),
            graph_capture_after=max(0, _env_int("IRODORI_OPT_GRAPH_CAPTURE_AFTER", 1)),
            codec_fold_weight_norm=_env_bool("IRODORI_OPT_CODEC_FOLD_WN", True),
            reference_cache=_env_bool("IRODORI_OPT_REF_CACHE", True),
            reference_cache_entries=max(1, _env_int("IRODORI_OPT_REF_CACHE_ENTRIES", 8)),
            cpu_cast=_env_bool("IRODORI_OPT_CPU_CAST", True),
            watermark=_env_bool("IRODORI_OPT_WATERMARK", False),
            text_bucket=max(1, _env_int("IRODORI_OPT_TEXT_BUCKET", 16)),
            speaker_bucket=max(1, _env_int("IRODORI_OPT_SPEAKER_BUCKET", 64)),
            compile_dit=_env_bool("IRODORI_OPT_COMPILE_DIT", False),
            compile_codec=_env_bool("IRODORI_OPT_COMPILE_CODEC", False),
            decode_chunk_frames=max(0, _env_int("IRODORI_OPT_DECODE_CHUNK", 96)),
            decode_overlap_frames=max(0, _env_int("IRODORI_OPT_DECODE_OVERLAP", 16)),
            decode_autocast_bf16=_env_bool("IRODORI_OPT_DECODE_AUTOCAST", True),
            codec_cudnn=_env_choice("IRODORI_OPT_CODEC_CUDNN", "auto", {"auto", "0", "1"}),
            text_encoder_device=_env_choice("IRODORI_OPT_TE_DEVICE", "model", {"model", "cpu"}),
            encode_chunk_frames=max(0, _env_int("IRODORI_OPT_ENCODE_CHUNK", 96)),
            encode_overlap_frames=max(0, _env_int("IRODORI_OPT_ENCODE_OVERLAP", 32)),
            vram_limit_mb=max(0, _env_int("IRODORI_OPT_VRAM_LIMIT_MB", 3840)),
            empty_cache_after_request=_env_bool("IRODORI_OPT_EMPTY_CACHE", False),
            skip_init=_env_bool("IRODORI_OPT_SKIP_INIT", True),
            prebake=_env_bool("IRODORI_OPT_PREBAKE", True),
            load_parallel=_env_bool("IRODORI_OPT_LOAD_PARALLEL", True),
        )

    def codec_use_cudnn(self) -> bool:
        if self.codec_cudnn == "1":
            return True
        if self.codec_cudnn == "0":
            return False
        import torch

        return not bool(getattr(torch.version, "hip", None))

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
