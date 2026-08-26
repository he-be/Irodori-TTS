import os as _os

# Local deployment: reduce caching-allocator fragmentation so `memory_reserved` stays close to
# `memory_allocated` (must be set before the CUDA allocator initializes).
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

"""Irodori-TTS package: text-conditioned RF diffusion over DACVAE latents."""

from .config import ModelConfig, TrainConfig
from .lora import LORA_TARGET_PRESETS
from .model import TextToLatentRFDiT
from .tokenizer import PretrainedTextTokenizer

__all__ = [
    "LORA_TARGET_PRESETS",
    "ModelConfig",
    "PretrainedTextTokenizer",
    "TextToLatentRFDiT",
    "TrainConfig",
]
