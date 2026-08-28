import os as _os

# Metal-only deployment (Apple Silicon, this machine). PyTorch's MPS backend can
# silently run unsupported ops on the CPU when PYTORCH_ENABLE_MPS_FALLBACK=1; this
# branch forbids that, so force it off before torch is imported and let an
# unsupported op raise instead.
_os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

"""Irodori-TTS package: text-conditioned RF diffusion over DACVAE latents (Metal / MPS build)."""

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
