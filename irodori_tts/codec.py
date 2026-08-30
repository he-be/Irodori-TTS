from __future__ import annotations

import contextlib
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from .fast_init import skip_random_init
from .opt_config import get_opt_config

_CODEC_DEFAULT = object()


def _import_dacvae():
    # Prefer installed package; fallback to local clone at ../dacvae.
    try:
        from dacvae import DACVAE
    except ImportError:
        local_repo = Path(__file__).resolve().parents[2] / "dacvae"
        if local_repo.exists():
            sys.path.insert(0, str(local_repo))
        from dacvae import DACVAE
    return DACVAE


def resolve_codec_weights(repo_id: str) -> str:
    """Local path of the DACVAE ``weights.pth`` for ``repo_id`` (downloads if needed)."""
    location = str(repo_id).strip()
    if location.startswith("hf://"):
        location = location[len("hf://") :]
    if not Path(location).exists() and "/" in location and not location.endswith(".pth"):
        try:
            resolved = hf_hub_download(repo_id=location, filename="weights.pth")
            print(f"[codec] dacvae: hf://{repo_id} -> {resolved}", flush=True)
            return resolved
        except Exception:
            # Let DACVAE.load surface a clearer error if this is not a valid path/repo.
            return location
    return location


def _load_dacvae_weights(dacvae_cls, location: str) -> torch.nn.Module:
    """Build a DACVAE from a ``weights.pth`` without the throwaway random init.

    ``audiotools``'s ``BaseModel.load`` first tries ``torch.package``, then builds
    the module (0.6 s of ``uniform_`` for 107 M params) and loads the state dict
    with ``strict=False``.  Here the state dict is loaded with ``strict=True``
    instead, which is what makes skipping the init safe: a tensor the checkpoint
    does not cover raises rather than staying uninitialized.
    """
    payload = torch.load(location, map_location="cpu", weights_only=True)
    if not (
        isinstance(payload, dict)
        and isinstance(payload.get("state_dict"), dict)
        and isinstance(payload.get("metadata"), dict)
    ):
        # Packaged (torch.package) or otherwise unusual checkpoint: let audiotools do it.
        return dacvae_cls.load(location)

    metadata = payload["metadata"]
    class_keys = set(inspect.signature(dacvae_cls).parameters)
    kwargs = {k: v for k, v in dict(metadata.get("kwargs", {})).items() if k in class_keys}
    with skip_random_init(get_opt_config().skip_init):
        model = dacvae_cls(**kwargs)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.metadata = metadata
    return model


def patchify_latent(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert latent from (B, T, D) -> (B, T//patch, D*patch).
    Extra tail tokens are dropped.
    """
    if patch_size <= 1:
        return latent
    bsz, seq_len, dim = latent.shape
    usable = (seq_len // patch_size) * patch_size
    latent = latent[:, :usable]
    latent = latent.reshape(bsz, usable // patch_size, dim * patch_size)
    return latent


def unpatchify_latent(patched: torch.Tensor, patch_size: int, latent_dim: int) -> torch.Tensor:
    """
    Convert latent from (B, T_p, D*patch) -> (B, T_p*patch, D).
    """
    if patch_size <= 1:
        return patched
    return patched.reshape(patched.shape[0], patched.shape[1] * patch_size, latent_dim)


def fold_weight_norm_(model: torch.nn.Module) -> int:
    """
    Fold legacy ``torch.nn.utils.weight_norm`` hooks into plain weights (in place).

    At inference the weight ``g * v / ||v||`` is recomputed on every forward by a
    pre-hook; folding it once yields bit-identical weights and removes those kernels.
    Only modules carrying a ``WeightNorm`` hook are touched.
    """
    from torch.nn.utils.weight_norm import WeightNorm, remove_weight_norm

    count = 0
    for module in model.modules():
        names = [
            hook.name for hook in module._forward_pre_hooks.values() if isinstance(hook, WeightNorm)
        ]
        for name in names:
            remove_weight_norm(module, name=name)
            count += 1
    return count


@dataclass
class DACVAECodec:
    model: torch.nn.Module
    sample_rate: int
    latent_dim: int
    device: torch.device
    dtype: torch.dtype
    deterministic_encode: bool
    deterministic_decode: bool
    normalize_db: float | None

    @classmethod
    def load(
        cls,
        repo_id: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        deterministic_encode: bool = True,
        deterministic_decode: bool = True,
        normalize_db: float | None = -16.0,
        fold_weight_norm: bool = True,
        prebaked_state: dict[str, torch.Tensor] | None = None,
        prebaked_kwargs: dict | None = None,
    ) -> DACVAECodec:
        DACVAE = _import_dacvae()

        if prebaked_state is not None:
            # Bundle path: the tensors are already folded, already in the target
            # dtype and already on the target device (see prebake.py).
            with skip_random_init(get_opt_config().skip_init):
                model = DACVAE(**dict(prebaked_kwargs or {}))
            model = model.eval()
            if fold_weight_norm:
                fold_weight_norm_(model)
            model.load_state_dict(prebaked_state, strict=True, assign=True)
            model = model.to(device)
        else:
            location = resolve_codec_weights(repo_id)
            model = _load_dacvae_weights(DACVAE, location).eval().to(device)
            if fold_weight_norm:
                folded = fold_weight_norm_(model)
                print(f"[codec] folded weight_norm on {folded} conv layers", flush=True)
            if dtype is not None:
                model = model.to(dtype=dtype)

        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "alpha"):
            decoder.alpha = 0.0
            if hasattr(decoder, "wm_model"):
                # Irodori checkpoints were trained without the DACVAE watermark branch.
                # Keep decode output mono while skipping that encode/decode path.
                def _watermark_passthrough(
                    x: torch.Tensor,
                    message: torch.Tensor | None = None,
                    _decoder=decoder,
                ) -> torch.Tensor:
                    del message
                    return _decoder.wm_model.encoder_block.forward_no_conv(x)

                decoder.watermark = _watermark_passthrough

        if deterministic_decode:
            cls._configure_deterministic_decode(model=model, device=device)

        model_dtype = next(model.parameters()).dtype
        # Infer latent dimension by encoding a tiny random signal.
        # This probe is load-bearing beyond the shape it reports: it is the first
        # conv on this device, and it fixes the cuDNN algorithm choice for the
        # encoder. Taking the latent dim from a manifest and skipping the probe
        # changes every later encode bit-for-bit (see 11-load-time.md).
        dummy = torch.zeros(1, 1, 2048, device=device, dtype=model_dtype)
        with torch.inference_mode():
            z = model.encode(dummy)  # (B, D, T)
        latent_dim = int(z.shape[1])
        return cls(
            model=model,
            sample_rate=int(model.sample_rate),
            latent_dim=latent_dim,
            device=torch.device(device),
            dtype=model_dtype,
            deterministic_encode=bool(deterministic_encode),
            deterministic_decode=bool(deterministic_decode),
            normalize_db=None if normalize_db is None else float(normalize_db),
        )

    @staticmethod
    def _configure_deterministic_decode(model: torch.nn.Module, device: str | torch.device) -> None:
        decoder = getattr(model, "decoder", None)
        wm_model = getattr(decoder, "wm_model", None)
        msg_processor = getattr(wm_model, "msg_processor", None)
        if msg_processor is None:
            return
        nbits = int(msg_processor.nbits)
        message_device = torch.device(device)

        def _fixed_message(batch_size: int) -> torch.Tensor:
            return torch.zeros((batch_size, nbits), dtype=torch.float32, device=message_device)

        wm_model.random_message = _fixed_message

    @staticmethod
    def _normalize_loudness(
        wav: torch.Tensor, sample_rate: int, target_db: float | None
    ) -> torch.Tensor:
        if target_db is None:
            return wav
        wav_device = wav.device
        wav = wav.to(dtype=torch.float32)
        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav[0]
            elif wav.shape[1] == 1:
                wav = wav[:, 0]
            else:
                wav = wav.mean(dim=0)
        if wav.ndim != 1:
            raise ValueError(
                "normalize_loudness expects a mono waveform with shape (T,) "
                f"or singleton-channel (1, T)/(T, 1), got {tuple(wav.shape)}"
            )

        try:
            from audiotools import AudioSignal
        except Exception as exc:
            raise RuntimeError(
                "audiotools is required when normalize_db is set. "
                "Install audiotools or disable normalize_db."
            ) from exc

        signal = AudioSignal(wav.unsqueeze(0).unsqueeze(0), int(sample_rate))
        signal.normalize(float(target_db))
        signal.ensure_max_of_audio()
        normalized = signal.audio_data
        if not isinstance(normalized, torch.Tensor):
            normalized = torch.as_tensor(normalized)
        normalized = normalized.to(dtype=torch.float32, device=wav_device)
        normalized = normalized.squeeze()
        if normalized.ndim != 1:
            raise RuntimeError(
                "audiotools normalization returned an unexpected waveform shape "
                f"{tuple(normalized.shape)}"
            )
        return normalized

    def _encode_window(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode one waveform window to (B, D, T_latent) without chunking."""
        if self.deterministic_encode:
            required_paths_present = (
                hasattr(self.model, "encoder")
                and hasattr(self.model, "_pad")
                and hasattr(self.model, "quantizer")
                and hasattr(self.model.quantizer, "in_proj")
            )
            if not required_paths_present:
                raise RuntimeError(
                    "deterministic_encode=True requires encoder/_pad/quantizer.in_proj on DACVAE model."
                )
            z = self.model.encoder(self.model._pad(waveform))
            mean, _scale = self.model.quantizer.in_proj(z).chunk(2, dim=1)
            return mean
        return self.model.encode(waveform)  # (B, D, T)

    @staticmethod
    def _conv_backend_ctx():
        """MIOpen on gfx900 falls back to a naive kernel for dilated conv1d (20x slower than
        torch's own im2col+GEMM path); see docs/experiments/12-igpu-offload.md."""
        if get_opt_config().codec_use_cudnn():
            return contextlib.nullcontext()
        return torch.backends.cudnn.flags(enabled=False)

    def encode_waveform(self, *args, **kwargs) -> torch.Tensor:
        with self._conv_backend_ctx():
            return self._encode_waveform_impl(*args, **kwargs)

    def decode_latent(self, *args, **kwargs) -> torch.Tensor:
        with self._conv_backend_ctx():
            return self._decode_latent_impl(*args, **kwargs)

    @torch.inference_mode()
    def _encode_waveform_impl(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        normalize_db: float | None | object = _CODEC_DEFAULT,
        ensure_max: bool | None = None,
        chunk_frames: int | None = None,
        overlap_frames: int = 32,
    ) -> torch.Tensor:
        """
        Input:
          waveform: (B, C, T) or (C, T)
          normalize_db: Optional target loudness (LUFS-like dB) applied before encode
          ensure_max: If True and normalize_db is None, scale down only when abs peak exceeds 1.0
        Output:
          latent: (B, T_latent, D_latent)

        ``chunk_frames`` mirrors :meth:`decode_latent`: the waveform is encoded in
        hop-aligned windows with ``overlap_frames`` of context per side and only the
        centre frames are kept, so the transient VRAM scales with the window instead of
        the reference length. Loudness normalization stays global (it runs on the whole
        waveform before windowing), and the tail window keeps the model's own reflect
        padding, so the result matches a full encode to float error
        (docs/experiments/09-vram-safe-operating-point.md).
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform ndim=3, got shape={tuple(waveform.shape)}")

        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        if normalize_db is _CODEC_DEFAULT:
            effective_normalize_db = self.normalize_db
        elif normalize_db is None:
            effective_normalize_db = None
        else:
            effective_normalize_db = float(normalize_db)
        # audiotools normalization already applies ensure_max_of_audio(), so codec-side
        # peak scaling is only needed when normalization is disabled.
        effective_ensure_max = (
            effective_normalize_db is None and bool(ensure_max) if ensure_max is not None else False
        )

        waveform = waveform.to(dtype=torch.float32)
        if effective_normalize_db is not None or effective_ensure_max:
            # Keep behavior deterministic per utterance by normalizing each waveform independently.
            processed: list[torch.Tensor] = []
            for wav in waveform.squeeze(1):
                if effective_normalize_db is not None:
                    wav = self._normalize_loudness(
                        wav, sample_rate=self.sample_rate, target_db=effective_normalize_db
                    )
                wav = wav.squeeze()
                if wav.ndim != 1:
                    raise RuntimeError(
                        "Expected mono per-item waveform after preprocessing, "
                        f"got shape={tuple(wav.shape)}"
                    )
                if effective_ensure_max:
                    peak = wav.abs().max()
                    if torch.isfinite(peak) and peak > 1.0:
                        wav = wav * (1.0 / float(peak))
                processed.append(wav)
            waveform = torch.stack(processed, dim=0).unsqueeze(1)

        waveform = waveform.to(self.device, dtype=self.dtype)
        hop = int(self.model.hop_length)
        samples = int(waveform.shape[-1])
        total = -(-samples // hop)  # ceil: frames a full encode would produce
        if chunk_frames is None or chunk_frames <= 0 or total <= chunk_frames + 2 * overlap_frames:
            encoded = self._encode_window(waveform)
            return encoded.transpose(1, 2).contiguous()  # (B, T, D)

        # Fixed-size windows (repeating shapes for cuDNN); a short remainder is merged
        # into the previous window instead of being encoded as a tiny chunk.
        bounds: list[tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(total, start + chunk_frames)
            if total - end < max(2 * overlap_frames, chunk_frames // 2):
                end = total
            bounds.append((start, end))
            start = end
        pieces: list[torch.Tensor] = []
        for start, end in bounds:
            lo = max(0, start - overlap_frames)
            hi = min(total, end + overlap_frames)
            window = waveform[..., lo * hop : min(samples, hi * hop)]
            z = self._encode_window(window)
            pieces.append(z[..., start - lo : end - lo].clone())
            del z, window
        encoded = torch.cat(pieces, dim=-1)
        return encoded.transpose(1, 2).contiguous()  # (B, T, D)

    @torch.inference_mode()
    def _decode_latent_impl(
        self,
        latent: torch.Tensor,
        *,
        chunk_frames: int | None = None,
        overlap_frames: int = 64,
        autocast_bf16: bool = False,
    ) -> torch.Tensor:
        """
        Input:
          latent: (B, T, D)
        Output:
          audio: (B, 1, samples)

        With ``chunk_frames`` the latent is decoded in overlapping windows and only the
        centre of each window is kept. The decoder is a finite-receptive-field conv stack,
        so with ``overlap_frames`` well above that field the result matches full decode
        (verified in docs/experiments/06-memory.md) while the transient VRAM scales with
        the window instead of the utterance length.
        """
        if latent.ndim != 3:
            raise ValueError(f"Expected latent ndim=3, got shape={tuple(latent.shape)}")
        z = latent.transpose(1, 2).contiguous().to(self.device, dtype=self.dtype)  # (B, D, T)
        total = int(z.shape[-1])
        if autocast_bf16 and self.device.type == "cuda" and self.dtype == torch.float32:
            # Decode-only reduced precision: weights stay fp32 (so reference *encode* is
            # unchanged); convs run in bf16 and activations are half the size.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.decode_latent(
                    latent, chunk_frames=chunk_frames, overlap_frames=overlap_frames
                )
            return out.float()
        if chunk_frames is None or chunk_frames <= 0 or total <= chunk_frames + 2 * overlap_frames:
            return self.model.decode(z)
        hop = int(self.model.hop_length)
        pieces: list[torch.Tensor] = []
        # Fixed-size windows (so cuDNN sees repeating shapes); a short remainder is merged
        # into the previous window instead of being decoded as a tiny chunk.
        bounds: list[tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(total, start + chunk_frames)
            if total - end < max(2 * overlap_frames, chunk_frames // 2):
                end = total
            bounds.append((start, end))
            start = end
        for start, end in bounds:
            lo = max(0, start - overlap_frames)
            hi = min(total, end + overlap_frames)
            audio = self.model.decode(z[:, :, lo:hi])
            keep_from = (start - lo) * hop
            keep_to = keep_from + (end - start) * hop
            if hi == total:
                keep_to = audio.shape[-1]  # keep the true tail as decoded
            pieces.append(audio[:, :, keep_from:keep_to])
            del audio
        return torch.cat(pieces, dim=-1)

    def encode_file(self, path: str | Path) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(str(path))
        except RuntimeError:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            wav = torch.from_numpy(data)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.T
        wav = wav.unsqueeze(0)  # (1, C, T)
        return self.encode_waveform(wav, sr).cpu()
