"""Prebaked runtime bundles: store the *result* of loading instead of redoing it.

A normal load reads the FP32 checkpoint, builds the module, copies 714 tensors in
and casts them to the runtime dtype; the codec additionally unpickles a 410 MB
``weights.pth`` and folds its ``weight_norm`` hooks.  All of that is a pure
function of (checkpoint, codec, precision), so it can be done once and written
out as two safetensors files that load straight onto the GPU.

The bundle stores exactly the tensors a freshly loaded runtime holds, so the
weights it produces are bit-identical to the slow path (see
docs/experiments/11-load-time.md).

Layout of ``<root>/<fingerprint>/``::

    manifest.json      identity of the sources + everything needed to rebuild the modules
    model.safetensors  DiT state dict in the runtime dtype
    codec.safetensors  DACVAE state dict, weight_norm already folded

A bundle is used only when every field in ``manifest.json`` still matches the
current request; otherwise it is ignored and the slow path runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

PREBAKE_VERSION = 1

MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.safetensors"
CODEC_NAME = "codec.safetensors"


def default_root() -> Path:
    override = os.environ.get("IRODORI_OPT_PREBAKE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "irodori-tts" / "prebake"


def _file_identity(path: str | Path) -> dict:
    p = Path(path)
    st = p.stat()
    return {"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def source_identity(
    *,
    checkpoint: str | Path,
    codec_repo: str,
    model_precision: str,
    codec_precision: str,
    model_device_type: str,
    codec_device_type: str,
    fold_weight_norm: bool,
) -> dict:
    """The part of a bundle's identity that is knowable without touching the hub.

    The codec's ``weights.pth`` is identified in the manifest instead (see
    ``file_identity``): resolving it goes through ``hf_hub_download``, and the
    whole point of the bundle is to not pay for that on the fast path.
    """
    return {
        "prebake_version": PREBAKE_VERSION,
        "torch_version": torch.__version__,
        "checkpoint": _file_identity(checkpoint),
        "codec_repo": str(codec_repo),
        "model_precision": str(model_precision),
        "codec_precision": str(codec_precision),
        "model_device_type": str(model_device_type),
        "codec_device_type": str(codec_device_type),
        "fold_weight_norm": bool(fold_weight_norm),
    }


def file_identity(path: str | Path) -> dict:
    return _file_identity(path)


def fingerprint(identity: dict) -> str:
    blob = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class PrebakeBundle:
    directory: Path
    manifest: dict

    @property
    def model_path(self) -> Path:
        return self.directory / MODEL_NAME

    @property
    def codec_path(self) -> Path:
        return self.directory / CODEC_NAME

    def model_config(self) -> dict:
        return json.loads(self.manifest["model_config_json"])

    def text_encoder_config(self) -> dict | None:
        raw = self.manifest.get("text_encoder_config_json")
        return None if raw is None else json.loads(raw)

    def train_config(self) -> dict | None:
        raw = self.manifest.get("train_config_json")
        return None if raw is None else json.loads(raw)

    def load_model_state(self, device: str) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        return load_file(str(self.model_path), device=device)

    def load_codec_state(self, device: str) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        return load_file(str(self.codec_path), device=device)


def bundle_dir(root: Path, identity: dict) -> Path:
    return Path(root) / fingerprint(identity)


def find(root: Path, identity: dict) -> PrebakeBundle | None:
    directory = bundle_dir(root, identity)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("identity") != identity:
        # Fingerprint collision or a hand-edited bundle: refuse it.
        return None
    for name in (MODEL_NAME, CODEC_NAME):
        if not (directory / name).is_file():
            return None
    recorded = manifest.get("codec_weights")
    if isinstance(recorded, dict):
        try:
            if _file_identity(recorded["path"]) != recorded:
                return None
        except (OSError, KeyError):
            return None
    return PrebakeBundle(directory=directory, manifest=manifest)


def write(
    root: Path,
    *,
    identity: dict,
    model_state: dict[str, torch.Tensor],
    codec_state: dict[str, torch.Tensor],
    model_config_json: str,
    text_encoder_config_json: str | None,
    train_config_json: str | None,
    dacvae_kwargs: dict,
    codec_weights: dict | None,
    codec_latent_dim: int,
    codec_sample_rate: int,
    extra: dict | None = None,
) -> Path:
    from safetensors.torch import save_file

    directory = bundle_dir(root, identity)
    directory.mkdir(parents=True, exist_ok=True)

    def _dump(state: dict[str, torch.Tensor], path: Path) -> None:
        cpu_state = {k: v.detach().to("cpu").contiguous() for k, v in state.items()}
        save_file(cpu_state, str(path))

    _dump(model_state, directory / MODEL_NAME)
    _dump(codec_state, directory / CODEC_NAME)

    manifest = {
        "identity": identity,
        "model_config_json": model_config_json,
        "text_encoder_config_json": text_encoder_config_json,
        "train_config_json": train_config_json,
        "dacvae_kwargs": dacvae_kwargs,
        "codec_weights": codec_weights,
        "codec_latent_dim": int(codec_latent_dim),
        "codec_sample_rate": int(codec_sample_rate),
    }
    if extra:
        manifest.update(extra)
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False)
    )
    return directory
