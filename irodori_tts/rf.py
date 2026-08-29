from __future__ import annotations

import math

import torch

from .ane_dit import get_ane_runner
from .model import TextToLatentRFDiT
from .opt_config import OptConfig, get_opt_config
from .speaker_inversion import SPEAKER_INVERSION_UNCOND_MODES


def _make_rng(seed: int, device: torch.device) -> tuple[torch.Generator, torch.device]:
    # Metal-only build: the noise is drawn on the MPS device itself (no CPU fallback).
    return torch.Generator(device=device).manual_seed(seed), device


def sample_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(z)
    return t.clamp(min=t_min, max=t_max)


def sample_stratified_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    """
    Stratified sampling for logit-normal timesteps.

    u ~ stratified U(0, 1), z = mean + std * Phi^{-1}(u), t = sigmoid(z)
    """
    if batch_size <= 0:
        return torch.empty((0,), device=device)
    u = (
        torch.arange(batch_size, device=device, dtype=torch.float32)
        + torch.rand(batch_size, device=device)
    ) / float(batch_size)
    u = u.clamp(1e-6, 1.0 - 1e-6)
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    z = torch.erfinv(2.0 * u - 1.0) * (2.0**0.5)
    z = z * std + mean
    t = torch.sigmoid(z)
    # Randomize assignment order so dataset ordering does not correlate with t bins.
    t = t[torch.randperm(batch_size, device=device)]
    return t.clamp(min=t_min, max=t_max)


def rf_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Straight line interpolation: x_t = (1-t) x0 + t z.
    return (1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise


def rf_velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # For x_t = (1-t) x0 + t z, velocity is d/dt x_t = z - x0.
    return noise - x0


def temporal_score_rescale(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: float | torch.Tensor,
    rescale_k: float,
    rescale_sigma: float,
) -> torch.Tensor:
    """
    Temporal score rescaling from https://arxiv.org/pdf/2510.01184.
    """
    t_value = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
    if t_value >= 1.0:
        return v_pred
    one_minus_t = 1.0 - t_value
    snr = (one_minus_t * one_minus_t) / (t_value * t_value)
    sigma_sq = float(rescale_sigma) * float(rescale_sigma)
    ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / float(rescale_k) + 1.0)
    return (ratio * (one_minus_t * v_pred + x_t) - x_t) / one_minus_t


def scale_speaker_kv_cache(
    context_kv_cache: list[tuple[torch.Tensor, ...]],
    scale: float,
    max_layers: int | None = None,
) -> None:
    """
    In-place scaling of speaker K/V tensors in precomputed context cache.
    """
    if max_layers is None:
        n_layers = len(context_kv_cache)
    else:
        n_layers = max(0, min(int(max_layers), len(context_kv_cache)))
    for i in range(n_layers):
        layer_kv = context_kv_cache[i]
        if len(layer_kv) < 4:
            raise ValueError(
                f"Expected at least 4 tensors in context KV cache entry, got {len(layer_kv)}"
            )
        k_speaker = layer_kv[2]
        v_speaker = layer_kv[3]
        k_speaker.mul_(scale)
        v_speaker.mul_(scale)


@torch.inference_mode()
def _sample_euler_rf_cfg_legacy(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    sequence_length: int,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    speaker_state_override: torch.Tensor | None = None,
    speaker_mask_override: torch.Tensor | None = None,
    speaker_uncond_mode: str = "mask",
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    cfg_scale: float | None = None,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    use_context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
) -> torch.Tensor:
    """
    Euler sampling over RF ODE with text/reference/caption conditioning CFG.

    Returns:
      latent sequence in patched space, shape (B, sequence_length, patched_latent_dim)
    """
    device = model.device
    dtype = model.dtype
    batch_size = text_input_ids.shape[0]
    latent_dim = model.cfg.patched_latent_dim

    rng, rng_device = _make_rng(seed=seed, device=device)
    x_t = torch.randn(
        (batch_size, sequence_length, latent_dim), device=rng_device, dtype=dtype, generator=rng
    )
    if rng_device != device:
        x_t = x_t.to(device=device)
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)

    if cfg_scale is not None:
        # Backward compatibility for old single-scale caller.
        cfg_scale_text = float(cfg_scale)
        cfg_scale_caption = float(cfg_scale)
        cfg_scale_speaker = float(cfg_scale)
    if not model.cfg.use_speaker_condition_resolved:
        cfg_scale_speaker = 0.0
        speaker_kv_scale = None
    speaker_uncond_mode = str(speaker_uncond_mode).strip().lower()
    if speaker_uncond_mode not in SPEAKER_INVERSION_UNCOND_MODES:
        raise ValueError(
            f"speaker_uncond_mode must be one of {sorted(SPEAKER_INVERSION_UNCOND_MODES)}, "
            f"got {speaker_uncond_mode!r}"
        )

    cfg_guidance_mode = str(cfg_guidance_mode).strip().lower()
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected one of: independent, joint, alternating."
        )

    init_scale = 0.999
    t_schedule_mode_norm = str(t_schedule_mode).strip().lower()
    sway_coeff_value = float(sway_coeff)
    if not math.isfinite(sway_coeff_value):
        raise ValueError(f"sway_coeff must be finite, got {sway_coeff!r}.")
    if t_schedule_mode_norm == "linear":
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    elif t_schedule_mode_norm == "sway":
        # F5-TTS-style Sway Sampling. Negative sway_coeff densifies the noise
        # side of the schedule (early steps); positive densifies the data side.
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        u = u + sway_coeff_value * (torch.cos(0.5 * math.pi * u) + u - 1.0)
        u = u.clamp(0.0, 1.0)
    else:
        raise ValueError(
            f"Unsupported t_schedule_mode={t_schedule_mode!r}. Expected 'linear' or 'sway'."
        )
    t_schedule = (1.0 - u) * init_scale
    if not bool(torch.all(t_schedule[:-1] > t_schedule[1:]).item()):
        raise ValueError("t_schedule must be strictly decreasing; adjust num_steps or sway_coeff.")
    use_independent_cfg = cfg_guidance_mode == "independent"
    use_joint_cfg = cfg_guidance_mode == "joint"
    use_alternating_cfg = cfg_guidance_mode == "alternating"

    (
        text_state_cond,
        text_mask_cond,
        speaker_state_cond,
        speaker_mask_cond,
        caption_state_cond,
        caption_mask_cond,
    ) = model.encode_conditions(
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        ref_latent=ref_latent,
        ref_mask=ref_mask,
        caption_input_ids=caption_input_ids,
        caption_mask=caption_mask,
        speaker_state_override=speaker_state_override,
        speaker_mask_override=speaker_mask_override,
        speaker_uncond_mode=speaker_uncond_mode,
    )
    text_state_uncond = torch.zeros_like(text_state_cond)
    text_mask_uncond = torch.zeros_like(text_mask_cond)
    speaker_state_uncond = None
    speaker_mask_uncond = None
    if model.cfg.use_speaker_condition_resolved:
        if speaker_state_cond is None or speaker_mask_cond is None:
            raise RuntimeError(
                "Speaker conditioning is enabled but encoded speaker state is missing."
            )
        if speaker_uncond_mode == "noise":
            speaker_noise = torch.randn(
                speaker_state_cond.shape,
                device=rng_device,
                dtype=speaker_state_cond.dtype,
                generator=rng,
            )
            if rng_device != device:
                speaker_noise = speaker_noise.to(device=device)
            speaker_state_uncond = speaker_noise * speaker_state_cond.std().clamp_min(1e-6)
            speaker_mask_uncond = torch.ones_like(speaker_mask_cond)
        else:
            speaker_state_uncond = torch.zeros_like(speaker_state_cond)
            speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)
    caption_state_uncond = None
    caption_mask_uncond = None
    if model.cfg.use_caption_condition:
        if caption_state_cond is None or caption_mask_cond is None:
            raise RuntimeError(
                "Caption conditioning is enabled but encoded caption state is missing."
            )
        caption_state_uncond = torch.zeros_like(caption_state_cond)
        caption_mask_uncond = torch.zeros_like(caption_mask_cond)

    has_text_cfg = cfg_scale_text > 0
    has_caption_cfg = (
        model.cfg.use_caption_condition
        and cfg_scale_caption > 0
        and caption_mask_cond is not None
        and bool(caption_mask_cond.any().item())
    )
    has_speaker_cfg = cfg_scale_speaker > 0

    def _bundle(
        *,
        text_state: torch.Tensor,
        text_mask_val: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask_val: torch.Tensor | None,
        caption_state: torch.Tensor | None,
        caption_mask_val: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return (
            text_state,
            text_mask_val,
            speaker_state,
            speaker_mask_val,
            caption_state,
            caption_mask_val,
        )

    cond_bundle = _bundle(
        text_state=text_state_cond,
        text_mask_val=text_mask_cond,
        speaker_state=speaker_state_cond,
        speaker_mask_val=speaker_mask_cond,
        caption_state=caption_state_cond,
        caption_mask_val=caption_mask_cond,
    )
    enabled_cfg_names: list[str] = []
    cfg_scales: dict[str, float] = {}
    if has_text_cfg:
        enabled_cfg_names.append("text")
        cfg_scales["text"] = float(cfg_scale_text)
    if has_speaker_cfg:
        enabled_cfg_names.append("speaker")
        cfg_scales["speaker"] = float(cfg_scale_speaker)
    if has_caption_cfg:
        enabled_cfg_names.append("caption")
        cfg_scales["caption"] = float(cfg_scale_caption)

    independent_bundles = [cond_bundle]
    independent_names = ["cond"]
    if use_independent_cfg:
        for name in enabled_cfg_names:
            independent_names.append(name)
            independent_bundles.append(
                _bundle(
                    text_state=text_state_uncond if name == "text" else text_state_cond,
                    text_mask_val=text_mask_uncond if name == "text" else text_mask_cond,
                    speaker_state=(
                        speaker_state_uncond if name == "speaker" else speaker_state_cond
                    ),
                    speaker_mask_val=(
                        speaker_mask_uncond if name == "speaker" else speaker_mask_cond
                    ),
                    caption_state=(
                        caption_state_uncond if name == "caption" else caption_state_cond
                    ),
                    caption_mask_val=(
                        caption_mask_uncond if name == "caption" else caption_mask_cond
                    ),
                )
            )
    cfg_batch_mult = len(independent_bundles)

    def _cat_optional_tensors(values: list[torch.Tensor | None]) -> torch.Tensor | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        if len(present) != len(values):
            raise ValueError("Cannot concatenate optional condition tensors with mixed presence.")
        return torch.cat(present, dim=0)

    independent_text_state = torch.cat([bundle[0] for bundle in independent_bundles], dim=0)
    independent_text_mask = torch.cat([bundle[1] for bundle in independent_bundles], dim=0)
    independent_speaker_state = _cat_optional_tensors([bundle[2] for bundle in independent_bundles])
    independent_speaker_mask = _cat_optional_tensors([bundle[3] for bundle in independent_bundles])
    independent_caption_state = _cat_optional_tensors([bundle[4] for bundle in independent_bundles])
    independent_caption_mask = _cat_optional_tensors([bundle[5] for bundle in independent_bundles])

    joint_uncond_bundle = _bundle(
        text_state=text_state_uncond,
        text_mask_val=text_mask_uncond,
        speaker_state=speaker_state_uncond,
        speaker_mask_val=speaker_mask_uncond,
        caption_state=caption_state_uncond,
        caption_mask_val=caption_mask_uncond,
    )

    alternating_bundles: dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ] = {
        "text": _bundle(
            text_state=text_state_uncond,
            text_mask_val=text_mask_uncond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        ),
        "caption": _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            caption_state=caption_state_uncond,
            caption_mask_val=caption_mask_uncond,
        ),
    }
    if has_speaker_cfg:
        alternating_bundles["speaker"] = _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_uncond,
            speaker_mask_val=speaker_mask_uncond,
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        )

    # Force-speaker scaling operates on projected speaker K/V, so it requires context KV caches.
    effective_use_context_kv_cache = bool(use_context_kv_cache or (speaker_kv_scale is not None))

    context_kv_cond = None
    context_kv_cfg = None
    context_kv_joint_uncond = None
    context_kv_alternating: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    if effective_use_context_kv_cache:
        context_kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
            caption_state=caption_state_cond,
        )
        if use_independent_cfg and cfg_batch_mult > 1:
            context_kv_cfg = model.build_context_kv_cache(
                text_state=independent_text_state,
                speaker_state=independent_speaker_state,
                caption_state=independent_caption_state,
            )
        elif use_joint_cfg:
            if enabled_cfg_names:
                context_kv_joint_uncond = model.build_context_kv_cache(
                    text_state=joint_uncond_bundle[0],
                    speaker_state=joint_uncond_bundle[2],
                    caption_state=joint_uncond_bundle[4],
                )
        elif use_alternating_cfg:
            for name in enabled_cfg_names:
                bundle = alternating_bundles[name]
                context_kv_alternating[name] = model.build_context_kv_cache(
                    text_state=bundle[0],
                    speaker_state=bundle[2],
                    caption_state=bundle[4],
                )
    if speaker_kv_scale is not None:
        scale_speaker_kv_cache(
            context_kv_cache=context_kv_cond,
            scale=float(speaker_kv_scale),
            max_layers=speaker_kv_max_layers,
        )
        if context_kv_cfg is not None:
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cfg,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
        for cache in context_kv_alternating.values():
            scale_speaker_kv_cache(
                context_kv_cache=cache,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
    speaker_kv_active = speaker_kv_scale is not None

    for i in range(num_steps):
        t = t_schedule[i]
        t_next = t_schedule[i + 1]
        tt = torch.full((batch_size,), t, device=device, dtype=dtype)

        use_cfg = bool(enabled_cfg_names) and (cfg_min_t <= t.item() <= cfg_max_t)
        if use_cfg:
            if use_independent_cfg:
                x_t_cfg = torch.cat([x_t] * cfg_batch_mult, dim=0).to(dtype)
                tt_cfg = tt.repeat(cfg_batch_mult)
                v_out = model.forward_with_encoded_conditions(
                    x_t=x_t_cfg,
                    t=tt_cfg,
                    text_state=independent_text_state,
                    text_mask=independent_text_mask,
                    speaker_state=independent_speaker_state,
                    speaker_mask=independent_speaker_mask,
                    caption_state=independent_caption_state,
                    caption_mask=independent_caption_mask,
                    context_kv_cache=context_kv_cfg,
                )
                chunks = v_out.chunk(cfg_batch_mult, dim=0)
                v = chunks[0]
                for name, chunk in zip(independent_names[1:], chunks[1:], strict=True):
                    v = v + cfg_scales[name] * (chunks[0] - chunk)
            else:
                v_cond = model.forward_with_encoded_conditions(
                    x_t=x_t.to(dtype),
                    t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    caption_state=caption_state_cond,
                    caption_mask=caption_mask_cond,
                    context_kv_cache=context_kv_cond,
                )
                if use_joint_cfg:
                    if len(enabled_cfg_names) > 1:
                        joint_scales = [cfg_scales[name] for name in enabled_cfg_names]
                        if max(joint_scales) - min(joint_scales) > 1e-6:
                            raise ValueError(
                                "cfg_guidance_mode='joint' expects equal enabled guidance scales; "
                                "set matching text/speaker/caption scales or use --cfg-scale."
                            )
                    joint_scale = cfg_scales[enabled_cfg_names[0]]
                    v_uncond_joint = model.forward_with_encoded_conditions(
                        x_t=x_t.to(dtype),
                        t=tt,
                        text_state=joint_uncond_bundle[0],
                        text_mask=joint_uncond_bundle[1],
                        speaker_state=joint_uncond_bundle[2],
                        speaker_mask=joint_uncond_bundle[3],
                        caption_state=joint_uncond_bundle[4],
                        caption_mask=joint_uncond_bundle[5],
                        context_kv_cache=context_kv_joint_uncond,
                    )
                    v = v_cond + joint_scale * (v_cond - v_uncond_joint)
                elif use_alternating_cfg:
                    alt_name = enabled_cfg_names[i % len(enabled_cfg_names)]
                    alt_bundle = alternating_bundles[alt_name]
                    v_uncond_alt = model.forward_with_encoded_conditions(
                        x_t=x_t.to(dtype),
                        t=tt,
                        text_state=alt_bundle[0],
                        text_mask=alt_bundle[1],
                        speaker_state=alt_bundle[2],
                        speaker_mask=alt_bundle[3],
                        caption_state=alt_bundle[4],
                        caption_mask=alt_bundle[5],
                        context_kv_cache=context_kv_alternating.get(alt_name),
                    )
                    v = v_cond + cfg_scales[alt_name] * (v_cond - v_uncond_alt)
                else:
                    raise RuntimeError(f"Unexpected cfg_guidance_mode: {cfg_guidance_mode}")
        else:
            v = model.forward_with_encoded_conditions(
                x_t=x_t.to(dtype),
                t=tt,
                text_state=text_state_cond,
                text_mask=text_mask_cond,
                speaker_state=speaker_state_cond,
                speaker_mask=speaker_mask_cond,
                caption_state=caption_state_cond,
                caption_mask=caption_mask_cond,
                context_kv_cache=context_kv_cond,
            )

        if rescale_k is not None and rescale_sigma is not None:
            v = temporal_score_rescale(
                v_pred=v,
                x_t=x_t,
                t=t,
                rescale_k=float(rescale_k),
                rescale_sigma=float(rescale_sigma),
            )

        if (
            speaker_kv_active
            and speaker_kv_min_t is not None
            and (t_next < speaker_kv_min_t)
            and (t >= speaker_kv_min_t)
        ):
            inv_scale = 1.0 / float(speaker_kv_scale)
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cond,
                scale=inv_scale,
                max_layers=speaker_kv_max_layers,
            )
            if context_kv_cfg is not None:
                scale_speaker_kv_cache(
                    context_kv_cache=context_kv_cfg,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            for cache in context_kv_alternating.values():
                scale_speaker_kv_cache(
                    context_kv_cache=cache,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            speaker_kv_active = False

        x_t = x_t + v * (t_next - t)

    return x_t


# ---------------------------------------------------------------------------
# Fast path (single-machine optimization): no per-step host<->device syncs,
# precomputed additive attention masks, optional condition reuse, and a
# self-contained per-step function (one Metal command stream per step).
# ---------------------------------------------------------------------------

_Bundle = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


def _slice_bundle(bundle: _Bundle, start: int, stop: int) -> _Bundle:
    return tuple(None if t is None else t[start:stop] for t in bundle)  # type: ignore[return-value]


def _slice_kv(
    kv: list[tuple[torch.Tensor, ...]] | None, start: int, stop: int
) -> list[tuple[torch.Tensor, ...]] | None:
    if kv is None:
        return None
    return [tuple(t[start:stop] for t in layer) for layer in kv]


def _index_bundle(bundle: _Bundle, rows: list[int]) -> _Bundle:
    idx = None
    out = []
    for t in bundle:
        if t is None:
            out.append(None)
            continue
        if idx is None:
            idx = torch.tensor(rows, device=t.device)
        out.append(t.index_select(0, idx))
    return tuple(out)  # type: ignore[return-value]


def _index_kv(
    kv: list[tuple[torch.Tensor, ...]] | None, rows: list[int]
) -> list[tuple[torch.Tensor, ...]] | None:
    if kv is None:
        return None
    idx = torch.tensor(rows, device=kv[0][0].device)
    return [tuple(t.index_select(0, idx) for t in layer) for layer in kv]


def _cat_optional(values: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("Cannot concatenate optional condition tensors with mixed presence.")
    return torch.cat(present, dim=0)


def _cat_bundles(bundles: list[_Bundle]) -> _Bundle:
    return (
        torch.cat([b[0] for b in bundles], dim=0),
        torch.cat([b[1] for b in bundles], dim=0),
        _cat_optional([b[2] for b in bundles]),
        _cat_optional([b[3] for b in bundles]),
        _cat_optional([b[4] for b in bundles]),
        _cat_optional([b[5] for b in bundles]),
    )


def temporal_score_rescale_tensor(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    rescale_k: float,
    rescale_sigma: float,
) -> torch.Tensor:
    """Tensor-only variant of :func:`temporal_score_rescale` (graph capturable)."""
    t32 = t.to(torch.float32).reshape(-1, 1, 1)
    one_minus_t = 1.0 - t32
    snr = (one_minus_t * one_minus_t) / (t32 * t32)
    sigma_sq = float(rescale_sigma) * float(rescale_sigma)
    ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / float(rescale_k) + 1.0)
    out = (ratio * (one_minus_t * v_pred.float() + x_t.float()) - x_t.float()) / one_minus_t
    return torch.where(t32 >= 1.0, v_pred.float(), out).to(v_pred.dtype)


class _FastSamplerState:
    """Per-request constants of the sampler step."""

    def __init__(
        self,
        *,
        model: TextToLatentRFDiT,
        cond_bundle: _Bundle,
        context_kv_cond: list[tuple[torch.Tensor, ...]] | None,
        cfg_batch_mult: int,
        independent_bundle: _Bundle | None,
        context_kv_cfg: list[tuple[torch.Tensor, ...]] | None,
        independent_names: list[str],
        joint_bundle: _Bundle | None,
        context_kv_joint: list[tuple[torch.Tensor, ...]] | None,
        alternating_bundles: dict[str, _Bundle],
        context_kv_alternating: dict[str, list[tuple[torch.Tensor, ...]]],
        enabled_cfg_names: list[str],
        cfg_scales: dict[str, float],
        cfg_guidance_mode: str,
        rescale_k: float | None,
        rescale_sigma: float | None,
        latent_len: int,
        latent_mask: torch.Tensor | None,
        ane_runner: object | None = None,
        ane_gpu_branches: int = 0,
        ane_gpu_cond: bool = True,
        ane_nocfg_gpu: bool = False,
        ane_candidates: bool = True,
    ) -> None:
        self.model = model
        # Neural Engine path (13-ane.md): contexts are built lazily per bundle; any miss
        # (no enumerated shape fits, batch > 1) turns the ANE off for this request.
        self.ane_runner = ane_runner
        self.ane_gpu_branches = int(ane_gpu_branches)
        # The cond branch is amplified by (1 + sum of guidance scales) in the CFG combination,
        # so by default it is the branch that goes to the (more precise) GPU.
        self.ane_gpu_cond = bool(ane_gpu_cond)
        self.ane_nocfg_gpu = bool(ane_nocfg_gpu)
        # num_candidates=2: candidate 0 on the ANE, candidate 1 on the GPU, concurrently.
        self.ane_candidates = bool(ane_candidates)
        self._cand_cache: dict[str, object] = {}
        self.ane_active = ane_runner is not None
        self._ane_ctx: dict[str, object] = {}
        self.cond_bundle = cond_bundle
        self.context_kv_cond = context_kv_cond
        self.cfg_batch_mult = cfg_batch_mult
        self.independent_bundle = independent_bundle
        self.context_kv_cfg = context_kv_cfg
        self.independent_names = independent_names
        self.joint_bundle = joint_bundle
        self.context_kv_joint = context_kv_joint
        self.alternating_bundles = alternating_bundles
        self.context_kv_alternating = context_kv_alternating
        self.enabled_cfg_names = enabled_cfg_names
        self.cfg_scales = cfg_scales
        self.cfg_guidance_mode = cfg_guidance_mode
        self.rescale_k = rescale_k
        self.rescale_sigma = rescale_sigma
        self.latent_len = latent_len
        self.latent_mask = latent_mask
        self.masks: dict[str, torch.Tensor | None] = {}

    def required_mask_names(self, *, use_cfg: bool, alt_index: int) -> list[tuple[str, _Bundle, bool]]:
        mode = self.cfg_guidance_mode
        if use_cfg and self.enabled_cfg_names:
            if mode == "independent":
                assert self.independent_bundle is not None
                return [("independent", self.independent_bundle, self.context_kv_cfg is not None)]
            out = [("cond", self.cond_bundle, self.context_kv_cond is not None)]
            if mode == "joint":
                assert self.joint_bundle is not None
                out.append(("joint", self.joint_bundle, self.context_kv_joint is not None))
            elif mode == "alternating":
                alt_name = self.enabled_cfg_names[alt_index % len(self.enabled_cfg_names)]
                out.append(
                    (
                        f"alt_{alt_name}",
                        self.alternating_bundles[alt_name],
                        self.context_kv_alternating.get(alt_name) is not None,
                    )
                )
            return out
        return [("cond", self.cond_bundle, self.context_kv_cond is not None)]

    def prepare_masks(self, *, use_cfg: bool, alt_index: int) -> None:
        for name, bundle, has_kv in self.required_mask_names(use_cfg=use_cfg, alt_index=alt_index):
            self.mask_for(name, bundle, has_kv)

    def mask_for(self, name: str, bundle: _Bundle, has_kv: bool) -> torch.Tensor | None:
        if not has_kv:
            return None
        cached = self.masks.get(name)
        if cached is None:
            bsz = bundle[1].shape[0]
            latent_mask = self.latent_mask
            if latent_mask is not None and latent_mask.shape[0] != bsz:
                latent_mask = latent_mask.expand(bsz, -1)
            cached = self.model.build_combined_attn_mask(
                latent_len=self.latent_len,
                text_mask=bundle[1],
                speaker_mask=bundle[3],
                caption_mask=bundle[5],
                latent_mask=latent_mask,
            )
            self.masks[name] = cached
        return cached

    def forward(
        self,
        x_t: torch.Tensor,
        tt: torch.Tensor,
        bundle: _Bundle,
        context_kv: list[tuple[torch.Tensor, ...]] | None,
        mask_name: str,
    ) -> torch.Tensor:
        return self.model.forward_with_encoded_conditions(
            x_t=x_t,
            t=tt,
            text_state=bundle[0],
            text_mask=bundle[1],
            speaker_state=bundle[2],
            speaker_mask=bundle[3],
            caption_state=bundle[4],
            caption_mask=bundle[5],
            latent_mask=self.latent_mask,
            context_kv_cache=context_kv,
            attn_mask=self.mask_for(mask_name, bundle, context_kv is not None),
        )

    def _ane_context(self, name: str, bundle: _Bundle) -> object | None:
        ctx = self._ane_ctx.get(name)
        if ctx is None and name not in self._ane_ctx:
            if bundle[2] is None or bundle[3] is None or bundle[4] is None or bundle[5] is None:
                ctx = None
            else:
                ctx = self.ane_runner.make_context(  # type: ignore[union-attr]
                    latent_len=self.latent_len,
                    text_state=bundle[0],
                    text_mask=bundle[1],
                    speaker_state=bundle[2],
                    speaker_mask=bundle[3],
                    caption_state=bundle[4],
                    caption_mask=bundle[5],
                )
            self._ane_ctx[name] = ctx
            if ctx is None:
                self.ane_active = False
        return ctx

    def release(self) -> None:
        if self.ane_runner is not None:
            for ctx in self._ane_ctx.values():
                if ctx is not None:
                    self.ane_runner.drop_context(ctx)  # type: ignore[attr-defined]
            self._ane_ctx = {}

    def _cfg_combine(self, v_branches: torch.Tensor, mult: int) -> torch.Tensor:
        chunks = v_branches.chunk(mult, dim=0)
        v = chunks[0]
        for name, chunk in zip(self.independent_names[1:], chunks[1:], strict=True):
            v = v + self.cfg_scales[name] * (chunks[0] - chunk)
        return v

    def _cand(self, name: str, build):  # noqa: ANN001
        value = self._cand_cache.get(name)
        if value is None and name not in self._cand_cache:
            value = build()
            self._cand_cache[name] = value
        return value

    def _velocity_ane_candidates(
        self, x_t: torch.Tensor, tt: torch.Tensor, t_value: float, *, use_cfg: bool
    ) -> torch.Tensor | None:
        """num_candidates=2: the whole request of candidate 0 runs on the ANE while candidate 1
        runs the regular MPS path, step by step in lockstep. Bundles are branch-major
        [cond(c0,c1), uncond_1(c0,c1), ...], so candidate c owns rows c, c+2, c+4, ..."""
        runner = self.ane_runner
        device, dtype = x_t.device, x_t.dtype
        if use_cfg and self.enabled_cfg_names:
            assert self.independent_bundle is not None
            mult = self.cfg_batch_mult
            rows0 = [2 * b for b in range(mult)]
            rows1 = [2 * b + 1 for b in range(mult)]
            bundle0 = self._cand("cfg_b0", lambda: _index_bundle(self.independent_bundle, rows0))
            bundle1 = self._cand("cfg_b1", lambda: _index_bundle(self.independent_bundle, rows1))
            kv1 = self._cand("cfg_kv1", lambda: _index_kv(self.context_kv_cfg, rows1))
            ctx = self._ane_context("cand0_cfg", bundle0)
            if ctx is None:
                return None
            runner.submit(ctx, x_t[0:1].repeat(mult, 1, 1), t_value)  # type: ignore[attr-defined]
            v1 = self.forward(x_t[1:2].repeat(mult, 1, 1), tt[1:2].repeat(mult), bundle1, kv1, "cand1_cfg")
            v0 = runner.wait().to(device=device, dtype=dtype)  # type: ignore[attr-defined]
            return torch.cat([self._cfg_combine(v0, mult), self._cfg_combine(v1, mult)], dim=0)
        bundle0 = self._cand("cond_b0", lambda: _index_bundle(self.cond_bundle, [0]))
        bundle1 = self._cand("cond_b1", lambda: _index_bundle(self.cond_bundle, [1]))
        kv1 = self._cand("cond_kv1", lambda: _index_kv(self.context_kv_cond, [1]))
        ctx = self._ane_context("cand0_cond", bundle0)
        if ctx is None:
            return None
        runner.submit(ctx, x_t[0:1], t_value)  # type: ignore[attr-defined]
        v1 = self.forward(x_t[1:2], tt[1:2], bundle1, kv1, "cand1_cond")
        v0 = runner.wait().to(device=device, dtype=dtype)  # type: ignore[attr-defined]
        return torch.cat([v0, v1], dim=0)

    def _velocity_ane(
        self, x_t: torch.Tensor, tt: torch.Tensor, t_value: float, *, use_cfg: bool
    ) -> torch.Tensor | None:
        """ANE (+ optional GPU branches) velocity for ``independent`` CFG; None = fall back."""
        runner = self.ane_runner
        if runner is None:
            return None
        if x_t.shape[0] == 2 and self.ane_candidates:
            return self._velocity_ane_candidates(x_t, tt, t_value, use_cfg=use_cfg)
        if x_t.shape[0] != 1:
            return None
        device, dtype = x_t.device, x_t.dtype
        if use_cfg and self.enabled_cfg_names:
            assert self.independent_bundle is not None
            mult = self.cfg_batch_mult
            n_gpu = max(0, min(self.ane_gpu_branches, mult - 1))
            n_ane = mult - n_gpu
            # Branch layout of the independent bundle is [cond, uncond_1, ...]. The GPU takes
            # a contiguous slice: the head (cond) when ane_gpu_cond, else the tail.
            if n_gpu > 0 and self.ane_gpu_cond:
                gpu_lo, gpu_hi = 0, n_gpu
                ane_lo, ane_hi = n_gpu, mult
            else:
                ane_lo, ane_hi = 0, n_ane
                gpu_lo, gpu_hi = n_ane, mult
            ctx = self._ane_context(
                f"cfg_ane{ane_lo}_{ane_hi}", _slice_bundle(self.independent_bundle, ane_lo, ane_hi)
            )
            if ctx is None:
                return None
            # Order matters: hand the ANE its work first, then enqueue the GPU branch(es) so
            # both units run the same step at the same time, then collect.
            runner.submit(ctx, x_t.repeat(n_ane, 1, 1), t_value)  # type: ignore[attr-defined]
            v_gpu: torch.Tensor | None = None
            if n_gpu > 0:
                v_gpu = self.forward(
                    x_t.repeat(n_gpu, 1, 1),
                    tt.repeat(n_gpu),
                    _slice_bundle(self.independent_bundle, gpu_lo, gpu_hi),
                    _slice_kv(self.context_kv_cfg, gpu_lo, gpu_hi),
                    f"independent_gpu{gpu_lo}_{gpu_hi}",
                )
            v_ane = runner.wait().to(device=device, dtype=dtype)  # type: ignore[attr-defined]
            chunks: list[torch.Tensor | None] = [None] * mult
            for i, chunk in enumerate(v_ane.chunk(n_ane, dim=0)):
                chunks[ane_lo + i] = chunk
            if v_gpu is not None:
                for i, chunk in enumerate(v_gpu.chunk(n_gpu, dim=0)):
                    chunks[gpu_lo + i] = chunk
            return self._cfg_combine(torch.cat(chunks, dim=0), mult)  # type: ignore[arg-type]
        if self.ane_nocfg_gpu:
            return None  # plain cond step on the GPU (the ANE idles in this phase)
        ctx = self._ane_context("cond", self.cond_bundle)
        if ctx is None:
            return None
        return runner.step(ctx, x_t, t_value).to(device=device, dtype=dtype)  # type: ignore[attr-defined]

    def velocity(
        self,
        x_t: torch.Tensor,
        tt: torch.Tensor,
        *,
        use_cfg: bool,
        step_index: int,
        t_value: float | None = None,
    ) -> torch.Tensor:
        """Predict velocity for one Euler step. ``tt`` has shape (B,)."""
        mode = self.cfg_guidance_mode
        if self.ane_active and mode == "independent" and t_value is not None:
            v_ane = self._velocity_ane(x_t, tt, t_value, use_cfg=use_cfg)
            if v_ane is not None:
                return v_ane
        if use_cfg and self.enabled_cfg_names:
            if mode == "independent":
                assert self.independent_bundle is not None
                mult = self.cfg_batch_mult
                x_t_cfg = torch.cat([x_t] * mult, dim=0)
                tt_cfg = tt.repeat(mult)
                v_out = self.forward(
                    x_t_cfg, tt_cfg, self.independent_bundle, self.context_kv_cfg, "independent"
                )
                chunks = v_out.chunk(mult, dim=0)
                v = chunks[0]
                for name, chunk in zip(self.independent_names[1:], chunks[1:], strict=True):
                    v = v + self.cfg_scales[name] * (chunks[0] - chunk)
                return v
            v_cond = self.forward(x_t, tt, self.cond_bundle, self.context_kv_cond, "cond")
            if mode == "joint":
                assert self.joint_bundle is not None
                joint_scale = self.cfg_scales[self.enabled_cfg_names[0]]
                v_uncond = self.forward(
                    x_t, tt, self.joint_bundle, self.context_kv_joint, "joint"
                )
                return v_cond + joint_scale * (v_cond - v_uncond)
            if mode == "alternating":
                alt_name = self.enabled_cfg_names[step_index % len(self.enabled_cfg_names)]
                v_uncond = self.forward(
                    x_t,
                    tt,
                    self.alternating_bundles[alt_name],
                    self.context_kv_alternating.get(alt_name),
                    f"alt_{alt_name}",
                )
                return v_cond + self.cfg_scales[alt_name] * (v_cond - v_uncond)
            raise RuntimeError(f"Unexpected cfg_guidance_mode: {mode}")
        return self.forward(x_t, tt, self.cond_bundle, self.context_kv_cond, "cond")

    def step(
        self,
        x_t: torch.Tensor,
        tt: torch.Tensor,
        dt: torch.Tensor,
        *,
        use_cfg: bool,
        step_index: int,
        t_value: float | None = None,
    ) -> torch.Tensor:
        """One Euler step: x_{t+dt} = x_t + v(x_t, t) * dt. ``dt`` has shape (1,)."""
        v = self.velocity(x_t, tt, use_cfg=use_cfg, step_index=step_index, t_value=t_value)
        if self.rescale_k is not None and self.rescale_sigma is not None:
            v = temporal_score_rescale_tensor(
                v_pred=v,
                x_t=x_t,
                t=tt,
                rescale_k=float(self.rescale_k),
                rescale_sigma=float(self.rescale_sigma),
            )
        return x_t + v * dt.to(v.dtype)


@torch.inference_mode()
def _sample_euler_rf_cfg_fast(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    sequence_length: int,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    speaker_state_override: torch.Tensor | None = None,
    speaker_mask_override: torch.Tensor | None = None,
    speaker_uncond_mode: str = "mask",
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    cfg_scale: float | None = None,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    use_context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
    encoded_conditions: _Bundle | None = None,
    has_caption: bool | None = None,
    opt: OptConfig | None = None,
) -> torch.Tensor:
    opt = get_opt_config() if opt is None else opt
    device = model.device
    dtype = model.dtype
    batch_size = text_input_ids.shape[0]
    latent_dim = model.cfg.patched_latent_dim

    rng, rng_device = _make_rng(seed=seed, device=device)
    x_t = torch.randn(
        (batch_size, sequence_length, latent_dim), device=rng_device, dtype=dtype, generator=rng
    )
    if rng_device != device:
        x_t = x_t.to(device=device)
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)

    if cfg_scale is not None:
        cfg_scale_text = float(cfg_scale)
        cfg_scale_caption = float(cfg_scale)
        cfg_scale_speaker = float(cfg_scale)
    if not model.cfg.use_speaker_condition_resolved:
        cfg_scale_speaker = 0.0
        speaker_kv_scale = None
    speaker_uncond_mode = str(speaker_uncond_mode).strip().lower()
    if speaker_uncond_mode not in SPEAKER_INVERSION_UNCOND_MODES:
        raise ValueError(
            f"speaker_uncond_mode must be one of {sorted(SPEAKER_INVERSION_UNCOND_MODES)}, "
            f"got {speaker_uncond_mode!r}"
        )
    cfg_guidance_mode = str(cfg_guidance_mode).strip().lower()
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected one of: independent, joint, alternating."
        )

    # Timestep schedule: computed on the device exactly like the legacy path (so the
    # values are bit-identical) but transferred to the host once instead of once per step.
    init_scale = 0.999
    t_schedule_mode_norm = str(t_schedule_mode).strip().lower()
    sway_coeff_value = float(sway_coeff)
    if not math.isfinite(sway_coeff_value):
        raise ValueError(f"sway_coeff must be finite, got {sway_coeff!r}.")
    if t_schedule_mode_norm == "linear":
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    elif t_schedule_mode_norm == "sway":
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        u = u + sway_coeff_value * (torch.cos(0.5 * math.pi * u) + u - 1.0)
        u = u.clamp(0.0, 1.0)
    else:
        raise ValueError(
            f"Unsupported t_schedule_mode={t_schedule_mode!r}. Expected 'linear' or 'sway'."
        )
    t_schedule = (1.0 - u) * init_scale
    t_list: list[float] = t_schedule.tolist()
    if any(a <= b for a, b in zip(t_list[:-1], t_list[1:], strict=False)):
        raise ValueError("t_schedule must be strictly decreasing; adjust num_steps or sway_coeff.")
    dt_schedule = t_schedule[1:] - t_schedule[:-1]  # (num_steps,), fp32 on device
    t_model = t_schedule.to(dtype=dtype)  # bf16/fp32 view of t for the model

    if encoded_conditions is None:
        encoded_conditions = model.encode_conditions(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            speaker_state_override=speaker_state_override,
            speaker_mask_override=speaker_mask_override,
            speaker_uncond_mode=speaker_uncond_mode,
        )
    (
        text_state_cond,
        text_mask_cond,
        speaker_state_cond,
        speaker_mask_cond,
        caption_state_cond,
        caption_mask_cond,
    ) = encoded_conditions

    text_state_uncond = torch.zeros_like(text_state_cond)
    text_mask_uncond = torch.zeros_like(text_mask_cond)
    speaker_state_uncond = None
    speaker_mask_uncond = None
    if model.cfg.use_speaker_condition_resolved:
        if speaker_state_cond is None or speaker_mask_cond is None:
            raise RuntimeError(
                "Speaker conditioning is enabled but encoded speaker state is missing."
            )
        if speaker_uncond_mode == "noise":
            speaker_noise = torch.randn(
                speaker_state_cond.shape,
                device=rng_device,
                dtype=speaker_state_cond.dtype,
                generator=rng,
            )
            if rng_device != device:
                speaker_noise = speaker_noise.to(device=device)
            speaker_state_uncond = speaker_noise * speaker_state_cond.std().clamp_min(1e-6)
            speaker_mask_uncond = torch.ones_like(speaker_mask_cond)
        else:
            speaker_state_uncond = torch.zeros_like(speaker_state_cond)
            speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)
    caption_state_uncond = None
    caption_mask_uncond = None
    if model.cfg.use_caption_condition:
        if caption_state_cond is None or caption_mask_cond is None:
            raise RuntimeError(
                "Caption conditioning is enabled but encoded caption state is missing."
            )
        caption_state_uncond = torch.zeros_like(caption_state_cond)
        caption_mask_uncond = torch.zeros_like(caption_mask_cond)

    has_text_cfg = cfg_scale_text > 0
    if has_caption is None:
        has_caption = bool(
            caption_mask_cond is not None and bool(caption_mask_cond.any().item())
        )
    has_caption_cfg = bool(
        model.cfg.use_caption_condition and cfg_scale_caption > 0 and has_caption
    )
    has_speaker_cfg = cfg_scale_speaker > 0

    cond_bundle: _Bundle = (
        text_state_cond,
        text_mask_cond,
        speaker_state_cond,
        speaker_mask_cond,
        caption_state_cond,
        caption_mask_cond,
    )
    enabled_cfg_names: list[str] = []
    cfg_scales: dict[str, float] = {}
    if has_text_cfg:
        enabled_cfg_names.append("text")
        cfg_scales["text"] = float(cfg_scale_text)
    if has_speaker_cfg:
        enabled_cfg_names.append("speaker")
        cfg_scales["speaker"] = float(cfg_scale_speaker)
    if has_caption_cfg:
        enabled_cfg_names.append("caption")
        cfg_scales["caption"] = float(cfg_scale_caption)

    def _uncond_bundle(drop: set[str]) -> _Bundle:
        return (
            text_state_uncond if "text" in drop else text_state_cond,
            text_mask_uncond if "text" in drop else text_mask_cond,
            speaker_state_uncond if "speaker" in drop else speaker_state_cond,
            speaker_mask_uncond if "speaker" in drop else speaker_mask_cond,
            caption_state_uncond if "caption" in drop else caption_state_cond,
            caption_mask_uncond if "caption" in drop else caption_mask_cond,
        )

    use_independent_cfg = cfg_guidance_mode == "independent"
    use_joint_cfg = cfg_guidance_mode == "joint"
    use_alternating_cfg = cfg_guidance_mode == "alternating"

    independent_bundles = [cond_bundle]
    independent_names = ["cond"]
    if use_independent_cfg:
        for name in enabled_cfg_names:
            independent_names.append(name)
            independent_bundles.append(_uncond_bundle({name}))
    cfg_batch_mult = len(independent_bundles)
    independent_bundle = _cat_bundles(independent_bundles) if cfg_batch_mult > 1 else None

    joint_bundle = _uncond_bundle({"text", "speaker", "caption"})
    if use_joint_cfg and len(enabled_cfg_names) > 1:
        joint_scales = [cfg_scales[name] for name in enabled_cfg_names]
        if max(joint_scales) - min(joint_scales) > 1e-6:
            raise ValueError(
                "cfg_guidance_mode='joint' expects equal enabled guidance scales; "
                "set matching text/speaker/caption scales or use --cfg-scale."
            )
    alternating_bundles: dict[str, _Bundle] = {}
    if use_alternating_cfg:
        for name in enabled_cfg_names:
            alternating_bundles[name] = _uncond_bundle({name})

    effective_use_context_kv_cache = bool(use_context_kv_cache or (speaker_kv_scale is not None))
    context_kv_cond = None
    context_kv_cfg = None
    context_kv_joint_uncond = None
    context_kv_alternating: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    if effective_use_context_kv_cache:
        context_kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
            caption_state=caption_state_cond,
        )
        if use_independent_cfg and independent_bundle is not None:
            context_kv_cfg = model.build_context_kv_cache(
                text_state=independent_bundle[0],
                speaker_state=independent_bundle[2],
                caption_state=independent_bundle[4],
            )
        elif use_joint_cfg and enabled_cfg_names:
            context_kv_joint_uncond = model.build_context_kv_cache(
                text_state=joint_bundle[0],
                speaker_state=joint_bundle[2],
                caption_state=joint_bundle[4],
            )
        elif use_alternating_cfg:
            for name in enabled_cfg_names:
                bundle = alternating_bundles[name]
                context_kv_alternating[name] = model.build_context_kv_cache(
                    text_state=bundle[0],
                    speaker_state=bundle[2],
                    caption_state=bundle[4],
                )
    all_kv_caches = [c for c in (context_kv_cond, context_kv_cfg) if c is not None] + list(
        context_kv_alternating.values()
    )
    if speaker_kv_scale is not None:
        for cache in all_kv_caches:
            scale_speaker_kv_cache(
                context_kv_cache=cache,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
    speaker_kv_active = speaker_kv_scale is not None

    state = _FastSamplerState(
        model=model,
        cond_bundle=cond_bundle,
        context_kv_cond=context_kv_cond,
        cfg_batch_mult=cfg_batch_mult,
        independent_bundle=independent_bundle,
        context_kv_cfg=context_kv_cfg,
        independent_names=independent_names,
        joint_bundle=joint_bundle if use_joint_cfg else None,
        context_kv_joint=context_kv_joint_uncond,
        alternating_bundles=alternating_bundles,
        context_kv_alternating=context_kv_alternating,
        enabled_cfg_names=enabled_cfg_names,
        cfg_scales=cfg_scales,
        cfg_guidance_mode=cfg_guidance_mode,
        rescale_k=rescale_k,
        rescale_sigma=rescale_sigma,
        latent_len=int(sequence_length),
        latent_mask=None,
        ane_runner=get_ane_runner(model, opt) if opt.ane else None,
        ane_gpu_branches=int(opt.ane_gpu_branches),
        ane_gpu_cond=bool(opt.ane_gpu_cond),
        ane_nocfg_gpu=bool(opt.ane_nocfg_gpu),
        ane_candidates=bool(opt.ane_candidates),
    )

    # Precompute every attention mask this request can use up-front (one allocation
    # per mask instead of one per step).
    state.prepare_masks(use_cfg=False, alt_index=0)
    if enabled_cfg_names:
        for alt_index in range(len(enabled_cfg_names) if use_alternating_cfg else 1):
            state.prepare_masks(use_cfg=True, alt_index=alt_index)

    for i in range(num_steps):
        t = t_list[i]
        tt = t_model[i : i + 1].expand(batch_size)
        dt = dt_schedule[i : i + 1]
        use_cfg = bool(enabled_cfg_names) and (cfg_min_t <= t <= cfg_max_t)
        x_t = state.step(x_t, tt, dt, use_cfg=use_cfg, step_index=i, t_value=t)

        t_next = t_list[i + 1]
        if (
            speaker_kv_active
            and speaker_kv_min_t is not None
            and (t_next < speaker_kv_min_t)
            and (t >= speaker_kv_min_t)
        ):
            inv_scale = 1.0 / float(speaker_kv_scale)
            for cache in all_kv_caches:
                scale_speaker_kv_cache(
                    context_kv_cache=cache,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            speaker_kv_active = False

    if state.ane_runner is not None:
        state.release()
        stats = state.ane_runner.reset_stats()  # type: ignore[attr-defined]
        if opt.ane_log:
            used = "ane" if state.ane_active else "fallback:mps"
            print(
                f"[ane] {used} steps={stats['steps']} predict={stats['predict_sec'] * 1000:.0f} ms "
                f"wait={stats['wait_sec'] * 1000:.0f} ms gpu_branches={state.ane_gpu_branches}",
                flush=True,
            )
    return x_t


def sample_euler_rf_cfg(
    model: TextToLatentRFDiT,
    *args: object,
    encoded_conditions: _Bundle | None = None,
    has_caption: bool | None = None,
    opt: OptConfig | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Dispatch to the fast sampler (default) or the legacy reference implementation."""
    opt = get_opt_config() if opt is None else opt
    if opt.fast_sampler:
        return _sample_euler_rf_cfg_fast(
            model,
            *args,  # type: ignore[arg-type]
            encoded_conditions=encoded_conditions,
            has_caption=has_caption,
            opt=opt,
            **kwargs,  # type: ignore[arg-type]
        )
    return _sample_euler_rf_cfg_legacy(model, *args, **kwargs)  # type: ignore[arg-type]
