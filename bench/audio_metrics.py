#!/usr/bin/env python3
"""Objective distance metrics between two WAV files (or directories of same-named WAVs).

Metrics:
  - max abs sample diff, RMS diff (dB relative to reference RMS)
  - log-mel spectral distance (LSD, dB) on 48 kHz audio, 128 mel bins
  - duration difference (ms)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torchaudio


def load(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    return wav, int(sr)


def log_mel(wav: torch.Tensor, sr: int) -> torch.Tensor:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0
    )(wav)
    # Floor at -80 dB relative to the peak so near-silent frames do not dominate the distance.
    floor = mel.max() * 1e-8
    return 10.0 * torch.log10(mel.clamp_min(floor))


def compare(a_path: Path, b_path: Path) -> dict[str, float]:
    a, sr_a = load(a_path)
    b, sr_b = load(b_path)
    assert sr_a == sr_b
    n = min(a.numel(), b.numel())
    dur_diff_ms = abs(a.numel() - b.numel()) / sr_a * 1000.0
    a, b = a[:n], b[:n]
    max_abs = (a - b).abs().max().item()
    rms_ref = a.pow(2).mean().sqrt().item()
    rms_diff = (a - b).pow(2).mean().sqrt().item()
    snr_db = 20.0 * math.log10(max(rms_ref, 1e-9) / max(rms_diff, 1e-9))
    la, lb = log_mel(a, sr_a), log_mel(b, sr_b)
    # Use a common floor for both signals.
    common = max(la.min().item(), lb.min().item())
    la, lb = la.clamp_min(common), lb.clamp_min(common)
    lsd = ((la - lb) ** 2).mean(dim=0).sqrt().mean().item()
    return {
        "max_abs": max_abs,
        "snr_db": snr_db,
        "lsd_db": lsd,
        "dur_diff_ms": dur_diff_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()
    a, b = Path(args.a), Path(args.b)
    pairs: list[tuple[Path, Path]]
    if a.is_dir():
        pairs = [(p, b / p.name) for p in sorted(a.glob("*.wav")) if (b / p.name).exists()]
    else:
        pairs = [(a, b)]
    for pa, pb in pairs:
        m = compare(pa, pb)
        print(
            f"{pa.name} vs {pb.name}: max_abs={m['max_abs']:.3f} snr={m['snr_db']:.1f}dB "
            f"lsd={m['lsd_db']:.2f}dB dur_diff={m['dur_diff_ms']:.0f}ms"
        )


if __name__ == "__main__":
    main()
