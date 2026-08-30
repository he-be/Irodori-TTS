#!/usr/bin/env python3
"""Long-run churn on one device: cycle requests of very different lengths for N rounds under a
fixed allocator cap and record OOMs, allocator peaks, and the *sampled* (50 ms) peak of the
process-independent amdgpu GTT/VRAM counters (docs/experiments/12, 13.4).

  HSA_OVERRIDE_GFX_VERSION=9.0.0 IRODORI_OPT_TE_DEVICE=cpu IRODORI_OPT_VRAM_LIMIT_MB=2560 \
    .venv-rocm/bin/python bench/churn_igpu.py --rounds 6 --precision fp16 --codec-precision fp16
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "bench"))

from bench_runtime import DEFAULT_REF, INPUTS  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)


class SysfsPeak:
    def __init__(self, card_dir: Path, interval: float = 0.05) -> None:
        self.files = {k: card_dir / "device" / f"mem_info_{k}_used" for k in ("gtt", "vram")}
        self.peak = {k: 0 for k in self.files}
        self.baseline = {k: 0 for k in self.files}
        self.interval = interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def read(self) -> dict[str, int]:
        return {k: int(f.read_text()) // 2**20 for k, f in self.files.items()}

    def start(self) -> None:
        self.baseline = self.read()
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for k, v in self.read().items():
                    self.peak[k] = max(self.peak[k], v)
            except OSError:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)


def _amd_card() -> Path | None:
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        try:
            if (card / "device" / "driver").resolve().name == "amdgpu":
                return card
        except OSError:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--codec-precision", default="fp16")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--num-steps", type=int, default=12)
    ap.add_argument("--max-text", type=int, default=256)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ck = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
    card = _amd_card()
    sampler = SysfsPeak(card) if card else None
    if sampler:
        sampler.start()
    rt = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ck, model_device="cuda", model_precision=args.precision,
                   codec_device="cuda", codec_precision=args.codec_precision)
    )
    long_text = (str(INPUTS["long"]["text"]) * 4)[: args.max_text]
    caption = INPUTS["caption_noref"]["caption"]
    plan = [
        ("short", dict(text=str(INPUTS["short"]["text"]), ref_wav=DEFAULT_REF)),
        ("text_max", dict(text=long_text, ref_wav=DEFAULT_REF)),
        ("caption_noref", dict(text=str(INPUTS["caption_noref"]["text"]), caption=caption, no_ref=True)),
        ("long", dict(text=str(INPUTS["long"]["text"]), ref_wav=DEFAULT_REF)),
        ("medium", dict(text=str(INPUTS["medium"]["text"]), ref_wav=DEFAULT_REF)),
        ("worst30", dict(text=long_text, caption=caption, ref_wav=DEFAULT_REF)),
    ]
    log: list[dict] = []
    oom = 0
    t_all = time.perf_counter()
    for r in range(args.rounds):
        for name, kw in plan:
            req = SamplingRequest(num_steps=args.num_steps, t_schedule_mode="sway", seed=1234 + r, **kw)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                res = rt.synthesize(req)
                torch.cuda.synchronize()
                ok, secs = True, float(res.audio.shape[-1]) / float(res.sample_rate)
            except RuntimeError as e:  # noqa: BLE001
                ok, secs = False, 0.0
                oom += 1
                print(f"  !! {name}: {str(e)[:120]}", flush=True)
            entry = {
                "round": r, "case": name, "ok": ok, "audio_s": round(secs, 2),
                "wall_s": round(time.perf_counter() - t0, 2),
                "peak_alloc_mib": torch.cuda.max_memory_allocated() // 2**20,
                "reserved_mib": torch.cuda.memory_reserved() // 2**20,
                "gtt_peak_mib": sampler.peak["gtt"] if sampler else None,
            }
            log.append(entry)
            print(f"[{r:02d}] {name:14s} ok={ok} audio={secs:5.1f}s wall={entry['wall_s']:6.1f}s "
                  f"alloc_peak={entry['peak_alloc_mib']} reserved={entry['reserved_mib']} "
                  f"gtt_peak={entry['gtt_peak_mib']}", flush=True)
    if sampler:
        sampler.stop()
    summary = {
        "rounds": args.rounds, "requests": len(log), "oom": oom,
        "total_s": round(time.perf_counter() - t_all, 1),
        "max_peak_alloc_mib": max(e["peak_alloc_mib"] for e in log),
        "max_reserved_mib": max(e["reserved_mib"] for e in log),
        "sysfs_baseline_mib": sampler.baseline if sampler else None,
        "sysfs_peak_mib": sampler.peak if sampler else None,
        "cap_mb": __import__("irodori_tts.opt_config", fromlist=["x"]).get_opt_config().vram_limit_mb,
        "te_device": __import__("irodori_tts.opt_config", fromlist=["x"]).get_opt_config().text_encoder_device,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps({"summary": summary, "log": log}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
