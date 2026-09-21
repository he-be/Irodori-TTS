#!/usr/bin/env python3
"""Segment-by-segment long-script benchmark (what the longform Gradio app does).

Splits a ~3 minute script with the production splitter and synthesizes every segment in
order, reporting time-to-first-audio, total wall, and the realtime factor of the whole run.

    uv run --no-sync python bench/bench_longform.py --tag m1_ane --num-steps 12
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)
from irodori_tts.text_segmentation import split_script  # noqa: E402

SCRIPT = """
自分の書いた原稿を読み上げさせたい。クラウドの音声合成サービスは文字数で課金されるし、書きかけの原稿を毎回どこかに送りたくない。
何より、一度決めた声が来月も同じである保証が欲しい。ローカルで動くなら、この三つが全部解決する。

イロドリTTSは、日本語の音声合成モデルだ。フローマッチングで音声の潜在表現を生成して、コーデックで波形に戻す。
数秒の参照音声を渡すと、その声で喋る。テキストで落ち着いた低い声で、と指示して声を作ることもできる。
日本語の読みが素直で、ライセンスはMIT。学習コードまで公開されている。

今回の実機はマックミニだ。アップルのM1チップ、メモリは十六ギガバイト。ファンがついているので、長時間まわしても性能が落ちにくい。
これをネットワーク越しの読み上げサーバーとして使えるかどうかを、実際に測って確かめた。

測ってみると、素の状態では音声一秒あたり一秒以上かかっていた。実時間より遅い。
そこで、積分のステップ数を四十から十二に減らし、ニューラルエンジンに計算を逃がし、コンパイルを効かせた。
結果として、三分の原稿がどれくらいで音声になるのかが、この測定の答えになる。
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    p.add_argument("--ref", default=str(REPO_ROOT / "outputs" / "sample.wav"))
    p.add_argument("--num-steps", type=int, default=12)
    p.add_argument("--t-schedule-mode", default="sway")
    p.add_argument("--sway-coeff", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tag", default="longform")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    segments = split_script(SCRIPT)
    ck = download_hf_checkpoint(args.hf_checkpoint)
    t0 = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=ck,
            model_device="mps",
            model_precision="fp16",
            codec_device="mps",
            codec_precision="fp32",
        )
    )
    load_sec = time.perf_counter() - t0

    def request(text: str, seed: int) -> SamplingRequest:
        return SamplingRequest(
            text=text,
            ref_wav=args.ref,
            num_steps=int(args.num_steps),
            t_schedule_mode=args.t_schedule_mode,
            sway_coeff=float(args.sway_coeff),
            num_candidates=1,
            cfg_guidance_mode="independent",
            seed=seed,
            trim_tail=True,
        )

    # warm the caches the way a running server would already be warm
    runtime.synthesize(request(segments[0].text, args.seed))
    torch.mps.synchronize()

    rows = []
    audio_total = 0.0
    wall0 = time.perf_counter()
    first_done = None
    for i, seg in enumerate(segments):
        t = time.perf_counter()
        out = runtime.synthesize(request(seg.text, args.seed + i))
        torch.mps.synchronize()
        dt = time.perf_counter() - t
        if first_done is None:
            first_done = time.perf_counter() - wall0
        audio = out.audio if hasattr(out, "audio") else out[0]
        secs = float(audio.shape[-1]) / float(getattr(out, "sample_rate", 44100))
        audio_total += secs
        rows.append({"i": i, "chars": len(seg.text), "wall_ms": dt * 1000, "audio_s": secs})
        print(f"[{args.tag}] seg {i:2d} {len(seg.text):3d} chars -> {secs:5.2f}s audio in {dt*1000:7.0f} ms", flush=True)
    wall = time.perf_counter() - wall0

    out = {
        "tag": args.tag,
        "segments": len(segments),
        "load_sec": load_sec,
        "audio_seconds_total": audio_total,
        "wall_seconds_total": wall,
        "rtf_total": wall / audio_total,
        "time_to_first_audio_s": first_done,
        "per_segment_ms_median": statistics.median(r["wall_ms"] for r in rows),
        "rows": rows,
        "config": vars(args),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"[bench] wrote {args.output}")


if __name__ == "__main__":
    main()
