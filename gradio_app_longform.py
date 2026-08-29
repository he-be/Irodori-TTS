#!/usr/bin/env python3
"""Gradio UI for reading a long script: split it, synthesize segment by segment,
fix a misread segment by editing its text, and write one wav for the whole thing.

Why segments: the runtime synthesizes a whole utterance in one shot and gets less
efficient the longer it is (73 / 79 / 136 ms of DiT per second of audio at 7.2 /
11.8 / 28.8 s, and 16 sampler steps past 20 s), and a single request has a 30 s
ceiling. Sequential segments of 7-12 s keep the cost per second flat and make a
misread a local, cheap fix instead of a full re-run
(docs/experiments/14-step-count.md, 15-decode-ane.md).
"""

from __future__ import annotations

import os as _os

# Persistent server: the one-time inductor/MPS compile (~20 s DiT + ~4 s codec at the first
# request) pays for itself (-17% wall, -30% decode; docs/experiments/12-metal-port.md).
# Override with IRODORI_OPT_COMPILE_DIT=0 / IRODORI_OPT_COMPILE_CODEC=0 for a fast first request.
_os.environ.setdefault("IRODORI_OPT_COMPILE_DIT", "1")
_os.environ.setdefault("IRODORI_OPT_COMPILE_CODEC", "1")
# RF step on the Neural Engine with the cond CFG branch on the GPU (13-ane.md). The first
# start with an empty ~/.cache/irodori-tts/ane builds the Core ML packages (minutes);
# `uv run python bench/build_ane.py --shapes full` does that ahead of time.
_os.environ.setdefault("IRODORI_OPT_ANE", "1")
_os.environ.setdefault("IRODORI_OPT_ANE_GPU_BRANCHES", "1")
_os.environ.setdefault("IRODORI_OPT_ANE_SHAPES", "full")

import argparse
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
import torchaudio

from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS, build_emoji_palette
from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    download_hf_checkpoint,
    get_cached_runtime,
    save_wav,
)
from irodori_tts.speaker_inversion import is_speaker_inversion_safetensors_path
from irodori_tts.text_segmentation import (
    CHARS_PER_SECOND,
    DEFAULT_MAX_CHARS,
    HARD_MAX_CHARS,
    split_script,
)

# 64 segments x ~12 s is a bit over 12 minutes of speech. The rows are built up front
# (Gradio needs a fixed component tree) and hidden until a split fills them.
MAX_SEGMENTS = 64
OUTPUT_ROOT = Path("gradio_outputs_longform")

# This build has one backend: the RF step runs on the Neural Engine with the cond CFG branch
# on the GPU, everything else (encoders, codec) on MPS fp16 (13-ane.md).
MODEL_DEVICE = "mps"
MODEL_PRECISION = "fp16"
CODEC_DEVICE = "mps"
CODEC_PRECISION = "fp16"
BACKEND_LABEL = (
    "backend: RF step on ANE (Core ML) + cond CFG branch on GPU / encoders + codec on MPS fp16"
)

# Same defaults as gradio_app.py's Advanced panel, which this UI does not expose.
CONTEXT_KV_CACHE = True
SPEAKER_KV_MIN_T = 0.9
REF_NORMALIZE_DB = -16.0
REF_ENSURE_MAX = True

# A blank line in the script gets this multiple of the inter-segment gap.
PARAGRAPH_GAP_SCALE = 2.0


def _default_checkpoint() -> str:
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *(
                path
                for path in Path(".").glob("**/checkpoint_*.safetensors")
                if not is_speaker_inversion_safetensors_path(path)
            ),
        ]
    )
    if not candidates:
        return "Aratako/Irodori-TTS-v4.1-Small"
    return str(candidates[-1])


def _on_t_schedule_mode_change(mode: str) -> object:
    return gr.update(interactive=str(mode).strip().lower() == "sway")


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _coerce_gradio_file_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        return None
    candidate = getattr(value, "name", None)
    if candidate is not None and str(candidate).strip():
        return str(candidate)
    text = str(value).strip()
    return text or None


def _resolve_ref_wavs(uploaded_audio: object) -> list[str]:
    if uploaded_audio is None:
        return []
    values = uploaded_audio if isinstance(uploaded_audio, (list, tuple)) else [uploaded_audio]
    paths = [_coerce_gradio_file_path(value) for value in values]
    return [path for path in paths if path is not None]


def _resolve_speaker_embedding(
    uploaded_embedding: object,
    speaker_embedding_path_raw: str | None,
) -> str | None:
    uploaded_path = _coerce_gradio_file_path(uploaded_embedding)
    raw_path = None
    if speaker_embedding_path_raw is not None and str(speaker_embedding_path_raw).strip():
        raw_path = str(speaker_embedding_path_raw).strip()
    if uploaded_path is not None and raw_path is not None:
        raise ValueError("Use either speaker embedding upload or speaker embedding path, not both.")
    return uploaded_path if uploaded_path is not None else raw_path


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    resolved = download_hf_checkpoint(checkpoint)
    print(f"[longform] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _build_runtime_key(checkpoint: str) -> RuntimeKey:
    return RuntimeKey(
        checkpoint=_resolve_checkpoint_path(checkpoint),
        model_device=MODEL_DEVICE,
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=MODEL_PRECISION,
        codec_device=CODEC_DEVICE,
        codec_precision=CODEC_PRECISION,
        compile_model=False,
        compile_dynamic=False,
    )


def _load_model(checkpoint: str) -> str:
    runtime_key = _build_runtime_key(checkpoint)
    _, reloaded = get_cached_runtime(runtime_key)
    status = "loaded model into memory" if reloaded else "model already loaded; reused runtime"
    return f"{status}\ncheckpoint: {runtime_key.checkpoint}"


def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


# ---------------------------------------------------------------- request parameters


@dataclass(frozen=True)
class _Params:
    """The shared settings, read fresh from the UI on every generate / regenerate.

    The field order here is the order of `param_components` in build_ui(); both must match.
    """

    checkpoint: str
    ref_wavs: list[str]
    speaker_embedding: str | None
    num_steps: int
    base_seed: int | None
    duration_scale: float
    t_schedule_mode: str
    sway_coeff: float
    cfg_guidance_mode: str
    cfg_scale_text: float
    cfg_scale_speaker: float
    gap_seconds: float
    trim_silence: bool

    @classmethod
    def from_args(cls, args: tuple[object, ...]) -> _Params:
        (
            checkpoint,
            uploaded_audio,
            uploaded_speaker_embedding,
            speaker_embedding_path_raw,
            num_steps,
            seed_raw,
            duration_scale,
            t_schedule_mode,
            sway_coeff,
            cfg_guidance_mode,
            cfg_scale_text,
            cfg_scale_speaker,
            gap_seconds,
            trim_silence,
        ) = args

        ref_wavs = _resolve_ref_wavs(uploaded_audio)
        speaker_embedding = _resolve_speaker_embedding(
            uploaded_embedding=uploaded_speaker_embedding,
            speaker_embedding_path_raw=speaker_embedding_path_raw,
        )
        if ref_wavs and speaker_embedding is not None:
            raise ValueError("Reference audio and speaker embedding are mutually exclusive.")

        return cls(
            checkpoint=str(checkpoint),
            ref_wavs=ref_wavs,
            speaker_embedding=speaker_embedding,
            num_steps=int(num_steps),
            base_seed=_parse_optional_int(seed_raw, "seed"),
            duration_scale=float(duration_scale),
            t_schedule_mode=str(t_schedule_mode),
            sway_coeff=float(sway_coeff),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker),
            gap_seconds=float(gap_seconds),
            trim_silence=bool(trim_silence),
        )


# ---------------------------------------------------------------- audio helpers


def _trim_edges(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Drop the near-silent head and tail so segment joins do not stack up pauses.

    The threshold is relative to the segment's own peak (-40 dB) and 50 ms is kept on
    each side, so a soft onset survives. Unverified by ear; the checkbox turns it off.
    """
    if audio.numel() == 0:
        return audio
    envelope = audio.abs().amax(dim=0)
    peak = float(envelope.max())
    if peak <= 0.0:
        return audio
    voiced = (envelope > peak * (10.0 ** (-40.0 / 20.0))).nonzero()
    if voiced.numel() == 0:
        return audio
    margin = int(0.05 * sample_rate)
    start = max(0, int(voiced[0].item()) - margin)
    end = min(int(envelope.shape[-1]), int(voiced[-1].item()) + 1 + margin)
    return audio[:, start:end]


def _load_wav(path: str) -> tuple[torch.Tensor, int]:
    audio, sample_rate = torchaudio.load(str(path))
    return audio.to(dtype=torch.float32), int(sample_rate)


def _concat_segments(segments: list[dict], gap_seconds: float) -> tuple[torch.Tensor, int]:
    pieces: list[torch.Tensor] = []
    sample_rate: int | None = None
    for segment in segments:
        path = segment.get("path")
        if not path or not Path(path).is_file():
            continue
        audio, file_rate = _load_wav(path)
        if sample_rate is None:
            sample_rate = file_rate
        elif file_rate != sample_rate:
            raise ValueError(f"sample rate mismatch: {file_rate} != {sample_rate}")
        if pieces:
            gap = gap_seconds * (PARAGRAPH_GAP_SCALE if segment.get("paragraph_break") else 1.0)
            silence = torch.zeros(audio.shape[0], max(0, int(gap * sample_rate)))
            pieces.append(silence)
        pieces.append(audio)
    if not pieces or sample_rate is None:
        raise ValueError("No generated segment to join yet.")
    return torch.cat(pieces, dim=-1), sample_rate


# ---------------------------------------------------------------- state


def _new_state() -> dict:
    return {"run_dir": None, "segments": [], "full_path": None}


def _segment_entry(text: str, paragraph_break: bool) -> dict:
    return {
        "text": text,
        # The text that produced `path`; a mismatch is what marks a segment stale.
        "generated_text": None,
        "paragraph_break": bool(paragraph_break),
        "path": None,
        "seconds": 0.0,
        "seed": None,
        "wall": 0.0,
        "version": 0,
    }


def _run_dir(state: dict) -> Path:
    if not state.get("run_dir"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state["run_dir"] = str(OUTPUT_ROOT / f"run_{stamp}")
    path = Path(state["run_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _segment_label(index: int, segment: dict) -> str:
    seconds = float(segment.get("seconds") or 0.0)
    if seconds > 0.0:
        return f"{index + 1}: {seconds:.1f} s (seed {segment.get('seed')})"
    return f"{index + 1}"


def _text_label(index: int, segment: dict) -> str:
    estimate = len(segment["text"]) / CHARS_PER_SECOND
    return f"セグメント {index + 1} — {len(segment['text'])} 文字 / 推定 {estimate:.1f} s"


def _total_seconds(state: dict) -> float:
    return sum(float(seg.get("seconds") or 0.0) for seg in state["segments"])


def _status_text(state: dict, lines: list[str]) -> str:
    segments = state["segments"]
    done = sum(1 for seg in segments if seg.get("path"))
    stale = sum(1 for seg in segments if seg.get("path") and seg["generated_text"] != seg["text"])
    header = [
        f"セグメント: {len(segments)}  生成済み: {done}"
        + (f"  要再生成: {stale}" if stale else ""),
        f"生成済み音声の合計: {_total_seconds(state):.1f} s",
    ]
    if state.get("full_path"):
        header.append(f"結合ファイル: {state['full_path']}")
    return "\n".join([*header, "", *lines])


# ---------------------------------------------------------------- synthesis


def _synthesize(runtime, text: str, seed: int, params: _Params):
    return runtime.synthesize(
        SamplingRequest(
            text=text,
            ref_wav=None,
            ref_wavs=params.ref_wavs or None,
            ref_latent=None,
            ref_embed=params.speaker_embedding,
            no_ref=not params.ref_wavs and params.speaker_embedding is None,
            ref_normalize_db=REF_NORMALIZE_DB,
            ref_ensure_max=REF_ENSURE_MAX,
            num_candidates=1,
            decode_mode="sequential",
            seconds=None,
            duration_scale=params.duration_scale,
            max_ref_seconds=None,
            max_text_len=None,
            num_steps=params.num_steps,
            seed=seed,
            cfg_guidance_mode=params.cfg_guidance_mode,
            cfg_scale_text=params.cfg_scale_text,
            cfg_scale_speaker=params.cfg_scale_speaker,
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=CONTEXT_KV_CACHE,
            speaker_kv_scale=None,
            speaker_kv_min_t=SPEAKER_KV_MIN_T,
            speaker_kv_max_layers=None,
            t_schedule_mode=params.t_schedule_mode,
            sway_coeff=params.sway_coeff,
            trim_tail=True,
            lora_adapter=None,
        ),
        log_fn=lambda msg: print(msg, flush=True),
    )


def _generate_one(runtime, state: dict, index: int, seed: int, params: _Params) -> dict:
    """Synthesize one segment, write its wav, and record it in `state`."""
    segment = state["segments"][index]
    text = segment["text"].strip()
    if text == "":
        raise ValueError(f"セグメント {index + 1} が空です。")

    wall0 = time.perf_counter()
    result = _synthesize(runtime, text, seed, params)
    wall = time.perf_counter() - wall0

    audio = result.audio.float().cpu()
    if params.trim_silence:
        audio = _trim_edges(audio, result.sample_rate)

    segment["version"] = int(segment.get("version", 0)) + 1
    out_path = save_wav(
        _run_dir(state) / f"seg_{index + 1:03d}_v{segment['version']}.wav",
        audio,
        result.sample_rate,
    )
    segment["path"] = str(out_path)
    segment["generated_text"] = segment["text"]
    segment["seconds"] = float(audio.shape[-1]) / float(result.sample_rate)
    segment["seed"] = int(result.used_seed)
    segment["wall"] = wall
    return segment


def _write_full(state: dict, gap_seconds: float) -> str:
    audio, sample_rate = _concat_segments(state["segments"], gap_seconds)
    stamp = datetime.now().strftime("%H%M%S")
    out_path = save_wav(_run_dir(state) / f"full_{stamp}.wav", audio, sample_rate)
    state["full_path"] = str(out_path)
    return str(out_path)


# ---------------------------------------------------------------- event handlers


def _audio_updates(state: dict, changed: set[int]) -> list[object]:
    """No-op updates everywhere except the segments that actually changed, so the
    other players keep their playback position."""
    updates: list[object] = []
    for index in range(MAX_SEGMENTS):
        if index not in changed:
            updates.append(gr.update())
            continue
        segment = state["segments"][index]
        updates.append(gr.update(value=segment.get("path"), label=_segment_label(index, segment)))
    return updates


def _split(text: str, max_chars: float, state: dict) -> tuple[object, ...]:
    if str(text).strip() == "":
        raise gr.Error("テキストを入力してください。")
    segments = split_script(str(text), max_chars=int(max_chars))
    if not segments:
        raise gr.Error("分割できるテキストがありません。")
    if len(segments) > MAX_SEGMENTS:
        raise gr.Error(
            f"{len(segments)} セグメントになりました。上限は {MAX_SEGMENTS} です。"
            "テキストを分けるか、1 セグメントの文字数を増やしてください。"
        )

    # Keep the audio of segments whose text is unchanged, so editing the tail of a
    # script and splitting again does not throw away what is already generated.
    reusable: dict[str, list[dict]] = {}
    for old in state.get("segments", []):
        if old.get("path"):
            reusable.setdefault(old["generated_text"], []).append(old)

    new_state = _new_state()
    new_state["run_dir"] = state.get("run_dir")
    for segment in segments:
        entry = _segment_entry(segment.text, segment.paragraph_break)
        pool = reusable.get(segment.text)
        if pool:
            old = pool.pop(0)
            entry.update(
                {
                    "generated_text": old["generated_text"],
                    "path": old["path"],
                    "seconds": old["seconds"],
                    "seed": old["seed"],
                    "wall": old["wall"],
                    "version": old["version"],
                }
            )
        new_state["segments"].append(entry)

    row_updates = [gr.update(visible=i < len(segments)) for i in range(MAX_SEGMENTS)]
    text_updates = [
        gr.update(
            value=new_state["segments"][i]["text"],
            label=_text_label(i, new_state["segments"][i]),
        )
        if i < len(segments)
        else gr.update(value="")
        for i in range(MAX_SEGMENTS)
    ]
    audio_updates = [
        gr.update(
            value=new_state["segments"][i].get("path"),
            label=_segment_label(i, new_state["segments"][i]),
        )
        if i < len(segments)
        else gr.update(value=None)
        for i in range(MAX_SEGMENTS)
    ]

    estimate = sum(len(s.text) for s in segments) / CHARS_PER_SECOND
    reused = sum(1 for seg in new_state["segments"] if seg.get("path"))
    lines = [
        f"{len(segments)} セグメントに分割しました（推定 {estimate:.0f} s = {estimate / 60:.1f} 分）。",
        *([f"うち {reused} 本は前回の音声をそのまま使えます。"] if reused else []),
        "テキストを直してから「生成」を押してください。",
    ]
    return (
        *row_updates,
        *text_updates,
        *audio_updates,
        gr.update(value=None),
        _status_text(new_state, lines),
        new_state,
    )


def _generate(*args: object):
    state: dict = args[0]
    force_all = bool(args[1])
    seg_texts = [str(x) for x in args[2 : 2 + MAX_SEGMENTS]]
    params = _Params.from_args(tuple(args[2 + MAX_SEGMENTS :]))

    state = dict(state)
    state["segments"] = [dict(seg) for seg in state.get("segments", [])]
    if not state["segments"]:
        raise gr.Error("先に「テキストを分割」を押してください。")

    # The textboxes are the source of truth: the user may have fixed a misreading.
    for index, segment in enumerate(state["segments"]):
        segment["text"] = seg_texts[index].strip()

    targets = [
        index
        for index, segment in enumerate(state["segments"])
        if force_all or not segment.get("path") or segment["generated_text"] != segment["text"]
    ]
    if not targets:
        yield (
            *_audio_updates(state, set()),
            gr.update(),
            _status_text(state, ["生成が必要なセグメントはありません。"]),
            state,
        )
        return

    runtime, reloaded = get_cached_runtime(_build_runtime_key(params.checkpoint))
    lines = [
        f"runtime: {'reloaded' if reloaded else 'reused'}",
        f"{len(targets)} / {len(state['segments'])} セグメントを生成します。",
    ]
    yield (*_audio_updates(state, set()), gr.update(), _status_text(state, lines), state)

    run0 = time.perf_counter()
    for order, index in enumerate(targets, start=1):
        segment = state["segments"][index]
        seed = (
            params.base_seed + index
            if params.base_seed is not None
            else random.randrange(2**31 - 1)
        )
        lines.append(f"[{order}/{len(targets)}] セグメント {index + 1} を生成中…")
        yield (*_audio_updates(state, set()), gr.update(), _status_text(state, lines), state)
        try:
            _generate_one(runtime, state, index, seed, params)
        except Exception as exc:  # surface the failure but keep what is already done
            lines[-1] = f"[{order}/{len(targets)}] セグメント {index + 1} で失敗: {exc}"
            yield (
                *_audio_updates(state, set()),
                gr.update(),
                _status_text(state, lines),
                state,
            )
            raise gr.Error(f"セグメント {index + 1} の生成に失敗しました: {exc}") from exc
        lines[-1] = (
            f"[{order}/{len(targets)}] セグメント {index + 1}: "
            f"{segment['seconds']:.1f} s / {segment['wall']:.2f} s "
            f"(RTF {segment['wall'] / max(segment['seconds'], 1e-6):.3f}, seed {segment['seed']})"
        )
        yield (*_audio_updates(state, {index}), gr.update(), _status_text(state, lines), state)

    full_path = _write_full(state, params.gap_seconds)
    wall = time.perf_counter() - run0
    total = _total_seconds(state)
    lines.append(
        f"完了: 音声 {total:.1f} s を {wall:.1f} s で生成 (RTF {wall / max(total, 1e-6):.3f})。"
    )
    yield (
        *_audio_updates(state, set()),
        gr.update(value=full_path),
        _status_text(state, lines),
        state,
    )


def _regenerate_one(index: int, args: tuple[object, ...]):
    state: dict = args[0]
    seg_text = str(args[1])
    params = _Params.from_args(tuple(args[2:]))

    state = dict(state)
    state["segments"] = [dict(seg) for seg in state.get("segments", [])]
    if index >= len(state["segments"]):
        raise gr.Error("このセグメントは存在しません。もう一度分割してください。")

    segment = state["segments"][index]
    segment["text"] = seg_text.strip()
    runtime, _ = get_cached_runtime(_build_runtime_key(params.checkpoint))
    # A re-take should differ even when the text is unchanged, so the seed is always new.
    seed = random.randrange(2**31 - 1)
    try:
        _generate_one(runtime, state, index, seed, params)
    except Exception as exc:
        raise gr.Error(f"セグメント {index + 1} の生成に失敗しました: {exc}") from exc

    lines = [
        f"セグメント {index + 1} を再生成しました: "
        f"{segment['seconds']:.1f} s / {segment['wall']:.2f} s (seed {segment['seed']})"
    ]
    full_update: object = gr.update()
    if any(seg.get("path") for seg in state["segments"]):
        full_path = _write_full(state, params.gap_seconds)
        full_update = gr.update(value=full_path)
        lines.append("結合ファイルを更新しました。")
    return (
        gr.update(value=segment["path"], label=_segment_label(index, segment)),
        full_update,
        _status_text(state, lines),
        state,
    )


def _rejoin(state: dict, gap_seconds: float):
    state = dict(state)
    state["segments"] = [dict(seg) for seg in state.get("segments", [])]
    try:
        full_path = _write_full(state, float(gap_seconds))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return (
        gr.update(value=full_path),
        _status_text(state, ["結合ファイルを書き出しました。"]),
        state,
    )


# ---------------------------------------------------------------- UI


def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()

    with gr.Blocks(title="Irodori-TTS Longform") as demo:
        gr.Markdown("# Irodori-TTS 長文読み上げ")
        gr.Markdown(
            "長い原稿を 7〜12 秒のセグメントに分割し、頭から順に生成して 1 本の wav にまとめます。"
            "読み間違いがあったセグメントはテキストを直して「再生成」すれば、そのセグメントと"
            "結合ファイルだけが更新されます。"
        )
        gr.Markdown(BACKEND_LABEL)

        state = gr.State(_new_state())

        with gr.Row():
            checkpoint = gr.Textbox(
                label="Checkpoint (.pt/.safetensors or HF repo id)",
                value=default_checkpoint,
                scale=4,
            )
            load_model_btn = gr.Button("Load Model", scale=1)
            clear_cache_btn = gr.Button("Unload Model", scale=1)
        model_status = gr.Textbox(label="Model Status", interactive=False)

        script_text = gr.Textbox(
            label="原稿（長文可）",
            lines=12,
            placeholder="ここに読み上げたい文章を貼り付けてください。空行は段落の区切りとして扱います。",
            elem_id="irodori-text-input",
        )
        build_emoji_palette(script_text, open=False)

        with gr.Tabs():
            with gr.Tab("Reference Audio"):
                gr.Markdown(
                    "**Long-reference tip:** Upload multiple clean, shorter clips from the same "
                    "speaker and arrange them in the desired order. すべてのセグメントで同じ参照を"
                    "使うので、声は 1 本目のキャッシュが再利用されます。"
                )
                uploaded_audio = gr.File(
                    label="Reference Audio Uploads (optional; concatenated in displayed order)",
                    type="filepath",
                    file_count="multiple",
                    file_types=["audio"],
                    allow_reordering=True,
                )
            with gr.Tab("Speaker Embedding"):
                with gr.Row():
                    uploaded_speaker_embedding = gr.File(
                        label="Speaker Embedding Upload (.speaker.safetensors, optional)",
                        type="filepath",
                        file_count="single",
                        scale=1,
                    )
                    speaker_embedding_path_raw = gr.Textbox(
                        label="Speaker Embedding Path (.speaker.safetensors, optional)",
                        value="",
                        scale=1,
                    )

        with gr.Accordion("分割 / 結合", open=True):
            with gr.Row():
                max_chars = gr.Slider(
                    label=f"1 セグメントの最大文字数（約 {CHARS_PER_SECOND:.1f} 文字/秒）",
                    minimum=20,
                    maximum=HARD_MAX_CHARS,
                    value=DEFAULT_MAX_CHARS,
                    step=5,
                )
                gap_seconds = gr.Slider(
                    label=f"セグメント間の無音 (秒、空行は {PARAGRAPH_GAP_SCALE:g} 倍)",
                    minimum=0.0,
                    maximum=1.5,
                    value=0.25,
                    step=0.05,
                )
                trim_silence = gr.Checkbox(label="前後の無音をトリム", value=True)
            gr.Markdown(
                "「テキストを分割」は原稿から一覧を作り直します（セグメント側の編集は上書きされます。"
                "テキストが変わっていないセグメントの音声はそのまま残ります）。"
                "「つなぎ直して書き出し」は生成せずに無音の長さだけ反映します。"
            )
            with gr.Row():
                split_btn = gr.Button("テキストを分割", variant="secondary")
                rejoin_btn = gr.Button("つなぎ直して書き出し")

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                # 12 sway steps, raised to 16 past 20 s of output by the runtime's auto-step
                # floor (docs/experiments/14-step-count.md).
                num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=12, step=1)
                seed_raw = gr.Textbox(
                    label="Seed (blank=random; 指定するとセグメント i は seed+i)", value=""
                )
                duration_scale = gr.Slider(
                    label="Duration Scale", minimum=0.5, maximum=1.5, value=1.0, step=0.01
                )
            with gr.Row():
                t_schedule_mode = gr.Dropdown(
                    label="Time Schedule", choices=["linear", "sway"], value="sway"
                )
                sway_coeff = gr.Slider(
                    label="Sway Coeff", minimum=-1.0, maximum=1.5, value=-1.0, step=0.1
                )
                cfg_guidance_mode = gr.Dropdown(
                    label="CFG Guidance Mode",
                    choices=["independent", "joint", "alternating"],
                    value="independent",
                )
            with gr.Row():
                cfg_scale_text = gr.Slider(
                    label="CFG Scale Text", minimum=0.0, maximum=10.0, value=3.0, step=0.1
                )
                cfg_scale_speaker = gr.Slider(
                    label="CFG Scale Speaker", minimum=0.0, maximum=10.0, value=5.0, step=0.1
                )

        with gr.Row():
            generate_btn = gr.Button("生成（未生成 / 編集されたセグメントのみ）", variant="primary")
            force_all = gr.Checkbox(label="すべて生成し直す", value=False)

        status = gr.Textbox(label="進捗", lines=10, interactive=False)
        full_audio = gr.Audio(label="全体（結合）", type="filepath", interactive=False)

        seg_rows: list[gr.Row] = []
        seg_texts: list[gr.Textbox] = []
        seg_audios: list[gr.Audio] = []
        seg_buttons: list[gr.Button] = []
        with gr.Column():
            for index in range(MAX_SEGMENTS):
                with gr.Row(visible=False) as row:
                    seg_texts.append(
                        gr.Textbox(
                            label=f"セグメント {index + 1}",
                            lines=3,
                            scale=4,
                            show_label=True,
                        )
                    )
                    seg_audios.append(
                        gr.Audio(
                            label=f"{index + 1}",
                            type="filepath",
                            interactive=False,
                            scale=3,
                            min_width=200,
                        )
                    )
                    seg_buttons.append(gr.Button("再生成", scale=1, min_width=90))
                seg_rows.append(row)

        # Shared settings, in the order _Params.from_args unpacks them.
        param_components = [
            checkpoint,
            uploaded_audio,
            uploaded_speaker_embedding,
            speaker_embedding_path_raw,
            num_steps,
            seed_raw,
            duration_scale,
            t_schedule_mode,
            sway_coeff,
            cfg_guidance_mode,
            cfg_scale_text,
            cfg_scale_speaker,
            gap_seconds,
            trim_silence,
        ]

        split_btn.click(
            _split,
            inputs=[script_text, max_chars, state],
            outputs=[*seg_rows, *seg_texts, *seg_audios, full_audio, status, state],
        )
        generate_btn.click(
            _generate,
            inputs=[state, force_all, *seg_texts, *param_components],
            outputs=[*seg_audios, full_audio, status, state],
        )
        rejoin_btn.click(
            _rejoin,
            inputs=[state, gap_seconds],
            outputs=[full_audio, status, state],
        )
        for index, button in enumerate(seg_buttons):

            def _regen(*args: object, _index: int = index):
                return _regenerate_one(_index, args)

            button.click(
                _regen,
                inputs=[state, seg_texts[index], *param_components],
                outputs=[seg_audios[index], full_audio, status, state],
            )

        t_schedule_mode.change(
            _on_t_schedule_mode_change, inputs=[t_schedule_mode], outputs=[sway_coeff]
        )
        load_model_btn.click(_load_model, inputs=[checkpoint], outputs=[model_status])
        clear_cache_btn.click(_clear_runtime_cache, outputs=[model_status])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio longform reader for Irodori-TTS.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    # One request at a time: the runtime is a single cached model on one GPU + ANE worker.
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        css=EMOJI_PALETTE_CSS,
    )


if __name__ == "__main__":
    main()
