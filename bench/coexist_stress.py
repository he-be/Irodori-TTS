#!/usr/bin/env python3
"""VLM (llama-swap / llama-server) + Irodori-TTS co-existence stress (experiment 10).

The VLM is driven over HTTP; the TTS runs in a subprocess (bench/coexist_tts_worker.py)
so load/unload is a real CUDA context appearing and disappearing.  A background
nvidia-smi sampler records the whole-GPU timeline, which is what actually decides
whether the two fit.

Scenarios (--scenario):
  vlm_only    VLM tasks back to back, no TTS
  tts_only    TTS synth + playback-length sleep, no VLM traffic
  concurrent  both at once (the main case)
  tts_churn   VLM tasks back to back; TTS process loads, speaks, exits, repeats
  vlm_swap    TTS loop continuous; VLM unloaded and reloaded through llama-swap
  pipeline    VLM writes a line, TTS speaks it while the next VLM request runs

Example:
  uv run --no-sync python bench/coexist_stress.py --scenario concurrent --duration 180 \
      --output docs/experiments/results/10_concurrent.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "bench" / "coexist_tts_worker.py"

# 6-18 s of speech each: the operating point in the brief (one speaker, batch 1).
TTS_TEXTS = [
    "こんにちは、私はAIです。これは音声合成のテストです。",
    "今日は朝から雨が降っていましたが、午後になると雲の切れ間から日差しが差し込みました。",
    "画像に写っているのは、色とりどりの楕円が重なり合った抽象的なデザインです。"
    "具体的な物の形は描かれていません。",
    "音声合成の速度を改善するには、まず現状の処理時間を正確に測定し、"
    "どの段階に時間がかかっているのかを把握することが重要です。",
    "会議の要点をまとめます。第一に、来週までに試作品を仕上げること。"
    "第二に、計測結果を共有すること。第三に、次の打ち合わせを金曜日に設定することです。",
]

VLM_PROMPT_SHORT = "日本の四季について、二文で説明してください。"
VLM_PROMPT_LONG = (
    "次の文章を要約し、要点を三つ挙げてください。\n"
    + "日本の四季は春夏秋冬に分かれ、それぞれ異なる自然の表情を見せる。" * 60
)


def now() -> float:
    return time.perf_counter()


class GpuSampler:
    """Whole-GPU memory / utilization timeline from nvidia-smi."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, int, int]] = []
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )

        def _reader() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) == 2:
                    try:
                        self.samples.append((now(), int(parts[0]), int(parts[1])))
                    except ValueError:
                        pass

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def window(self, t0: float, t1: float) -> dict:
        mem = [m for (t, _u, m) in self.samples if t0 <= t <= t1]
        util = [u for (t, u, _m) in self.samples if t0 <= t <= t1]
        if not mem:
            return {}
        return {
            "mem_max_mib": max(mem),
            "mem_median_mib": statistics.median(mem),
            "util_mean": round(statistics.mean(util), 1),
            "n": len(mem),
        }


class VlmClient:
    def __init__(self, base_url: str, model: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.image_b64: str | None = None

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            # llama-swap proxies the upstream error body; it is the only place the
            # llama-server side of a failure is visible from here.
            body = ""
            try:
                body = exc.read().decode(errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"HTTP {exc.code}: {body}") from None

    def unload(self) -> dict:
        t0 = now()
        try:
            with urllib.request.urlopen(self.base_url + "/unload", timeout=120):
                pass
            return {"event": "vlm_unload", "ok": True, "wall": now() - t0}
        except Exception as exc:  # noqa: BLE001
            return {"event": "vlm_unload", "ok": False, "error": str(exc)[:300], "wall": now() - t0}

    def text_task(self, prompt: str, n_predict: int, timeout: float = 900) -> dict:
        payload = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": False,
                   "temperature": 1.0, "top_k": 64}
        if self.model:
            payload["model"] = self.model
        t0 = now()
        try:
            d = self._post("/completion", payload, timeout)
            t = d.get("timings", {})
            return {"event": "vlm_text", "ok": True, "wall": now() - t0,
                    "prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"),
                    "gen_n": t.get("predicted_n"), "gen_tps": t.get("predicted_per_second"),
                    "content": (d.get("content") or "")[:400]}
        except Exception as exc:  # noqa: BLE001
            return {"event": "vlm_text", "ok": False, "wall": now() - t0, "error": str(exc)[:300]}

    def image_task(self, image_path: str, n_predict: int, timeout: float = 900) -> dict:
        if self.image_b64 is None:
            self.image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        payload = {"messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + self.image_b64}},
            {"type": "text", "text": "この画像に写っているものを日本語で簡潔に説明してください。"}]}],
            "max_tokens": n_predict, "temperature": 1.0, "top_k": 64}
        if self.model:
            payload["model"] = self.model
        t0 = now()
        try:
            d = self._post("/v1/chat/completions", payload, timeout)
            usage = d.get("usage", {})
            return {"event": "vlm_image", "ok": True, "wall": now() - t0,
                    "prompt_n": usage.get("prompt_tokens"), "gen_n": usage.get("completion_tokens"),
                    "content": (d["choices"][0]["message"]["content"] or "")[:400]}
        except Exception as exc:  # noqa: BLE001
            return {"event": "vlm_image", "ok": False, "wall": now() - t0, "error": str(exc)[:300]}

    def speak_task(self, n_predict: int = 80, timeout: float = 900) -> dict:
        """Ask for one short spoken-style line (used by the pipeline scenario)."""
        prompt = ("次の話題について、話し言葉で一文だけ、五十文字程度で述べてください。"
                  "記号や箇条書きは使わないでください。話題: 今日の天気と気分。\n")
        return self.text_task(prompt, n_predict, timeout)


class TtsProc:
    """The TTS subprocess: start() == load, stop() == unload."""

    def __init__(self, env: dict[str, str], log: list[dict]) -> None:
        self.env = env
        self.log = log
        self.proc: subprocess.Popen[str] | None = None
        self._q: queue.Queue[dict] = queue.Queue()
        self._reader: threading.Thread | None = None
        self.stderr_tail: list[str] = []

    def start(self) -> dict:
        t0 = now()
        env = dict(os.environ)
        env.update(self.env)
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env, cwd=str(REPO_ROOT),
        )

        def _read_out() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass

        def _read_err() -> None:
            assert self.proc is not None and self.proc.stderr is not None
            for line in self.proc.stderr:
                self.stderr_tail.append(line.rstrip()[:300])
                del self.stderr_tail[:-40]

        self._reader = threading.Thread(target=_read_out, daemon=True)
        self._reader.start()
        threading.Thread(target=_read_err, daemon=True).start()
        self._expect("ready", timeout=180)
        rec = self.send({"cmd": "load"}, timeout=900)
        rec["proc_start_wall"] = now() - t0
        return rec

    def _expect(self, event: str, timeout: float) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self._q.get(timeout=1.0)
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    return {"event": event, "ok": False,
                            "error": f"worker died rc={self.proc.returncode}",
                            "crash": True, "stderr": self.stderr_tail[-8:]}
                continue
            if msg.get("event") == event or not msg.get("ok", True):
                return msg
        return {"event": event, "ok": False, "error": "timeout", "timeout": True}

    def send(self, msg: dict, timeout: float = 600) -> dict:
        assert self.proc is not None and self.proc.stdin is not None
        expect = {"load": "loaded", "unload": "unloaded", "synth": "synth",
                  "stats": "stats", "reset_peak": "reset_peak"}[msg["cmd"]]
        try:
            self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            return {"event": expect, "ok": False, "error": "worker pipe closed", "crash": True,
                    "stderr": self.stderr_tail[-8:]}
        rec = self._expect(expect, timeout)
        if not rec.get("ok", True) and self.stderr_tail:
            rec.setdefault("stderr", self.stderr_tail[-8:])
        return rec

    def stop(self) -> dict:
        t0 = now()
        if self.proc is None:
            return {"event": "tts_stop", "ok": True, "wall": 0.0}
        try:
            if self.proc.stdin is not None and self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self.proc.kill()
            self.proc.wait(timeout=30)
        rc = self.proc.returncode
        self.proc = None
        return {"event": "tts_stop", "ok": True, "wall": now() - t0, "rc": rc}


class Recorder:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.t0 = now()

    def add(self, rec: dict) -> dict:
        rec = dict(rec)
        rec["t"] = round(now() - self.t0, 3)
        with self.lock:
            self.events.append(rec)
        status = "ok" if rec.get("ok", True) else "FAIL"
        extra = ""
        if rec.get("event") == "synth" and rec.get("ok"):
            extra = f" audio={rec['audio_seconds']:.2f}s wall={rec['wall']*1000:.0f}ms rtf={rec['rtf']:.3f}"
        elif rec.get("event", "").startswith("vlm_") and rec.get("ok"):
            extra = (f" prompt_n={rec.get('prompt_n')} gen_n={rec.get('gen_n')}"
                     f" wall={rec['wall']:.2f}s")
        elif not rec.get("ok", True):
            extra = " " + str(rec.get("error"))[:200]
        print(f"[{rec['t']:7.1f}s] {rec.get('event'):12s} {status}{extra}", flush=True)
        return rec


def vlm_loop(client: VlmClient, rec: Recorder, stop: threading.Event, args, image: str) -> None:
    """Keep the VLM busy: alternate short text, long-prefill text, and image tasks."""
    i = 0
    while not stop.is_set():
        kind = i % 3
        if kind == 0:
            r = client.text_task(VLM_PROMPT_SHORT, args.vlm_n_predict)
        elif kind == 1:
            r = client.text_task(VLM_PROMPT_LONG, args.vlm_n_predict)
        else:
            r = client.image_task(image, args.vlm_n_predict)
        r["i"] = i
        rec.add(r)
        i += 1
        if args.vlm_gap > 0:
            stop.wait(args.vlm_gap)


def tts_loop(tts: TtsProc, rec: Recorder, stop: threading.Event, args) -> None:
    """Synthesize, then sleep for the audio length (single speaker, batch 1)."""
    i = 0
    consecutive = 0
    while not stop.is_set():
        r = tts.send({"cmd": "synth", "text": TTS_TEXTS[i % len(TTS_TEXTS)], "id": i})
        r["i"] = i
        rec.add(r)
        i += 1
        consecutive = 0 if r.get("ok") else consecutive + 1
        if r.get("crash") or consecutive >= args.max_failures:
            # a TTS that cannot synthesize any more is the finding; stop hammering it
            rec.add({"event": "tts_giving_up", "ok": False,
                     "error": f"{consecutive} consecutive failures"})
            stop.set()
            return
        if args.playback and r.get("ok") and r.get("audio_seconds"):
            stop.wait(float(r["audio_seconds"]))


def run(args) -> dict:
    image = args.image
    client = VlmClient(args.vlm_url, args.vlm_model)
    rec = Recorder()
    sampler = GpuSampler(args.sample_ms)
    sampler.start()
    tts_env = {}
    for item in args.env:
        k, _, v = item.partition("=")
        tts_env[k] = v
    if args.ref:
        tts_env["COEXIST_REF"] = args.ref

    stop = threading.Event()
    tts: TtsProc | None = None
    threads: list[threading.Thread] = []
    t_start = now()
    try:
        if args.scenario in ("vlm_only", "concurrent", "tts_churn", "vlm_swap", "pipeline"):
            rec.add({"event": "vlm_warmup", **client.text_task("こんにちは。", 8, timeout=1200)})

        if args.scenario in ("tts_only", "concurrent", "vlm_swap", "pipeline"):
            tts = TtsProc(tts_env, rec.events)
            loaded = rec.add({**tts.start(), "event": "tts_load"})
            if not loaded.get("ok"):
                stop.set()
            else:
                # one warmup synth so caches / graphs reach the steady state
                rec.add({**tts.send({"cmd": "synth", "text": TTS_TEXTS[0], "id": -1}),
                         "warmup": True})
                tts.send({"cmd": "reset_peak"})

        t_start = now()
        deadline = t_start + args.duration

        if stop.is_set():
            # TTS could not even load: that is the whole result, summarize and stop.
            rec.add({"event": "aborted", "ok": False, "error": "TTS failed to load"})
        elif args.scenario == "vlm_only":
            vlm_loop_thread = threading.Thread(target=vlm_loop, args=(client, rec, stop, args, image))
            threads.append(vlm_loop_thread)
            vlm_loop_thread.start()

        elif args.scenario == "tts_only":
            assert tts is not None
            th = threading.Thread(target=tts_loop, args=(tts, rec, stop, args))
            threads.append(th)
            th.start()

        elif args.scenario == "concurrent":
            assert tts is not None
            for target, targs in ((vlm_loop, (client, rec, stop, args, image)),
                                  (tts_loop, (tts, rec, stop, args))):
                th = threading.Thread(target=target, args=targs)
                threads.append(th)
                th.start()

        elif args.scenario == "tts_churn":
            th = threading.Thread(target=vlm_loop, args=(client, rec, stop, args, image))
            threads.append(th)
            th.start()
            cycle = 0
            while now() < deadline and not stop.is_set():
                w = TtsProc(tts_env, rec.events)
                r = rec.add({**w.start(), "event": "tts_load", "cycle": cycle})
                if not r.get("ok"):
                    stop.set()
                    w.stop()
                    break
                for k in range(args.churn_synths):
                    s = rec.add({**w.send({"cmd": "synth",
                                           "text": TTS_TEXTS[k % len(TTS_TEXTS)], "id": k}),
                                 "cycle": cycle})
                    if s.get("crash") or not s.get("ok"):
                        stop.set()
                        break
                    if args.playback and s.get("audio_seconds"):
                        time.sleep(float(s["audio_seconds"]))
                rec.add({**w.stop(), "event": "tts_unload", "cycle": cycle})
                cycle += 1

        elif args.scenario == "vlm_swap":
            assert tts is not None
            th = threading.Thread(target=tts_loop, args=(tts, rec, stop, args))
            threads.append(th)
            th.start()
            cycle = 0
            while now() < deadline and not stop.is_set():
                for _ in range(2):
                    if now() >= deadline or stop.is_set():
                        break
                    rec.add({**client.text_task(VLM_PROMPT_LONG, args.vlm_n_predict), "cycle": cycle})
                    rec.add({**client.image_task(image, args.vlm_n_predict), "cycle": cycle})
                rec.add({**client.unload(), "cycle": cycle})
                time.sleep(args.swap_gap)
                cycle += 1

        elif args.scenario == "pipeline":
            assert tts is not None
            # VLM produces a line; TTS speaks it while the next VLM request already runs.
            pending: dict | None = None
            i = 0
            spoken: list[dict] = []
            while now() < deadline and not stop.is_set():
                box: dict = {}

                def _ask(box=box, i=i) -> None:
                    r = client.text_task(VLM_PROMPT_SHORT if i % 2 else VLM_PROMPT_LONG,
                                         args.vlm_n_predict)
                    box.update(r)

                th = threading.Thread(target=_ask)
                th.start()  # next VLM request runs while we speak the previous one
                if pending is not None:
                    text = (pending.get("content") or "").strip().replace("\n", " ")
                    text = text[:120] or TTS_TEXTS[i % len(TTS_TEXTS)]
                    s = rec.add({**tts.send({"cmd": "synth", "text": text, "id": i}),
                                 "source": "vlm", "chars": len(text)})
                    spoken.append(s)
                    if s.get("crash"):
                        stop.set()
                    if args.playback and s.get("ok") and s.get("audio_seconds"):
                        time.sleep(float(s["audio_seconds"]))
                th.join()
                rec.add({**box, "i": i})
                pending = box if box.get("ok") else None
                i += 1
        else:
            raise SystemExit(f"unknown scenario {args.scenario}")

        while now() < deadline and not stop.is_set() and threads:
            time.sleep(0.5)
    finally:
        stop.set()
        for th in threads:
            th.join(timeout=120)
        t_end = now()
        if tts is not None:
            rec.add({**tts.send({"cmd": "stats"})})
            rec.add({**tts.stop(), "event": "tts_unload"})
        time.sleep(1.0)
        sampler.stop()

    return summarize(args, rec, sampler, t_start, t_end)


def summarize(args, rec: Recorder, sampler: GpuSampler, t_start: float, t_end: float) -> dict:
    ev = rec.events

    def sel(name: str, ok: bool = True) -> list[dict]:
        return [e for e in ev if e.get("event") == name and bool(e.get("ok")) == ok
                and not e.get("warmup")]

    def stats(vals: list[float]) -> dict | None:
        if not vals:
            return None
        s = sorted(vals)
        return {"n": len(s), "median": round(statistics.median(s), 3),
                "p95": round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], 3),
                "max": round(s[-1], 3)}

    synth = sel("synth")
    text = sel("vlm_text")
    img = sel("vlm_image")
    failures = [e for e in ev if not e.get("ok", True)]
    ooms = [e for e in failures if e.get("oom")]
    crashes = [e for e in failures if e.get("crash")]

    return {
        "scenario": args.scenario,
        "duration_s": round(t_end - t_start, 1),
        "gpu": sampler.window(t_start, t_end),
        "gpu_whole_run": sampler.window(rec.t0, t_end),
        "tts": {
            "count": len(synth),
            "wall": stats([e["wall"] for e in synth]),
            "rtf": stats([e["rtf"] for e in synth]),
            "audio_seconds_total": round(sum(e["audio_seconds"] for e in synth), 1),
            "peak_alloc_mib": round(max((e.get("peak_alloc_mib", 0) for e in synth), default=0), 1),
            "peak_reserved_mib": round(max((e.get("peak_reserved_mib", 0) for e in synth), default=0), 1),
            "load_wall": stats([e["wall"] for e in ev if e.get("event") == "tts_load" and e.get("ok")]),
        },
        "vlm": {
            "text_count": len(text), "image_count": len(img),
            "prefill_tps": stats([e["prompt_tps"] for e in text if e.get("prompt_tps")]),
            "gen_tps": stats([e["gen_tps"] for e in text if e.get("gen_tps")]),
            "text_wall": stats([e["wall"] for e in text]),
            "image_wall": stats([e["wall"] for e in img]),
        },
        "failures": {"total": len(failures), "oom": len(ooms), "crash": len(crashes),
                     "first": failures[:5]},
        "events": ev,
        "gpu_timeline": [(round(t - rec.t0, 2), u, m) for (t, u, m) in sampler.samples],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   choices=["vlm_only", "tts_only", "concurrent", "tts_churn", "vlm_swap", "pipeline"])
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--vlm-url", default="http://127.0.0.1:8080")
    p.add_argument("--vlm-model", default=None, help="model name for llama-swap routing")
    p.add_argument("--vlm-n-predict", type=int, default=128)
    p.add_argument("--vlm-gap", type=float, default=0.0)
    p.add_argument("--image", default=str(REPO_ROOT / "bench" / "assets" / "coexist_image.png"))
    p.add_argument("--ref", default=None)
    p.add_argument("--playback", action="store_true", default=True)
    p.add_argument("--no-playback", dest="playback", action="store_false")
    p.add_argument("--churn-synths", type=int, default=3)
    p.add_argument("--swap-gap", type=float, default=5.0)
    p.add_argument("--sample-ms", type=int, default=100)
    p.add_argument("--max-failures", type=int, default=5,
                   help="stop the TTS loop after this many consecutive failures")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for the TTS worker")
    p.add_argument("--tag", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    record = run(args)
    record["tag"] = args.tag or args.scenario
    record["args"] = vars(args)
    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("events", "gpu_timeline")}, ensure_ascii=False, indent=2)[:4000])
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"wrote {args.output}")
    sys.exit(1 if record["failures"]["total"] else 0)


if __name__ == "__main__":
    main()
