# 05: 参照音声キャッシュ（L1: codec latent / L2: speaker state）

日付: 2026-08-26

## 1. 目的 / 仮説

同じ参照音声を繰り返し使う（対話用途・Gradio）場合、毎 request の
「wav 読込 → loudness normalize (audiotools, CPU) → codec encode → speaker encoder」（約 54 ms）を省く。

## 2. 変更内容（`irodori_tts/inference_runtime.py`）

- **L1**: 参照 wav 1 clip → codec latent (CPU tensor)。
  key = (解決済み path, size, mtime_ns, ファイル SHA-256, normalize_db, ensure_max,
  単一 clip 時の max_ref_seconds, codec repo / device / dtype, deterministic_encode)。
  `deterministic_encode=False` のときは bypass。
- **L2**: 結合・trim・bucket pad 後の `ref_latent` の内容 SHA-1 + shape + mask 数 + model variant（base / LoRA path）
  + model dtype + batch + speaker_uncond_mode → `encode_conditions` が返す (speaker_state, speaker_mask)。
  hit 時は `speaker_state_override` として渡す（既存の Speaker Inversion 用経路。normal 経路と同じ tensor になる）。
- 両方とも entry 数 LRU（既定 8）。`ref_cache_stats` に hit / miss を記録。

## 3. 検証

- `bench/check_equivalence.py` では runtime 構築後に L1/L2 が有効なので、2 回目以降は L2 hit 経由。
  FP32 で legacy（cache なし）と max abs diff ≤ 1.1e-3（02/03 と同じ crop 由来の差）、
  同一条件の反復は bit 一致（`repeat_maxdiff=0`）。
- 別 request で参照音声を変えた場合は key が変わるので miss（内容 hash）。

## 4. 結果

`prepare_reference`: 54 ms → **0.9 ms**（L1+L2 hit、`02_opt_bf16.json`）。
`predict_duration` も speaker encoder が省けるため 34 → 27 ms。

ablation E → F（[03-cuda-graph.md](03-cuda-graph.md)）: short 549 → **491 ms**、long 1550 → **1490 ms**（-58 ms）。

## 5. 注意

- cache miss 時（初めての参照音声）は従来どおり 54 ms かかる。初回生成の速度として報告しない。
- L1 は wav ファイルの SHA-256 を毎回計算する（700 KB で < 1 ms）。ファイルが書き換われば miss。
- L2 の値は GPU 上に保持（speaker state (1, 1+T/4, 768) bf16 ≈ 数百 KB）。entry 8 なら無視できる。
- LoRA adapter を切り替えると variant が key に入るので誤再利用しない（graph も同時に破棄）。
