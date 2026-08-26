# 01: 環境プローブ・推論経路調査・ベースライン計測

日付: 2026-08-26 / commit: 8224daf (未変更の main)

## 1. 目的

最適化前の事実を固定する。

- 実行環境の能力（GPU, dtype 対応, VRAM）
- 推論 call graph と、既に存在する最適化機能
- FP32（現行デフォルト）と BF16（既存オプション）の warm 計測

## 2. 環境プローブ

| 項目 | 値 |
|---|---|
| GPU | RTX 5060 Ti 16 GB, compute capability 12.0, 36 SM, 空き 15.4 GiB |
| torch | 2.10.0+cu128 / CUDA 12.8 / triton 3.6.0 / torchao 0.16.0 |
| BF16 | 対応 |
| `torch.compile` | triton あり（使用可）。既存 `--compile-model` オプションは未検証 |

## 3. 推論経路の調査（`infer.py` → `InferenceRuntime.synthesize`）

```
InferenceRuntime.from_key
  ├ load safetensors (CPU, FP32 714 tensors 2.9 GB)
  ├ TextToLatentRFDiT: ModernBERT-ja-310m backbone (25 layers, 768) + projector
  │   + ReferenceLatentEncoder (8 layers, 768) + DiT (12 layers, dim 1280, 20 heads)
  │   + DurationPredictor
  ├ model.to(cuda)  ← FP32 のまま GPU へ (ロード時ピーク 3.4 GB)
  ├ _move_inference_module → param ごとに BF16 cast
  ├ DACVAECodec.load (weight_norm 付き conv, decoder の watermark 枝は無効化)
  └ SilentCipherWatermarker (codec device)

synthesize (batch=1)
  ├ tokenize_text        : max_text_len=256 に固定長 padding
  ├ caption tokenize     : max_caption_len=512 に固定長 padding（caption 空でも 512）
  ├ prepare_reference    : wav load → loudness normalize (audiotools, CPU) → codec encode
  ├ predict_duration     : model.encode_conditions (text/speaker/caption 全部) → duration predictor
  ├ sample_rf            : sample_euler_rf_cfg
  │    ├ model.encode_conditions  ← ★ predict_duration と同じ計算を再実行
  │    ├ build_context_kv_cache ×(cond, cfg batch)
  │    └ 40 steps × forward_with_encoded_conditions (CFG independent: batch 3〜4)
  │         各 layer で mask cat / self_mask ones 生成 / bool mask → SDPA
  │         t_schedule[i].item() で毎 step GPU 同期
  ├ unpatchify / crop
  ├ decode_latent        : codec.decode (GPU) + find_flattening_point (Python ループ, T 回の GPU 同期)
  └ silentcipher_watermark: 48k→44.1k resample + STFT + CNN (GPU) → CPU
```

既に存在する最適化機能:

- `--model-precision bf16` / `--codec-precision bf16`
- `--context-kv-cache`（デフォルト有効。text/speaker/caption の K/V projection を step 前に計算）
- `--compile-model`（`torch.compile`。LoRA 併用不可）
- `--ref-latent`（参照音声の encode 済み latent を渡す）
- SDPA (`F.scaled_dot_product_attention`) は全 attention で使用済み
- 決定的 codec encode/decode
- Sway sampling / step 数削減（品質変化型）

不足している／改善余地のある点:

1. duration 用と sampling 用で `encode_conditions` が二重実行
2. text 256 / caption 512 の固定長 padding が encoder と context K/V の両方に乗る
   （caption 空でも 512 個の masked key を毎 step 参照）
3. RF ループ内の `.item()` 同期、`torch.all().item()`、`caption_mask.any().item()`
4. attention mask を毎 layer・毎 step で cat 生成
5. CUDA Graph なし（BF16 では step 時間が長さにほぼ依存せず、launch overhead 支配）
6. `find_flattening_point` が Python ループ（T 回の `.std()`/`.mean()` 同期）
7. codec の weight_norm が推論時も毎 forward で weight を再計算
8. FP32 モデル全体を GPU に置いてから cast（ロード時ピーク VRAM）
9. 参照音声のキャッシュなし（同じ参照音声でも毎回 load / normalize / encode）
10. warm 後の `memory_reserved` が 8.6〜9 GB（allocated 2.3 GB）と大きい

## 4. 計測

```bash
uv run --no-sync python bench/bench_runtime.py --precision fp32 --tag 01_baseline_fp32 \
  --inputs short medium long caption_noref --warmup 2 --repeats 10 \
  --output docs/experiments/results/01_baseline_fp32.json
uv run --no-sync python bench/bench_runtime.py --precision bf16 --tag 01_baseline_bf16 ...
```

条件: 40 steps, linear, CFG independent (text 3.0 / speaker 5.0 / caption 3.0), seed 1234,
codec fp32 / cuda, watermark 有効, reference = `outputs/sample.wav` (7.28 s)。

### 4.1 FP32（現行デフォルト）

| 入力 | 音声長 | wall median | p95 | RTF | GPU util mean | peak alloc |
|---|---|---|---|---|---|---|
| short | 6.48 s | 1316 ms | 1318 | 0.203 | 95 % | 4406 MiB |
| medium | 10.96 s | 2049 ms | 2054 | 0.187 | 97 % | 4881 MiB |
| long | 28.76 s | 4495 ms | 4638 | 0.156 | 97 % | 6761 MiB |
| caption_noref | 7.32 s | 1355 ms | 1356 | 0.185 | 96 % | 4495 MiB |

stage median (ms): short: prepare_reference 54 / predict_duration 50 / sample_rf 1090 / decode 81 / watermark 40

ロード後 allocated 3460 MiB, warm 後 reserved 9062 MiB。

### 4.2 BF16（既存オプション）

| 入力 | 音声長 | wall median | p95 | RTF | GPU util mean | peak alloc |
|---|---|---|---|---|---|---|
| short | 6.48 s | 915 ms | 949 | 0.141 | 68 % | 2969 MiB |
| medium | 11.00 s | 1081 ms | 1099 | 0.098 | 79 % | 3447 MiB |
| long | 28.80 s | 1935 ms | 1946 | 0.067 | 95 % | 5329 MiB |
| caption_noref | 7.32 s | 868 ms | 1023 | 0.119 | 74 % | 3058 MiB |

stage median (ms): short: prepare_reference 53 / predict_duration 34 / sample_rf 706 / decode 81 / watermark 40
medium: sample_rf 794, long: sample_rf 1330

ロード後 allocated 2023 MiB, warm 後 reserved 8660 MiB。

## 5. 観察

- FP32 は GPU util 95 % 超で **compute-bound**（RTX 5060 Ti の FP32 は TF32 無効のため遅い）。
- BF16 では `sample_rf` が short 706 → medium 794 ms と長さにほぼ比例しない。
  1 step ≈ 17.6 ms（short）は計算量から見て大きすぎ、**launch overhead / 小カーネル支配**。
  GPU util も 68〜79 % に落ちる → CUDA Graph が効く領域。
- BF16 短文では decode (81 ms) + watermark (40 ms) + reference (53 ms) + duration (34 ms) で
  約 210 ms、`sample_rf` 以外が 23 % を占める。
- 同一 seed で 10 回とも音声 hash 一致（決定的）。
- FP32 と BF16 の音声は当然一致しない（別品質判定が必要 → 聴感では両方問題なし、後述の実験で再確認）。

## 6. 次のアクション

- 02: 出力保持型の変更（condition 再利用、text/caption crop、同期除去、mask 事前計算、
  tail 検出のベクトル化、ロード時 cast 順序）を実装し、FP32/BF16 両方で hash/誤差を確認。
- 03: RF step の CUDA Graph 化。
- 04: codec の weight_norm fold、codec BF16、decode の compile 検討。
- 05: 参照音声 / speaker state キャッシュ。
