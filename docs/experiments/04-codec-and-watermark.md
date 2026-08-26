# 04: Codec（weight_norm fold / BF16 / compile）と watermark

日付: 2026-08-26

## 1. 現状（01 の BF16 ベースライン、short 6.5 s）

| stage | ms | 備考 |
|---|---|---|
| decode_latent | 81 | DACVAE decoder (FP32, GPU) + tail 検出 |
| silentcipher_watermark | 40 | 48k→44.1k resample + STFT + CNN + 逆変換、long では 173 ms |

## 2. watermark 無効化

利用者指示により、このPC・ローカル限定で SilentCipher watermark を **無効化**（`IRODORI_OPT_WATERMARK=0` が既定）。
SilentCipher モデルのロード自体をスキップするので VRAM も約 120 MB 減る（ロード後 allocated 2023 → 1905 MiB）。
再有効化は `IRODORI_OPT_WATERMARK=1`。

## 3. weight_norm fold（出力保持）

`torch.nn.utils.weight_norm`（旧 API, forward pre-hook）が推論のたびに `g * v / ||v||` を再計算している。
ロード時に `remove_weight_norm` で 62 層を fold。

`bench/check_codec_fold.py fp32`: 3 長さすべて **bitwise 一致**。
decode 74.5 → 73.2 ms / 122.0 → 120.5 / 315.2 → 313.8 ms（約 1.5 % 改善、peak VRAM 変化なし）。
効果は小さいが害はないので既定 on。

## 4. codec BF16（品質変化型・別判定）

`bench/check_codec_fold.py bf16`（fold 済み BF16 vs FP32、同一 latent）:

| 音声長 | decode FP32 | decode BF16 | peak alloc FP32 | peak alloc BF16 | max abs diff |
|---|---|---|---|---|---|
| 6.5 s | 74.5 ms | **39.6 ms** | 1666 MiB | **1323 MiB** | 5.1e-2 |
| 11.0 s | 122.0 | **67.8** | 2146 | **1563** | 4.1e-2 |
| 28.8 s | 315.3 | **177.8** | 4030 | **2507** | 4.0e-2 |

decode は activation の帯域律速なので BF16 で約 1.8 倍。VRAM ピークも 1.6 倍小さい。
差は波形の 4〜5 %（-26 dB 程度）で、決定論的な同一 latent に対する差なので聴感評価の対象。
→ 採否は 06 で（`--codec-precision bf16`）。

## 5. torch.compile（decoder）— 不採用

`bench/check_codec_compile.py`（`torch.compile(decoder, dynamic=True)`）は 28.8 s 相当の latent で
compile 中に GPU 15 GB を使い切り OOM。VRAM 上限を優先する方針（06）と相性が悪く、
decode は activation 帯域律速で Snake の fusion による伸びしろも限定的なので不採用。
`IRODORI_OPT_COMPILE_CODEC` はスイッチだけ残し既定 off。

## 6. VRAM ピークの主因

long (28.8 s) の request peak ≈ 5.1 GB は codec decode の transient（FP32 で +3 GB、106 MiB/秒）が主因。
→ [06-memory.md](06-memory.md)（chunk decode で一定化）
