# 07: DiT の torch.compile、VRAM 上限込みの最終設定、BF16 品質

日付: 2026-08-26

## 1. DiT forward の torch.compile（`IRODORI_OPT_COMPILE_DIT`）

03 の残り: 1 step 9.7 ms のうち RMSNorm / AdaLN / RoPE / gate などの fp32 elementwise が帯域を食う。
`forward_with_encoded_conditions` を `torch.compile(dynamic=True)` で fuse し、その compiled callable を
そのまま CUDA Graph に capture する（capture 前の side-stream warmup で compile が走る）。

結果（`03_compile_dit_bf16.json`、chunk なし・上限なし時点）:

| 入力 | sample_rf 03 → compile | wall |
|---|---|---|
| short | 387 → **314 ms**（1 step 7.9 ms） | 491 → 419 ms |
| long | 1144 → **936 ms** | 1490 → 1284 ms |

注意: 初回の compile（dynamic）に十数秒かかる。warm 後は graph replay なので影響しない。
Inductor が "Not enough SMs to use max_autotune_gemm mode"（36 SM）を出すが問題なし。

## 2. 最終設定（このPC既定）

| 項目 | 値 | 根拠 |
|---|---|---|
| model precision | bf16 | 01 |
| codec precision | fp32（bf16 は選択肢） | 04 |
| watermark | off | 利用者指示 |
| condition 再利用 / crop / 同期除去 / mask 事前計算 | on | 02 |
| CUDA Graph（bucket S=32 / text=16 / speaker=64、LRU 12） | on | 03 |
| DiT compile | on（下記の等価性確認後） | 07 |
| codec weight_norm fold | on | 04 |
| reference cache L1/L2（8 entry） | on | 05 |
| decode chunk / overlap | 96 / 16 frame | 06 |
| VRAM 上限 | 3584 MB（+ CUDA context） | 06 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 06 |

### 最終ベンチ（`07_final_bf16.json`、warmup 3 / repeats 10）

| 入力 | 音声長 | wall median | p95 | RTF | GPU util | torch peak alloc | **nvidia-smi max** |
|---|---|---|---|---|---|---|---|
| short | 6.48 s | **450 ms** | 452 | 0.069 | 92 % | 2513 MiB | **3369 MiB** |
| medium | 10.96 s | **646 ms** | 649 | 0.059 | 94 % | 2582 | 3429 |
| long | 28.80 s | **1406 ms** | 1408 | 0.049 | 97 % | 2587 | 3437 |
| caption_noref | 7.32 s | **481 ms** | 481 | 0.066 | 100 % | 2514 | 3437 |

stage (short): predict_duration 27 / sample_rf 314 / decode 106（chunk 化で +31 ms）。
ロード後 allocated 1873 MiB、warm 後 reserved 3094 MiB。**nvidia-smi 上のピーク 3.4 GB（CUDA context 込み）で
4 GB 目標を達成**。long でも peak alloc が short と同じ（2.5〜2.6 GB）= 長さ非依存。

### 01 からの推移（wall median）

| 入力 | 01 FP32 既定 | 01 BF16 | 03 全部入り | 07 最終（compile + chunk + 上限） |
|---|---|---|---|---|
| short | 1316 ms | 915 | 491 | **450**（×2.9） |
| medium | 2049 | 1081 | 675 | **646**（×3.2） |
| long | 4495 | 1935 | 1490 | **1406**（×3.2） |
| caption_noref | 1355 | 868 | 506 | **481**（×2.8） |
| request peak alloc (long) | 6761 MiB | 5329 | 5114 | **2587** |

### codec BF16 の場合（`07_final_codec_bf16.json`、品質変化型）

short 394 / medium 601 / long 1214 / caption 404 ms、peak alloc 2.1 GB、nvidia-smi max 2977 MiB。
ただし codec 全体を BF16 にすると参照音声の **encode も BF16** になり、speaker state と duration 予測が
変わる（short の音声長 6.48 → 6.72 s）。decode だけ BF16 にする `IRODORI_OPT_DECODE_AUTOCAST=1`
（weights は FP32 のまま autocast、encode は不変）を用意した → 結果は §4。

## 3. BF16 品質について

FP32 と BF16 は同一 seed でも別の波形になる（02 参照: 40 step の積分でカオス的に発散）。
同じことは FP32 でも seed を変えれば起きるので、比較の物差しとして
「FP32 seed 1234 vs FP32 seed 4321（＝別サンプル）」の距離を並べる。

`bench/gen_quality_wavs.py`（watermark なし、wav は `outputs/quality/<set>/`）+ `bench/audio_metrics.py`
（log-mel 128 bin、-80 dB floor の LSD、波形 SNR）:

| 比較 | short | medium | long | caption_noref |
|---|---|---|---|---|
| **別サンプルの物差し**: FP32 s1234 vs FP32 s4321 | LSD 15.4 dB / SNR -2.9 | 16.4 / -3.2 | 16.2 / -3.1 | 22.0 / -2.6 |
| FP32 legacy vs **BF16 最終**（同 seed） | **1.8 dB / 6.7 dB** | **0.7 / 12.6** | **2.6 / 4.7** | **0.4 / 21.1** |
| BF16 最終 vs + codec 全体 BF16 | 14.3 / -1.7（長さ +240 ms） | 20.3 / -2.8（+600 ms） | 15.6 / -2.0（+800 ms） | 2.1 / 43.6 |

読み方:

- 同じ seed の FP32 と BF16 最終は、別 seed 同士（LSD 15〜22 dB、SNR ≈ -3 dB = 無相関）より
  **1 桁近い**（LSD 0.4〜2.6 dB、SNR 5〜21 dB）。BF16 は「同じ軌道を少しずれて辿っている」状態で、
  発話内容・話者・長さは同じ。波形 bit 一致はしないが、別サンプル扱いにはならない。
- codec 全体 BF16 は参照音声の encode が変わるため speaker state → duration → 軌道が変わり、別サンプル並みの
  距離になる（参照なしの caption_noref では decode のみの差になり LSD 2.1 dB / SNR 43.6 dB と小さい）。
  → codec を BF16 にするなら **decode だけ**（`IRODORI_OPT_DECODE_AUTOCAST=1`）にする。
- 最終判断は聴感で。`outputs/quality/` に 4 セット × 4 入力を置いた。

## 4. decode-only BF16（`IRODORI_OPT_DECODE_AUTOCAST`、既定 on）

codec の重みは FP32 のまま、decode だけ `torch.autocast(bf16)` で実行。参照 encode は不変なので
speaker state / duration は FP32 codec と同一。

`07_final_decode_autocast.json`（compile on、chunk 96/16、上限 3584）:

| 入力 | wall median | decode_latent | torch peak alloc | nvidia-smi max |
|---|---|---|---|---|
| short | **401 ms**（codec FP32: 450） | 57 ms（106） | 2417 MiB | **3079 MiB** |
| long | **1234 ms**（1406） | 271 ms（441） | 2465 | 3079 |

指標（vs BF16 最終 codec FP32、同 seed）: short LSD 2.1 dB / SNR 19 dB、long 1.5 / 12.9、長さ差 0。

### 聴感（利用者による全曲聴き比べ、2026-08-26）

- `fp32_legacy_s1234` vs `bf16_opt_codecbf16_s1234`（codec 全体 BF16）: 読み・声は区別つかないが、
  後者は「風呂で話しているような」こもった音質劣化が若干ある → **不採用**。
- `bf16_opt_decode_autocast_s1234`（decode のみ BF16）: 劣化なし（「完璧」） → **既定に採用**。

## 5. 最終既定のまとめ

| 項目 | 既定 |
|---|---|
| model bf16 / codec fp32 重み + decode autocast bf16 | on |
| watermark | off |
| 02 の出力保持型一式、CUDA Graph、codec fold、reference cache | on |
| decode chunk / overlap | 96 / 16 |
| VRAM 上限 | 3584 MB（nvidia-smi 上 ≈ 3.1 GB ピーク） |
| DiT compile | **CLI は off**（初回 45〜80 s の compile が一発実行に見合わない）。常駐プロセスでは `IRODORI_OPT_COMPILE_DIT=1` で -70〜-200 ms/req。Gradio 2 本は起動時に自動で on にする（利用者は Gradio を使わないため未検証） |

CLI 既定（compile なし）の目安: short ≈ 470 ms、long ≈ 1.45 s、nvidia-smi ピーク ≈ 3.1 GB。
