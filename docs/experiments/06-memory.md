# 06: VRAM プロファイルと削減

日付: 2026-08-26

## 1. request 中の transient（`bench/profile_memory.py`、bf16、全最適化、codec FP32）

| 入力 | 常駐 | request peak alloc | reserved | encode_conditions | sample_rf | decode_latent |
|---|---|---|---|---|---|---|
| short 6.5 s | 1956 MiB | 2641 MiB | 3008 | +1 | +25 | **+685** |
| long 28.8 s | 2009 | 5051 | 6294 | +2 | +37 | **+3041** |
| caption_noref 7.3 s | 2028 | 2801 | 3450 | +1 | +25 | **+773** |

ロード直後: allocated 1905 MiB（model bf16 ≈ 1.45 GB + codec fp32 ≈ 0.3 GB + 定数）、peak 2011 MiB
（旧経路は FP32 を GPU に置いてから cast していたので 3460 MiB）。
Graph static: 6 entries / 2 const sets で 58 MiB。

**結論: request のピークはほぼ全部 codec decode の activation。** DiT 側は 40 MB 以下。

## 2. 対策

### 2.1 codec decode のコスト構造（`bench/check_decode_transient.py`）

codec の hop_length は **1920**（25 fps、48 kHz）。decode の transient と時間は latent frame 数にほぼ比例:
**約 4.2 MiB / frame ≒ 106 MiB / 秒、0.45 ms / frame**（FP32, eager）。
28.8 s = 720 frame → +3046 MiB。これが request ピークの正体。

`torch.backends.cudnn.benchmark=True` は時間・transient とも約 12 % 減るが、shape ごとに autotune が走る
（可変長では初回コストが読めない）ので採用しない。
`CUDNN_CONV_WSCAP_DBG` は効果なし（workspace ではなく activation 本体が大きい）。

### 2.2 overlap 付き chunk decode（出力保持）

decoder は有限受容野の conv スタック（latent 換算で ±10 frame 程度）なので、latent を `chunk_frames` ごとに
前後 `overlap_frames` 付きで decode し中央だけ残せば全体 decode と同じ波形になる。先頭・末尾は本来の端を
そのまま使う。末尾の半端は前の window に merge する（極端に短い window は decode 長が変わるため）。

`bench/check_codec_chunk.py fp32`（transient は request 中の追加分、時間は decode のみ）:

| frames (音声長) | 全体 decode | chunk 64 / ovl 16 | chunk 96 / ovl 16 | max abs diff |
|---|---|---|---|---|
| 162 (6.5 s) | 73 ms / +686 MiB | 116 ms / +407 MiB | 104 ms / +474 MiB | ≤ 1.9e-3 |
| 275 (11.0 s) | 121 ms / +1164 MiB | 224 ms / +410 MiB | 185 ms / +544 MiB | ≤ 3.4e-3 |
| 720 (28.8 s) | 314 ms / +3046 MiB | 627 ms / +418 MiB | 439 ms / +552 MiB | ≤ 2.3e-3 |
| ≤ 96 | 同一 | bitwise 一致 | bitwise 一致 | 0 |

差 1〜3e-3 は chunk 長が変わることで cuDNN が別アルゴリズムを選ぶ float 誤差（FP32 の全体 decode 同士でも
長さが変われば同程度の差が出る）。**transient は長さによらず一定**になる。

既定: `IRODORI_OPT_DECODE_CHUNK=96`（3.84 s）/ `IRODORI_OPT_DECODE_OVERLAP=16`（0.64 s）。
時間は long で +125 ms（decode 314 → 439 ms）だが、DiT compile の -208 ms で相殺できる（07）。
より厳しく抑えるなら 64/16（+410 MiB）、速度優先なら 0（chunk なし）。

### 2.3 ハード上限（`IRODORI_OPT_VRAM_LIMIT_MB`、既定 3584）

`torch.cuda.set_per_process_memory_fraction()` で caching allocator に上限を掛ける。上限に達すると
allocator は cache を解放してから再試行し、それでも足りなければ OOM 例外を出す —
つまり **黙って膨らむことはない**。確認: chunk なし + 上限 4096 で long を回すと
"4.00 GiB allowed" で OOM になる（意図どおり）。
CUDA context（このドライバ・torch では約 0.4〜0.5 GB）は allocator の外なので、
nvidia-smi 上の使用量 ≈ 上限 + context。4 GB 目標なら上限 3584 MB。

### 2.2 codec BF16（品質変化型）

decode transient が半分になる（04 参照）。chunk decode と併用可。

### 2.3 reserved の縮小

allocator の断片化で reserved が allocated の 1.5〜2 倍になる。単一プロセスで他アプリと GPU を
共有しないなら実害はないが、必要なら `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
