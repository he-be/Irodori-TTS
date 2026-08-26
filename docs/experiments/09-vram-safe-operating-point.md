# 09: VRAM ハード上限 3072 MB の安全運用化

日付: 2026-08-26

## 1. 目的

08 では「代表入力（最長 28.8 s、参照 7.28 s、caption なし）」で上限 3072 MB が通ることを確認したが、
余白が ~100 MB しかなく、長い caption や長い参照音声は未計測だった。本実験の目的は
**OOM 限界値ではなく安全運用値を確定すること**。すなわち

- チェックポイントが宣言する入力上限（`max_text_len=256` / `max_caption_len=512` /
  `ref_max_seconds=120`、生成長は `SamplingRequest.max_seconds=30`）を実際に入れて計測する
- 長時間プロセスの状態（CUDA Graph の LRU が埋まった状態）で計測する
- 品質は保持する

運用ポリシーとして **参照音声は 30 s まで**を最悪ケースとする（120 s は方針外だが参考値として計測）。

## 2. 分かったこと（3 つのスケーリング欠陥）

代表入力では見えず、上限入力で初めて出る問題が 3 つあった。いずれも「入力長やセッション長に対して
VRAM が単調増加する」タイプで、上限を下げる前にこれらを潰す必要があった。

### 2.1 参照音声の encode が chunk 化されていない

decode は 06 で chunk 化済みだったが、**encode は全長を一度に通していた**。
transient は参照長にほぼ比例して **約 72 MiB / 秒**:

| 参照長 | frames | 全体 encode（時間 / transient） |
|---|---|---|
| 7.28 s | 182 | 41 ms / +513 MiB |
| 30 s | 750 | 166 ms / +2115 MiB |
| 120 s | 3000 | 664 ms / **+8460 MiB** |

つまり上限 3584 でも 12 s を超える参照音声は OOM する。既定の `outputs/sample.wav`（7.28 s）で
たまたま収まっていただけだった。

**対策**: `DACVAECodec.encode_waveform()` に decode と同じ overlap 付き窓を実装
（`irodori_tts/codec.py`）。window は hop 境界に揃えるので内部窓では `_pad` が no-op、
末尾窓だけモデル本来の reflect pad が効き、全体 encode と同じ構造になる。
ラウドネス正規化は窓分割より前（全体）で行うので影響を受けない。

### 2.2 CUDA Graph の const set がバイト数で無制限

const set は「条件 state / mask / context K/V キャッシュ」の **静的コピー**で、
参照長・caption 長に比例して膨らむ。LRU は **個数（12）でしか制限していなかった**:

| 入力 | const set 1 個のサイズ | 内訳 |
|---|---|---|
| 長文 + 7.28 s 参照 | 80 MiB | speaker 65 tok / text 256 tok |
| 長文 + caption 512 + 120 s 参照 | **436 MiB** | speaker 769 tok = 225、caption 432 tok = 127、text 256 tok = 75 |

12 entry 分溜まると static だけで GB 級になる。

**対策**: `IRODORI_OPT_GRAPH_MAX_STATIC_MB`（既定 256）でバイト予算を導入。
予算超過で LRU evict、**1 個で予算を超える const set はそもそも capture しない**（eager 実行）。

### 2.3 CUDA Graph の private pool が単調増加する（最大の要因）

全 graph が 1 つの private pool を共有していたため、entry を evict しても
**pool の segment はドライバに返らない**。stress を流すと pool が 512 → 1400 MiB まで増え、
`max_entries` を 12 → 4 に減らしても変わらなかった（pool は entry 数ではなく
「その pool で今までに capture した形状の総量」で決まる）。

**対策**: graph ごとに専用 pool を持たせ（`IRODORI_OPT_GRAPH_SHARED_POOL=0`、新既定）、
evict 時に `graph.reset()` → `empty_cache()` で回収する。pool は **124〜272 MiB** で頭打ちになった。
graph 間のブロック再利用は失うが、実測で速度差はなかった。

### 2.4 capture そのもののピーク

上限 3072 では、長い latent の capture 時（const set のコピー + private pool の workspace +
生きている K/V が同時に存在する）に OOM する。一方 03 の通り **長文では graph の利得は小さい**
（long は compute-bound、sample_rf 1143 ms は eager と同等）。

**対策**: `IRODORI_OPT_GRAPH_MAX_LATENT`（既定 384 frame = 15.4 s）を超える latent は eager 実行。
短文・中文（graph の利得が大きい領域）は従来どおり replay する。

## 3. 変更内容

| ファイル | 変更 |
|---|---|
| `irodori_tts/codec.py` | `encode_waveform(chunk_frames, overlap_frames)`、`_encode_window()` に分離 |
| `irodori_tts/cuda_graph.py` | static バイト予算 / oversize const set の capture 回避 / latent 長で capture 回避 / entry ごとの private pool と `release()`（reset + empty_cache） |
| `irodori_tts/inference_runtime.py` | 参照 encode に chunk を配線、graph runner に新オプションを配線 |
| `irodori_tts/opt_config.py` | 新スイッチ 5 個 + 既定値の変更 |
| `bench/stress_vram.py` | 新規: 上限入力での VRAM stress（下記） |
| `bench/check_codec_encode_chunk.py` | 新規: chunk encode の一致・時間・transient |

既定値の変更:

| スイッチ | 旧 | 新 | 理由 |
|---|---|---|---|
| `IRODORI_OPT_VRAM_LIMIT_MB` | 3584 | **3072** | 本実験の結論 |
| `IRODORI_OPT_GRAPH_MAX_ENTRIES` | 12 | **6** | 12 では pool が予算を超える（cap 2944 で OOM） |
| `IRODORI_OPT_GRAPH_SHARED_POOL` | (共有) | **0 = entry ごと** | 2.3 |
| `IRODORI_OPT_GRAPH_MAX_LATENT` | — | **384** | 2.4 |
| `IRODORI_OPT_GRAPH_MAX_STATIC_MB` | — | **256** | 2.2 |
| `IRODORI_OPT_ENCODE_CHUNK` / `_OVERLAP` | — | **96 / 32** | 2.1 |

`IRODORI_OPT_DECODE_CHUNK` は 96 のまま（64 に下げても peak は 10 MiB しか変わらず、
long の decode が 271 → 298 ms 悪化するだけだった）。

## 4. 計測方法

```bash
# 上限入力での stress（text 256 tok / caption 512 tok / 参照 15-120 s / 生成 30 s）
uv run --no-sync python bench/stress_vram.py --tag <tag> \
  --env IRODORI_OPT_VRAM_LIMIT_MB=<cap> --output docs/experiments/results/<tag>.json
```

`--graph-fill 14`（既定）で長さの違う 14 リクエストを先に流し、**CUDA Graph の LRU が埋まった状態**で
各ケースを測る。上限の判定対象は `max_memory_reserved`（重み + transient + graph private pool + 断片）。
`peak_alloc` と private pool の内訳（`memory_snapshot()` の segment_pool_id≠0）も記録する。

## 5. 結果

### 5.1 上限入力の VRAM（上限なし、既定設定）

encode chunk と graph の対策前後（`results/09_stress_nocap.json`、`09_defaults_final.json`）:

| ケース | 対策前 peak_alloc | 対策後 peak_alloc | 対策前 pool | 対策後 pool |
|---|---|---|---|---|
| text_max（256 tok, 参照 7.28 s） | 2751 MiB | 2561 | 636 | 210 |
| caption_max（+ caption 512 tok） | 2967 | 2561 | 780 | 210 |
| ref30（参照 30 s） | 4851 | 2562 | 1152 | 210 |
| worst（text + caption + 参照 30 s） | — | 2562 | — | 210 |
| ref120（方針外） | **11448** | 2563 | 1400 | 210 |
| worst_ref120（方針外） | 4162 ※ | 2615 | 1400 | 210 |

※ 対策前の worst_ref120 が ref120 より小さいのは、直前のケースで参照 latent が L1 キャッシュに
乗っていて encode を再実行しないため。

### 5.2 上限の探索（既定設定、上限のみ変更）

| 上限 | ポリシー内（参照 ≤30 s） | 方針外（参照 120 s） | 備考 |
|---|---|---|---|
| 3584（旧既定） | OK | OK | 余白過大 |
| **3072（新既定）** | **OK** | **OK** | peak_reserved 3008〜3060、nvidia-smi 3307〜3359 MiB |
| 2944 | OK（peak_reserved 2944 = 上限に張り付き） | OOM | 余白ゼロ |
| 2816 | OOM | OOM | |
| 2688 / 2560 | OOM | OOM | |

**ポリシー内の OOM 境界は 2816〜2944 の間**。新既定 3072 はそこから **約 130〜250 MB の余白**を持つ。
さらに、方針外の入力（120 s 参照 + 512 token caption）まで通るという二重の余裕がある。

対策前（08 時点の設定）は同じ 3072 でも上限入力では全ケース OOM だったので、
「3072 が通る」の中身が変わっている: 08 は代表入力のみ、09 は宣言上限入力 + graph LRU 満杯。

### 5.3 速度（`bench/bench_runtime.py`、bf16、warmup 3 / repeats 10）

| 入力 | 旧既定 wall_median | 新既定 wall_median | 旧 p95 | 新 p95 |
|---|---|---|---|---|
| short (6.5 s) | 474 ms | **473** | 629 | 474 |
| medium (11.0 s) | 652 | **653** | 653 | 655 |
| long (28.8 s) | 1445 | **1446** | 1447 | **1613** |
| caption_noref (7.3 s) | 485 | **485** | 486 | 486 |

median は誤差内で同じ。long の p95 だけ +166 ms 悪化する: 28.8 s = 720 frame は
`GRAPH_MAX_LATENT=384` を超えて eager 実行になるため、CPU launch のばらつきが出る
（median が変わらないのは 03 の通り long が compute-bound だから）。

nvidia-smi の最大使用量は 3305 → 3195 MiB（代表入力）、上限入力の stress でも 3359 MiB。

### 5.4 chunk encode 単体（`bench/check_codec_encode_chunk.py`、FP32）

| 参照長 | 全体 encode | chunk 96 / ovl 32 | latent maxdiff |
|---|---|---|---|
| 7.28 s | 41 ms / +513 MiB | 63 ms / +361 MiB | 1.6e-3 |
| 30 s | 166 ms / +2115 MiB | 272 ms / +456 MiB | 1.4e-3 |
| 120 s | 664 ms / +8460 MiB | 1088 ms / +472 MiB | 8.0e-4 |

overlap を 8 → 64 に増やしても差は 6e-4 程度で頭打ちになる（受容野の切り落としではなく、
入力長が変わることによる cuDNN のアルゴリズム選択差。06 の chunk decode と同じ性質）。
時間は参照 1 回につき +20〜400 ms 増えるが、L1 キャッシュがあるので同じ参照音声では初回のみ。

## 6. 品質確認

- **graph の変更（2.2〜2.4）は出力保持**。`IRODORI_OPT_GRAPH_MAX_LATENT=0` と `384` で
  FP32 の long を生成し、音声 SHA-256 が **完全一致**（`206b9f54…`）。
  eager 実行と graph replay が同じ結果を返すことの再確認でもある。
- **chunk encode は参照 latent が float 誤差レベルで変わる**ので、FP32 で音声を A/B した
  （`bench/audio_metrics.py`、encode chunk 0 vs 96）:

  | 入力 | max_abs | SNR | LSD | 長さ差 |
  |---|---|---|---|---|
  | short | 0.107 | 34.9 dB | 0.57 dB | 0 ms |
  | medium | 0.014 | 45.4 dB | 0.59 dB | 0 ms |
  | long | 0.027 | 44.9 dB | 0.63 dB | 0 ms |

  07 の物差しで見ると、**既に採用済みの BF16 化（LSD 0.4〜2.6 dB / SNR 5〜21 dB）より 1 桁小さく**、
  別サンプル（LSD 15〜22 dB / SNR ≈ -3 dB）とは比較にならない。
  参照 latent は bf16 にキャストされて DiT に入る（相対誤差 ~4e-3）ので、chunk 由来の摂動
  （相対 3e-4）はその丸め誤差より小さい。
- `bench/check_equivalence.py --precision fp32` は本変更の前後どちらも MISMATCH
  （maxdiff 8e-3〜2e-2、変更後はむしろ 4e-3〜1e-2 と同等以下）。**変更前から存在する差**で、
  本実験で悪化はしていない。BF16 での比較は 02 の通り意味を持たない。

## 7. 採否

**採用。既定を上限 3072 MB とする。**

- 品質: 出力保持（graph）+ BF16 化より 1 桁小さい摂動（chunk encode）。
- 速度: median 不変。long の p95 のみ +166 ms。
- 余白: ポリシー内の最悪入力に対して 130〜250 MB。llama.cpp 側には
  16 GB - (3072 + CUDA context ≈ 290 MB) ≈ **12.9 GB** を残せる。

### 未計測（この余白を消費しうるもの）

| 項目 | 影響 | 対応 |
|---|---|---|
| LoRA アダプタ | 重み増加 + `_set_variant()` で graph 全 clear | 使うなら再計測が必要 |
| `num_candidates > 1` | sample_rf がバッチ化される（README の前提はバッチ 1） | 上限を上げるか候補数 1 で運用 |
| `IRODORI_OPT_WATERMARK=1` | SilentCipher モデルが常駐 | 上限を上げる |
| FP32 精度 | 重みだけで 2.9 GB | 従来どおり `IRODORI_OPT_VRAM_LIMIT_MB=0` |

### さらに下げるには（未実施）

3072 のうち 1873 MB は常駐重み（DiT 684 + ModernBERT 600 + speaker encoder 115 +
duration 42 + codec fp32 ~300 MiB）で、request transient は 690 MB 程度まで落ちている。
2.5 GB 級を狙うなら 08 の §5 と同じく重みを動かすしかない
（ModernBERT の CPU 常駐 -600 MB / +80〜100 ms、int8 化 -300 MB / 要聴取）。
