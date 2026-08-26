# 10: VLM (llama-swap) との同居 — OOM で落ちない運用点

日付: 2026-08-26

## 1. 目的 / 前提

同じ 16 GB の GPU で、llama-swap 上の VLM（`gemma4-26b-a4b`）と Irodori-TTS を**同時に**動かす。
要件は「速いこと」ではなく「**どちらも OOM で落ちないこと**」。

想定シチュエーション: VLM が常に何らかのタスクを走らせている裏で、TTS が 6〜18 s の音声を
合成しては再生する（スピーカー 1 個・バッチ 1 なので、再生時間だけインターバルが空く）。

| 項目 | 値 |
|---|---|
| GPU | RTX 5060 Ti 16 GB（`nvidia-smi` の total = 16311 MiB、デスクトップ常駐 13 MiB） |
| VLM | `gemma-4-26B-A4B` Q4_0 **14488 MB** + mmproj f16 **1193 MB** + draft(MTP) Q4_0 **252 MB** = 約 **15.9 GB** |
| VLM 設定 | `~/LLM/config.yaml` の `gemma4-26b-a4b` を**変更しない**（`-c 8192 -b/-ub 1024 -np 1 -fa on -cram 0`、draft-mtp） |
| TTS | 09 の既定（bf16、decode/encode chunk、CUDA Graph、VRAM 上限） |
| llama.cpp | b10430 (4c1a0af40) / llama-swap v251 |

VLM の重みだけで GPU 容量とほぼ同じなので、**エキスパートの CPU オフロード量（`--n-cpu-moe`）が
唯一の配分ノブ**になる。本実験はその値と TTS 側の設定を同時に決める。

## 2. 結論（先に）

| 側 | 設定 | `nvidia-smi` |
|---|---|---|
| VLM | `--n-cpu-moe 11`（`config.yaml` は変更せず環境変数で注入） | 12021 MiB（画像 prefill 時 +242） |
| TTS | `IRODORI_OPT_CUDA_GRAPH=0` + `IRODORI_OPT_VRAM_LIMIT_MB=3072` | 2857 MiB |
| 合計 | | **14882 MiB / 16311**（余白 1429 MiB） |

この組み合わせで 420 s の同時実行（TTS 37 発 = 音声 391 s、VLM テキスト 117 + 画像 58 リクエスト）を
**OOM 0 / クラッシュ 0** で完走した。ロード / アンロードを繰り返す 3 シナリオも完走した。

起動コマンド: `bench/coexist_llama_swap.sh`（既定 `NCMOE=11`）と

```bash
IRODORI_OPT_CUDA_GRAPH=0 IRODORI_OPT_VRAM_LIMIT_MB=3072 uv run --no-sync python infer.py ...
```

**副産物として、09 で決めた TTS 既定の上限 3072 MB が過小だったことが分かった**（§5）。
既定を **3840** に変更した。同居時は上のレシピ（Graph off）で 3072 に**戻す**のが正しい。

## 3. 分かったこと

### 3.1 `-fit on` は「同居」には使えない（load 時の空き容量に依存するため）

`config.yaml` は `-fit on` で、未指定の `-ngl` / `--n-cpu-moe` を自動決定する。
実装（`common/fit.cpp:562`）は

```
target = (その時点の空き VRAM) - margin      # margin = -fitt、既定 1024 MiB
```

つまり **モデルサイズが load した瞬間の GPU の空きで決まる**。TTS が常駐していない瞬間に
llama-swap が VLM を load すると、VLM は 15181 MiB を掴む。その後 TTS を起動すると:

```
tts_load  OutOfMemoryError: ... GPU 0 has a total capacity of 15.52 GiB of which 13.31 MiB is free.
          Process (llama-server) has 14.80 GiB memory in use. ... this process has 690.00 MiB in use.
```

TTS は**モデルのロードすら完了できない**（`results/10_fit_hazard.json`）。
llama-swap は要求が来た時に load するので、この順序は日常的に起こる。

→ **配分は静的に固定する**。ただし `config.yaml` は変更しない要件があるので、llama.cpp の
環境変数で注入する:

```bash
LLAMA_ARG_N_GPU_LAYERS=99 LLAMA_ARG_N_CPU_MOE=11 ./llama-swap --config config.yaml
```

`-ngl` をユーザーが設定すると fit は
`n_gpu_layers already set by user to 99, abort` を出して**自分を無効化する**（`common/fit.cpp:377`、
例外は `common_fit_params` が捕まえて警告にする）。`-fit on` が cmd に残っていても実害はない。

### 3.2 `--n-cpu-moe` 1 段 = 約 406 MiB（このモデルは 31 層）

`llama-fit-params` によればこのモデルは 31 層。各段の実測（load 直後 / 画像 prefill 時のピーク、
prefill と生成は 1029 token プロンプト + 128 token 生成の中央値）:

| `--n-cpu-moe` | static | 画像 prefill peak | prefill tok/s | gen tok/s |
|---|---|---|---|---|
| 0 | — | — | **load 失敗**（16 GB に入らない） | — |
| 4 | 14863 | 14959 | 1235 | 161 |
| 6 | 14057 | 14153 | — | — |
| 8 | 13245 | 13339 | 888 | 79 |
| 10 | 12429 | 12527 | 763〜853 | 55〜66 |
| **11** | **12021** | **12263** | 778 | 61 |
| 12 | 11611 | 11709 | 675〜741 | 57〜63 |
| (参考) `-fit on` 既定 margin 1024 | 15181 | 15351 | 1408〜1420 | 106〜124 |

- 静的容量は 1 段あたり **約 406 MiB** 単調に減る。
- 画像 prefill の transient は **+96〜242 MiB** だけ（mmproj の graph は 1 回目で確保されて以後増えない）。
- 生成速度は 4→8 段で大きく落ち、8 段以降はほぼ横ばい（MTP speculative の受理率のばらつきの方が大きい）。
  **10 と 12 の間に実質的な速度差はない**ので、この領域では余白を取る方が得。

### 3.3 実際に使える VRAM は 16311 MiB ではなく約 15.9 GB

`--n-cpu-moe 10`（VLM 12.4 GB）+ TTS 上限 3072 の同時実行では、`nvidia-smi` が **15888 MiB** まで
上がったところで両方が壊れた（`results/10_concurrent_ncmoe10_fail.json`）:

- TTS: `... total capacity of 15.52 GiB of which 31.31 MiB is free` → 物理的に空きが無い OOM
- llama-server: 174 s 時点で **プロセスごと落ちた**（llama-swap のログに
  `upstream process exited unexpectedly`、以後 502 が続き、llama-swap が再起動）

**llama.cpp は実行中の VRAM 不足からは回復しない**（ggml は abort する）。一方 TTS 側は例外を投げて
プロセスは生き残る。したがって安全側の設計は「**llama.cpp に必要量を静的に確保させ、
TTS を上限で縛る**」になる。運用目安は `nvidia-smi` 合計 **15.4 GB 以下**。

### 3.4 TTS の CUDA Graph は同居では割に合わない

TTS を上限なしで 420 s（39 リクエスト、5 種類の文長の巡回）走らせ、`max_memory_reserved` の
頭打ちを見た:

| TTS 設定 | reserved 頭打ち | 頭打ちまで | `nvidia-smi` 実効 | synth wall median |
|---|---|---|---|---|
| CUDA Graph あり（既定） | **3458 MiB** | 21 リクエスト | 3979 MiB | 0.70〜0.76 s |
| `IRODORI_OPT_CUDA_GRAPH=0` | **2438 MiB** | 4 リクエスト | 2857 MiB | 0.77 s |

Graph を切ると **1020 MiB 減り、頭打ちも早い**。代償は synth 1 発あたり **+70 ms**
（6〜18 s の音声に対して RTF 0.074 → 0.082、再生インターバルの中に完全に隠れる）。
1020 MiB は `--n-cpu-moe` 2.5 段分に相当するので、**同居では Graph を切って VLM に回す方が有利**。

（Graph の利得が大きいのは 03 の通り短文を連射する場合で、
「合成 → 音声長だけ待つ」本シナリオでは待ち時間に埋もれる。）

## 4. 変更内容

| ファイル | 変更 |
|---|---|
| `bench/coexist_stress.py` | 新規。VLM を HTTP で、TTS を別プロセスで駆動し、`nvidia-smi` の全体タイムラインを取りながら 6 シナリオを回す |
| `bench/coexist_tts_worker.py` | 新規。TTS 側のワーカー（別プロセス = load/unload が本物の CUDA context の生成/破棄になる）。OOM を捕まえて生き延び、`oom` フラグ付きで報告する |
| `bench/coexist_llama_swap.sh` | 新規。`config.yaml` を変更せず `LLAMA_ARG_N_CPU_MOE` を注入して llama-swap を起動する |
| `bench/assets/coexist_image.png` | 新規。画像 prefill 用の 1920x1440 テスト画像 |
| `irodori_tts/opt_config.py` | `IRODORI_OPT_VRAM_LIMIT_MB` の既定を **3072 → 3840**（§5） |

## 5. 09 の上限 3072 MB は過小だった（既定を 3840 に変更）

同居とは独立の問題として、**既定設定のまま TTS を長時間走らせると OOM する**ことが分かった。
VLM を常駐させただけ（トラフィックなし）で TTS を回すと、13 発目で止まる:

```
resv trace (MiB): 2510 2514 2634 2816 2910 2910 2952 2956 3000 3070 3072 3072 3072 → OOM
```

`max_memory_reserved` が上限 3072 に張り付いて、codec decode の 46 MiB が取れなくなる。
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（リポジトリの既定）でも、
`IRODORI_OPT_EMPTY_CACHE=1`（3068 まで）でも、`IRODORI_OPT_GRAPH_MAX_STATIC_MB=96` でも**変わらない**。
再利用できない領域は CUDA Graph の private pool（168 MiB）と、リクエストごとに増える graph static だから。

**09 の 3072 という数字は、上限を掛けた状態で測った `max_memory_reserved` だった** ——
上限そのものが測定値を頭打ちにするので、自己参照的な値になっていた。上限を外して測り直すと:

| 測り方 | 必要な reserved |
|---|---|
| 宣言上限入力の stress（`bench/stress_vram.py`、graph LRU 充填済み） | **3124 MiB** |
| 5 種類の文長を巡回する長時間実行（本実験、39 リクエスト） | **3458 MiB** |
| 同上、`IRODORI_OPT_CUDA_GRAPH=0` | 2438 MiB |
| 宣言上限入力の stress、`IRODORI_OPT_CUDA_GRAPH=0` | 2558 MiB |

→ 既定（Graph あり）は **3840 MB** とした。3458 に対して 382 MiB の余白がある。
検証: 既定 3840 で 420 s / 39 リクエストを完走し、reserved は 3458 で頭打ち、
`nvidia-smi` は 3755 MiB（`results/10_ttsonly_cap3840.json`）。

Graph を切る同居レシピでは 2558 が最悪なので **3072 で 514 MiB の余白**があり、
09 以前と同じ数字のままより厳しい保証になる。

## 6. 計測方法

```bash
# VLM: config.yaml は触らず、環境変数で配分を固定して llama-swap を起動
NCMOE=11 ./bench/coexist_llama_swap.sh

# 同時実行（本命）
uv run --no-sync python bench/coexist_stress.py --scenario concurrent --duration 420 \
  --vlm-model gemma4-26b-a4b \
  --env IRODORI_OPT_CUDA_GRAPH=0 --env IRODORI_OPT_VRAM_LIMIT_MB=3072 \
  --output docs/experiments/results/10_concurrent.json

# 他のシナリオ: vlm_only / tts_only / tts_churn / vlm_swap / pipeline
```

シナリオの内容:

| シナリオ | 内容 |
|---|---|
| `vlm_only` | VLM のみ。短プロンプト → 長プロンプト（1029 token）→ 画像 を連続で回す |
| `tts_only` | TTS のみ。合成 → 音声長だけ sleep（再生の代用）を繰り返す |
| `concurrent` | 上記 2 つを同時に。TTS は llama-swap の都合を一切見ない |
| `tts_churn` | VLM を回したまま、TTS プロセスの load → 合成 ×3 → exit を繰り返す |
| `vlm_swap` | TTS を回したまま、llama-swap の `/unload` と再ロードを繰り返す |
| `pipeline` | VLM が書いた文をそのまま TTS が読み上げ、読み上げ中に裏で次の VLM リクエストを走らせる |

TTS は別プロセスなので、load = CUDA context 生成、unload = プロセス終了で、VRAM が本当に返る。
判定は `nvidia-smi` の全体タイムライン（100 ms 間隔）と、両側の失敗イベント数。

## 7. 結果（`--n-cpu-moe 11` + Graph off + 上限 3072）

| シナリオ | 時間 | TTS | VLM | 失敗 | `nvidia-smi` max |
|---|---|---|---|---|---|
| `vlm_only` | 240 s | — | text 72 / image 36 | 1（§8） | 12263 |
| `concurrent` | 420 s | 37 発 / 音声 391 s | text 117 / image 58 | **OOM 0 / crash 0**（1 は §8） | **14882** |
| `tts_churn` | 420 s | 11 サイクル / 33 発 | text 123 / image 62 | OOM 0 / crash 0（2 は §8） | 14878 |
| `vlm_swap` | 420 s | 38 発 / 音声 401 s | 18 回 unload→再ロード | OOM 0 / crash 0（3 は §8） | 14872 |
| `pipeline` | 420 s | 20 発 / 音声 405 s | text 22 | OOM 0 / crash 0（1 は §8） | 14856 |

### 7.1 同居のコスト

`vlm_only`（240 s）と `concurrent`（420 s）の比較。VLM は同じ 3 種類のタスクを回している。

| 指標 | 単独 | 同居 | 差 |
|---|---|---|---|
| TTS synth wall median | 0.771 s | 0.920 s | **+19 %**（RTF 0.074 → 0.090） |
| TTS synth wall p95 | 1.028 s | 1.218 s | +18 % |
| VLM 短プロンプト応答 median / p95 | 1.426 / 2.819 s | 1.560 / 3.068 s | +9 % / +9 % |
| VLM 長プロンプト応答 median / p95 | 3.282 / 4.007 s | 3.437 / 4.314 s | +5 % / +8 % |
| VLM 画像応答 median / p95 | 1.839 / 2.121 s | 1.902 / 2.402 s | +3 % / +13 % |
| VLM prefill tok/s（1029 token） | 778 | 776 | ±0 |
| VLM 生成 tok/s（長プロンプト） | 61.2 | 61.3 | ±0 |

**スループットはほぼ落ちず、遅延だけが 3〜9 % 増える。**
TTS は「合成 → 音声長だけ待つ」なので GPU を掴んでいる時間が短く（単独時の GPU util は 6.8 %）、
VLM から見ると散発的な割り込みにしかならない。逆に TTS 側は VLM が GPU を占有している最中に
合成するので +19 % を被るが、RTF 0.09 なら 18 s の音声でも合成は 1.2 s 程度で、
再生インターバルに十分収まる。

**譲り合いは何もしなくても起きる**。両方 CUDA の既定ストリーム優先度で走るだけで、
どちらも待たされはするが破綻しない。llama-swap 側に TTS の存在を知らせる必要はない。

### 7.2 ロード / アンロードの挙動

- **TTS の load は VLM 稼働中でも 10.1〜11.4 s**（11 サイクルの中央値 10.8 s）。単独時は 8.3〜8.7 s。
- **VLM の再ロードは TTS 常駐中でも成功する**（18 回）。llama-swap のオンデマンドロードを
  引き起こしたリクエストの wall は p95 11.8 s / max 15.1 s（ロード時間込み）。
  この間 TTS の合成は止まらない。
- 配分が静的なので**どちらの順序でも入る**: TTS 常駐時の空きは 16311-2857 = 13454 > 12021、
  VLM 常駐時の空きは 16311-12263 = 4048 > 2857。

### 7.3 パイプライン（VLM → TTS 読み上げ）

VLM が生成した文（先頭 120 文字）をそのまま読み上げ、読み上げ中に次の VLM リクエストを
バックグラウンドで走らせた。20 発すべて成功。音声長は 12.0〜26.0 s（120 文字上限のため
6〜18 s の想定より長め）で、その長い入力でも TTS の reserved は **2558 MiB**
（= §5 の宣言上限 stress と同じ値）に収まった。VLM の生成と TTS の合成が重なる区間でも
`nvidia-smi` は 14856 MiB を超えない。

## 8. 残った失敗（VRAM 起因ではない）

採用した運用点の 5 シナリオ（VLM リクエスト計 568 件）で、失敗は HTTP 500 が **8 件（1.4 %）**だけ。
中身は

```
{"error":{"code":500,"message":"The model produced output that does not match the expected
 Content-only format","type":"server_error"}}
```

llama-server 側のチャットテンプレート/パーサのエラーで、**VRAM とは無関係**
（`--reasoning off` の設定でモデルが想定外の形式を出した時に発生する）。
内訳は `concurrent` 1 / `tts_churn` 2 / `vlm_swap` 3 / `pipeline` 1 / `vlm_only` 1 で、
**TTS が居ない `vlm_only` でも出る**（= 同居とは無関係）。llama-server は落ちず、次のリクエストは通る。
アプリ側でリトライすれば実害はない。

## 9. 採否

**採用。** 同居の運用点を以下とする。

| | 設定 |
|---|---|
| llama-swap 起動 | `NCMOE=11 ./bench/coexist_llama_swap.sh`（= `LLAMA_ARG_N_GPU_LAYERS=99 LLAMA_ARG_N_CPU_MOE=11`） |
| TTS | `IRODORI_OPT_CUDA_GRAPH=0` + `IRODORI_OPT_VRAM_LIMIT_MB=3072` |
| 予算 | VLM 12.0〜12.3 GB + TTS 2.9 GB = 14.9 GB / 16.3 GB（余白 1.4 GB） |

あわせて `IRODORI_OPT_VRAM_LIMIT_MB` の既定を 3840 に変更した（単独運用時の正しい値）。

### 未計測 / 注意

| 項目 | 影響 |
|---|---|
| `gemma4-12b`（config.yaml のもう 1 つのモデル） | `-c 262144 -np 20` で KV が桁違い。同居させるなら別途計測が必要 |
| VLM の context を 8192 より増やす | KV が増えるので `--n-cpu-moe` を上げ直す必要がある |
| TTS の LoRA / `num_candidates > 1` / watermark | 09 と同じく上限の外 |
| TTS 側の DiT compile（`IRODORI_OPT_COMPILE_DIT=1`） | 本実験では未使用。compile と Graph off の組み合わせは未計測 |
| 画像サイズ | 1920x1440 と 640x480 のみ。より大きい画像は mmproj の transient が増える |
