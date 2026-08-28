# 推論高速化・省VRAM化 実験ノート

このディレクトリは **このPC・この環境専用** の推論最適化実験の記録である。
移植性は考慮しない（他環境での再現は保証しない）。

> **ブランチ注記 (`metal-local`)**: 01〜11 は RTX 5060 Ti 機での記録（`main` の履歴として
> 凍結）。この Mac (M3 Pro / Metal) での作業は [12-metal-port.md](12-metal-port.md) 以降。
> 12 以降の計測は `bench/bench_runtime.py`（MPS 版）で取り、結果 JSON は `results/metal_*.json`。

## 前提・制約

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB (sm_120 / Blackwell, 36 SM) |
| Driver / CUDA UMD | 610.43.02 / 13.3 |
| PyTorch | 2.10.0+cu128, triton 3.6.0, torchao 0.16.0, transformers 5.12.1 |
| Python | 3.10.20 (`.venv`, uv 管理) |
| CPU / RAM | 12 threads / 27 GiB |
| モデル | `Aratako/Irodori-TTS-v4.1-Small` (推奨のまま。FP32 safetensors 2.9 GB, 714 tensors) |
| Codec | `Aratako/Semantic-DACVAE-Japanese-32dim` (48 kHz) |
| バッチ数 | 常に 1 (`num_candidates=1`) |
| 禁止事項 | TTS モデル自体の置き換え |
| 目標 | VRAM 消費の削減、GPU 使用率の向上、推論速度（warm 後の反復時間）の短縮 |

## ドキュメント運用ルール

- 1 実験 = 1 ファイル。`NN-slug.md` の連番で追加する。
- `00-index.md` が目次。実験を追加したら 1 行追記する。
- 巨大な 1 ファイルに進捗を追記し続けない。実験が終わったらそのファイルは原則凍結し、
  続きは新しい番号のファイルに書く。
- 生の計測 JSON は `results/` に置く（`bench/bench_runtime.py` の出力）。
- 各実験ファイルの構成:
  1. 目的 / 仮説
  2. 変更内容（差分の要点、対象ファイル）
  3. 計測方法（コマンド）
  4. 結果（表）
  5. 品質確認（音声 hash 一致 / 不一致、聴感メモ）
  6. 採否と次のアクション

## 計測方法

```bash
# warm ベンチ（モデル 1 回ロード → warmup → 反復計測 → JSON）
uv run --no-sync python bench/bench_runtime.py \
  --precision bf16 --tag <tag> \
  --inputs short medium long caption_noref \
  --warmup 2 --repeats 10 \
  --output docs/experiments/results/<tag>.json
```

- `wall_median` / `wall_p95`: `synthesize()` 1 回の壁時計時間（warm、reference 音声は毎回再ロード・再エンコード）
- `stages_median_ms`: runtime 内の stage timing の median
- `rtf_median`: wall_median / 生成音声長
- `gpu_util`: 計測区間中に `nvidia-smi` を 50 ms 間隔でサンプリングした utilization.gpu
- `cuda_mem_peak`: 計測区間中の `max_memory_allocated` / `max_memory_reserved`
- `audio_hashes`: 生成音声 (float32 PCM) の SHA-256。同一条件で 1 種類なら決定的

代表入力は `bench/bench_runtime.py` の `INPUTS` に定義（短文 / 中文 / 長文 / caption+no-ref）。
参照音声は `outputs/sample.wav`（7.28 s, 48 kHz）。

## ツール一覧（`bench/`）

| スクリプト | 用途 |
|---|---|
| `bench_runtime.py` | warm ベンチ（stage timing / RTF / GPU util / VRAM / hash → JSON） |
| `bench_load.py` | ロード/アンロードのプロファイル（子プロセスで cold load、phase 内訳、11） |
| `run_ablation.sh` | 最適化スイッチを 1 つずつ足す ablation を順に実行 |
| `check_equivalence.py` | legacy 経路と最適化経路を同一 runtime で比較（hash / max abs diff） |
| `check_codec_fold.py` | codec weight_norm fold の bit 一致確認と decode 時間・VRAM |
| `check_codec_compile.py` | codec decoder の eager / compile / bf16 比較 |
| `profile_memory.py` | request 中の stage 別 transient VRAM |
| `stress_vram.py` | 宣言上限入力（text 256 / caption 512 / 参照 15-120 s）での VRAM stress（09） |
| `check_codec_encode_chunk.py` | 参照 encode の chunk 化の一致確認と時間・VRAM（09） |
| `gen_quality_wavs.py` | 品質比較用 wav セット生成（watermark なし） |
| `audio_metrics.py` | 2 つの wav の max abs / SNR / log-mel LSD / 長さ差 |
| `coexist_stress.py` | llama-swap の VLM と TTS の同居 stress（同時実行 / ロード churn / パイプライン、10） |
| `coexist_tts_worker.py` | 同居 stress の TTS 側ワーカー（別プロセス = 本物の load/unload） |
| `coexist_llama_swap.sh` | `config.yaml` を変更せず `--n-cpu-moe` を注入して llama-swap を起動（10） |

ロード用の道具は `prebake_runtime.py`（事前計算バンドルの生成 / `--list` / `--prune`、11 参照）。

最適化スイッチは `irodori_tts/opt_config.py` の `IRODORI_OPT_*` 環境変数（既定は全部 on、watermark は off）。

## 品質の扱い

- **出力保持型**（数値経路を変えない: 重複計算の除去、同期除去、mask 事前計算、CUDA Graph 等）は
  音声 hash の一致、または float 誤差レベルの一致を要求する。
- **品質が変わり得る型**（FP32→BF16、step 削減、codec 量子化 等）は別実験として分け、
  hash 一致を要求せず、聴感 + 指標で判断する。
- 出力保持型でも attention の key 長や conv の長さが変わると float 誤差が出る。判定は **FP32 で
  max abs diff ≤ 1e-3**。BF16 では 40 step の Euler 積分がこの誤差をカオス的に増幅するので、
  BF16 同士の bit 比較はしない（02 参照）。

## 現在の既定（11 時点）

`infer.py` / Gradio とも bf16 が既定。`IRODORI_OPT_*` は全部 on（watermark off、codec 重み fp32 +
decode のみ bf16 autocast、decode chunk 96/16、参照 encode chunk 96/32、CUDA Graph は
entry 6 / static 予算 256 MB / entry ごと pool / latent 384 frame 超は eager、
**VRAM 上限 3840 MB**）。DiT compile は常駐プロセス向けに
`IRODORI_OPT_COMPILE_DIT=1`（初回 45〜80 s、07 参照）。
FP32 で動かすときは `IRODORI_OPT_VRAM_LIMIT_MB=0` にすること（FP32 重みだけで 2.9 GB）。

ロードは skip-init / prebake バンドル / 並列ロードが既定で on（`IRODORI_OPT_SKIP_INIT`,
`IRODORI_OPT_PREBAKE`, `IRODORI_OPT_PREBAKE_DIR`, `IRODORI_OPT_LOAD_PARALLEL`）。
バンドルは `prebake_runtime.py` で 1 回作る。無ければ黙って通常経路になる（11 参照）。

上限 3840 MB は、上限を外して測った実必要量（宣言上限入力の stress で 3124 MiB、
文長の異なるリクエストを巡回する長時間実行で 3458 MiB）に約 380 MiB の余白を足した値（10 参照）。
09 の 3072 は上限を掛けた状態で測った値で、**長時間実行では 13 リクエスト目で OOM した**。
LoRA・`num_candidates>1`・watermark 有効はこの余白の外なので、使うなら再計測する。

### 他プロセスと GPU を共有するとき（10 参照）

llama-swap の VLM などと同居させる場合は

```bash
IRODORI_OPT_CUDA_GRAPH=0 IRODORI_OPT_VRAM_LIMIT_MB=3072
```

を使う。CUDA Graph を切ると常駐量が `nvidia-smi` で 3979 → **2857 MiB** に下がり
（reserved の頭打ちが 3458 → 2438 MiB）、代償は synth 1 発あたり +70 ms だけ。
VLM 側は `NCMOE=11 ./bench/coexist_llama_swap.sh` で静的に 12.0 GB に固定する
（`-fit on` のままだと load 時の空き容量しだいで VLM が 15.2 GB を掴み、TTS がロードできなくなる）。
