# 11: ロード時間 — 「重みを運ぶ時間」はもう払っていなかった

日付: 2026-08-26

## 1. 目的 / 仮説

初回ロードに約 9 秒かかる。重みは FP32 で 2.9 GB、PCIe で運ぶだけなら 1 秒もかからないはずで、
GPU 帯域から見て遅すぎる。**事前に計算して置いておけないか**、が出発点。

仮説は「ロードの大半はディスク→GPU の転送とキャストであり、済ませた状態を保存すれば消える」。
先に結論を言うと、**この仮説は外れていた**（§3）。転送は元から 0.8 秒しかなく、
残りは全部 Python 側の無駄と import だった。事前計算は効いたが、効いた理由は転送ではない。

## 2. 計測方法

`bench/bench_load.py` を新規追加した。ロードは**毎回まっさらな子プロセス**で行う
（= CUDA context の生成込み、実運用の「起動して喋らせる」と同じ条件）。
内訳は `IRODORI_OPT_LOAD_TRACE=1` のとき `inference_runtime` が記録する phase timing から取る。

```bash
uv run --no-sync python bench/bench_load.py --repeats 3 --tag 11_final \
  --synth --reload --output docs/experiments/results/11_final.json
```

| 指標 | 意味 |
|---|---|
| `t_process_to_loaded` | 子プロセス起動 → runtime が使える状態になるまでの壁時計 |
| `t_import_torch` / `t_cuda_init` / `t_from_key` | その内訳（親からの `time.monotonic()` 基準） |
| phases | `ckpt_read` / `model_construct` / `load_state_dict` / `to_device` / `tokenizer` / `codec_load` … |
| `t_first_synth` | ロード直後の 1 発目の `synthesize()`（= 体感の「喋り出すまで」の残り） |
| `t_second_load` | **同一プロセス内**の 2 回目のロード（import は払い済み） |
| `smi_*` / `rss_mib` | 自プロセスの `nvidia-smi` 実測と RSS |

数値はすべて 3 回の median。GPU は他プロセスなし（`nvidia-smi` 13 MiB）。

## 3. ベースラインの内訳（変更前 = 259d846）

`results/11_baseline.json`。**9.55 s** の中身:

| 区間 | 秒 | 中身 |
|---|---|---|
| `import torch` | 1.00 | 削れない |
| `import irodori_tts` | 0.18 | |
| CUDA context | 0.20 | |
| HF hub の解決 | 0.27 | `snapshot_download` の HTTP ラウンドトリップ（キャッシュ済みでも走る） |
| `model_construct` | **4.19** | transformers の import ≈ 2.0 + **乱数初期化 1.8** + モジュール構築 0.7 |
| `ckpt_read` | 0.02 | safetensors は mmap なのでここでは読まない |
| `load_state_dict` | 0.11 | `assign=True` なので mmap テンソルを差すだけ |
| `to_device` | **0.78** | FP32 → BF16 キャスト + H2D。**「重みを運ぶ」実費はこれだけ** |
| `tokenizer` + `caption_tokenizer` | 0.51 + 0.37 | 同じ tokenizer を 2 回読んでいる |
| `codec_load` | **1.93** | `torch.load` 0.4 + **乱数初期化 0.6** + weight_norm fold + probe encode 0.55 |

cProfile で見ると、`model_construct` の 4.19 s のうち **1.82 s は `uniform_` 一つ**だった
（`nn.Linear.reset_parameters` の kaiming 初期化、496 回、766 M パラメータ）。
その直後の `load_state_dict` が 714 個のテンソルを**全部**上書きするので、この 1.8 秒は丸ごと捨てている。
codec も同じ構造で 0.6 s。

つまりベースライン 9.55 s の内訳は **転送 0.8 s / 捨てる初期化 2.4 s / import 3.2 s / その他 3.1 s**。
「GPU 帯域から見て遅すぎる」という見立ては正しく、原因は帯域ではなかった。

## 4. 変更内容

| # | 変更 | ファイル |
|---|---|---|
| A | 上書きされる乱数初期化をやめる | `irodori_tts/fast_init.py`（新規）、`inference_runtime.from_key`、`codec.py` |
| B | text/caption tokenizer が同一なら 1 個だけ読む | `inference_runtime.from_key` |
| C | **事前計算バンドル**（prebake） | `irodori_tts/prebake.py`（新規）、`prebake_runtime.py`（新規） |
| D | import の裏で重み・tokenizer・codec を並列ロード | `inference_runtime.from_key` |
| E | 計測ハーネス | `bench/bench_load.py`（新規）、`_load_phase` トレース |

スイッチは `IRODORI_OPT_SKIP_INIT` / `IRODORI_OPT_PREBAKE` / `IRODORI_OPT_PREBAKE_DIR` /
`IRODORI_OPT_LOAD_PARALLEL`（既定は全部 on）。

### A. 乱数初期化のスキップ

`fast_init.skip_random_init()` は `torch.nn.init` の**乱数系だけ**（`uniform_` / `normal_` /
`kaiming_*` / `xavier_*` / `trunc_normal_` / `orthogonal_` / `sparse_`）を no-op に差し替える。
`zeros_` / `ones_` / `constant_` は残す（checkpoint に無いバッファを埋めている可能性があるため）。

安全性の根拠は**その後の `load_state_dict` が strict であること**に尽きる:

- DiT: checkpoint の 714 キーとモデルの 714 キーが完全一致（過不足 0）。`load_state_dict` の
  既定は `strict=True` なので、埋め忘れがあれば例外になる。
- codec: `audiotools` の `BaseModel.load` は **`strict=False`** で読んでいた（= 欠けたキーは
  乱数のまま残る）。ここは自前の `_load_dacvae_weights()` に置き換え、`strict=True` にした。
  ついでに毎回失敗する `torch.package` の試行も消えた。実測でも欠けキーは 0。

### B. tokenizer の共有

v4.1-Small は text と caption で同じ tokenizer（checkpoint 同梱の `tokenizer/`、`add_bos` も同じ）を
指している。source / `local_files_only` / `add_bos` が一致するときだけ同じインスタンスを使い回す。
推論時の wrapper は状態を持たないので副作用はない。

### C. 事前計算バンドル（prebake）

「ロードした結果」をそのまま保存する。生成は

```bash
uv run --no-sync python prebake_runtime.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small
# -> ~/.cache/irodori-tts/prebake/<fingerprint>/  (1871 MiB)
#    manifest.json / model.safetensors (1462 MiB, BF16) / codec.safetensors (410 MiB, fold 済み)
```

作り方は「一度**遅い経路で**ロードして、出来上がった `state_dict()` をそのまま書く」。
だから中身は定義上、通常経路が作る tensor と bit 一致する。

| バンドルが消すもの | 効果 |
|---|---|
| FP32 → BF16 のキャスト（`to_device` 0.78 s） | ほぼ 0（読んだ時点で BF16、`device="cuda"` で直接 GPU へ） |
| ディスクから読む量 2.9 GB → 1.46 GB | 転送も半分 |
| codec の `torch.load`（410 MB の pickle）と weight_norm fold | `codec_load` 1.25 → 0.26 s |
| ロード中の FP32 中間コピー | `nvidia-smi` 2216 → 2136 MiB、reserved 2054 → 1974 MiB |

**使う/使わないの判定は自動**。`manifest.json` の `identity`（checkpoint のパス/サイズ/mtime、
codec repo、precision、device 種別、fold フラグ、torch バージョン、prebake フォーマット版）が
今の要求と完全一致し、かつ manifest に記録した codec `weights.pth` の stat も一致したときだけ使う。
一つでも合わなければ黙って通常経路に落ちる（実測で確認: §7）。

### D. 並列ロード

`model_construct` の中身のうち約 2.0 s は `from transformers import AutoConfig, AutoModel` で、
これは**純粋に GIL を握った Python**。一方

- safetensors の GPU への読み込み（Rust + `cudaMemcpy`）
- tokenizer の読み込み（`tokenizers`、Rust）
- codec の構築と probe encode（大半が C++/CUDA）

はどれも GIL を離す。そこで `from_key` の頭で `ThreadPoolExecutor(3)` に

1. バンドルの DiT 重み読み込み
2. text/caption tokenizer
3. **codec 一式**（読み込み → 構築 → fold → probe encode）

を投げ、メインスレッドは transformers の import とモジュール構築を続ける。
join した時点でどれも完了済みで、phase timing は 3 つとも 0.00 s になる。

スレッド間の import 競合を避けるため、pool に投げる前にメインスレッドで `import transformers`
だけ済ませておく（phase `transformers_import` ≈ 0.4 s）。

## 5. 結果

`--repeats 3` の median。すべて同じ checkpoint / bf16 / 既定スイッチ。

| 設定 | プロセス起動→ロード完了 | `from_key` | 同プロセス 2 回目 | 1 発目の synth | JSON |
|---|---|---|---|---|---|
| 変更前（259d846） | **9.55 s** | 7.90 | — | — | `11_baseline` |
| 全スイッチ off（= 変更前 + tokenizer 共有） | 9.03 s | 7.38 | 4.73 s | 1.14 s | `11_A_alloff` / `11_alloff_synth` |
| + A 乱数初期化スキップ | 6.59 s | 4.95 | — | — | `11_B_skipinit` |
| + C prebake | 6.05 s | 4.41 | — | — | `11_C_prebake` |
| + D 並列ロード（**既定**） | **5.08 s** | 3.39 | **1.44 s** | 1.14 s | `11_final` / `11_final_synth` |
| 既定 + `HF_HUB_OFFLINE=1` | **4.83 s** | 3.36 | — | — | `11_offline` |

phase 内訳（既定、`results/11_final.json`）:

```
torch 0.96 | pkg 0.17 | cuda 0.25 | transformers_import 0.32 | model_construct 2.98
| bundle_model_read 0.00 | load_state_dict 0.01 | to_device 0.01
| tokenizer 0.00 | caption_tokenizer 0.00 | codec_load 0.00
```

**重み・tokenizer・codec は完全に隠れて 0.00 s になった。** 残っているのは Python の import と
モジュール構築だけで、これが 5.0 s のうち 4.7 s を占める（`t_process_to_loaded` 5.00、
`t_second_load` 1.42 の再測でも同じ）。

メモリ:

| | 変更前 | 既定 |
|---|---|---|
| ロード直後 `nvidia-smi` | 2216 MiB | **2136 MiB** |
| ロード直後 `max_memory_reserved` | 2054 MiB | **1974 MiB** |
| synth 1 発後 `nvidia-smi` | 2686 MiB | 2686 MiB |
| アンロード後 `nvidia-smi` | 320 MiB | 320 MiB（CUDA context が残る） |
| プロセス RSS | 1846 MiB | 1887 MiB |
| unload | 0.36 s | 0.36 s |

## 6. 品質確認

- **音声 hash 一致**: `bench/bench_runtime.py` の 4 入力（short / medium / long / caption_noref）で、
  変更前（`IRODORI_OPT_SKIP_INIT=0 IRODORI_OPT_PREBAKE=0 IRODORI_OPT_LOAD_PARALLEL=0`）と
  既定の SHA-256 が**全一致**（`results/11_defaults_before.json` vs `11_defaults_after.json`）。
- **warm 性能も同一**: wall median は short 473 / 473、medium 652 / 653、long 1444 / 1446、
  caption_noref 484 / 485 ms。`max_reserved` も 2990538752 B で一致。
- prebake の codec テンソルは通常経路の `state_dict()` と 317 個すべて `torch.equal` で一致。

つまり A〜D は 02 の分類でいう**出力保持型**で、hash 一致を満たしている。

## 7. 落とし穴: codec の probe encode は「latent_dim を測るため」だけではなかった

`DACVAECodec.load` の末尾には、ダミー波形を 1 回 encode して latent 次元を得るコードがある。
バンドルには latent_dim を書いてあるので**この probe は省ける**と考えて省いたところ、
参照音声を使う 3 入力の hash が変わった（`caption_noref` だけ一致 = 参照 encode 経路が犯人）。

切り分け: 同じ重み（317 テンソル bit 一致を確認済み）で `outputs/sample.wav` を encode すると

| | latent hash |
|---|---|
| 通常経路 | `46685b936da34a8c719781af` |
| バンドル + probe 省略 | `db2bf8caf7d1222de07335ba` |
| バンドル + probe あり | `46685b936da34a8c719781af` |

この probe は**そのデバイスでの最初の conv** であり、cuDNN のアルゴリズム選択を固定している。
省くと後続の encode が bit 単位で変わる。よって **probe は残す**（`prebaked_latent_dim` という
API は作らずに消した）。代わりに D の並列化で probe ごと裏に回したので、時間的な損はない。

なお probe を並列スレッドに移しても hash は変わらなかった（§6 の hash 一致はその状態での確認）。

## 8. 使い方

```bash
# 1 回だけ: バンドルを作る（checkpoint を変えたら作り直す）
uv run --no-sync python prebake_runtime.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small
uv run --no-sync python prebake_runtime.py --list     # 一覧
uv run --no-sync python prebake_runtime.py --prune    # 全部消す

# 以降は何もしなくてよい。オフラインなら更に 0.2 s 速い
HF_HUB_OFFLINE=1 uv run --no-sync python infer.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small ...
```

無効化: `IRODORI_OPT_PREBAKE=0`（バンドルを無視）、`IRODORI_OPT_SKIP_INIT=0`（乱数初期化を戻す）、
`IRODORI_OPT_LOAD_PARALLEL=0`（直列ロード）。置き場所は `IRODORI_OPT_PREBAKE_DIR`。

バンドルが無い / 古い / FP32 を要求した場合は自動で通常経路になる。実測:

| ケース | 結果 |
|---|---|
| `--precision fp32`（バンドルは bf16 用） | 通常経路、6.39 s |
| `IRODORI_OPT_PREBAKE_DIR` が存在しない | 通常経路、5.93 s |
| manifest の torch バージョンを書き換え | 通常経路、6.11 s |
| manifest の codec `weights.pth` サイズを書き換え | 通常経路、5.97 s |
| `IRODORI_OPT_CUDA_GRAPH=0`（同居レシピ、10 参照） | バンドル使用、5.11 s |

## 9. 採否と次のアクション

**採用**（A〜D すべて既定 on）。cold start は 9.55 → **5.08 s**（`HF_HUB_OFFLINE=1` で 4.83 s）、
同一プロセスの再ロードは 4.73 → **1.44 s**。喋り出すまで（ロード + 1 発目）は 10.19 → **6.22 s**。

**残っているのは Python の import とモジュール構築だけ**なので、この方向での改善はここで頭打ち。
これ以上を望むならロード自体を無くすしかない:

| 案 | 見込み | 状態 |
|---|---|---|
| 常駐プロセス（unix socket 経由の合成サーバ）+ 薄いクライアント | ロード 0 s、リクエストは `t_first_synth` の 1.14 s から | 未着手（次の実験候補） |
| import 済みプロセスを fork して CUDA だけ子で初期化 | 約 2 s まで（CUDA context は fork 不可なので子で作り直し） | 未計測、複雑 |
| `model_construct` の 2.9 s を削る | transformers の `AutoConfig`/`AutoModel` import が 1.5 s。ModernBERT を直接 import する方が**遅かった**（2.71 s） | 打ち止め |

### 未計測 / 注意

| 項目 | 影響 |
|---|---|
| 量子化 checkpoint | torchao の tensor subclass は safetensors に入らないので prebake 不可（builder が例外を出す）。skip-init と並列化は効く |
| LoRA アダプタ | バンドルはベース重みのみ。アダプタは従来どおり毎回読む |
| バンドルの容量 | 1 checkpoint あたり 1.87 GB。checkpoint を増やすと `--prune` が要る |
| `IRODORI_OPT_COMPILE_DIT=1` | compile の 45〜80 s はロードとは別枠（07 参照）。バンドルは compile 結果を持たない |
