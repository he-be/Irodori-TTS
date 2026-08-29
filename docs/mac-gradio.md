# Mac (Apple Silicon) で Gradio を動かす

対象: `metal-local` ブランチ / M3 Pro (18 GB) / macOS 15.7.5。
背景と測定の詳細は [12-metal-port.md](experiments/12-metal-port.md)（Metal 専用化）と
[13-ane.md](experiments/13-ane.md)（Neural Engine 併用）にあります。

**実測** = 手元で数字を取ったもの。**未確認** = 根拠なし、断定しない。

## 1. 何が動くのか

RF step（DiT）を **Neural Engine (ANE)** で、CFG の cond 分岐を **GPU (MPS)** で同時に回します。
text / speaker / caption encoder と codec decode は GPU です。

| 処理 | 実行先 |
|---|---|
| RF step の uncond CFG 2 分岐、CFG なし区間 | ANE（Core ML、別プロセス） |
| RF step の cond CFG 1 分岐 | GPU (MPS) |
| encoder 各種 / codec decode | GPU (MPS) |

素の MPS eager 比 **1.50×**（short で RTF 0.480 → 0.319、実測）。

## 2. 初回だけ必要な準備

```bash
uv sync
uv run python bench/build_ane.py --shapes full
```

`build_ane.py` は Core ML パッケージを `~/.cache/irodori-tts/ane/` に作ります（**実測**: export 約 5 分 +
ANE コンパイル約 30 分、ディスク 8〜10 GB）。これをやらずにアプリを起動すると、最初のリクエストの中で
同じコンパイルが走って数十分ブロックします。

2 回目以降のプロセス起動では OS がコンパイル結果をキャッシュしているので、パッケージあたり 0.2 s で
ロードされます（**実測**: 6 パッケージで 2.3 s）。

## 3. 起動

```bash
# 参照音声クローン（Speaker Inversion 入力あり）
uv run python gradio_app.py --server-port 7860

# caption（スタイルプロンプト）でのボイスデザイン
uv run python gradio_app_voicedesign.py --server-port 7861
```

ブラウザで `http://127.0.0.1:7860` / `:7861`。停止は Ctrl-C（子プロセスの ANE worker も一緒に落ちます）。

両アプリとも起動時に以下を既定にしています（`gradio_app*.py` の冒頭、`os.environ.setdefault` なので
シェル側で指定すればそちらが勝ちます）:

```
IRODORI_OPT_ANE=1              ANE 経路を有効化
IRODORI_OPT_ANE_GPU_BRANCHES=1 cond 分岐を GPU に回す
IRODORI_OPT_ANE_SHAPES=full    23 種の latent 長 × 2 プロファイル
IRODORI_OPT_COMPILE_DIT=1      GPU 側 DiT を torch.compile
IRODORI_OPT_COMPILE_CODEC=1    codec decode を torch.compile
```

デバイスと精度は **mps + fp16 固定**で、UI から選べません。デバイスの選択肢は元々 `mps` の 1 択、
精度も bf16 は 40 step の積分で誤差が発散して別の読みになる（12-metal-port.md 5-6）ため、
選ぶ意味がないので UI から外してあります。

## 4. 画面の見方

**Timing** パネル:

```
[timing] prepare_reference: 4.7 ms       ← 1 桁 ms なら参照キャッシュが効いている
[timing] sample_rf: 1573.7 ms
[timing] decode_latent (sequential): 663.9 ms
[timing] wall: 2.295 s  audio: 7.20 s  RTF: 0.319
```

**Run Log** パネル:

```
info: speaker state served from L2 cache.                        ← 参照キャッシュのヒット
info: rf step on ANE + GPU (steps=40, predict=1486 ms, gpu_branches=1)
```

最後の行が `rf step on MPS (ANE fallback)` になっていたら、そのリクエストは ANE を使わず
MPS だけで走っています（条件は 6 節）。

## 5. 参照音声を 1 回だけ変換してテキストを差し替える

**既定で有効です。追加の操作は要りません。** 参照音声を 1 回アップロードしたら、あとはテキストだけ
書き換えて Generate を押せば、2 回目以降は参照の変換を丸ごとスキップします。

**実測**（同一プロセス、同じ参照、テキストのみ差し替え）:

| | prepare_reference |
|---|---:|
| 1 回目 | 571.5 ms |
| 2 回目以降 | 1.2〜1.5 ms |

キャッシュは 2 段（`IRODORI_OPT_REF_CACHE=1`、8 エントリ）:

- **L1**: wav ファイル → reference latent（codec encode の結果）
- **L2**: reference latent → speaker state（DiT の speaker encoder 出力）

外れる条件（`inference_runtime.py` の `_reference_l1_key` / l2_key）:

- 同じファイルでも**アップロードし直す**（Gradio の一時パスが変わる）→ 一度上げたら触らない
- 参照ファイルを変える、`max_ref_seconds` を変える → L1 から作り直し
- **候補数 (`num_candidates`) を変える**、`speaker_uncond_mode` を変える → L2 だけ作り直し
- 9 個以上の参照を使い回す（LRU 8 エントリから溢れる。`IRODORI_OPT_REF_CACHE_ENTRIES` で増やせる）

text / steps / seed / cfg / duration の変更はキャッシュに影響しません。

## 6. 注意点

### 初回リクエストは遅い

ANE パッケージのロード（6 個で 2.3 s）と `torch.compile`（DiT 約 20 s + codec 約 4 s）が最初の
リクエストに乗ります。**実測**: 同じ short 入力で 1 回目 6.92 s → 2 回目 2.18 s。
速度を見るときは 2 回目以降の数字を見てください。

コンパイルを待ちたくない場合は `IRODORI_OPT_COMPILE_DIT=0 IRODORI_OPT_COMPILE_CODEC=0` を付けて起動
します（ANE 経路には影響しません。GPU 分岐の compile は −1.5% しか効かない）。

### ANE から MPS に落ちる条件

request 単位でフォールバックします。Run Log の `rf step on ...` で判別できます。

- 生成が **約 30 秒（768 frame）を超える**（batch 3 のパッケージは S ≤ 768 までしか列挙していない）
- **候補数が 3 以上**（2 は候補 0 = ANE / 候補 1 = GPU で並列、3 以上は MPS）
- テキストが 256 トークン超、参照が約 40 秒超、caption が 64 トークン超
  （**実測**: 100 字程度の caption は ANE のまま）

落ちても出力は正しく、遅くなるだけです。

### 二重起動しない

ポート違いで 2 アプリを同時に上げるのは問題ありませんが、**同じアプリを 2 つ起動しない**でください。
各プロセスがモデル（約 1.9 GB）と ANE worker を別々に持ち、18 GB の unified memory を食い合います。
起動に失敗して `Cannot find empty port in range: 7860-7860` が出たら、既に上がっているので
`lsof -nP -iTCP:7860 -sTCP:LISTEN` で確認してください。

### ANE キャッシュを移動・リネームしない

`~/.cache/irodori-tts/ane/*.mlmodelc` をリネームしたり移動したりすると、OS 側の ANE コンパイル
キャッシュが無効になります（**実測**: 移動後のロードが 0.2 s → 244 s に戻った）。整理するときは
再コンパイル前提で。

### CPU フォールバックは無効

パッケージ import 時に `PYTORCH_ENABLE_MPS_FALLBACK=0` を強制しています。MPS 未対応の op に当たると
CPU に落ちずに例外になります（黙って 10 倍遅くなるのを防ぐため）。

### CLI (`infer.py`) は ANE off

単発起動ではパッケージのロードとコンパイルが見合わないため、CLI は ANE も compile も無効が既定です。
色々試すなら Gradio を上げっぱなしにするのが一番速いです。

## 7. 参考値（実測、fp16、3 回中央値、13-ane.md 5 節）

| 入力 | 音声長 | MPS eager | ANE + GPU | RTF |
|---|---:|---:|---:|---:|
| short | 7.20 s | 3459 ms | **2299 ms** | 0.319 |
| medium | 11.84 s | 5872 ms | **3931 ms** | 0.331 |
| long | 28.84 s | 16450 ms | **11210 ms** | 0.393 |
| caption + no-ref | 7.32 s | 3463 ms | **2323 ms** | 0.316 |
| short × 2 候補 | | 6.40 s | **4.16 s** | |

品質は fp32 出力を基準に SNR 23.8 dB / LSD 0.16 dB（short）。聴感では MPS 版と区別がつきません
（ユーザー確認、2026-08-29、13-ane.md 6 節）。
