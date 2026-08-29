# Mac (Apple Silicon) で Gradio を動かす

対象: `metal-local` ブランチ / M3 Pro (18 GB) / macOS 15.7.5。
背景と測定の詳細は [12-metal-port.md](experiments/12-metal-port.md)（Metal 専用化）、
[13-ane.md](experiments/13-ane.md)（Neural Engine 併用）、
[14-step-count.md](experiments/14-step-count.md)（sway 12 step 既定化 + 長さによる自動引き上げ）に
あります。

**実測** = 手元で数字を取ったもの。**未確認** = 根拠なし、断定しない。

## 1. 何が動くのか

RF step（DiT）を **Neural Engine (ANE)** で、CFG の cond 分岐を **GPU (MPS)** で同時に回します。
text / speaker / caption encoder と codec decode は GPU です。

| 処理 | 実行先 |
|---|---|
| RF step の uncond CFG 2 分岐、CFG なし区間 | ANE（Core ML、別プロセス） |
| RF step の cond CFG 1 分岐 | GPU (MPS) |
| encoder 各種 / codec decode | GPU (MPS) |

さらに sampler の step 数を 40 → 12（sway sampling、20 秒以上の出力は自動で 16）に落としてあります。
素の MPS eager / 40 step 比で **3.4×**（short で RTF 0.480 → 0.142、Gradio の既定 = ANE + compile、実測）。
内訳は step 削減 2.06× / ANE 1.35× / compile 1.23×（7 節）。

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
精度も bf16 は積分誤差が発散して別の読みになる（12-metal-port.md 5-6）ため、選ぶ意味がないので
UI から外してあります。

サンプラの既定は **Num Steps 12 / Time Schedule sway / Sway Coeff −1.0** です（14-step-count.md）。
さらに **出力が 20 秒以上になる request は自動で 16 step に引き上げます**（`IRODORI_OPT_AUTO_STEPS=0`
で無効化）。適用されると Run Log に `info: auto steps 12 -> 16 for a 32.3s output.` が出ます。
これは**下限としてのみ**働くので、UI で 40 を指定した request が下げられることはありません。

**8 step まで下げないでください**: 15 秒以下なら聴き分けられませんが、長文で 8 kHz 以上のノイズが
4〜5 dB 増えます（14-step-count.md 3-5）。また同じ step 数なら linear より sway の方が明確に良いので、
Time Schedule は sway のままにしてください。

## 4. 画面の見方

**Timing** パネル:

```
[timing] prepare_reference: 1.2 ms       ← 1 桁 ms なら参照キャッシュが効いている
[timing] sample_rf: 522.7 ms
[timing] decode_latent: 462.7 ms
[timing] wall: 1.023 s  audio: 7.20 s  RTF: 0.142
```

12 step では **decode と sample_rf がほぼ拮抗します**（compile 済みで 463 vs 523 ms）。
40 step 時代は sample_rf が decode の 2.4 倍で、内訳の見え方が変わっています。

**Run Log** パネル:

```
info: speaker state served from L2 cache.                        ← 参照キャッシュのヒット
info: auto steps 12 -> 16 for a 32.3s output.   ← 20 秒以上の出力のときだけ出る
info: rf step on ANE + GPU (steps=12, predict=430 ms, gpu_branches=1)
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
リクエストに乗ります。**実測**: 同じ short 入力で 1 回目 6.92 s → 2 回目 2.18 s（40 step 当時の測定。
step 数を変えても初回に乗る固定費は同じ）。
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
サンプラの既定（`--num-steps 12 --t-schedule-mode sway` と auto-step）は Gradio と揃えてあります。upstream と
同じ出力が要るときは `--num-steps 40 --t-schedule-mode linear` を明示してください。

## 7. 参考値（実測、fp16、3 回中央値、13-ane.md 5 節 / 14-step-count.md 3 節）

同じ step 数で MPS と ANE を並べた比較（fp16、3 回中央値、seed 1234、括弧内は RTF）。
long は auto-step で 16 step が適用された値です。

**40 step linear（改善前の基準）**:

| 構成 | short | medium | long | caption_noref |
|---|---:|---:|---:|---:|
| MPS eager | 3459 ms (0.480) | 5872 (0.496) | 16450 (0.570) | 3463 (0.473) |
| MPS + compile | 2860 (0.397) | 4883 (0.412) | 14039 (0.487) | 2863 (0.391) |
| ANE + GPU | 2299 (0.319) | 3931 (0.332) | 11210 (0.389) | 2323 (0.317) |

**sway 12 step（現在の既定）**:

| 構成 | short | medium | long | caption_noref |
|---|---:|---:|---:|---:|
| MPS eager | 1678 ms (0.233) | 2716 (0.229) | 8908 (0.309) | 1689 (0.231) |
| MPS + compile | 1277 (0.177) | 2039 (0.172) | 6984 (0.242) | 1235 (0.169) |
| ANE + GPU | 1247 (0.173) | 2086 (0.176) | 6697 (0.232) | 1241 (0.169) |
| **ANE + GPU + compile（Gradio の既定）** | **1023 (0.142)** | **1713 (0.145)** | **5776 (0.200)** | **1025 (0.140)** |

short での寄与の分離: step 40 → 12 が **2.06×**、そこに ANE が **1.35×**、compile が **1.23×** 乗って
合計 **3.38×**（MPS eager 40 step の 3459 ms → 1023 ms）。

**注意**: 13-ane.md の「ANE で 1.50×」は 40 step 前提の数字です。12 step では sample_rf の比重が
下がるので ANE の効きは 1.25〜1.35× に縮み、MPS + compile (1277 ms) と ANE eager (1247 ms) は
ほぼ並びます。それでも ANE を既定にしているのは、compile が初回 20 s かかりプロセスを跨げないのに対し、
ANE のパッケージは 2 回目以降 0.2 s でロードできるためです。

内訳では短縮分の大半が sample_rf です（short: 2639 → 523 ms、5.0×）。decode は 783 → 463 ms（1.7×）
で、いまや **decode の方が sample_rf より重い**（既定の short で 463 vs 523 ms、ほぼ拮抗）。

品質は聴感で判断しています（ユーザー確認、2026-08-29）。ANE と MPS の差、および 12 step と 40 step の
差はどちらも聴き分けられませんでした。波形距離（LSD）は step 数を変えると原理的に大きく出る
（別のサンプルになるため）ので、step 数の判定には使えません。詳細は 14-step-count.md 3-2。

既知の問題（step 数とは無関係、14-step-count.md 5 節）: **末尾が 1 秒ほど切れることがある**（40 step
linear でも発生）。**30 秒を超える文章は `max_seconds` の既定で途中で切れる**（UI・CLI に露出なし）ので
分割して投げてください。
