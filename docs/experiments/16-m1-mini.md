# 16. M1 Mac mini を TTS サーバーとして使う（性能実測と ANE の適用可否）

作成: 2026-09-21（ブランチ `metal-local`、12〜15 の続き）

**実測** = この M1 mini で数字を取ったもの（条件併記）。**導出** = 実測からの計算。**未確認** = 根拠なし、断定しない。
比較対象の M3 Pro の数値は 12〜15 の既存実測（`docs/experiments/results/metal_*.json`）。

## 1. 目的 / 仮説

- 目的: 手元の M1 Mac mini を **ネットワーク越しの読み上げサーバー**として使えるかを測る。
  ブランチは M3 Pro 向けに書いた `metal-local` をそのまま（コード変更なし）。
- 仮説:
  1. M1 の GPU は M3 Pro の半分以下なので、そのままでは実時間に届かない。
  2. ANE の世代差（公称 M1 11 TOPS / M3 18 TOPS、いずれも 16 コア。**未実測**）は GPU の世代差より小さいので、
     **ANE の相対的な価値は M1 の方が高い**はず。
  3. 常駐サーバーなら、1 リクエストのレイテンシだけでなく**同時実行時のスループット**が効く。

## 2. 実機

| 項目 | M1 mini（今回） | M3 Pro（12〜15） |
|---|---|---|
| チップ | Apple M1（CPU 4P+4E / GPU 8 / ANE 16 コア）| Apple M3 Pro（CPU 12 / GPU 18 / ANE 16）|
| メモリ | 16 GB unified | 18 GB unified |
| OS | macOS 27.0 (26A428) | macOS 15.7.5 |
| PyTorch | 2.10.0 / Python 3.12.14（uv）| 2.10.0 / 3.12.11 |
| `torch.mps.recommended_max_memory()` | 12,124 MiB | 12,288 MiB |
| 電源・冷却 | AC 給電、ファンあり、ヘッドレス | バッテリー駆動可、ファンレス寄り |
| 設置 | LAN 上、ssh のみ | 手元 |

セットアップは `uv sync` と prebake のみ（`prebake_runtime.py`、バンドル 1871 MiB）。
コードの M1 向け変更は**不要**だった（実測: 4 入力すべて完走、MPS fallback なし）。

### 2-1. GPU のピーク（実測、4096³ matmul 10 回平均 / 512 MiB コピー）

| | M1 mini | M3 Pro | 比 |
|---|---:|---:|---:|
| matmul fp32 | 1.58 TFLOPS | 5.33 | 0.30× |
| matmul fp16 | **2.35** | 5.63 | 0.42× |
| matmul bf16 | **1.13** | 5.71 | 0.20× |
| コピー帯域 (R+W) | 61.2 GB/s | （理論 150 GB/s） | |
| 読み出し帯域 | 62.8 GB/s | | |

**M1 では bf16 が fp16 の半分**（M3 Pro は同速）。この世代に bf16 の実装が無いことの現れで、
`--precision bf16` は M1 では選んではいけない（fp16 固定）。帯域は理論 68.25 GB/s の 90%。

## 3. 測定方法

```bash
# GPU ピーク
uv run --no-sync python bench/probe_m1.py
# 合成時間（4 入力、fp16、seed 1234、warmup 後に 3〜5 回中央値、間に cooldown 5 s）
uv run --no-sync python bench/bench_runtime.py --precision fp16 --tag m1_ane_only_sway12_compile \
  --env IRODORI_OPT_ANE=1 --env IRODORI_OPT_ANE_SHAPES=dev --env IRODORI_OPT_ANE_GPU_BRANCHES=0 \
  --env IRODORI_OPT_COMPILE_DIT=1 --env IRODORI_OPT_COMPILE_CODEC=1 \
  --inputs short medium long caption_noref --num-steps 12 --t-schedule-mode sway --sway-coeff -1.0 \
  --warmup 1 --repeats 3 --cooldown 5 --output docs/experiments/results/m1_ane_only_sway12_compile.json
# ANE のステップ単体（配置と数値）/ decode の ANE 適合性
uv run --no-sync python bench/check_ane.py --input short --shapes dev
uv run --no-sync python bench/probe_decode_ane.py 180
# サーバー特性
uv run --no-sync python bench/bench_load.py --repeats 3 --synth
uv run --no-sync python bench/bench_longform.py --tag m1_longform_ane   # 本実験で新設
uv run --no-sync python bench/bench_runtime.py ... --inputs long --repeats 10 --cooldown 0  # 持続
```

測定前に mac mini を再起動している。理由は 7 節（`audiomxd` の spin）。

## 4. 結果

### 4-1. 合成時間（実測、fp16、sway 12 step、括弧内は RTF、long は auto-step で 16 step）

| 構成 | short (7.20 s) | medium (11.84 s) | long (28.84 s) | caption_noref (7.32 s) |
|---|---:|---:|---:|---:|
| 40 step linear（素の状態） | 9626 ms (1.337) | 16364 (1.382) | 42790 (1.484) | 9818 (1.341) |
| MPS eager | 4770 (0.662) | 7529 (0.636) | 25726 (0.892) | 4931 (0.674) |
| MPS + compile | 3403 (0.473) | 5520 (0.466) | 17711 (0.614) | 3450 (0.471) |
| ANE + GPU cond 分岐 | 3804 (0.528) | 6011 (0.508) | 15427 (0.535) | 3868 (0.528) |
| ANE + GPU cond 分岐 + compile | 3101 (0.431) | 4922 (0.416) | **12848 (0.445)** | 3124 (0.427) |
| ANE 全分岐 | 3166 (0.440) | 5132 (0.433) | 15087 (0.523) | 3215 (0.439) |
| **ANE 全分岐 + compile（M1 の最良）** | **2687 (0.373)** | **4335 (0.366)** | 13152 (0.456) | **2749 (0.376)** |
| 参考: M3 Pro の既定 | 1023 (0.142) | 1713 (0.145) | 5776 (0.200) | 1025 (0.140) |

素の状態は **RTF 1.34 = 実時間より遅い**。最良構成で **RTF 0.37**、M3 Pro の既定の 2.6 倍の時間。
素からの短縮は **3.58×**（9626 → 2687）で、内訳は step 削減 2.02× × ANE 1.27× × compile 1.40×（導出）。

### 4-2. 段別の内訳（実測、short、ms）

| 構成 | wall | sample_rf | decode_latent |
|---|---:|---:|---:|
| MPS eager | 4770 | 2693 (56%) | 1923 (40%) |
| MPS + compile | 3403 | 1871 | 1381 |
| ANE 全分岐 | 3166 | **1114** | 1900 |
| ANE 全分岐 + compile | 2687 | 1132 (42%) | 1394 (52%) |

M1 と M3 Pro は**どの段も一様に 2.8 倍**の関係にある（eager 同条件で wall 2.84×、rf 2.80×、decode 2.83×）。
M1 固有の偏りがあるわけではない。ただし ANE を入れると DiT だけが 2.4× 速くなるので、
**M1 でも律速は codec decode に移る**（1394 vs 1132 ms）。これは M3 Pro と同じ構図（463 vs 523 ms、14 節）。

### 4-3. ANE のステップ単体（実測、`check_ane.py --shapes dev`、latent 180）

| 形 | MPS fp16 eager | ANE | 倍率 | M3 Pro の同形 |
|---|---:|---:|---:|---|
| batch 1 × 180 | 116.6 ms | **37.4 ms** | 3.1× | 25.5 → 10.3 ms (2.5×) |
| batch 3 × 180 | 297.0 ms | **102.4 ms** | 2.9× | 67.9 → 36.8 ms (1.85×) |

**ANE の相対的な効きは M1 の方が大きい**（仮説 2 は当たり）。絶対値で見ると
M1 の ANE は M3 Pro の 2.8 倍遅いだけなのに対し、M1 の GPU は M3 Pro の 4.4 倍遅い。

数値差は ANE vs MPS fp16 で rel 3.6e-3（batch 1）/ 1.9e-3（batch 3）。M3 Pro と同程度で、
fp16 同士の差の範囲。**M1 向けに数式を書き直す必要は無かった**（13 節の exp 化・x/64 スケールがそのまま効く）。

### 4-4. GPU への cond 分岐の分割は、M1 では逆効果（実測）

M3 Pro では CFG の 3 分岐を ANE 2 本 + GPU 1 本に割るのが最速だった（13 節）。M1 では逆になる。

- ANE batch 3 = 102.4 ms、GPU batch 1 = 116.6 ms → **分割すると GPU が律速**（導出）。
- 実測でも short 3804（分割）vs 3166（ANE 全分岐）、compile 込みで 3101 vs **2687**。

例外は long（28.84 s、latent 721）で、分割版が 12848 ms と全分岐版 13152 ms をわずかに上回る。
batch 3 × 768 は ANE 側が重く（`S_MAX_BY_BATCH` の上限形）、GPU に 1 本逃がす方が釣り合うため（導出）。

→ **M1 では `IRODORI_OPT_ANE_GPU_BRANCHES=0` が既定であるべき**（M3 Pro 向けの既定は 1）。

### 4-5. codec decode は M1 でも ANE に載らない（実測、`probe_decode_ane.py 180`）

15 節の M3 Pro の結論を持ち込まず、この実機で取り直した。結果は同じだった。

| 実行先 | predict | SNR（torch fp32 比） |
|---|---:|---|
| Core ML `CPU_AND_NE` | 1335 ms | 40.9 dB |
| Core ML `ALL` | 2050 ms | 54.1 dB |
| Core ML `CPU_AND_GPU` | 1199 ms | 59.5 dB |
| PyTorch MPS fp16 autocast eager | 1835 ms | 60.5 dB |
| 参考: 本番経路の compile 済み MPS decode | 1394 ms（4-2 の実測） | — |

- compute plan: ANE 69 op に対し **CPU 落ちが 120 op**（`sin` 29、`mul` 44、`pow` 22、`conv` 19、
  `conv_transpose` 4、`tanh` 1）。Snake 活性化（`sin`/`pow`）と転置畳み込みが丸ごと落ちる。
- `CPU_AND_NE` は速いが **SNR 40.9 dB まで劣化**する（fp16 MPS は 60.5 dB）。採用できない。
- Core ML GPU（1199 ms）は compile 済み MPS（1394 ms）より 1.16× 速いが、**GPU を占有する**ので
  ANE と並走させる構成では意味がない（導出）。

### 4-6. サーバー特性（実測）

**起動**（`bench_load.py --repeats 3 --synth`、prebake 有効、compile なし）

| | プロセス起動 → ロード完了 | 初回 synthesize |
|---|---:|---:|
| MPS | 4.46 s | +8.28 s |
| ANE (dev, 3 package) | 4.46 s | +8.93 s |

ANE パッケージのロードは、OS 側のコンパイル済みキャッシュが効いていれば **1 個あたり 0.2〜1.2 s**
（実測、`check_ane` のプロセスで `a_b1` 1.2 s → `a_b3` 0.2 s）。
`torch.compile` は初回 20 秒級でプロセスを跨げないので、**再起動の多い運用では ANE が有利**（13 節と同じ）。

**メモリ**（`profile_memory.py`、1 プロセス）

| | 常駐 | リクエスト中ピーク | driver |
|---|---:|---:|---:|
| ロード後 | 1873 MiB | — | 2605 MiB |
| short 生成中 | 1873 | 2305 MiB | 3734 |
| long 生成中 | 1873 | 3081 MiB | 6134 |

段別の一時増分は decode が支配的（short +495 MiB、long +1208 MiB）。16 GB なら 1 プロセスは余裕。

**持続負荷**（long を cooldown 0 で 10 連発）

中央値 13078 ms / p95 13182 / 最小 13073 → **ばらつき 0.8%、サーマルの落ち込みなし**。
ファンのある mini は、連続生成でもクロックを維持する（M3 Pro のノートでは cooldown を挟んでいた）。

**長文**（`bench_longform.py`、原稿を 12 セグメントに自動分割、音声 107.1 s）

| 構成 | 総時間 | RTF | 最初の音が出るまで | セグメント中央値 |
|---|---:|---:|---:|---:|
| MPS eager | 73.8 s | 0.689 | 2.93 s | 6247 ms |
| MPS + compile | 52.4 s | 0.489 | 2.08 s | 4459 ms |
| **ANE 全分岐 + compile** | **44.3 s** | **0.414** | **2.05 s** | 3951 ms |

3 分の原稿なら **74 秒前後**（導出、107.1 s で 44.3 s の比例外挿）。M3 Pro は同じ作業が 35 秒前後。
頭から流し始められるまでは 2 秒。

### 4-7. 同時実行 — ここが M1 mini で ANE を使う最大の理由（実測）

short を repeats 5、cooldown 0 で、1 プロセス単独 / 2 プロセス同時で測った。

| 構成 | 単独 | 2 プロセス同時（各プロセスの中央値） | スループット |
|---|---:|---|---:|
| MPS + compile | 3417 ms | 6020 / 6127 ms | **1.12×** |
| **ANE 全分岐 + compile** | 2672 ms | **2649 / 2772 ms** | **1.97×** |

MPS だけの構成では 2 本目がそのまま GPU の取り合いになり、各リクエストが 1.8 倍遅くなる（スループットはほぼ増えない）。
ANE 構成では **片方の DiT（ANE）ともう片方の decode（GPU）が別の演算器で重なる**ため、
各リクエストのレイテンシを**まったく落とさずに**スループットがほぼ 2 倍になる。

15 節が「セグメント単位のヘテロ・パイプライン（導出 1.6×）」として構想していたものが、
**2 ワーカーを並べるだけで実現している**（実測 1.97×）。

メモリは 2 プロセスで PhysMem 15 GB 使用・compressor 3.7 GB・**swap 0**。16 GB での 2 ワーカーは成立するが上限。
3 ワーカーは未測定（**未確認**、swap に入る可能性が高い）。

## 5. ANE パッケージのビルドコスト（実測、M1 での最大の落とし穴）

| shape セット | 変換（coremltools） | OS 側 ANE コンパイル | ディスク | 合計 |
|---|---:|---:|---:|---:|
| `full`（6 package × 17〜23 形状）| 5563 s = 93 分 | 1 package 目で 762 s、**打ち切り** | 8.2 GB | **完走せず** |
| `dev`（3 package × 3 形状）| 2011 s = 34 分 | 84 + 299 + 502 s = 15 分 | 3.8 GB | **48 分** |

`dev` の 48 分のうち 13 分ほどは、打ち切った `full` の `ANECompilerService` が CPU を握ったままだったための
待ちである（下記）。競合が無ければ **35 分前後**（導出）。

- M3 Pro では `full` が約 35 分・8.4 GB（14 節・mac-gradio.md）。M1 では**変換だけで 93 分**かかり、
  そのあとの OS 側 ANE コンパイル（`ANECompilerService`、シングルスレッド 100%）が 1 package 762 秒。
  6 package 分で**数時間**になる見込み（導出）。今回は打ち切って `dev` に切り替えた。
- 個別の変換時間: `a_b1` 67 s / `a_b2` 64 s / `a_b3` **1328 s** / `b_b1` 63 s / `b_b2` **2540 s** / `b_b3` 1423 s。
  batch 3 と広いコンテキスト（profile b）が極端に重い。
- 打ち切った `ANECompilerService` は **root 所有で SIGKILL も通らず**、クライアントを kill しても
  CPU 100% を数時間抱え続ける（実測）。次のビルドはその後ろで待たされる。`sudo pkill -9 ANECompilerService` が必要。

**4-1 の ANE 行は `dev` セット（latent bucket 192 / 320 / 768）での実測**である点に注意。
このセットは bench の 4 入力をちょうど覆うが、任意長の原稿では bucket 間の長さが上の bucket まで
パディングされる（例: latent 400 → 768、約 1.9 倍の無駄、導出）。**本番には `full` が要る**。
`full` のビルドは未完了（**未確認**: 完走後の速度が dev と同じかどうか）。

## 6. 結論と推奨構成

1. **TTS サーバーとしては成立する**。最良構成で RTF 0.37〜0.46（実時間の 2.2〜2.7 倍速）、
   3 分の原稿で 74 秒、最初の音まで 2 秒、常駐 1.9 GB、持続負荷でも落ちない。
2. **素のままでは実時間に届かない**（RTF 1.34）。step 削減と ANE と compile の 3 つが揃って初めて実用域。
3. **M1 では ANE の価値が M3 Pro より高い**。ステップ単体で 3 倍速く、
   何より **2 ワーカーでスループットがほぼ 2 倍**になる（MPS のみだと 1.12 倍）。
4. **M3 Pro 向けの既定を 2 つ変えるべき**:
   - `IRODORI_OPT_ANE_GPU_BRANCHES=0`（M1 では GPU が律速になり逆効果、4-4）
   - bf16 は選ばない（fp16 の半分、2-1）
5. **導入コストは ANE パッケージのビルド**。`full` は M1 で数時間（未完走）。
   ビルド済みキャッシュは `~/.cache/irodori-tts/ane` にあるので、**M3 Pro 側で作って配る**のが筋だが、
   OS 側のコンパイルキャッシュはマシン固有なので初回ロードは各機で走る（**未確認**: mlpackage の可搬性）。

推奨の起動環境変数（M1 mini）:

```sh
IRODORI_OPT_ANE=1 IRODORI_OPT_ANE_SHAPES=full IRODORI_OPT_ANE_GPU_BRANCHES=0 \
IRODORI_OPT_COMPILE_DIT=1 IRODORI_OPT_COMPILE_CODEC=1 \
  uv run --no-sync python gradio_app.py     # 2 ワーカー並べるとスループットは約 2 倍
```

## 7. 測定環境で見つけた問題（本題ではないが実測）

- 再起動前、`audiomxd` が 71% CPU・`configd` が 43% CPU を起動以来ずっと焼いていた（合計 1.1 コア相当）。
  再起動後も `audiomxd` は残った（`headless-slim.sh` が `com.apple.audio.AudioComponentRegistrar.daemon`
  など 3 つの audio 系を disable したままなのが原因と**推測**、**未確認**）。ベンチはこの状態で取っている。
  1 コア分の常時負荷があるので、解消すれば 4 節の数字は**わずかに良くなる可能性がある**（未確認）。
- `hf_xet` 経由のダウンロードがこの回線で停止する（89 KB で無進捗）。`HF_HUB_DISABLE_XET=1` で 9.6 MB/s。

## 8. 残件

- `full` shape セットのビルド完走と、dev との速度比較（4-1 の ANE 行の一般性の確認）。
- 3 ワーカー以上の同時実行（16 GB でどこまで並べられるか）。
- long（latent 721）での ANE / GPU 分岐の釣り合い（4-4 の例外の詰め）。
- `~/.cache/irodori-tts/ane` を機械間でコピーできるか（M3 Pro で焼いて mini に配れるか）。
- 品質の聴感確認。今回は速度のみで、M1 の ANE 出力は**聴いていない**（数値差は M3 Pro と同程度）。
