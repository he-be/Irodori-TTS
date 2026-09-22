# 16. M1 Mac mini を TTS サーバーとして使う（性能実測と ANE の適用可否）

作成: 2026-09-21 / 更新: 2026-09-22（`full` shape セットを一晩焼いた結果、5 節。同日、統合ログで機序を確定し 5-2 を訂正）（ブランチ `metal-local`、12〜15 の続き）

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
  そのあとの OS 側 ANE コンパイルは 1 package あたり 757〜4518 秒（`ANECompilerService`、シングルスレッド 100%）。
- 個別の変換時間: `a_b1` 67 s / `a_b2` 64 s / `a_b3` **1328 s** / `b_b1` 63 s / `b_b2` **2540 s** / `b_b3` 1423 s。
  batch 3 と広いコンテキスト（profile b）が極端に重い。
- 打ち切った `ANECompilerService` は **root 所有で SIGKILL も通らず**、クライアントを kill しても
  CPU 100% を数時間抱え続ける（実測、最長 11 時間 39 分）。次のビルドはその後ろで待たされる。
  `sudo pkill -9 ANECompilerService` が必要。

### 5-1. `full` セットを一晩かけて焼いた結果（実測、2026-09-21 夜 → 09-22 朝）

変換済みの 6 package（8.2 GB）を残したまま `build_ane.py --shapes full` を再投入した。

| 段 | 所要 |
|---|---|
| ビルド完走（OS 側 ANE コンパイルのみ、変換はキャッシュ済み） | 19:38:26 → 23:36:51 = **3 時間 58 分** |
| 内訳（package ごと）| a_b1 757 s / a_b2 **4474 s** / a_b3 2088 s / b_b1 774 s / b_b2 **4102 s** / b_b3 2101 s |

**結果は失敗だった。`full` セットは M1 の ANE に載らず、CPU で走る。**

ステップ単体（`check_ane.py --input short`、latent 180、同じ実機）:

| 形 | MPS fp16 | ANE（`dev`、3 bucket）| ANE（`full`、23 bucket）|
|---|---:|---:|---:|
| batch 1 × 180 | 117.7 ms | **37.4 ms** | **329.8 ms** |
| batch 3 × 180 | 299.0 ms | **102.4 ms** | **955.1 ms** |

batch 3 の 955 ms は、同じ入力の **CPU fp32 リファレンス（756 ms）より遅い**。
13-ane.md 5-1 が M3 Pro で見つけた「batch 3 を 768 超まで列挙すると 315 秒かけてコンパイルしたあと
CPU で走る」現象が、M1 では **`full` セット全体で起きている**。機序は 5-2 の統合ログで確定した
（ANE コンパイルの失敗 → Core ML が黙って CPU 実行）。

エンドツーエンド（sway 12 step、fp16、compile あり、括弧内 RTF）:

| 構成 | short | medium | long | caption_noref |
|---|---:|---:|---:|---:|
| `dev` + ANE 全分岐 | **2687 (0.373)** | **4335 (0.366)** | 13152 (0.456) | **2749 (0.376)** |
| ANE なし（MPS + compile）| 3403 (0.473) | 5520 (0.466) | 17711 (0.614) | 3450 (0.471) |
| `full` + ANE 全分岐 | 10625 (1.476) | 18808 (1.589) | 73742 (2.557) | 10710 (1.463) |
| `full` + GPU cond 分岐 | 8678 (1.205) | 14736 (1.245) | 50174 (1.740) | 8710 (1.190) |

`full` は **ANE を切った素の MPS より 3 倍遅い**（short の sample_rf が 1132 → 9048 ms）。
`full` で GPU 分岐ありの方が速いのは、3 分岐のうち 1 本が遅い ANE 経路を離れて GPU に載るからで、
4-4 の結論とは無関係（導出）。長文通しも 44.3 s → **165.1 s**、最初の音まで 2.05 s → 7.03 s。

### 5-2. `full` は ANE コンパイルに失敗している（実測、統合ログ。当初の「キャッシュ容量」説を訂正）

ビルド完走後のベンチは、**プロセスを起動するたびに ANE コンパイルを払い直した**
（`a_b1` を 5 回、毎回 757〜766 秒。`a_b3` を 4 回、毎回 2017〜2088 秒。`dev` は別プロセスで 0.2〜1.2 秒）。

当初これを「OS 側キャッシュの容量上限で追い出されている」と解釈したが、**誤りだった**。
mini の統合ログ（`/usr/bin/log show --predicate 'process == "ANECompilerService" OR process == "aned"'`、
sudo 不要、保存: `results/m1_ane_compiler_log_20260921.txt`）に、コンパイルの**失敗**が残っている。

```
(ANECompiler) RegAlloc: failed to allocate tensors with fixed L2 hint.        ← 多数、これ単体は致命でない
(ANECompiler) ZinCompilerCore::RunRegisterSpilling(): - Register spiller failure for ANE 0.
aned: Compilation failed ... Domain=com.apple.appleneuralengine.compiler Code=1
aned: Model load failed ... isPreCompiledModel=0
```

| 失敗時刻（09-21 夜のビルド、19:38:26 開始）| 直前からの経過 | 対応する package |
|---|---:|---|
| 19:51:09 | 763 s | a_b1 |
| 21:05:43 | 4474 s | a_b2 |
| 21:40:30 | 2087 s | a_b3 |
| 21:53:24 | 774 s | b_b1 |
| 23:01:46 | 4102 s | b_b2 |
| 23:36:47 | 2101 s | b_b3 |

経過時間が 5-1 の package ごとの所要と一致する（実測）。つまり **6 package すべてが ANE コンパイルに失敗**し、
Core ML はエラーを出さずに CPU で実行していた。以降のベンチでも同じ失敗が 12 回記録されている
（23:49〜翌 05:51）。**失敗したコンパイルはキャッシュされない**ので、プロセスごとに同じ失敗を
757〜4500 秒かけてやり直していた。5-1（CPU 落ち）と 5-2（毎回再コンパイル）は**同じ一つの原因**である。

13-ane.md 5-1 が M3 Pro で記録した「B=3 を 1536 frame まで列挙した package は、315 s かけて
コンパイルしたあと CPU 実行になり、**OS キャッシュにも乗らない（毎プロセス 315 s）**」と外から見える症状は同じだが、
**機序は別**（訂正、09-22、17 節 2-3 で M3 Pro のログを実測）。M3 Pro のエラーは `Invalid TD exists` /
`RunCodeGenObjectGen() failed`（コード生成）で spiller failure は 0 件、M1 は `Register spiller failure`（メモリ割り当て）。
「同じ署名」と書いた前版は誤り。

何が spiller を失敗させるのかは**未確認**。手元のデータが与える制約は次のとおり（attention score
`B×20×S×K` の fp16 サイズ、導出）:

| package | 列挙数 | 最大形 | 最大 attention score | 結果 |
|---|---:|---|---:|---|
| M1 `dev` a_b3 | 3 | 3 × 768 | 80 MiB | ANE で動く |
| M1 `full` a_b3 | 17 | 3 × 768 | 80 MiB（ctx パディング +16 で 84）| **失敗** |
| M1 `full` a_b1 | 23 | 1 × 1536 | 98 MiB | **失敗** |
| M3 Pro `full` a_b2 | 23 | 2 × 1536 | 197 MiB | ANE で動く |
| M3 Pro B=3 × 1536 | 23 | 3 × 1536 | 295 MiB | 失敗（13 節 5-1）|

ANE のアドレス空間は 2 機で桁が違う（実測、`ioreg -c AppleARMIODevice -r -w0 | grep -A 25 dart-ane`）:
M1 は `dart,t8020`・`vm-size` = 0xe0000000 = **3.5 GiB**（32 bit）、M3 Pro は `dart,t8110`・`vm-size` が 48 bit 級。
package 1 個の重みは 0.68 GB なので、M1 では `full` の 6 package（4.1 GB）は重みだけで空間を超える（導出。
package 間で重みが共有されない場合）。コンパイル失敗との因果は**未確認**だが、「列挙した全形状の
作業領域の合計」が効くという仮説と整合する。公開情報（非公式、エラー文言そのものの一致例は無し）は
17 節の計画に整理する。

`full` a_b3 は `dev` a_b3 と**最大形が同じなのに失敗している**。したがって「最大形だけで決まる」では
説明できず、列挙数（または列挙全体の合計）か、bucket ごとの ctx パディング +i が効いている。
最大形の上限も別にあるかもしれない（a_b1）。この 2 軸は分けて測る必要がある（8 節）。

**訂正（09-22、17 節 2-2 で実測）**: 上の「アドレス空間」「列挙数」「ctx パディング」の各仮説は単独では成り立たない。
23 形の列挙も 1×1536 単独も ctx パディングも 1 ブロックなら全部載る。batch 1 で落ちるのは
**192 未満の小さい形と大きい形を同じ package に入れ、かつ 2 層以上**のとき。192 以上に絞れば
**batch 1 は 1536 まで 18 bucket が 12 層で載る**（実測、715 s）。batch 3 は 192 以上 12 形でも 12 層で落ち、
載る列挙数の上限は 6 と 9 の間（17 節 2-2）。

**したがって 4-1 の ANE 行（`dev` セット）が M1 の到達点である**（→ 09-22 時点では batch 1 について更新済み、上記）。
ただし `dev` は bucket が 192 / 320 / 768 の 3 つしかないので、任意長の原稿では上の bucket まで
パディングされる（例: latent 400 → 768、約 1.9 倍の無駄、導出）。bucket 間の長さでは
その分 ANE 側が重くなる。**M1 に載る最大の bucket 数を探すのが次の作業**（8 節）。

## 6. 結論と推奨構成

1. **TTS サーバーとしては成立する**。最良構成で RTF 0.37〜0.46（実時間の 2.2〜2.7 倍速）、
   3 分の原稿で 74 秒、最初の音まで 2 秒、常駐 1.9 GB、持続負荷でも落ちない。
2. **素のままでは実時間に届かない**（RTF 1.34）。step 削減と ANE と compile の 3 つが揃って初めて実用域。
3. **M1 では ANE の価値が M3 Pro より高い**。ステップ単体で 3 倍速く、
   何より **2 ワーカーでスループットがほぼ 2 倍**になる（MPS のみだと 1.12 倍）。
4. **M3 Pro 向けの既定を 2 つ変えるべき**:
   - `IRODORI_OPT_ANE_GPU_BRANCHES=0`（M1 では GPU が律速になり逆効果、4-4）
   - bf16 は選ばない（fp16 の半分、2-1）
5. **現行の `full` shape セットは M1 では使えない**（5-1 / 5-2）。6 package とも ANE コンパイルに失敗して
   CPU に落ち、ANE なしより 3 倍遅くなる。失敗はキャッシュされないので、プロセス起動のたびに
   20〜75 分の失敗コンパイルを払う。M3 Pro 向けの既定がそのまま害になる 3 つ目。
   失敗の要因は 17 節で分離した（09-22）: 「M1 には多 bucket が載らない」は誤りで、192 未満の bucket を外せば
   batch 1 は 1536 まで 18 bucket が載る。batch 2 / 3 は 6 形まで。M1 向けの `m1` セット（17 節 4-4）を使う。
   ロード時にコンパイル失敗を統合ログで検出して
   MPS に戻す安全弁を `ane_worker.py` / `ane_dit.py` に入れた（17 節 3 節）。
6. **導入コストは ANE パッケージのビルド**（`dev` で 35 分前後、`full` は一晩かけて焼いて失敗）。
   ビルド済みキャッシュは `~/.cache/irodori-tts/ane` にあるが、**M3 Pro で焼いたものを配れるかは未確認**。
   OS 側のコンパイルは各機で走る。

推奨の起動環境変数（M1 mini、現時点）:

```sh
IRODORI_OPT_ANE=1 IRODORI_OPT_ANE_SHAPES=dev IRODORI_OPT_ANE_GPU_BRANCHES=0 \
IRODORI_OPT_COMPILE_DIT=1 IRODORI_OPT_COMPILE_CODEC=1 \
  uv run --no-sync python gradio_app.py     # 2 ワーカー並べるとスループットは約 2 倍
```

`dev` は bucket が 3 つしかないので、latent がその間に来る原稿ではパディング分だけ損をする。
bucket を増やせる上限が分かるまでは、これが M1 で確実に ANE に載る唯一のセットである（実測）。

## 7. 測定環境で見つけた問題（本題ではないが実測）

- 再起動前、`audiomxd` が 71% CPU・`configd` が 43% CPU を起動以来ずっと焼いていた（合計 1.1 コア相当）。
  再起動後も `audiomxd` は残った（`headless-slim.sh` が `com.apple.audio.AudioComponentRegistrar.daemon`
  など 3 つの audio 系を disable したままなのが原因と**推測**、**未確認**）。ベンチはこの状態で取っている。
  1 コア分の常時負荷があるので、解消すれば 4 節の数字は**わずかに良くなる可能性がある**（未確認）。
- `hf_xet` 経由のダウンロードがこの回線で停止する（89 KB で無進捗）。`HF_HUB_DISABLE_XET=1` で 9.6 MB/s。

## 8. 残件

- （済、17 節）**ANE コンパイル失敗の要因分離（最優先）**。5-2 の表のとおり、列挙数と最大形の 2 軸が混ざっている。
  単一形状 package で最大形の上限を、最大形を固定した列挙数違いの package で列挙数の上限を、別々に測る。
  判定は速度ではなく統合ログ（`Compilation failed` の有無）で行う。ctx パディング +i、profile b、
  同時ロード数、OS 版（mini は macOS 27.0、M3 Pro は 15.7.5）も別の要因として切り分ける。
- （済、17 節 2-5）変換 93 分の内訳。大半は `ct.convert` が変換後に行う ANE コンパイルで、`skip_model_load=True` で
  捨てられる（既定にした）。`a_b2` 64 s / `a_b3` 1328 s の非単調は未確認のまま。
- `full` を焼いたあと `dev` のロードが遅くなっていないかの再確認。
- 3 ワーカー以上の同時実行（16 GB でどこまで並べられるか）。`full` で試して再コンパイルに埋もれたため未測定。
- long（latent 721）での ANE / GPU 分岐の釣り合い（4-4 の例外の詰め）。
- `~/.cache/irodori-tts/ane` を機械間でコピーできるか。公開情報では ANE のコンパイル結果はチップ世代・OS build・
  `.mlmodelc` のパスに紐づくので、配れるのは変換済み `.mlpackage` まで（機種非依存、**未実測**）。
- 品質の聴感確認。今回は速度のみで、M1 の ANE 出力は**聴いていない**（数値差は M3 Pro と同程度）。
