# 13. Apple Neural Engine で RF step を回す（ANE + GPU のヘテロ実行）

作成: 2026-08-29（ブランチ `metal-local`、12 の続き）

**実測** = 手元で数字を取ったもの。**導出** = 実測からの計算。**未確認** = 根拠なし。

## 1. 目的 / 仮説

12 の結論は「MPS 上の DiT は演算律速（190 µs/token、fp16 matmul ピーク 5.6 TFLOPS の約 50%）で、
残る大玉は step 数の削減」だった。この Mac には GPU とは別の演算器 **Neural Engine (ANE, 16 コア)**
があり、PyTorch からは触れないが Core ML 経由なら使える。

仮説:
1. DiT の RF step（`forward_with_encoded_conditions`）は `nn.Linear` + SDPA + elementwise だけなので
   Core ML に変換でき、全 op が ANE に載る。
2. ANE の fp16 実効スループットは GPU の matmul ピークを超え、DiT の step が速くなる。
3. DiT が ANE に移ると **GPU が空く**。空いた GPU を同じ request の中で使う方法として
   - **CFG 分岐のヘテロ分割**: `cfg_guidance_mode=independent` は cond / uncond-text / uncond-speaker の
     3 分岐を batch 3 で 1 forward にしている。この 3 本を ANE 2 本 + GPU 1 本に分けて同一 step 内で並走させる。
   - **候補の並列生成**: `num_candidates=2` の 2 本目を GPU で同時に回し、レイテンシを増やさず候補を増やす。
   - 参考（一般論、本実験では未実装）: 常駐サーバで request N の codec decode（GPU）と request N+1 の DiT（ANE）
     を重ねるパイプライン。

## 2. 前提の実測（合成モデルによるプローブ）

DiT と同じ形のスタック（12 層 × [RMSNorm, 20-head SDPA, RMSNorm, SwiGLU 1280→3680]、248M params）を
coremltools 9.0 で変換（`jit.trace`、fp16、macOS15 target）し、MPS fp16 eager と比較した。

### 2-1. forward 時間（実測）

| 形 (batch × latent) | MPS fp16 eager | ANE (`CPU_AND_NE`) | 倍率 | 場面 |
|---|---:|---:|---:|---|
| 3 × 180 | 67.9 ms (3.96 TFLOPS) | **36.8 ms (7.3 TFLOPS)** | 1.85× | short の CFG あり前半 |
| 2 × 180 | — | 24〜25 ms（導出: KV 入力あり 38.7 ms から入力コピー分を除く） | | ヘテロ分割の ANE 側 |
| 1 × 180 | 25.5 ms | **10.3 ms (8.7 TFLOPS)** | 2.5× | short の CFG なし後半 |
| 3 × 720 | 352 ms | 227 ms（`ALL` だと 182 ms） | 1.55〜1.9× | long |

- compute plan: `linear / sdpa / silu / rsqrt / reduce_mean / mul / add / reshape / transpose` 全部が
  `MLNeuralEngineComputeDevice`。フォールバックなし。
- 出力は MPS fp16 と max |diff| 2e-4。
- Core ML の GPU 実行（`CPU_AND_GPU`）は 58 ms で PyTorch MPS eager より 15% 速い（参考）。

### 2-2. 運用上の制約（実測）

| 項目 | 結果 |
|---|---|
| 形の可変性 | `EnumeratedShapes` で 6 種の (B, S) を 1 モデルに同梱 → 全部 ANE で動き、定常速度は固定形と同じ。形ごとの初回呼び出し +40〜120 ms |
| ロード | 同梱した形はロード時に ANE 向けコンパイル: 6 形で 15〜16 s。**コンパイル済み `.mlmodelc` を再利用すれば 2 プロセス目は 0.0 s**（`compile_model` をやり直すと毎回 5 s 以上） |
| 入力コピー | context KV（B=3, 12 層, 256 key）47 MB を毎 step 入力に渡すと 36.8 → **58.6 ms**。KV は step の中で計算する方が安い |
| GIL | `coremltools` の `predict()` は **GIL を握る**（Python busy 51 ms + ANE 740 ms → 並走 794 ms）。スレッドでは GPU と重ねられない → ANE は **別プロセス** |
| 変換 | coremltools 9.0 は torch 2.10 の `jit.trace` を食う（未テスト版の警告のみ）。`ct.models.MLModel(path)` はこの環境で C++ 例外 → `compile_model` + `CompiledMLModel` |

### 2-3. 実モデルの寸法（実測、`bench_runtime.py` の 4 入力）

| 入力 | x_t (B,S,32) | text (B,Lt,512) | speaker (B,Ls,768) | caption (B,Lc,512) | key 合計 |
|---|---|---|---|---|---|
| short | 3/1 × 180 | 13 | 46 | 1 | 240 |
| medium | 3/1 × 296 | 24 | 46 | 1 | 367 |
| long | 3/1 × 721 | 51 | 46 | 1 | 819 |
| caption_noref | 3/1 × 183 | 13 | 2 | 13 | 211 |

- CFG あり（t ≥ 0.5 の前半 20 step）は batch 3 = [cond, uncond-text, uncond-speaker]（no-ref は uncond-caption）、
  後半 20 step は batch 1。
- context は 60〜100 key と小さいので、KV 投影（12 層 × 6 本）を step 内で計算しても +4% 程度（導出）。
- **残差ストリームの |h| は block 3 以降で数百、block 11 で 2300** に達する。fp16 で `x*x` を取ると
  65504 を超えて inf になるため、Core ML 版の RMSNorm / AdaLN / out_norm は `x/64` に事前スケールして
  から 2 乗する（数学的に同一、fp16 で 4000 まで安全）。MPS 経路の残差ストリーム自体は元々 fp16 なので、
  精度差が出るのは norm の内部だけ。

## 3. 設計と、ANE コンパイラに受け入れさせるまでの試行錯誤

### 3-1. 最終形（`irodori_tts/ane_dit.py`, `ane_worker.py`）

- Core ML モデルの入力: `x_t (B,S,32)`, `t_embed (B,S,512)`（**トークンごとに複製**、後述）,
  `text_state (B,Lt,512)`, `speaker_state (B,Ls,768)`, `caption_state (B,Lc,512)`,
  加算マスク 4 本 `mask_self (B,1,1,S)` / `mask_text (B,1,1,Lt)` / `mask_speaker` / `mask_caption`
  （0 / −1e4、モデル内で cat）, `rope (S,64,2)`（cos/sin をペアごとに複製）。出力 `v (B,S,32)`。
- 中身は `TextToLatentRFDiT` の重みをそのまま使う wrapper `AneStepModule`。context の KV 投影は
  step の中で毎回計算する（KV を入力で渡すと 47 MB/step のコピーで +22 ms、2-2）。
- **package = (batch B, context プロファイル) ごとに 1 つ**、その中で latent 長 S だけを
  `EnumeratedShapes` で列挙する。`dev` は profile a (64/64/16) × B∈{1,2,3} × S∈{192,320,768} の 3 package、
  `full` は profile a, b × B∈{1,2,3} × 23 bucket の 6 package。変換済み `.mlpackage` と
  コンパイル済み `.mlmodelc` は `~/.cache/irodori-tts/ane/<hash>/` に置く（重み・形・wrapper 版で keyed）。
- 実行は子プロセス（`spawn`）。入出力は shared memory、合図は pipe。親は step ごとに
  `submit()`（x_t 35 KB と t_embed をコピーして送信、非ブロック）→ GPU 分岐を enqueue → `wait()`。
  往復オーバーヘッドは **predict との差で 0.1 ms 未満**（実測、`[ane]` ログの predict / wait）。
- サンプラ（`rf.py` `_FastSamplerState._velocity_ane`）: `independent` CFG の分岐 [cond, uncond…] を
  ANE と GPU に分ける。既定は **GPU = cond 1 本、ANE = uncond 2 本**（`IRODORI_OPT_ANE_GPU_BRANCHES=1`,
  `IRODORI_OPT_ANE_GPU_COND=1`）。no-CFG 区間（t < 0.5）は ANE の B=1。列挙形状に収まらない request
  （B>3、context が profile を超える、`num_candidates>1`）は request 単位で MPS 経路に落ちる。

### 3-2. ANE コンパイラが黙って CPU に落とす条件（実測、1 ブロック縮小モデルで二分探索）

Core ML は ANE で実行できない program を **エラーにせず CPU で実行する**（`CPU_AND_NE` 指定時）。
compute plan（`MLComputePlan`）で各 op の preferred device を見ないと気付けない。
12 層モデルの実測では ANE なら 3×192 で 57 ms、CPU に落ちると 700 ms。落ちる条件は以下。

| 条件 | 結果 | 回避策 |
|---|---|---|
| `torch.jit.trace` 経由で SDPA を変換 | coremltools が q/k/v のバッチ次元シンボルを別物と見て `shape→slice→concat→broadcast_to` を挿入 → 全 op CPU | **`torch.export` + `dynamic_shapes`** で変換（`.run_decompositions({})` が必要） |
| 列挙形状の入力と**静的形状の入力**が混在（未使用でも） | 全 op CPU | 全入力を列挙にする |
| 複数入力の列挙リストに**重複**がある | Core ML は index で組み合わせを結ぶがリストを重複除去するため組み合わせがずれ、predict が "unknown exception" / ロード失敗 | B とプロファイルを package ごとに固定、context のパディング長を bucket ごとに +1 して全入力の形を一意にする |
| `(B,1,C)` を S 方向にブロードキャスト（AdaLN の shift/scale/gate、`expand_as` や ones との matmul でも同じ） | 全 op CPU | timestep embedding を `(B,S,512)` に複製して渡し、cond_module と低ランク AdaLN を**トークンごと**に計算（演算 +9%） |
| context の K/V を self の K/V と rank-4 `(B,L,H,hd)` で `cat` | 全 op CPU | rank-3 `(B,L,H*hd)` で cat してから unflatten |
| `attn_mask` 付き SDPA、または rank-5 の RoPE（`unflatten → stack`） | 全 op CPU | attention は `matmul → +mask → softmax → matmul` を手で書く。RoPE は `x·cos + (x@P)·sin`（P はペア入れ替えの定数 64×64） |

### 3-3. ANE の数値精度（実測、CPU fp32 基準の相対誤差）

同じ Core ML モデルを compute unit だけ変えて比較すると、GPU 2.0e-3 / CPU 1.2e-2 / **ANE 1.1e-1**
（B=1、1 forward）。op 単体では linear 3.9e-4、attention logit 3.6e-4、reduce_mean 5.2e-4 と GPU 並みだが、
**silu 1.5e-3 / sigmoid 2.3e-3 / tanh 6.1e-4**（GPU は 3e-4 / 1.7e-4 / 1.6e-4）と LUT 近似の活性化関数だけ
桁が違い、この偏りが 12 ブロックで積み上がっていた。`exp` は 4.5e-4 なので、silu / sigmoid / tanh を
`1/(1+exp(−clamp(x)))` ベースに書き換えた結果:

| ブロック数 | ANE（LUT 活性化） | ANE（exp 活性化） | GPU |
|---|---:|---:|---:|
| 1 | 1.2e-2 | 9.8e-4 | 4.9e-4 |
| 12（1 forward） | 1.2e-1 | **4.7e-3** | 1.6e-3 |

残差ストリーム |h| は最終ブロックで 2300 に達するので、RMS は `x/64` で 2 乗する（`mean(a*a)` を素で書くと inf）。

## 4. 計測方法

```bash
uv run python bench/bench_runtime.py --precision fp16 --tag metal_ane_gpucond \
  --env IRODORI_OPT_ANE=1 --env IRODORI_OPT_ANE_GPU_BRANCHES=1 \
  --inputs short medium long caption_noref --warmup 1 --repeats 3 --cooldown 5 \
  --save-wav-dir outputs/ane_gpucond --output docs/experiments/results/metal_ane_gpucond.json
uv run python bench/audio_metrics.py outputs/quality/q_fp32_short.wav outputs/ane_gpucond/metal_ane_gpucond_short.wav
```

step 単位の検証は `bench/check_ane.py`（実入力を捕捉して MPS fp16 / ANE / CPU fp32 を比較、`--units` で
同じモデルを GPU / CPU で走らせて ANE の精度を切り分ける）。

## 5. 結果（実測、dev 形状、fp16、3 回中央値）

| 構成 | short | medium | long | caption_noref |
|---|---:|---:|---:|---:|
| MPS eager（12） | 3459 ms (0.480) | 5872 | 16450 | 3463 |
| MPS compile（12） | 2860 (0.397) | 4883 | 14039 | 2863 |
| ANE のみ（全分岐 ANE） | 2847 (0.395) | 5063 | 13751 | 2828 |
| **ANE + GPU 1 分岐**（`metal_ane_gpucond.json`） | **2295 (0.319)** | **3921 (0.331)** | 11338 (0.393) | **2311 (0.316)** |
| ANE + GPU 1 分岐、no-CFG も GPU | 2446 (0.340) | 4152 | — | 2448 |
| ANE + GPU 1 分岐 + `compile_dit`（`metal_ane_gpucond_compile.json`） | 2259 (0.314) | 3871 | 11311 | 2282 |
| ANE + GPU 1 分岐、**`full` 形状**（`metal_ane_full.json`、Gradio の既定） | 2299 (0.319) | 3931 | 11210 | 2323 |

- ANE 単独の DiT は short で 40 step 2026 ms（B=3 step 66 ms、B=1 step 21 ms）。MPS eager の
  sample_rf 2639 ms より速いが compile 済み MPS と同等で、**単独では大勝ちではない**。
- GPU に cond 分岐を並走させると CFG 区間の step が max(ANE 2 本 ≈ 44 ms, GPU 1 本 ≈ 37 ms) になり、
  short で eager 比 **1.51×、compile 比 1.25×**。GPU 分岐を compile しても 2259 ms（−1.5%）で、
  CFG 区間は ANE 側が律速。
- 合成プローブ（2-1、3×180 で 36.8 ms）に比べ実 wrapper の B=3 step は 66 ms。差はトークンごとの
  cond（+9%）、context 投影（+4%）、gate linear、exp 活性化、手書き attention の分。

### 5-0. 候補並列（実測、short、`num_candidates=2`、warm 1 回）

| 構成 | wall | sample_rf | decode |
|---|---:|---:|---:|
| MPS のみ、batch 2（`metal_mps_cand2.json`） | 6.40 s | 5055 ms | 1294 ms |
| **候補 0 = ANE（B=3 → B=1）、候補 1 = GPU、同時**（`IRODORI_OPT_ANE_CANDIDATES=1` 既定） | **4.16 s** | 2806 ms | 1305 ms |
| 参考: 1 候補（ANE + GPU cond） | 2.29 s | 1576 ms | 661 ms |

- 2 候補目のコストは +1.9 s（MPS batch 2 だと +4.1 s）。CFG 区間は GPU 側の B=3 forward（≈ 96 ms）が律速で、
  ANE 側（B=3 で 66 ms）は待つ。step 単位のロックステップなので「片方が終わったら次へ」はできない。
- Gradio の候補 2 つ生成が 1.54× 速くなる、というのがこの機能の意味。3 候補以上は MPS 経路に落ちる。

### 5-1. ビルドとロード（実測、`bench/build_ane.py --shapes full`）

- export（`torch.export` + coremltools 変換）: 23 形 × 6 package で **323 s**（1 package 45〜60 s）。
- worker の初回ロード = ANE コンパイル: **1 package 4〜6 分**、6 package で 36 分（合計 2158 s）。
  OS がコンパイル結果をキャッシュするので、**2 プロセス目以降は 1 package 0.2 s**。
- ディスク: `~/.cache/irodori-tts/ane/<hash>/` に `.mlpackage` + `.mlmodelc`（fp16 重み 0.68 GB × 2）×
  6 package = **8.2 GB**。dev 形状は 4.1 GB。
- 常駐メモリは `peak_alloc`（MPS 側）に変化なし。worker プロセス側の RSS は未計測。
- **B=3 を 1536 frame まで列挙した package は壊れる**: ANE コンパイルに 315 s かかった上で CPU 実行になり、
  OS キャッシュにも乗らない（毎プロセス 315 s）。B=1 / B=2 の 23 形は問題ない。B=3 は S ≤ 768
  （12 形）に絞った（`S_MAX_BY_BATCH`）。それを超える request で B=3 が要る構成（ANE のみ、候補並列）は
  MPS に落ちる。原因は未確認（中間テンソル 3×20×1536×2100 の fp16 で 390 MB/層、と推定）。
- キャッシュは package 単位（`<pkg>_<hash>.mlmodelc`）に keyed。形リストを 1 つ変えても他は作り直さない。
- **`.mlmodelc` をリネーム / 移動すると OS の ANE コンパイルキャッシュは効かなくなる**（実測: 移動後の
  `a_b1` ロードが 0.2 s → 244 s に戻った）。キャッシュ dir の整理は再コンパイル前提で行うこと。

## 6. 品質確認（実測、fp32 出力基準、`audio_metrics.py`）

| 構成 | short | medium | caption_noref |
|---|---|---|---|
| fp16 MPS（12、基準の目安） | SNR 24.4 dB / LSD 0.13 dB | 10.9 / 0.23 | 24.9 / 0.17 |
| ANE のみ | 11.7 / 0.18 | 5.5 / **1.10** | 11.3 / 0.38 |
| **ANE + GPU=cond** | **23.8 / 0.16** | 9.5 / 0.51 | 12.5 / 0.52 |
| ANE + GPU=cond + no-CFG も GPU | 23.9 / 0.15 | 9.5 / 0.51 | 12.5 / 0.52 |

- cond 分岐の誤差は CFG 合成で (1 + s_text + s_speaker) ≈ 9 倍になるので、cond を GPU に置くだけで
  short は fp16 と同等になる。uncond 分岐（×3〜5）の ANE 誤差が medium / caption の残差。
- no-CFG 区間（t < 0.5）を GPU にしても指標は動かない → その区間は ANE のままでよい。
- **聴感（ユーザー確認、2026-08-29）**: `ane_dev`（ANE のみ）/ `ane_gpucond` / `ane_gpucond_nocfggpu` の
  3 セット（short / medium / caption_noref）を聴き比べて **全部区別がつかない**。medium の LSD 1.10 dB
  （ANE のみ）ですら聴感の閾値以下で、bf16 の 8.4 dB（別の読みになる）とは質が違う。指標上は
  GPU=cond の方が fp32 に近いので既定はそちらにするが、品質面で ANE のみを避ける理由はない。

## 7. 採否と次のアクション

採用（Gradio 2 アプリの既定。CLI `infer.py` は単発で package ビルド / ロードのコストが見合わないので off のまま）:
- `IRODORI_OPT_ANE=1 IRODORI_OPT_ANE_GPU_BRANCHES=1 IRODORI_OPT_ANE_SHAPES=full`
  （GPU = cond 分岐、ANE = uncond 分岐、no-CFG 区間は ANE、`num_candidates=2` は候補 0 = ANE / 候補 1 = GPU）。
- 効果: short **3459 → 2299 ms（RTF 0.480 → 0.319、eager 比 1.50×、compile 比 1.24×）**、medium 5872 → 3931、
  long 16450 → 11210、caption_noref 3463 → 2323。2 候補は 6.40 → 4.16 s。
- 初回だけ `uv run python bench/build_ane.py --shapes full`（export 5 分 + ANE コンパイル 30 分、8〜10 GB）。
  2 回目以降のロードは package あたり 0.2 s。

不採用 / 据え置き:
- ANE のみ（GPU を使わない）: compile 済み MPS と同速で意味が薄い。GPU と組んで初めて効く。
- no-CFG 区間の GPU 実行: 品質指標が動かず遅くなるだけ。
- GPU 分岐の `compile_dit`: −1.5% で誤差の範囲（Gradio は他の理由で compile on のまま）。
- context KV を入力で渡す設計、jit.trace 経由の変換、B=3 × 1536 frame の列挙（3-2 / 5-1）。

未確認 / 次のアクション:
1. ANE の残り誤差（1 forward で GPU の 3 倍、4.7e-3）: reduce_mean を linear で置き換える案は未計測。聴感では
   差が出ていないので優先度は低い。
2. `full` 形状セットの bucket 粒度（32〜128 刻み）と profile b（長文・長い参照）の実測は未実施。
   ベンチの 4 入力は全部 profile a で、bucket は dev と同じ 192 / 320 / 768 に当たる。
3. worker プロセスの RSS と、6 package を同時にロードしたときのメモリ。
4. step 数削減（sway sampling、`t_schedule_mode=sway` は runtime に実装済み）は ANE と直交する。
   ANE 経路で 8〜10 step にしたときの品質は未確認。
5. サーバ運用でのパイプライン（request N の decode と N+1 の DiT の重ね合わせ）は未実装。
