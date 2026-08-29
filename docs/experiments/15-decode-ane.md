# 15. codec decode の ANE 化プローブと、DiT / decode パイプラインの試算

作成: 2026-08-29（ブランチ `metal-local`、14 の続き）

**実測** = 手元で数字を取ったもの（条件併記）。**導出** = 実測からの計算。**未確認** = 根拠なし、断定しない。

## 1. 目的 / 問い

14 で律速が codec decode に移った（short: wall 1023 ms のうち decode 463 ms、14 §4-2）。そこで出た問いは 2 つ:

1. DiT を ANE、decode を GPU で回しているのだから、**両者を並列**にできないか。
2. そもそも **decode を ANE でやった方が速い**のではないか。

## 2. 前提: 並列にできる形（構造の話、実測不要）

decode の入力は最終 step の latent **全体**。RF sampler は時間軸を逐次生成しないので、前半フレームを先に
decode することもできない。したがって **同一 request の中で DiT と decode を重ねる余地はない**。
重ねられるのは次の 3 形だけ:

| 形 | 効く場面 | 前提 |
|---|---|---|
| request N の decode(GPU) と N+1 の DiT(ANE) | 常駐サーバで連続 request、長文の自動分割バッチ | Gradio の単発利用では効かない |
| decode 自体を chunk で ANE / GPU に振り分け | 単発 request にも効く | **ANE で decode が動くこと**（§4） |
| `num_candidates=2` の decode を ANE / GPU に分ける | 候補並列時のみ | 同上 |

単発 request で速くしたいなら「decode を ANE に載せられるか」が全て。§4 でプローブした。
分割バッチ前提の 1 つ目は §5 で試算した。

## 3. decoder の中身（実測、short 180 frame = 7.2 s）

`bench/decode_flops.py`。watermark 分岐は `alpha=0` で無効（`codec.py` の `_watermark_passthrough`）なので、
decode 経路は **conv / conv_transpose / Snake (x + sin²(αx)/α) / ELU だけ**の純 conv スタック。LSTM は watermark 側にしかない。
hop 1920、48 kHz、decoder 80M params（主経路 70.6M）。

| stage | 形 (C, T) | GFLOP | Snake 層 |
|---|---|---:|---:|
| out_proj + stage 0 | (1024→1536, 180) | 4 | 0 |
| stage 1 (×12) | (768, 2160) | 71 | 7 |
| stage 2 (×10) | (384, 21600) | 178 | 7 |
| stage 3 (×8) | (192, 172800) | **357** | 7 |
| stage 4 (×2) | (96, 345600) | 178 | 7 |
| 合計 | | **789** | 28 |

- 789 GFLOP を 463 ms（既定 = fp16 autocast + `compile_codec`）→ **1.7 TFLOPS**。DiT は同じ GPU で 4 TFLOPS
  出ている（13 §2-1）ので、decode は GPU 効率が半分以下。チャネル 96〜384 の kernel-7 conv と、
  33M 要素に対する Snake（sin / pow の elementwise）28 層が効いている（**導出**）。
- 理論下限は 789 GFLOP / 5.6 TFLOPS = 141 ms。463 ms はその 3.3 倍。

## 4. Core ML 変換プローブ（実測）

```bash
uv run python bench/probe_decode_ane.py 180     # outputs/decode_ane/ に package、初回 ANE compile 約 2 分
```

`codec.decode`（out_proj + conv stack + passthrough）を `torch.export` → coremltools 9.0、fp16 mlprogram、
macOS15 target、固定形 (1, 32, 180)。変換は 9 s で通る。MIL ops: conv 27 / conv_transpose 4 / sin 29 / pow 29 /
mul 58 / add 41 / tanh 1。

### 4-1. compute plan（CPU_AND_NE）

189 op 中 **ANE に載ったのは 81**。CPU に落ちたもの:

| op | CPU 落ち | 内訳 |
|---|---:|---|
| conv_transpose | 4 / 4 | 全部（stride 12 / 10 は長さに関係なく落ちる。stride 8 / 2 は長さが収まれば載る、§4-4） |
| conv | 19 / 27 | 長さ 21600 以上（stage 2〜4）が全部 |
| sin / pow | 21 / 29 ずつ | 同上 |
| mul | 42 / 58 | 同上 |

ANE に載ったのは stage 0〜1（長さ ≤ 2160）だけ。§4-4 の 8 frame 実行では長さ 15360 まで載るので、
境界は 15360〜21600 の間。ANE の既知の軸長上限 16384 と整合する（上限値そのものは **未確認**）。

### 4-2. 速度と数値

| compute units | predict（中央値 5 回） | vs torch fp32 decode |
|---|---:|---|
| CPU_AND_NE | 790 ms | max diff 1.92、**SNR −22 dB**（壊れている） |
| ALL | 429 ms | max diff 0.92、**SNR −0.6 dB**（壊れている） |
| CPU_AND_GPU（Core ML の GPU 実行） | 436 ms | max diff 0.0019、SNR 59.6 dB |
| 参考: MPS fp16 autocast eager | 636 ms | max diff 0.0018、SNR 60.5 dB |
| 参考: MPS fp16 autocast + compile（12 §5-4 / 14 §4-2） | 453〜463 ms | — |

- ANE に載った部分（stage 0〜1）だけで数値が破綻する。CPU_AND_GPU は正しいので、conv_transpose や
  fp16 そのものではなく ANE 実行の問題。Snake の `sin` が ANE では LUT なのが疑わしい（13 §3 の silu と同じ型）が、
  **原因の切り分けは未実施**。
- Core ML の GPU 実行 436 ms は、既定の compile 済み MPS（453〜463 ms）とほぼ同じ。**GPU decode は
  既に Core ML GPU と同水準**で、GPU 側の実装に伸びしろはない。

### 4-3. ANE で decode を速くするのに必要な工事

1. stride 12 / 10（kernel 24 / 20）の conv_transpose を等価な conv + reshape（pixel-shuffle 型）に書き換える
   （stride 8 / 2 は載るので、kernel か stride の上限が 16 前後にあると推測、**未確認**）。
2. 長さ上限。受容野は latent ±10 frame（06 §2.2）= 前後 19200 サンプルで、**overlap だけで上限を超える**。
   つまり chunk decode では原理的に収まらず、時間軸をチャネルに折り畳む（space-to-depth、dilation 1 / 3 / 9 の
   扱いが要る）しか道がない。**これが本質的な壁**。
3. Snake の ANE 数値（sin LUT）。全 op が ANE に載った §4-4 でも SNR −21.6 dB なので、置き場所ではなく
   ANE 実行そのものの問題。

### 4-4. 参考: 全部 ANE に載る長さでの速度（実測、8 frame = 0.32 s）

`bench/probe_decode_ane.py 8`。長さが 15360 サンプルに収まるので 189 op 中 **187 が ANE**（CPU 落ちは stride 12 / 10 の
conv_transpose 2 本だけ）。

| compute units | predict | 音声 1 秒あたり | vs torch fp32 |
|---|---:|---:|---|
| CPU_AND_NE | 15.6 ms | 49 ms/s | **SNR −21.6 dB**（壊れている） |
| ALL | 18.3 ms | 57 ms/s | SNR −20.2 dB |
| CPU_AND_GPU | 22.3 ms | 70 ms/s | SNR 61.6 dB |
| MPS fp16 autocast eager | 31.8 ms | 99 ms/s | SNR 62.1 dB |

- ほぼ全部 ANE で動く形なら Core ML GPU の 1.43×、MPS eager の 2×（180 frame での GPU は 64 ms/s なので、
  短い形は GPU 側に固定費が乗っていて対等な比較ではない）。ANE が conv で速い方向にあることの傍証にはなるが、
  **数値が壊れたままでは使えない**し、この長さは §4-3 の 2 により実用形にならない。

「ANE の方が速い」証拠は今回取れていない（載った部分がなく比較不能）。conv は ANE 向きなので理屈上の期待は
あるが、1〜2 を通してからでないと数字は出ない。**decode の高速化に ANE を使うのは見合わない**。

## 5. 分割バッチ時のパイプライン試算（導出）

長いスクリプトを自動分割してセグメントを連続処理する場合に限り、request N の decode(GPU) と N+1 の DiT(ANE)
を重ねられる。既存の実測から定常スループットを見積もる。

### 5-1. 前提（実測、既定 = ANE + GPU cond + compile、sway 12 step、`metal_ane_sway12_compile.json`）

| セグメント長 | wall | DiT (sample_rf) | decode (GPU) | その他 |
|---|---:|---:|---:|---:|
| short 7.2 s | 1023 ms | 523 | 463 | ~40 |
| medium 11.8 s | 1713 | 931 | 744 | ~40 |
| long 28.8 s（auto-step 16） | 5776 | 3911 | 1810 | ~55 |

パイプライン化すると GPU は decode で塞がるので、DiT は「ANE のみ」（全分岐 ANE）が基本形。
ANE のみは CFG step（sway 12 step では `cfg_min_t=0.5` により **9 / 12 step** が batch 3、16 step では 12 / 16）で
遅くなる。その差は 40 step 実測（`metal_ane_dev` vs `metal_ane_gpucond`、13 §4）から
**CFG 1 step あたり short +25 / medium +54 / long +121 ms**。

→ ANE のみ 12 step の DiT: short **748** / medium **1417** / long（16 step）**5363** ms（導出）。

GPU cond 分岐の GPU 時間: batch 1 forward（13 §2-1: 180 token 25.5 ms）× CFG step 数 → short ~225 / medium ~405 /
long ~1440 ms（導出、medium / long は token 数で外挿）。

### 5-2. 定常状態、セグメント 1 本あたり（導出）

定常スループットは「各デバイスの 1 セグメントあたり負荷の最大値」。GPU 側にはその他 ~40 ms を含める。

| 構成 | short 7.2 s | medium 11.8 s | long 28.8 s |
|---|---:|---:|---:|
| 現状（直列） | 1023 ms / RTF 0.142 | 1713 / 0.145 | 5776 / 0.200 |
| **A: ANE = DiT のみ ‖ GPU = decode** | max(748, 503) = **748 / 0.104（×1.37）** | max(1417, 784) = **1417 / 0.120（×1.21）** | 5363 / 0.186（×1.08） |
| B: ANE = DiT 2 分岐 ‖ GPU = decode + cond 分岐 | max(523, 503+225) = 728 / 0.101（×1.41） | max(931, 784+405) = 1189 / 0.100（×1.44） | 3911 / 0.136（×1.48） |
| **C: A + 空いた GPU がセグメント単位で DiT も担当** | **624 / 0.087（×1.64）** | **1081 / 0.091（×1.58）** | — |
| 理論下限（DiT 時間そのもの） | 523 / 0.073 | 931 / 0.079 | 3911 / 0.136 |

- **A** は素直だが、ANE のみの DiT が遅いぶん伸びが 1.2〜1.4× に留まる。GPU は 1/3 空く。
- **B** は数字上よいが、1 プロセスの MPS はキューが 1 本なので decode と cond 分岐が直列化する。
  step ごとに ANE が cond の結果を待つ構造のため、decode を細かく chunk しないと step が decode の
  後ろに並ぶ（chunk を細かくすると overlap の再計算が増える、12 §5-4）。実現性が低い。
- **C** は A の空いた GPU に、DiT をセグメント単位で回させる（MPS + compile の DiT: short 730 / medium 1254 ms、
  `metal_mps_sway12_compile.json`）。デバイス間の同期が step 単位でなくセグメント単位なので B の問題がない。
  GPU が担当する割合は short で 17 %、medium で 24 % で釣り合う。**現実的な最善**。

### 5-3. 長いスクリプトの例（300 秒、導出）

fill / drain の 1 段分を含む。セグメント境界の cond encoding や参照キャッシュ（L2）は 40 ms 級で無視できる。

| | 7.2 s × 42 本 | 11.8 s × 25 本 |
|---|---:|---:|
| 現状（直列） | 43.0 s | 42.8 s |
| A | 31.9 s | 36.2 s |
| C | **26.7 s** | **27.8 s** |

### 5-4. セグメント長

DiT は音声 1 秒あたり short 73 / medium 79 / long 136 ms と長さで悪化する（attention の二乗 + auto-step 16）一方、
decode は 63 ms/s で線形。短いほど DiT と decode が釣り合う。**7〜12 秒**が有利。

## 6. 採否 / 判断

- 単発 request（Gradio の通常利用）: **手を出さない**。並列化の余地がなく、ANE decode は載らない上に数値が壊れる。
- 長文の自動分割バッチ: **C で 1.6× 前後**が見込める。A だけだと 1.2〜1.4×。実装するなら
  - ANE worker（既に別プロセス）に**サンプリングループごと移す**（Euler 更新は S×32 floats なので CPU で十分）。
  - 主プロセスは GPU で decode と、キューが空いたときの GPU DiT を担当。セグメントのキューとスケジューラは新規。
- 本実験でリポジトリの推論経路は変更していない。追加したのは `bench/probe_decode_ane.py` と `bench/decode_flops.py` のみ。

## 7. 未確認 / 次のアクション

1. §5 の数字はすべて既存実測からの導出。パイプラインの実測は未実施。
2. セグメント分割そのものの聴感（継ぎ目の韻律・話速の揺れ）。速度とは別に分割方式の評価が要る。
3. ANE decode の数値破綻の原因（Snake の sin LUT 仮説）。8 frame の形（§4-4）は数秒で回るので、conv だけの
   部分モデルで切り分ければすぐ決着するが、§4-3 の 1〜2 が残る限り実用には繋がらないので優先度は低い。
4. decode を batch（複数セグメント同時）にしたときの GPU 効率。小チャネル conv なので上がる可能性はあるが未計測。
