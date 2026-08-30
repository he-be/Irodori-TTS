# 13. iGPU の DiT 高速化: AdaLN 射影のバッチ化と Linear 融合

## 1. 目的 / 仮説

実験 12 の carve-out 構成（iGPU fp16、sway 12 step、TE=cpu、上限 2560、chunk 96）は RTF が
short 1.06 / medium 0.98 / long 1.03 で、1 を切りきれていない。step を減らすのは聴感で不可なので、
**出力保持型**の変更だけで RTF < 1（12 step）、長文 16 step ≈ 1.0 を狙う。12 の 13.1 節のプロファイルから
候補は次の 4 つ。本ノートは 1 つ目と 2 つ目（3 節〜5 節が AdaLN、8 節〜10 節が Linear 融合）。

| 候補 | 根拠（12 の 13.1 節、short / long） | 見込み |
|---|---|---|
| **AdaLN 低ランク射影のバッチ化**（3〜5 節） | `3×1280 · 1280×192` 級の GEMM が 1 728 launch で 0.30 s。launch 律速（1 回 ~170 µs） | −0.3 s（固定、長さ無関係） |
| **DiT Linear の融合**（wq/wk/wv/gate、w1/w3、8〜10 節） | バッチ 1 の 4 step の GEMM が M=161 で 0.16〜0.62 TFLOPS | −0.3 s / −0.2 s（2 節） |
| 長文の GEMM 以外 7.5 s の分解 | softmax fp32 化・cast・elementwise が T に比例 | 長文の本丸 |
| codec decode の GEMM 形状 | im2col + GEMM が 0.57 TFLOPS（microbench 1.1） | 長文 −3 s |

## 2. 優先順位の根拠: Linear 融合の事前 microbench（iGPU fp16、`x @ W` 分割 vs 連結）

「Linear 融合の方が効くのでは」という問いに対し、実装前に GEMM 形状だけで確かめた。M=161 はバッチ 1 の
step、483 はバッチ 3 の短文、2157 は長文。

| M | wq/wk/wv/gate（4×1280 → 5120） | w1/w3（2×3680 → 7360） |
|---|---|---|
| 161 | 3.78 → 2.66 ms（**+30%**） | 2.85 → 3.88 ms（**−36%、遅くなる**） |
| 483 | 5.26 → 5.20 ms（+1%） | 7.77 → 5.82 ms（**+25%**） |
| 2157 | 22.3 → 22.0 ms（+1%） | 32.0 → 31.4 ms（+2%） |

- 効き方は形状依存で rocBLAS のカーネル選択次第。M=161 の w1/w3 は連結すると逆に遅い（N=7360 に悪い
  タイルが選ばれる）。長文は既に 1.27 TFLOPS で融合しても 1〜2%。
- 12 の 13.1 節に当てはめると short ≈ −0.3 s、long ≈ −0.2 s で、AdaLN バッチ化（−0.3 s 固定）と同程度。
  以前の「短文 −0.5 s」は楽観だった。M で分岐すれば損はしない（`nn.Linear` の weight は (N, K) なので
  融合 weight の行スライス `w[:3680]` / `w[3680:]` は連続ビュー、コピーもメモリ増もなし）。
- どちらか一方では短文 RTF 1.06 → 1 未満（−0.4 s 必要）に届かない。**両方やる前提で、確実で小さい
  AdaLN から**着手した（ユーザー判断 2026-08-30）。

## 3. 変更内容: 全 24 AdaLN の射影を step ごとに bmm 2 回で評価

`LowRankAdaLN`（各 DiffusionBlock に attention 用と MLP 用の 2 個 × 12 層 = 24 個）は timestep 条件
`cond_embed` [B, 1, 3D] だけから shift / scale / gate を作る（系列長に無関係）:

```
part = silu(cond_part);  mod = up(down(part)) + cond_part     # down: D→r (bias なし), up: r→D (bias あり)
gate = tanh(gate)
```

これを 24 × 3 part × 2 = 144 launch/step（12 step で 1 728）から、次の 2 launch/step にまとめる:

```
act  = silu(cond.view(B, 3, D)).transpose(0, 1)                       # [3, B, D]
down = bmm(act, W_down.view(3, 24·r, D)ᵀ)                              # [3, B, 24·r]   … 1 launch
up   = baddbmm(b_up, down.view(72, B, r), W_up.view(72, D, r)ᵀ)        # [72, B, D]     … 1 launch
mod  = up.view(3, 24, B, D) + cond ; gate = tanh(mod[2])
```

- `irodori_tts/model.py`: `LowRankAdaLN.compute_modulation()` / `forward(..., modulation=None)`、
  `DiffusionBlock.forward(..., modulation=None)`、`TextToLatentRFDiT.set_adaln_batching()` /
  `_build_adaln_stack()` / `_batched_adaln_modulations()`。`forward_with_encoded_conditions` が
  step ごとに 1 回だけ全層分を計算して各 block に渡す。
- **メモリ増なし**: 積み上げた `[3, 24, r, D]` などの tensor を作った後、元の `nn.Linear` の
  `weight.data` / `bias.data` をそのビューに付け替える。`state_dict()` / `load_state_dict()` はそのまま動く。
- **安全側の fallback**: 6 本の射影がすべて素の `nn.Linear`（peft の LoRA ラッパや量子化 tensor でない）
  のときだけ有効。step ごとにモジュール同一性と `data_ptr` を確認し、`.to()` や LoRA 適用で
  ずれたら再構築、再構築できなければ層別経路に戻る。学習時（`self.training`）は常に層別。
  `torch.compile` 中はチェックを飛ばす（graph break 回避）。
- `IRODORI_OPT_ADALN_BATCH=auto|0|1`（`opt_config.py`、既定 auto = **ROCm のみ有効**、4 節の理由）。
  runtime はロード後 `model.eval()` の直後に `set_adaln_batching()` を呼ぶ。

## 4. 等価性

### 4.1 単体（CPU fp32、小さい構成、乱数重み）

modulation 24 × 3 本、forward 出力とも **max abs diff 0.0**。`.to(float64)` 後の自動再構築、学習モードでの
層別 fallback、`state_dict` の往復も確認。

### 4.2 実モデル dGPU fp32（sway 12、graph off、上限なし、`results/13_dgpu_fp32_adaln{0,1}.json`）

| 入力 | 層別 vs バッチ（音声 wav、int16 正規化） | 同一設定の別プロセス（対照） |
|---|---|---|
| short | **ビット一致**（hash `c335d5b283d6`） | ビット一致 |
| caption_noref | max abs diff **6.8e-3**、diff RMS 2.7e-4、SNR 54 dB、\|diff\|>1e-3 が 1.06% のサンプル | ビット一致 |

DiT 1 step 単位で切り分けると:

| step の種類 | modulation maxdiff | 1 step の出力 maxdiff |
|---|---|---|
| B=3（CFG バッチ、t ≥ 0.5 の 8 step） | **0.0** | **0.0** |
| B=1（t < 0.5 の 4 step） | 1.9e-6（値の大きさ 13 → 相対 1.4e-7 = fp32 eps） | 7〜10e-6 |

- B=1 では cuBLAS が `F.linear`（M=1）を **gemv** 経路で計算し、bmm / 連結 linear とは K 方向の加算順が
  違う（down: K=1280 で 4.8e-7、up: K=192 で 1.9e-6）。B=3 では gemm 同士で完全一致。
  M=1 の gemv と同じ順序を bmm に取らせる手段はないので、**B=1 のビット一致は原理的に不可**。
- short は B=1 の 4 step で 7e-6 ずれても int16 に丸めると一致し、caption_noref は 4 step の
  カオス的増幅で 6.8e-3 まで開いた（02 以来の判定基準 fp32 ≤ 1e-3 を超える）。
- dGPU では CUDA Graph が launch を隠しているので、バッチ化の利得は graph なしでも 0.376 → 0.367 s
  （2%）しかない。「dGPU の既定経路は音声 hash 一致」を守る価値の方が大きいので、**既定 auto は ROCm
  のみ有効**とし、dGPU 既定の hash（short `c335d5b283d6`、caption_noref `f3c9e6fd374a`）は変更前と一致
  することを確認した（`results/13_dgpu_fp32_default` は保存せず、hash のみ照合）。
- iGPU の独立 CFG 経路はもともと `det=False` なので hash 比較の対象外（12 の 11 節）。数学的等価性は
  4.1 と B=3 のビット一致で担保する。

## 5. iGPU の結果（carve-out 構成、fp16、sway 12、`results/13_igpu_adaln_batch.json`）

| 入力 | 12 の基準 wall / RTF / sample_rf | **AdaLN バッチ化** wall / RTF / sample_rf | 差 |
|---|---|---|---|
| short (6.44 s) | 6.80 s / 1.056 / 4.78 s | **6.55 s / 1.017 / 4.53 s** | **−0.25 s** |
| long (28.76 s) | 29.48 s / 1.025 / 20.53 s | **29.20 s / 1.015 / 20.28 s** | **−0.28 s** |

- 見積もりどおり −0.25〜0.3 s（1 728 → 24 launch）。decode と peak alloc は不変（1 560 / 1 745 MiB）。
- 短文はまだ RTF 1.017。残り −0.11 s で 1 を切る。Linear 融合（2 節、−0.3 s 見込み）で届く見込み。

## 6. 採否

| 項目 | 採否 |
|---|---|
| AdaLN 射影のバッチ化（`IRODORI_OPT_ADALN_BATCH=auto`） | **採用**。ROCm では既定 on（−0.25〜0.28 s）、CUDA では既定 off（hash 不変。`=1` で強制可） |

## 7. Linear 融合の M 依存性: rocBLAS のカーネル選択は M に対して不連続

2 節の microbench は `x @ W`（(K, N) 配置）だったが、`nn.Linear` は `F.linear(x, W)`（(N, K) 配置）なので
測り直した（`bench/sweep_linear_fusion.py`、fp16、M = 64〜2400 を 32 刻み + 境界を 8 刻み）。

| M（= B × T） | qkvg 4 本 → 1 本 | w1/w3 2 本 → 1 本 | 現れる入力 |
|---|---|---|---|
| 56〜184 | **+48〜57%** | 96〜192 で **+19〜40%**、95 以下は −14〜19% | short のバッチ 1 step（T=161） |
| 192〜392 | +5〜6% | 200〜359 で **−18〜24%**、360 以上 +1% | medium のバッチ 1（T=272） |
| 392〜576 | +1〜2% | +1〜2% | short のバッチ 3（483） |
| 584〜856 | +1〜3% | **+17〜23%** | long のバッチ 1（719）、medium のバッチ 3（816） |
| 864〜1216 | +18〜23% | **−25%** | |
| 1224〜1631 | **−21〜26%** | **−23〜29%** | |
| 1632〜2111 | +1% | **−25%** | |
| 2112〜 | +1% | +1% | long のバッチ 3（2157） |

- 融合すると N が 1280 → 5120、3680 → 7360 になり、rocBLAS (Tensile, gfx900) が選ぶカーネルが変わる。
  その良し悪しが M の範囲ごとに ±25% で反転し、パディング（M を 64 / 128 の倍数に）では直らない
  （M=1400 → 1408 でも同じ）。分割側も M=161 が M=272 より遅いなど不連続。
- 全域を融合すると w1/w3 は合計で**損**（走査全体 1 327 → 1 497 ms）。範囲ごとに速い方を選ぶ
  「best-of」なら得（1 299 ms）。
- この PC 専用（移植性は放棄済み、torch 2.9.1+rocm6.3 固定）なので、**融合が損になる M の範囲を静的な表**
  にして分岐する: `model.LINEAR_FUSION_SKIP_RANGES = {"qkvg": [1217, 1631], "w1w3": [0, 95], [193, 359], [864, 2111]}`。
  境界には 8 刻みの余白を取り、不明な側は分割（= 従来どおり）に倒している。rocBLAS を更新したら
  `sweep_linear_fusion.py` で表を作り直す。

## 8. 変更内容: wq/wk/wv/gate と w1/w3 の融合（M の範囲表で分岐）

- `model._FusedLinearGroup`: bias なしで同じ入力を取る `nn.Linear` 群の weight を行方向に連結した
  1 本の (ΣN, K) tensor を作り、各 Linear の `weight.data` をその**行スライス（連続ビュー）**に付け替える
  （メモリ増なし、`state_dict` 不変）。`__call__(x)` は M = B × T が skip 範囲なら `None`（呼び側が
  従来の分割経路を使う）、そうでなければ 1 回の `F.linear` の出力を `torch.split` したビューを返す。
  後段の `reshape(B, T, heads, head_dim)` と `silu(h1) * h3` はビューのまま動く（コピーなし）。
- `JointAttention.forward`: `_fused_qkvg` があれば q / k / v / gate を一度に作る（gate は後で
  `sigmoid` に使う。fast 経路・legacy 経路とも）。`SwiGLU.forward`: `_fused_w13`。
- `TextToLatentRFDiT.set_linear_fusion(enabled, skip_ranges=None)`: 全 block 一括（1 つでも素の
  `nn.Linear` でなければ全体を無効化）。呼び出しごとに `data_ptr` で aliasing を確認し、`.to()` や
  LoRA で崩れていれば分割経路に落ちる（`torch.compile` 中は確認を省く）。
- `IRODORI_OPT_LINEAR_FUSE=auto|0|1`（既定 auto = ROCm のみ。理由は 4 節と同じ: N が変わればカーネルが
  変わり、dGPU の hash 一致が崩れる）。runtime はロード後に `set_adaln_batching` の次に呼ぶ。

## 9. 等価性と結果

- 単体（CPU fp32、小構成、skip 範囲なしで全 M 融合）: 出力 **max abs diff 0.0**。skip 範囲の判定、
  `.to()` 後の fallback（出力は分割経路と一致）、`state_dict` の往復も確認。
- dGPU 既定（auto → off）: fp32 の hash が変更前と一致（short `c335d5b283d6`、caption_noref `f3c9e6fd374a`）。

iGPU（carve-out 構成、fp16、sway 12、`results/13_igpu_fused.json`、AdaLN バッチ化と併用）:

| 入力 | 12 の基準 wall / RTF | +AdaLN（5 節） | **+AdaLN +融合** | 基準からの差 | sample_rf |
|---|---|---|---|---|---|
| short (6.44 s) | 6.80 s / 1.056 | 6.55 s / 1.017 | **6.33 s / 0.984** | **−0.47 s** | 4.78 → 4.31 s |
| medium (10.88 s) | 10.68 s / 0.982 | – | **10.22 s / 0.939** | **−0.46 s** | 7.31 → 6.84 s |
| long (28.76 s) | 29.48 s / 1.025 | 29.20 s / 1.015 | **29.05 s / 1.010** | −0.43 s | 20.53 → 20.12 s |
| caption_noref (7.32 s) | 7.52 s / 1.027 | – | **7.00 s / 0.957** | **−0.52 s** | 5.22 → 4.71 s |

- 融合分は short −0.22 s、long −0.15 s、medium ≈ −0.2 s で、7 節の表からの試算（short 0.23 / medium 0.26 /
  long 0.23）どおり。peak alloc は不変（1 558〜1 744 MiB）。
- **short / medium / caption は RTF < 1** になった。long は 1.01 で、16 step なら ≈ 1.3 のまま。

## 10. 採否と終了判断

| 項目 | 採否 |
|---|---|
| AdaLN 射影のバッチ化（`IRODORI_OPT_ADALN_BATCH=auto`） | **採用**（ROCm 既定 on、CUDA 既定 off） |
| Linear 融合 + M 範囲表（`IRODORI_OPT_LINEAR_FUSE=auto`、`LINEAR_FUSION_SKIP_RANGES`） | **採用**（同上）。この PC の rocBLAS 前提の表なので、環境を変えたら `sweep_linear_fusion.py` で再測定 |
| 長文の GEMM 以外 7.5 s の分解、codec decode の GEMM 形状（1 節の候補 3・4） | **着手せず**。ユーザー判断（2026-08-30）: 1 件あたり −0.2〜0.3 s の積み上げでは割に合わないので、やりかけの融合までで打ち切り |

iGPU の最終動作点（12 の 15.3 節の構成に本ノートの 2 つが既定で乗る）: sway 12 step で
**RTF short 0.98 / medium 0.94 / long 1.01 / caption 0.96**、VRAM 実使用 ≈ 2.7〜3.0 GB。
