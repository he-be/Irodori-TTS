# 13. iGPU の DiT 高速化 (1): AdaLN 低ランク射影のバッチ化

## 1. 目的 / 仮説

実験 12 の carve-out 構成（iGPU fp16、sway 12 step、TE=cpu、上限 2560、chunk 96）は RTF が
short 1.06 / medium 0.98 / long 1.03 で、1 を切りきれていない。step を減らすのは聴感で不可なので、
**出力保持型**の変更だけで RTF < 1（12 step）、長文 16 step ≈ 1.0 を狙う。12 の 13.1 節のプロファイルから
候補は次の 4 つ。本ノートはその 1 つ目。

| 候補 | 根拠（12 の 13.1 節、short / long） | 見込み |
|---|---|---|
| **AdaLN 低ランク射影のバッチ化**（本ノート） | `3×1280 · 1280×192` 級の GEMM が 1 728 launch で 0.30 s。launch 律速（1 回 ~170 µs） | −0.3 s（固定、長さ無関係） |
| DiT Linear の融合（wq/wk/wv/gate、w1/w3） | バッチ 1 の 4 step の GEMM が M=161 で 0.16〜0.62 TFLOPS | −0.3 s / −0.2 s（2 節） |
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

## 7. 次

1. Linear 融合（wq/wk/wv/gate → 1 本、w1/w3 → 1 本、M による分岐付き）。dGPU では同じ理由で
   bit 一致が崩れる可能性が高い（N が変わるとカーネルが変わる）ので、同様に ROCm 限定の auto にする。
2. 長文の GEMM 以外 7.5 s の分解（`bench/profile_synth_igpu.py --input long`）。
3. codec decode の GEMM 形状。
