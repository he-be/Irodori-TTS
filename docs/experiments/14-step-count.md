# 14. step 数の削減（sway sampling、8 step 既定化）

作成: 2026-08-29（ブランチ `metal-local`、13 の続き）

**実測** = 手元で数字を取ったもの（条件併記）。**導出** = 実測からの計算。**未確認** = 根拠なし、断定しない。

## 1. 目的 / 仮説

12 と 13 の結論はどちらも「残る大玉は step 数の削減」だった（12 §7 アクション 1、13 §7 アクション 4）。
RF sampler は Euler 積分の 40 step で、**時間は step 数にほぼ線形**（sample_rf = step 数 × forward）。
速度が線形に効くこと自体は自明なので、これまで測っていなかったのは**品質**の方。

`t_schedule_mode=sway`（F5-TTS の Sway Sampling）は runtime に実装済みで、負の `sway_coeff` は
t 列をノイズ側に密にする。少ない step でノイズ側を厚く取れば、同じ step 数でも linear より
40 step の解に近づくはず、というのが仮説。

きっかけはユーザーが Gradio で sway / 8 step を手で試して「速いし聴感で問題ない」と報告したこと。

## 2. 計測方法

```bash
uv run python bench/bench_runtime.py --precision fp16 --tag ane_sway8 \
  --env IRODORI_OPT_ANE=1 --env IRODORI_OPT_ANE_GPU_BRANCHES=1 --env IRODORI_OPT_ANE_SHAPES=full \
  --num-steps 8 --t-schedule-mode sway --sway-coeff -1.0 \
  --inputs short medium long caption_noref --warmup 1 --repeats 3 --cooldown 5 \
  --save-wav-dir outputs/steps/sway8 --output docs/experiments/results/metal_ane_sway8.json
uv run python bench/audio_metrics.py outputs/quality/q_fp32_short.wav outputs/steps/sway8/ane_sway8_short.wav
```

`bench_runtime.py` に `--t-schedule-mode` / `--sway-coeff` を追加した（この実験で新設）。
すべて ANE + GPU cond 分岐、fp16、seed 1234、3 回中央値。品質は **fp32 / 40 step / linear**
（`outputs/quality/q_fp32_*.wav`）を基準にした波形距離。

## 3. 結果（実測）

### 3-1. 速度

| 構成 | short (7.20 s) | medium (11.84 s) | long (28.84 s) | caption_noref (7.32 s) |
|---|---:|---:|---:|---:|
| 40 step linear（13 の既定） | 2299 ms (0.319) | 3931 (0.331) | 11210 (0.393) | 2323 (0.316) |
| sway 16 step | 1401 (0.195) | 2406 (0.203) | — | 1419 (0.194) |
| **sway 8 step** | **1069 (0.148)** | **1778 (0.150)** | **4671 (0.162)** | **1091 (0.149)** |
| linear 8 step | 1053 (0.144) | 1740 (0.147) | — | 1053 (0.144) |

40 step 比で **2.15×**（short）。12 の MPS eager (3459 ms) から数えると **3.2×**。

### 3-2. 品質（fp32 40 step linear 基準、`audio_metrics.py`）

| 構成 | short | medium | caption_noref |
|---|---|---|---|
| 40 step linear（ANE + GPU cond） | SNR 23.8 dB / LSD 0.16 dB | 9.5 / 0.51 | 12.5 / 0.52 |
| sway 16 step | 9.6 / 0.86 | 0.2 / 4.84 | −0.8 / 2.96 |
| **sway 8 step** | 4.5 / **2.63** | −1.2 / **6.73** | −1.1 / **3.57** |
| linear 8 step | 1.4 / 2.87 | −2.3 / **11.35** | −2.2 / 5.33 |

- **同じ 8 step なら sway が明確に良い**: medium で LSD 6.73 vs 11.35 dB、caption で 3.57 vs 5.33 dB。
  sway は t 列の作り方が変わるだけで**速度は同じ**（8 step 同士で wall 差 1〜2%、誤差の範囲）なので、
  step を削るなら sway 一択。
- **ただし LSD の絶対値は判定に使えない**。step を減らした出力は「劣化した 40 step」ではなく
  **別のサンプル**（同じ seed でも積分経路が変わる）なので、波形距離は原理的に大きく出る。
  この指標の校正: bf16 の medium 8.37 dB は実際に別の読みになる（12 §5-6）、ANE のみの medium
  1.10 dB は聴感で区別がつかない（13 §6）。sway8 の 6.73 dB は数値上 bf16 に近いが、
  bf16 のような破綻（積分誤差の発散）ではないので、**判定は聴感でしかできない**。

### 3-3. 聴感（ユーザー確認、2026-08-29）

Gradio で sway 8 step を試し、**問題なし**。medium（差が最大の入力）で fp32 40 step / sway16 /
sway8 / linear8 の 4 本を比較して既定を sway 8 step に決定。

### 3-4. 律速の移動（実測、short、sway 8 step）

| stage | 40 step | sway 8 step |
|---|---:|---:|
| sample_rf | 1574 ms | **378 ms** |
| decode_latent | 664 ms | **653 ms** |
| predict_duration | 53 ms | 37 ms |

8 step にすると **codec decode の方が重くなる**（wall 1069 ms のうち 653 ms）。
ここから先の高速化は DiT ではなく decode 側の話になる。

## 4. 採否

採用（既定を変更）:
- `gradio_app.py` / `gradio_app_voicedesign.py`: Num Steps 40 → **8**、Time Schedule linear → **sway**
  （Sway Coeff −1.0、これまで linear 既定のため grayout していたスライダを有効化）。
- `infer.py`: `--num-steps` 40 → **8**、`--t-schedule-mode` linear → **sway**。
  upstream の既定に戻すには `--num-steps 40 --t-schedule-mode linear`。
- ベンチ (`bench_runtime.py`) の既定は 40 / linear のまま（過去の測定と比較可能にするため）。

不採用 / 据え置き:
- linear 8 step（同速で品質が明確に劣る）。
- sway 16 step（8 step で聴感上の問題が出なかったため。品質を優先するなら UI で上げられる）。

## 5. 未確認 / 次のアクション

1. **decode が新しい律速**（short で wall の 61%）。12 §5-4 では `compile_codec` で −30%（3305 → 1785 ms,
   long）が出ているので、Gradio は既に有効。CLI / 非 compile 経路での改善余地は未計測。
2. sway_coeff は −1.0 のみ測定。他の値、および 4〜6 step の下限は未確認。
3. `cfg_min_t=0.5` は 40 step 前提の値（8 step だと CFG 区間が 4 step）。step 数と CFG 区間の
   組み合わせは未探索。
4. 8 step での候補並列（`num_candidates=2`）の実測は未実施。
