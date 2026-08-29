# 14. step 数の削減（sway sampling、既定 12 step + 長さによる自動引き上げ）

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
# 最終状態（既定 12 step、long は auto-step で 16 になる）
uv run python bench/bench_runtime.py --precision fp16 --tag ane_sway12 \
  --env IRODORI_OPT_ANE=1 --env IRODORI_OPT_ANE_GPU_BRANCHES=1 --env IRODORI_OPT_ANE_SHAPES=full \
  --num-steps 12 --t-schedule-mode sway --sway-coeff -1.0 \
  --inputs short medium long caption_noref --warmup 1 --repeats 3 --cooldown 5 \
  --save-wav-dir outputs/steps/sway12 --output docs/experiments/results/metal_ane_sway12.json
# step 数を振る場合は --num-steps / --t-schedule-mode を変える（8 / 12 / 16 / 24 / 40 を測定）
# ANE と step 数を分離するため、同じ step 数で MPS 側も測る（--env を外す / compile だけ付ける）
uv run python bench/bench_runtime.py --precision fp16 --tag mps_sway12 \
  --num-steps 12 --t-schedule-mode sway --sway-coeff -1.0 \
  --inputs short medium long caption_noref --warmup 1 --repeats 3 --cooldown 5 \
  --output docs/experiments/results/metal_mps_sway12.json
uv run python bench/audio_metrics.py outputs/quality/q_fp32_short.wav outputs/steps/sway12/ane_sway12_short.wav
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
| **sway 12 step（採用）** | **1247 (0.173)** | **2086 (0.176)** | **6697 (0.232)**※ | **1241 (0.169)** |
| sway 8 step | 1069 (0.148) | 1778 (0.150) | 4671 (0.162) | 1091 (0.149) |
| linear 8 step | 1053 (0.144) | 1740 (0.147) | — | 1053 (0.144) |

※ long (28.84 s) は auto-step（4-1）で 16 step が適用された値。

12 step は 40 step 比 **1.84×**（short）。12 の MPS eager (3459 ms) から数えると **2.77×**。

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

### 3-3. 聴感 1 回目: 8 step で決めかけた（ユーザー確認、2026-08-29）

Gradio で sway 8 step を試して「速いし聴感で問題ない」。medium で fp32 40 step / sway16 / sway8 /
linear8 を比較し、いったん **8 step を既定に採用**した。

### 3-4. 聴感 2 回目: 長文で破綻が見つかった（ユーザー確認、同日）

既定化した 8 step で 12 パターン（短文・数字・記号・疑問文・カタカナ・英語混在・早口言葉・絵文字・
長文・caption 2 種・超長文）を生成して確認した結果:

- **15 秒以下は 40 step と聴き分けられない**（3-3 の結論と一致）。
- **長文はノイズが増える**。
- 早口言葉で読み間違いが 1 回。

機械的な指標（クリップ数、ポーズ部の雑音床、末尾の減衰、最長無音）を同じ seed の 40 step 版と
突き合わせると、末尾の切れ・クリップ・無音はすべて 40 step 側にも同じだけ出ていて **step 数とは無関係**。
唯一相関したのが高域のエネルギーだった。

### 3-5. 長文の高域ノイズ（実測）

長文（32.3 s、`max_seconds` を外して全文生成）で step 数を振る:

| 構成 | wall (RTF) | **>8 kHz 成分比** | ポーズ部の雑音床 |
|---|---:|---:|---:|
| sway 8 step | 7.77 s (0.241) | **−21.1 dB** | −59.4 dB |
| sway 12 step | 7.38 s (0.229) | −26.4 dB | −60.1 dB |
| sway 16 step | 8.80 s (0.273) | −26.3 dB | −60.1 dB |
| sway 24 step | 11.79 s (0.365) | −25.7 dB | −59.6 dB |
| 40 step linear | 15.97 s (0.495) | −25.3 dB | −60.0 dB |

- **8 step のときだけ 8 kHz 以上が 4〜5 dB 多い**。12 step で解消し、それ以上増やしても動かない。
- 無音区間の雑音床は全構成同じ → 「静かな部分がザーッとする」型ではなく、音声そのものに乗る高域ノイズ。
- 3-4 の 12 サンプルで同じ指標を見ると、+2 dB を超えたのは 30 秒の説明調テキストだけ（+4.5 dB）。
  15 秒以下は全部 ±2 dB 以内。同じ 30 秒でも桃太郎は −0.5 dB なので、**長さだけでなく内容にも依存する**。
- LSD は長文だと 40 step 同士（sway vs linear）でも 9.9 dB 出るので、長文の判定には使えない。

### 3-6. 律速の移動（実測、short）

| stage | 40 step linear | sway 12 step |
|---|---:|---:|
| sample_rf | 1574 ms | **544 ms** |
| decode_latent | 664 ms | **661 ms** |
| predict_duration | 53 ms | 39 ms |

step を削ると **codec decode の方が重くなる**（wall 1247 ms のうち 661 ms）。
ここから先の高速化は DiT ではなく decode 側の話になる。

## 4. 採否

### 4-1. 既定（実装）

- **既定 12 step / sway / `sway_coeff` −1.0**（`gradio_app.py`, `gradio_app_voicedesign.py`, `infer.py`）。
- **長さによる自動引き上げ**: duration 予測の後、出力長が **20 秒以上なら 16 step** に上げる
  (`AUTO_STEP_FLOOR_BY_SECONDS` in `inference_runtime.py`)。**下限としてのみ働く**ので、
  明示的に高い step 数を指定した request は下げない。`IRODORI_OPT_AUTO_STEPS=0` で無効化。
  適用時は Run Log に `info: auto steps 12 -> 16 for a 32.3s output.` が出る。
- ベンチ (`bench_runtime.py`) の既定は 40 / linear のまま（過去の測定と比較可能にするため）。
  `--t-schedule-mode` / `--sway-coeff` をこの実験で追加した。

### 4-2. step 数と ANE の分離（実測、fp16、3 回中央値、seed 1234）

step 数を変えると同時に ANE の効きも変わる（後述）ので、**同じ step 数で MPS と ANE を並べた**。
long は auto-step で 16 step が適用された値。括弧内は RTF。

**40 step linear**（12 / 13 の測定）:

| 構成 | short | medium | long | caption_noref |
|---|---:|---:|---:|---:|
| MPS eager | 3459 ms (0.480) | 5872 (0.496) | 16450 (0.570) | 3463 (0.473) |
| MPS + compile | 2860 (0.397) | 4883 (0.412) | 14039 (0.487) | 2863 (0.391) |
| ANE + GPU cond | 2299 (0.319) | 3931 (0.332) | 11210 (0.389) | 2323 (0.317) |

**sway 12 step（20 秒以上は 16、この実験の測定）**:

| 構成 | short | medium | long | caption_noref |
|---|---:|---:|---:|---:|
| MPS eager | 1678 ms (0.233) | 2716 (0.229) | 8908 (0.309) | 1689 (0.231) |
| MPS + compile | 1277 (0.177) | 2039 (0.172) | 6984 (0.242) | 1235 (0.169) |
| ANE + GPU cond | 1247 (0.173) | 2086 (0.176) | 6697 (0.232) | 1241 (0.169) |
| **ANE + GPU cond + compile（Gradio の既定）** | **1023 (0.142)** | **1713 (0.145)** | **5776 (0.200)** | **1025 (0.140)** |

short での寄与（**導出**）:

| 比較 | 倍率 |
|---|---:|
| step 40 → 12（MPS eager 同士: 3459 → 1678） | **2.06×** |
| ANE 化（40 step eager 同士: 3459 → 2299） | 1.50× |
| **ANE 化（12 step eager 同士: 1678 → 1247）** | **1.35×** |
| compile（12 step MPS: 1678 → 1277） | 1.31× |
| ANE + compile vs MPS + compile（12 step: 1277 → 1023） | 1.25× |
| 全部（MPS eager 40 step → 既定: 3459 → 1023） | **3.38×** |

- **ANE の効きは step 数を減らすと薄くなる**（1.50× → 1.35×）。ANE が短縮するのは sample_rf だけで、
  step を減らすと wall に占める sample_rf の割合そのものが下がるため（**導出**）。
- 12 step では **MPS + compile (1277 ms) と ANE eager (1247 ms) がほぼ並ぶ**。ANE の優位は
  compile と足したときの 1.25× に縮む。ただし compile は初回 20 s のコンパイルが要り、
  プロセスを跨いで再利用できない（12 §5-3）ので、CLI では ANE 側だけが残る。
- 13 の「ANE で 1.50×」は 40 step 前提の数字であり、**現在の既定では 1.25〜1.35× が正しい**。

stage 内訳（short、既定 = ANE + compile）:

| stage | MPS eager 40 step | 既定 | 倍率 |
|---|---:|---:|---:|
| sample_rf | 2639 ms | **523 ms** | 5.0× |
| decode_latent | 783 ms | **463 ms** | 1.7× |
| その他 | 37 ms | 37 ms | 1.0× |

### 4-3. 不採用

- **8 step**: 長文で高域ノイズ（3-5）。15 秒以下では問題ないが、長さで既定を分けるより
  下限を 12 に揃える方が単純なので捨てた。
- linear 8 step（同速で品質が明確に劣る、3-2）。
- 24 step 以上（12 と 16 で指標が飽和している）。

## 5. 未確認 / 次のアクション

1. **末尾が 1 秒ほど切れる**（ユーザー確認）。40 step linear でも起きるので step 数とは無関係。
   `trim_tail`（`tail_window_size=20`, `tail_std_threshold=0.05`, `tail_mean_threshold=0.1`）か
   duration 予測の過小のどちらかだが、**切り分けは未実施**。
2. **早口言葉の読み間違い**が 8 step 固有か seed 依存かは未確認（8 step を 3 seed、16 / 40 step を
   1 seed 生成して比較待ちの状態で 8 step 自体を捨てたため、決着していない）。
3. `max_seconds` の既定 30 秒に当たると文の途中で切れる。CLI にも Gradio にも露出していない。
4. **decode が新しい律速**（short で wall の 53%）。Gradio は `compile_codec` 済み（12 §5-4 で −30%）。
   CLI / 非 compile 経路は未計測。
5. sway_coeff は −1.0 のみ測定。`cfg_min_t=0.5` は 40 step 前提の値で、step 数との組み合わせは未探索。
6. 12 step での候補並列（`num_candidates=2`）の実測は未実施。
7. **12 step では ANE の優位が 1.25× まで縮む**（4-2）。decode が律速なので、ANE 側の伸びしろより
   codec decode の高速化の方が効く可能性が高い。ANE package のビルドコスト（30 分、8〜10 GB）に
   見合うかは、CLI（compile が使えない）か常駐かで判断が分かれる。
