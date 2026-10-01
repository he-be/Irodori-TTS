# 19. M6 Mac mini の初回実測（GPU 単独 vs ANE）

状態: **5 節の 0〜4 と 5 の一部を実測**（2026-10-01）。持続負荷・2 ワーカー・`full` の可否は未着手。
凡例: **実測** / **導出** / **未確認**（12 節以降と同じ）。事前の見込みは 18 節。

## 1. 環境

M6 Mac mini（Mac18,5、GPU 12 コア、16 GB、macOS 27.0）、`metal-local` d3887d6 を rsync で配置、
Python 3.12.14 / torch 2.10.0 / coremltools 9.0（uv）。`prebake_runtime.py` のバンドルは 1871 MiB（M1 と同じ）。
`outputs/sample.wav`（参照音声）は gitignore なので別途コピーが要る。測定時 mflux / ComfyUI は停止（`pgrep` で確認）。

## 2. GPU のピーク（実測、`bench/probe_m1.py`）

| | M6 | M3 Pro | M1 |
|---|---:|---:|---:|
| matmul 4096³ fp32 | 4.84 TFLOPS | 5.33 | 1.58 |
| matmul fp16 | **18.66** | 5.63 | 2.35 |
| matmul bf16 | **18.66** | 5.71 | 1.13 |
| コピー帯域 (R+W) | **140.0 GB/s** | （理論 150） | 61.2 |
| 読み出し帯域 | **145.7 GB/s** | | 62.8 |

torch 2.10 の MPS でも tf / mflux の 18.6 がそのまま出る。帯域は M6 での初めての実測で、公称 153 GB/s の 92%。

## 3. GPU 単独（ANE off）の合成時間（実測、fp16、sway 12 step、warmup 1 / repeats 3 / cooldown 5）

wall 中央値 ms（括弧は RTF）。long は auto-step で 16 step。

| 構成 | short (7.20 s) | medium (11.84 s) | long (28.84 s) | caption_noref (7.32 s) |
|---|---:|---:|---:|---:|
| M6 MPS eager | 1113 (0.155) | 1851 (0.156) | 6411 (0.222) | 1114 (0.152) |
| **M6 MPS + compile** | **719 (0.100)** | **1220 (0.103)** | **4571 (0.159)** | **712 (0.097)** |
| M3 Pro MPS + compile（`metal_mps_sway12_compile`） | 1277 | 2039 | 6984 | 1235 |
| M3 Pro 最良（ANE + GPU cond + compile、14 節） | 1023 | | | |
| M1 最良（ANE 全分岐 + compile、16 節） | 2687 | 4335 | 13152 | 2749 |

- eager → compile → eager の順で回し、eager 2 回の差は 1% 以内（`m6_mps_eager_sway12{,_b}.json`）。compile も 2 回目（`_b`）で 1 ms 差。
  この長さの測定ではクロック低下（18 節 3-3）は見えない。
- **M6 は GPU だけで M3 Pro の最良（ANE 併用）より 1.42× 速い**（short 1023 → 719 ms）。

### 3-1. 段の内訳（short、compile）

| 段 | M3 Pro MPS | M6 MPS | 比 |
|---|---:|---:|---:|
| sample_rf（DiT 12 step） | 730 ms | 341 | **2.14×** |
| decode_latent（codec） | 472 | 345 | **1.37×** |

- DiT は 18 節の見込み（2〜3×）の下限側。long は 5107 → 3175 ms で 1.61× にとどまる（形が大きいほど伸びるという予想とは逆。未解析）。
- **decode は 789 GFLOP / 345 ms = 2.3 TFLOPS**（eager 550 ms = 1.4 TFLOPS）。fp16 ピーク 18.7 の 12% で、
  M3 Pro の 1.7 TFLOPS から 1.37× しか伸びていない。conv の大部分は NAX に乗っていないか、Snake 等の帯域律速が支配的（導出、帯域は 2.3× になったのに 1.37× なので後者だけでは説明しきれない）。
  **M6 では decode が wall の半分を占める**ので、次に削る対象はここ。

## 4. ANE（実測）

### 4-1. ステップ単体（`check_ane.py --input short --shapes dev --skip-cpu`）

| | MPS fp16 eager | ANE |
|---|---:|---:|
| B=1, S=180（bucket 192） | 18.0 ms | **14.6 ms** |
| B=3 | — | a_b3 が ANE コンパイル失敗 |

- **B=1 では ANE の方がまだ速い**（18 節の「GPU が 2.5× 速い側に逆転」は B=1 の小さい step では成り立たない）。数値は ANE vs MPS16 rel 3.5e-3（M3 Pro と同水準）。
- **batch 2 / 3 の package（a_b2、a_b3）は M6 の ANE でコンパイルに失敗する**。aned の `Compilation failed`、ANECompilerService は
  `com.apple.appleneuralengine.compiler Code=22`、約 7 s で即失敗（M1 の register spiller で 20〜75 分かけて落ちる失敗とは別物）。
  `ane_dit.ensure_packages` の弁が働き、`*.ane_failed.json` を残して以降は MPS に回る。変換は 1 package 17〜18 s、a_b1 の初回ロード 52 s。

### 4-2. 端から端（compile、`IRODORI_OPT_ANE_SHAPES=dev`）

| ANE_GPU_BRANCHES | short | medium | long | caption_noref | 実際に ANE で走ったか |
|---|---:|---:|---:|---:|---|
| 0（全分岐 ANE、B=3） | 719 | 1220 | 4568 | 712 | **否**（a_b3 失敗 → 全 step MPS） |
| 1（ANE B=2 + GPU 1） | 719 | 1220 | 4574 | 712 | **否**（a_b2 失敗 → 全 step MPS） |
| 2（ANE B=1 + GPU 2） | 776 (+8%) | 1153 (−6%) | **3940 (−14%)** | 772 (+8%) | 是 |
| ANE off（同じ並びで再測定） | 718 | 1221 | 4574 | 712 | — |

- 使える ANE 構成は gpu_branches 2 だけで、**短文は 8% 遅く、長文ほど得**（long sample_rf 3177 → 2536 ms）。
- short の gb2 は sample_rf 自体は 341 → 329 ms と縮んでいるのに wall が 57 ms 伸びる（段の合計に入らない時間。ANE worker との同期か、
  18 節 3-2 の ANE 同居による GPU 減速が decode 以外の所に出たか、未解析）。caption_noref は p95 1130 ms と揺れも大きい。

## 5. まとめと既定

1. **GPU（NAX）経路はコード変更なしで動き、M6 の既定にすべき**: short 719 ms / RTF 0.10、M3 Pro 最良の 1.42×、M1 最良の 3.7×。
2. ANE は「batch 1 の 1 分岐だけ」しか載らず、短文では損、long（28.8 s）で −14%。単発の対話用途の既定は ANE off。
3. 現状の `opt_config.py` では "Apple M6" は M3 Pro 既定（`full` + GPU 分岐 1）を受ける。ANE を有効にした Gradio app では
   b2 / b3 の package が毎回変換→失敗するまで MPS で動くだけで、結果は ANE off と同じになるが、初回に変換時間を払う（未計測、`full` は package 数が多い）。
4. 次の手: (a) decode の内訳（conv / conv_transpose / Snake の時間配分、18 節 4-2-3 の fp32 監査）、(b) long で gb2 が効く理由と short の +57 ms、
   (c) 持続負荷（3 分原稿）でのクロック低下、(d) `full` の b1 系だけを使う shape セット（M1 の `m1` セットと同じ発想）が長文で効くか。

## 6. 測定コマンド（M6 上）

```bash
~/.local/bin/uv run --no-sync python bench/probe_m1.py
HF_HUB_DISABLE_XET=1 ~/.local/bin/uv run --no-sync python prebake_runtime.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small
COMMON="--precision fp16 --inputs short medium long caption_noref --num-steps 12 --t-schedule-mode sway --sway-coeff -1.0 --warmup 1 --repeats 3 --cooldown 5"
bench/bench_runtime.py $COMMON --env IRODORI_OPT_ANE=0 [--env IRODORI_OPT_COMPILE_DIT=1 --env IRODORI_OPT_COMPILE_CODEC=1]
bench/bench_runtime.py $COMMON --env IRODORI_OPT_COMPILE_DIT=1 --env IRODORI_OPT_COMPILE_CODEC=1 \
  --env IRODORI_OPT_ANE=1 --env IRODORI_OPT_ANE_SHAPES=dev --env IRODORI_OPT_ANE_GPU_BRANCHES={0,1,2}
bench/check_ane.py --input short --shapes dev --skip-cpu
```

スクリプトは bash で回す（zsh は `$COMMON` を単語分割しない）。生データ: `results/m6_*.json`, `results/m6_check_ane_dev.txt`。
