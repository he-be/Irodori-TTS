# 12. iGPU (Radeon Vega 7 / gfx90c) への TTS オフロード

## 1. 目的 / 仮説

このPCの CPU は Ryzen 5 5500GT で、内蔵 GPU (Radeon Vega 7, gfx90c) が BIOS の UMA 設定で
**4096 MiB の RAM を占有**している。表示に使っているのは約 850 MiB で、残り約 3.2 GB は遊んでいる。
ここに TTS を詰め込み、dGPU (RTX 5060 Ti) を VLM に丸ごと明け渡せるか、そのとき **RTF 1 が出るか**
を確かめる。

前提として、聴感評価で **sway 12 step は linear 40 step と区別がつかない** ことが分かっているので、
本実験の動作点は `--num-steps 12 --t-schedule-mode sway`（sway_coeff -1.0）とする。
これまでの実験 (01-11) は linear 40 step 前提だった。

比較対象は同じ動作点での CPU 実行（iGPU が CPU に勝てなければ意味がない）。

## 2. ハードウェアと FLOP 収支

| 項目 | 値 |
|---|---|
| iGPU | AMD Radeon Vega 7 (Cezanne, `gfx90c`), 7 CU, 最大 ~1.9 GHz。理論 fp32 1.7 TFLOPS / fp16 (packed) 3.4 TFLOPS |
| iGPU メモリ | UMA carve-out 4096 MiB（`mem_info_vram_total`）、表示で 851 MiB 使用中。GTT 13.9 GB |
| メモリ帯域 | DDR4 dual channel、CPU と共有（理論 ~50 GB/s、iGPU 実効はそれ以下） |
| CPU | Zen 3 6C/12T, AVX2（AVX-512 / VNNI / BF16 なし） |
| ドライバ | amdgpu (in-tree, kernel 7.0), `/dev/kfd` と `renderD129` は logind の ACL でユーザーに rw |
| ROCm | 未導入。PyTorch の ROCm wheel が HIP/rocBLAS/MIOpen を同梱するのでシステム ROCm は不要 |

`torch.utils.flop_counter` で数えた 1 リクエスト分の演算量（短文 6.52 s、CFG independent = バッチ 3）:

| 段 | GFLOP | 備考 |
|---|---|---|
| RF 1 step | 175 | DiT 1280 dim × 12 層、バッチ 3、163 latent frame |
| RF 12 step | 2 099 | 音声 1 秒あたり 322 GFLOP |
| codec decode | 715 | 音声 1 秒あたり **110 GFLOP**（DAC-VAE 48 kHz デコーダは軽くない） |
| その他 | 245 | ModernBERT text encoder、speaker encoder、duration、参照 encode |
| 合計 | 3 059 | **音声 1 秒あたり 0.47 TFLOP** |

つまり RTF 1 には **0.47 TFLOPS の実効スループット**が要る。40 step ならこの 2.9 倍
（1.35 TFLOP/s）で、iGPU では望み薄。sway 12 step が前提になる理由がここにある。

## 3. CPU ベースライン（fp32, sway 12 step）

```bash
uv run --no-sync python bench/bench_runtime.py --device cpu --precision fp32 \
  --num-steps 12 --t-schedule-mode sway --inputs short medium long \
  --warmup 1 --repeats 3 --tag 12_cpu_sway12 --no-util \
  --output docs/experiments/results/12_cpu_sway12.json
```

| 入力 | 音声長 | wall median | RTF | sample_rf | decode_latent |
|---|---|---|---|---|---|
| short | 6.52 s | 11.35 s | **1.74** | 6.19 s | 4.98 s |
| medium | 11.00 s | 20.10 s | **1.83** | 10.29 s | 9.62 s |
| long | 28.80 s | 55.40 s | **1.92** | 27.51 s | 27.68 s |

- 実効 0.27 TFLOPS。torch 既定の 6 スレッド（物理コア）。12 スレッド（SMT）は逆に 9% 遅い。
- `IRODORI_OPT_DECODE_CHUNK=0`（chunk なし）にしても decode は速くならない（4.45 → 4.56 s）。
  chunk の重複分ではなく codec デコーダ自体が CPU で音声 1 秒あたり 0.75〜0.96 s かかる。
- **CPU では codec decode が RF と同じ重さ**。dGPU では decode は全体の 15% だったので、
  GPU から下ろすと codec の比重が跳ね上がる。iGPU 化でも codec が律速になり得る。

## 4. バックエンド選定

iGPU で PyTorch のモデルをそのまま動かせる経路は ROCm しかない（Vulkan には desktop 向けの
torch backend がなく、ONNX Runtime にも Linux/AMD で使える EP がない）。

- gfx90c は ROCm の公式対応外。`HSA_OVERRIDE_GFX_VERSION=9.0.0` で gfx900 (Vega 10) に見せるのが
  Ryzen APU の常套手段（ISA は同じ GFX9、MFMA なし）。
- PyTorch の ROCm wheel に同梱された rocBLAS の Tensile ライブラリを調べた結果
  （wheel 末尾の ZIP セントラルディレクトリを Range で取得して確認）:

| wheel | rocBLAS Tensile arch |
|---|---|
| torch 2.10.0+rocm7.1（`pyproject` の `rocm` extra） | gfx908 / 90a / 942 / 950 / 1030 / 110x / 115x / 120x — **gfx900 なし** |
| torch 2.10.0+rocm7.0 | 同上、gfx900 なし |
| torch 2.9.1+rocm6.4 | gfx908 以降のみ、gfx900 なし |
| **torch 2.9.1+rocm6.3** | **gfx900 / gfx906** あり |

  rocm7.1 で実際に `HSA_OVERRIDE_GFX_VERSION=9.0.0` を掛けると、最初の GEMM で
  `rocBLAS error: Cannot read TensileLibrary.dat ... for GPU arch : gfx900` で abort した。
  gfx908 に偽装すると MFMA 命令を含むカーネルが載って GPU が落ちるので不可。
- したがって **torch 2.9.1+rocm6.3** を `.venv-rocm` に入れる（`pyproject` の `torch>=2.10` より
  古いが、推論コードは 2.9 で動く。torchcodec は 0.9.1 の CPU wheel）。
  本体の `.venv`（cu128）と `uv.lock` には触れない。

```bash
UV_PROJECT_ENVIRONMENT=$PWD/.venv-rocm uv sync --frozen --extra rocm --no-dev   # 依存一式 (rocm7.1)
uv pip install --python .venv-rocm --index-url https://download.pytorch.org/whl/rocm6.3 \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  "torch==2.9.1+rocm6.3" "torchaudio==2.9.1+rocm6.3" "torchcodec==0.9.1"     # gfx900 入りに差し替え
HSA_OVERRIDE_GFX_VERSION=9.0.0 .venv-rocm/bin/python bench/probe_igpu.py
```

## 5. 変更内容

- `bench/bench_runtime.py`: `--device` / `--codec-device` / `--threads` / `--t-schedule-mode` /
  `--sway-coeff` を追加。CUDA 固有の呼び出しを device で分岐。ROCm では nvidia-smi の代わりに
  `/sys/class/drm/cardN/device/gpu_busy_percent` と `mem_info_vram_used` を 50 ms でサンプリング。
  JSON に device / schedule を記録。
- `irodori_tts/inference_runtime.py`: `precision="fp16"` を追加（CUDA/ROCm のみ）。GFX9 には bf16 の
  ハードウェア演算がないため、iGPU では fp16 が半精度の選択肢になる。`infer.py` の choices にも追加。
- `bench/probe_igpu.py`: HIP の認識、同梱 Tensile arch、DiT 形状の GEMM / codec 形状の conv1d /
  SDPA の実効 TFLOPS を測る。

## 6. iGPU プローブ（torch 2.9.1+rocm6.3, `HSA_OVERRIDE_GFX_VERSION=9.0.0`）

`bench/probe_igpu.py`:

| 演算 | fp32 | fp16 | bf16 |
|---|---|---|---|
| GEMM 486×1280×3680（DiT MLP、CFG バッチ 3） | 0.63 TFLOPS | **0.76** | 0.65 |
| GEMM 2048³ | 0.86 | **1.05** | 0.72 |
| conv1d 512→512 k7 L4096（codec 中段） | 0.70 | **1.11** | 0.61 |
| SDPA B3 H20 T162 D64 | 2.3 ms | 3.1 ms | – |

- 実効 0.6〜1.1 TFLOPS。必要量 0.47 TFLOP/s に対し理屈の上では RTF 0.5〜0.8 が狙える。
- **fp16 が最速**。GFX9 に bf16 演算器はなく、bf16 は変換込みのエミュレーションで fp32 より遅い。
  dGPU 側の既定 bf16 は iGPU では使わない。
- rocBLAS の Tensile は gfx900 用（gfx90c 専用ではない）。gfx90c は gfx900 と同じ GFX9 ISA
  なので動くが、チューニングは Vega 10 (64 CU) 向け。

## 7. carve-out (UMA 4 GB) は ROCm から見えない — 原因と回避策

| 観測 | 値 |
|---|---|
| `torch.cuda.mem_get_info()` total | 13 963 MiB（= `mem_info_gtt_total`。carve-out の 4096 MiB ではない） |
| 2 GiB を `hipMalloc` した前後 | `mem_info_vram_used` 869 → 873 MiB（不変）、`mem_info_gtt_used` 86 → **2 275 MiB** |
| KFD topology (`/sys/class/kfd/kfd/topology/nodes/1/mem_banks/0`) | 1 バンクのみ、`heap_type 1 (FB_PUBLIC)`、size 13.6 GiB |
| Vulkan (RADV) heaps | DEVICE_LOCAL 11.76 GiB / host 5.88 GiB。こちらも carve-out そのままではない |

つまり既定では **iGPU の演算は通常 RAM (GTT) を使い、遊んでいる 4 GB には一切触れない**。

原因は kernel の仕様。v7.0 の `drivers/gpu/drm/amd/amdgpu/amdgpu_ttm.c`:

```c
	if (adev->flags & AMD_IS_APU) {
		if (adev->gmc.real_vram_size < gtt_size)
			adev->apu_prefer_gtt = true;
	}
```

`apu_prefer_gtt` が立つと `amdgpu_amdkfd_get_local_mem_info()` は
`local_mem_size_public = ttm_tt_pages_limit()`（= GTT）を返し、VRAM を報告しない。
GTT の既定は RAM の約半分（13 963 MiB）で、carve-out 4 096 MiB より大きいので必ず GTT 優先になる。

**回避策（要再起動。15 節で実証済み）**: カーネル引数で GTT を carve-out 未満に絞る。

```
amdgpu.gttsize=4000      # 4000 < 4096 → apu_prefer_gtt = false → KFD が VRAM 4 GB を公開
```

このとき HIP のプールは carve-out（表示分を除き約 3.2 GB）になるので、TTS は fp16 で 3 GB 以内に
収める必要がある。副作用: iGPU のシステムメモリ側バッファ (GTT) がデスクトップ全体で 4 GB に
制限される（通常用途なら十分だが、iGPU で重い GL/Vulkan アプリを動かすなら注意）。
`amdgpu.gttsize` は `modinfo amdgpu` に存在することを確認済み。

ユーザー判断（2026-08-30）: **4 GB 以内で動くなら通常 RAM (GTT) でも可**。carve-out 化は
上記の再起動で後から切り替えられるので、まず GTT 上で性能とメモリを詰める。

## 8. codec decode が遅い原因: MIOpen の dilated conv1d

fp32 で iGPU に載せた最初の結果は RTF 4.19（`sample_rf` 5.3 s、`decode_latent` **21.8 s**）。
`bench/profile_codec_igpu.py` で decode を層別に見ると、fp32 / fp16 / fp16+`cudnn.benchmark` の
いずれも **`naive_conv_ab_nonpacked_fwd_nchw_*` が 94%**（17.5 s 中 16.5 s）。benchmark モードは
探索に 4〜7 分かけた上で同じ素朴カーネルに落ちる（= 他に applicable な solver がない）。

形状ごとの切り分け（fp16, `L=313 000` = 6.5 s 分の 48 kHz、最終段の 64 ch）:

| conv | MIOpen | MIOpen OFF (`torch.backends.cudnn.enabled=False`) |
|---|---|---|
| 64→64 k7 d1 | 68 ms (Tensile GEMM) | **48 ms** |
| 64→64 k7 **d3** | **854 ms (naive)** | **47 ms** |
| 64→64 k7 **d9** | **850 ms (naive)** | **47 ms** |
| 512→512 k7 d3 (L=39k) | **5 318 ms (naive)** | **228 ms** |
| 64→64 k1 | 7 ms | 6.5 ms |

- 入力の連続/非連続は無関係。**dilation > 1 のとき MIOpen (gfx900) は素朴カーネルしか持たない**。
  DAC-VAE デコーダは residual unit ごとに dilation 1/3/9 の conv を持つので、ほぼ全段がこれに当たる。
- MIOpen を切ると torch 自前の im2col + rocBLAS GEMM になり、18〜23 倍速い。dilation なしでも
  MIOpen より速い（MIOpen 側は Im2Col + GEMM の上に余計なコピーが入る）。
- 対策: `IRODORI_OPT_CODEC_CUDNN=auto|0|1`（既定 auto = ROCm では off、CUDA では on）。
  `DACVAECodec.encode_waveform` / `decode_latent` を `torch.backends.cudnn.flags(enabled=False)` で
  包む（`irodori_tts/codec.py`）。dGPU の経路は変わらない（cuDNN のまま）。

## 9. iGPU での TTS 全体（短文、sway 12 step、CUDA Graph off、上限なし）

| 構成 | wall | RTF | sample_rf | decode_latent | 備考 |
|---|---|---|---|---|---|
| fp32 / codec fp32 (MIOpen) | 27.3 s | 4.19 | 5.33 s | 21.8 s | peak alloc 4.0 GB、reserved 3.9 GB |
| fp16 / codec fp16 (MIOpen, benchmark) | 22.8 s | 3.56 | 5.29 s | 17.3 s | reserved 2.0 GB |
| **fp16 / codec fp16 (MIOpen off)** | **7.50 s** | **1.16** | 5.45 s | **1.89 s** | peak alloc 2.17 GB、reserved 2.56 GB |

- decode は 21.8 → 1.89 s（音声 1 秒あたり 0.29 s）。fp16 の重み・活性化でメモリは 4 GB に十分収まる。
- 残りの 73% は DiT（`sample_rf` 5.45 s = 12 step × 454 ms）。`bench/profile_synth_igpu.py` の内訳:
  - fp16 GEMM (rocBLAS Tensile `HHS`) 3.9 s（52%）。2.0 TFLOP を 3.9 s なので **実効 0.5 TFLOPS**。
    選ばれるタイルが 16×16 / 32×16 と小さい（M = 486 トークンの細長い GEMM）。
  - SDPA は gfx900 に flash 実装がなく math 経路（bmm + softmax）で約 0.8 s。
  - `_to_copy` 3 363 回 / `copy_` / `mul` 等の細かい elementwise が約 1 s。
  - **CPU 側の self time 合計 7.36 s ≒ GPU 時間 7.47 s**。GPU は 99% busy だが、Python の
    ディスパッチも張り付いており、HIP Graph で CPU 側を消せるかが次の論点。
- 参考: 同じ sway 12 step を dGPU (bf16, CUDA Graph) で回すと short 0.242 s / medium 0.334 s /
  long 0.760 s（RTF 0.037 / 0.031 / 0.026, `results/12_dgpu_bf16_sway12.json`）。iGPU は約 30 倍遅い。

## 10. iGPU fp16 本計測（sway 12、CFG independent、上限 3840 MB、MIOpen off）

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.0 IRODORI_OPT_CUDA_GRAPH=0 IRODORI_OPT_VRAM_LIMIT_MB=3840 IRODORI_OPT_PREBAKE=0 \
  .venv-rocm/bin/python bench/bench_runtime.py --device cuda --precision fp16 --codec-precision fp16 \
  --num-steps 12 --t-schedule-mode sway --inputs short medium long --warmup 1 --repeats 3 \
  --tag 12_igpu_fp16_nograph --output docs/experiments/results/12_igpu_fp16_nograph.json --save-wav-dir outputs/exp12
```

| 入力 | 音声長 | wall | **RTF** | sample_rf | decode_latent | peak alloc | peak reserved |
|---|---|---|---|---|---|---|---|
| short | 6.44 s | 7.53 s | **1.17** | 5.49 s | 1.88 s | 2 168 MiB | 3 280 MiB |
| medium | 10.88 s | 11.80 s | **1.08** | 8.38 s | 3.26 s | 2 231 MiB | 3 280 MiB |
| long | 28.76 s | 30.84 s | **1.07** | 21.80 s | 8.87 s | 2 353 MiB | 3 280 MiB |

- CPU (RTF 1.74〜1.92) の 1.6〜1.8 倍速い。長いほど RTF が良い（GEMM の M が大きくなり効率が上がる）。
- HIP Graph（`IRODORI_OPT_CUDA_GRAPH=1`）は動作するが short 7.50 s で変わらず（`results/12_igpu_fp16_graph.json`）。
  **GPU 律速**であり CPU 側は隠れている。graph 分のメモリを食うだけなので iGPU では off が正解。
- メモリは 4 GB 目標に対し reserved 3.28 GB（長文の chunk decode 時）。fp16 の重みは 1.7 GB。
- GTT 上で動いているので、この 3.3 GB は**通常 RAM から**取られる（carve-out ではない。7 節）。

## 11. RTF 1 を切るための追加実験（short / long、fp16、sway）

| 変更 | 種別 | short wall / RTF | long wall / RTF | 備考 |
|---|---|---|---|---|
| 基準（10 節: chunk 96/16, independent, 12 step） | – | 7.53 s / 1.17 | 30.84 s / 1.07 | |
| `IRODORI_OPT_DECODE_CHUNK=384` | 出力保持 | 7.27 s / **1.13** | – | decode 1.88 → 1.60 s。chunk 0（一括）も同じ 1.60 s。長文の peak alloc は 2.35 → 3.28 GB に増える |
| CFG `alternating`（バッチ 2） | 品質影響 | 9.42 s / 1.46 | 25.6 s / 0.89 | fast sampler は independent 専用で legacy 経路に落ちるため短文は逆に遅い |
| CFG `joint --cfg-scale 4`（バッチ 2） | 品質影響 | 9.29 s / 1.44 | 25.6 s / 0.89 | 同上 |
| **sway 10 step** | 品質影響 | 6.35 s / **0.985** | 26.2 s / **0.91** | |
| **sway 8 step** | 品質影響 | 5.50 s / **0.85** | 22.9 s / **0.80** | |

- dGPU で「独立 CFG のバッチ 3 を 1 回の forward にまとめる」fast sampler の恩恵が iGPU でも大きく、
  バッチを 2 に減らす CFG モードは legacy 経路のオーバーヘッド（同期・mask 再計算）で相殺される。
  バッチ 2 を活かすには fast sampler を alternating/joint に対応させる必要がある（未着手）。
- step 数はそのまま RF 時間に比例する。**12 → 10 で短文 RTF 0.985、長文 0.91**。
- 独立 CFG の fast 経路は ROCm では `det=False`（同一入力で hash が揺れる）。alternating/joint/legacy は
  det=True なので、fast sampler 内のどこか（SDPA math 経路か rocBLAS の reduction）が非決定的。
  聴感上の問題ではないが、iGPU では hash 比較による回帰検出は使えない。

聴感評価用 wav（`outputs/exp12/`、いずれも seed 1234、同じ参照音声）:

| ファイル | 内容 |
|---|---|
| `12_dgpu_bf16_linear40_{short,medium,long}.wav` | dGPU、従来既定（linear 40 step）— 基準 |
| `12_dgpu_bf16_sway12_{short,medium,long}.wav` | dGPU、sway 12 step |
| `12_igpu_fp16_nograph_{short,medium,long}.wav` | **iGPU fp16、sway 12 step**（本命） |
| `12_igpu_sway10_{short,long}.wav` / `12_igpu_sway8_{short,long}.wav` | iGPU fp16、step を減らした候補 |
| `12_igpu_alt_*.wav` / `12_igpu_joint4_*.wav` | iGPU fp16、CFG バッチ 2（参考。速度メリットなし） |

確認したいのは (1) fp16 + ROCm の経路（MIOpen off の im2col conv、fp16 の DiT）が dGPU bf16 と
比べて劣化していないか、(2) sway 10 / 8 が 12 と区別できるか。読みの分岐は seed 依存なので
データ的な類似度は見ない。

## 12. メモリ: 上限 3840 MB での stress と decode chunk の決定

`bench/stress_vram.py`（fp16、sway 12、graph off、上限 3840 MB。ROCm では `nvidia-smi` の代わりに
amdgpu の `mem_info_gtt_used` を記録）:

| decode chunk | 通常 9 ケース | `worst_ref120`（text 256 + caption 512 + 参照 120 s） | peak reserved |
|---|---|---|---|
| 384 | 8/9 ok（reserved 3 646 MiB） | **OOM**（500 MiB の確保に失敗） | – |
| **192** | ok | ok（peak alloc 3 498 MiB） | **3 720 MiB** |
| 96（dGPU 既定） | ok | ok | 3 680 MiB |

- 速度は chunk 192 と 384 で同じ（short 7.23 s / medium 11.2 s / long 29.9 s、`results/12_igpu_c192.json`）。
  chunk 96 → 192 で decode の重複計算が減り short −0.3 s。長文の peak alloc は 2.60 GB（384 だと 3.28 GB）。
- **iGPU の推奨は `IRODORI_OPT_DECODE_CHUNK=192`**。上限 3840 MB に対し最悪 3 720 MiB で 120 MiB しか
  余白がないが、ユーザー判断で**参照音声は 30 s 上限**（`--max-ref-seconds 30`）とするので、
  実運用の最悪は `worst`（参照 30 s + 最大 text/caption）= reserved 3 644 MiB になる。
- GTT の実使用（表示分 850 MiB 込み）は最大 4.1 GB。TTS 単体では約 3.2〜3.3 GB の通常 RAM を消費する。
- `infer.py` 本体も iGPU で動作確認済み（`outputs/exp12/12_igpu_infer_smoke.wav`。
  フラグは `--model-precision fp16 --codec-precision fp16 --model-device cuda --codec-device cuda`）。

## 13. 聴感評価と採否

ユーザー聴感（2026-08-30）:
- `12_igpu_fp16_nograph_*`（**iGPU fp16 / MIOpen off / sway 12**）: **OK**。fp16 化と ROCm の
  conv 経路による劣化は聴こえない。
- 長文は 12 step より **16 step にしたい**感触。**step を 12 未満に削るのは不可**（sway 10 / 8 は不採用）。
- 参照音声は 30 s 上限で可。

採否:

| 項目 | 採否 |
|---|---|
| iGPU への TTS オフロード（ROCm 6.3 + gfx900 偽装、fp16、MIOpen off、chunk 192、graph off、上限 3840） | **採用（動作点として成立）**。sway 12 で RTF **1.02〜1.12**、4 GB 以内 |
| step 削減で RTF < 1 | 不採用（聴感） |
| CFG バッチ 2（alternating / joint） | 保留。現状は legacy 経路で遅い。fast sampler 対応と聴感（`12_igpu_alt_*.wav`）が前提 |
| carve-out (UMA) の利用 | **採用**（15 節）。`amdgpu.gttsize=4000` で KFD のプールが VRAM 4 GiB に切り替わり、TE=cpu + 上限 2560 で bench / stress / churn すべて OOM なし。GTT 時より RTF 8〜10% 良い |
| `IRODORI_OPT_CODEC_CUDNN`（auto: ROCm で MIOpen off） | 採用。dGPU の経路は不変 |
| `bench_runtime.py` の `--device` / fp16 / sway / `--cfg-scale` / `--cudnn-benchmark`、`stress_vram.py` の fp16 / sway / ROCm 対応 | 採用 |

結論: **「iGPU で RTF 1」は 12 step で 1.02〜1.12 とほぼ達成だが、1 を切ってはいない。**
長文 16 step なら RTF ≈ 1.3 になる。step 以外で詰める余地（次の実験 13）:

1. **DiT の Linear 融合**（出力保持、float 誤差レベル）: `wq/wk/wv/gate` を 1 本（1280→5120）、
   MLP の `w1/w3` を 1 本（1280→7360）、AdaLN の `shift/scale/gate_down` を 1 本に。iGPU では
   M=486 の細長い GEMM が 16×16 タイルに落ちて実効 0.5 TFLOPS（microbench 0.76）なので、
   N を太くしてタイル効率と launch 数を稼ぐ。dGPU でも効くはず。
2. **codec decode の GEMM 形状**: im2col + GEMM が 0.57 TFLOPS（conv microbench 1.1）。長文では decode が
   25%（7.3 s）を占めるので、ここを半分にできれば長文 RTF −0.12。
3. fast sampler の alternating 対応（品質影響、聴感待ち）: DiT −33%。
4. SDPA math 経路と fp32↔fp16 cast（約 1.5 s/短文）の削減。gfx900 には Triton も flash もないので
   手書き融合は難しく、cast の発生源（RMSNorm / RoPE の fp32 計算）を fp16 に寄せられるかは品質次第。

見積もり: 1 + 2 で短文 RTF 1.12 → 0.9 台、長文 12 step 1.04 → 0.85 前後、長文 16 step ≈ 1.1。

### 13.1 DiT の GEMM 形状別プロファイル（fp16、sway 12、`record_shapes`）

| 形状 (M×K · K×N) | 回数 | short: 時間 / TFLOPS | long: 時間 / TFLOPS | 正体 |
|---|---|---|---|---|
| 483(2157)×1280 · 1280×3680 | 192 | 0.88 s / **0.99** | 3.67 s / **1.06** | MLP w1/w3、CFG バッチ 3 の 8 step |
| 483(2157)×1280 · 1280×1280 | 480 | 0.77 s / 0.99 | 3.21 s / 1.06 | attention の wq/wk/wv/wo/gate |
| 483(2157)×3680 · 3680×1280 | 96 | 0.71 s / 0.62 | 1.91 s / 1.02 | MLP w2 |
| 161(719)×1280 · 1280×3680 | 96 | 0.23 s / 0.62 | 0.65 s / 1.00 | **バッチ 1 の 4 step**（t < cfg_min_t=0.5 は CFG なし） |
| 161(719)×1280 · 1280×1280 | 240 | 0.44 s / **0.29** | 0.57 s / 0.99 | 同上 |
| 161(719)×3680 · 3680×1280 | 48 | 0.46 s / **0.16** | 0.65 s / 0.50 | 同上 |
| bmm 60×T×(T+60) · 60×(T+60)×64 | 96 | 0.17 s / 0.16 | **1.80 s / 0.24** | attention P·V（math 経路、N=64 が細い） |
| bmm 60×T×64 · 60×64×(T+60) | 96 | 0.13 s / 0.21 | 0.72 s / 0.60 | attention Q·Kᵀ |
| 3×1280 · 1280×192 ほか AdaLN 低ランク | 1 728 | **0.30 s** / ≈0 | 0.30 s | launch 律速（1 回 ~170 µs） |
| mm/bmm 合計 | | 4.37 s（sample_rf 5.47 s） | 14.3 s（sample_rf 21.8 s） | |

- 短文: GEMM 以外は 1.1 s。**バッチ 1 の 4 step（1.1 s）がバッチ 3 の 8 step（2.4 s）と 1 step あたり
  ほぼ同じ時間**。M=161 では 7 CU を埋められず、GEMM が占有率律速になる。
- 長文: GEMM 以外が **7.5 s** もある（softmax・fp16↔fp32 cast・elementwise が T に比例）。attention の
  bmm も 2.5 s。長文の RTF を下げるにはここが本丸。
- AdaLN の低ランク射影（timestep 条件のみに依存、系列長に無関係）は step ごとに全層分を 1 回の
  GEMM にまとめられる → 1 728 launch → 24 launch、−0.3 s（出力保持）。

次の実験 13 の優先順位（すべて出力保持型、聴感不要のはず）:
1. AdaLN 低ランク射影の全層バッチ化（−0.3 s 固定）
2. `wq/wk/wv/gate` と `w1/w3` の融合（特にバッチ 1 の step で効く。短文 −0.5 s 見込み）
3. 長文の GEMM 以外 7.5 s の分解（`profile_synth_igpu.py --input long`）— softmax の fp32 化や cast が
   疑わしい。attention を head 分割の手書き fp16 経路にして math 経路の中間バッファと cast を減らす
4. codec decode の GEMM 形状（0.57 TFLOPS → 1.1 目標、長文 −3 s）

これらで長文 12 step が RTF 0.8 台、16 step でも 1.0 前後に収まる見込み。

### 13.2 carve-out に収まるかの事前検証（上限 3072 MB を GTT 上で再現）

BIOS は非公式版で UMA を変更できないため、carve-out を使うなら **4096 − 表示 ≈ 850 MiB（表示は iGPU
固定、dGPU は表示に使わない）= 約 3.2 GB** に収める必要がある。再起動せずに同じ制約を再現するため
`IRODORI_OPT_VRAM_LIMIT_MB=3072`、decode chunk 96 で計測（`results/12_igpu_cap3072.json`,
`12_igpu_stress_cap3072.json`）:

| ケース | 結果 | peak alloc | reserved | GTT 実使用（allocator 外込み） |
|---|---|---|---|---|
| short / medium / long | ok、速度は上限なしと同じ（RTF 1.16 / 1.09 / 1.07） | 2.17 / 2.23 / 2.35 GB | 2.79 GB | – |
| text_max / caption_max / caption_max_noref / ref15 / ref30 | ok | 2.31〜2.87 GB | 3.01〜3.07 GB | **3.38〜3.44 GB** |
| worst（参照 30 s + text 256 + caption 512） | **OOM**（370 MiB の確保に失敗） | – | 3.07 GB | – |

- torch の allocator の外に **約 370 MiB**（HIP ランタイム、rocBLAS / im2col の作業領域）が乗る。
  つまり上限 3072 でも実使用は 3.4 GB で、carve-out の空き 3.2 GB には**入らない**。
- HIP には GTT への自動はみ出しがない（KFD は VRAM の確保上限で ENOMEM を返す）ので、溢れたら
  即 OOM（と当時は考えたが、VRAM プールに切り替えた後は TTM の eviction で GTT に退避する。15.1 節）。allocator 上限は **2.6 GB 程度**が必要で、現状の peak alloc（代表入力 2.35 GB、宣言上限
  2.87 GB）より 0.3〜0.5 GB 削らないと成立しない。
- 削り代（実験 13 のメモリ側）: ModernBERT text encoder（310M、fp16 0.6 GB）を CPU に置く
  （1 リクエスト 1 回、数十トークンなので CPU でも 100〜200 ms）、decode chunk 64 / encode chunk
  縮小で transient を削る、参照 30 s 上限。これで 2.6 GB 上限が見えたら `amdgpu.gttsize=4000` で
  再起動して pool の切り替わりを確認する（失敗しても GTT 4 GB は表示に十分で、引数を外せば戻る）。

### 13.3 ModernBERT を CPU に置く（`IRODORI_OPT_TE_DEVICE=cpu`）

BIOS（非公式版）で UMA を変えられないので、carve-out に入れるには TTS 側を約 3.2 GB
（allocator 上限 ≈ 2.6 GB）に収める必要がある。text/caption 共用の ModernBERT backbone
（310M、fp16 で 0.6 GB）を CPU fp32 に残し、射影後の state だけを DiT の device に送る。

変更: `opt_config.text_encoder_device`（`IRODORI_OPT_TE_DEVICE=model|cpu`、既定 model）、
`PretrainedTextBackbone.forward` が backbone の device で実行して結果を戻す、runtime の
`_dit_dtype()` と `TextToLatentRFDiT.device/dtype` が backbone を飛ばして DiT の dtype/device を返す
（`next(parameters())` が先頭登録の backbone を拾って fp32/CPU 扱いになる罠が 3 箇所あった）。
dGPU の既定経路は音声 hash 一致で不変。

| | TE=model（10 節） | **TE=cpu** |
|---|---|---|
| ロード後 alloc | 1 702 MiB | **1 092 MiB** |
| peak alloc short / medium / long | 2 168 / 2 231 / 2 353 MiB | **1 558 / 1 621 / 1 743 MiB** |
| RTF short / medium / long / caption | 1.16 / 1.09 / 1.07 / 1.08 | 1.17 / 1.09 / 1.07 / 1.13 |
| predict_duration（text encode 込み） | 157 / 152 / 172 / 183 ms | 165 / 178 / 208 / 241 ms |

stress（参照 ≤ 30 s の 6 ケース、chunk 96、`results/12_igpu_stress_tecpu_cap*.json`）:

| allocator 上限 | 結果 | worst の peak alloc / reserved | HIP 実使用（GTT） |
|---|---|---|---|
| **2560 MB** | **6/6 ok** | 2 390 / 2 544 MiB | **2 952 MiB** |
| 2304 MB | 4/6（caption_max, worst が OOM） | – | – |

- CPU 側の ModernBERT は数十トークンなので +10〜80 ms。RTF への影響なし。
- **carve-out 運用の候補構成**: `IRODORI_OPT_TE_DEVICE=cpu IRODORI_OPT_VRAM_LIMIT_MB=2560
  IRODORI_OPT_DECODE_CHUNK=96` で HIP 実使用 ≈ 2.95 GB。carve-out の空き ≈ 3.2 GB（表示 850〜870 MiB）
  に対し余白 ≈ 250 MiB。代表入力なら peak alloc 1.6〜1.7 GB でさらに余裕がある。
- 残る未検証事項は **`amdgpu.gttsize=4000` で再起動したときに KFD のプールが本当に VRAM に切り替わるか**
  （`torch.cuda.mem_get_info()` の total が 13 963 → 4 096 MiB になり、`mem_info_vram_used` が増えれば成功）。
  聴感確認用: `outputs/exp12/12_igpu_tecpu_{short,long}.wav`（ModernBERT が fp32/CPU になる分、
  state はわずかに変わる）。

### 13.4 上限テストの妥当性と長時間 churn

上限（`set_per_process_memory_fraction`）を掛けた計測で `peak_reserved ≤ 上限` になるのは同語反復なので、
根拠にしているのは (1) **OOM の有無**（上限に当たると allocator はキャッシュ解放→再試行→それでも不足なら
HIP OOM を投げる。2304 で 2 ケース、TE=model の 3072 で worst が実際に落ちており反証可能）、
(2) 上限と無関係な **`max_memory_allocated`**、(3) kernel 側から見た **`mem_info_gtt_used`**（torch の
allocator 外の HIP ランタイム・コードオブジェクト・作業領域を含む実保持量）の 3 つ。

残っていた穴「長時間実行での断片化」（09 では stress 通過後に 13 リクエスト目で OOM した前例）を
`bench/churn_igpu.py` で埋める: 長さの異なる 6 種（short / text 256 / caption+no-ref / long / medium /
text 256 + caption + 参照）を 6 周 = 36 リクエスト、上限 2560 + TE=cpu + chunk 96、GTT 使用量を 50 ms で
サンプリングして**実行中のピーク**を取る（`results/12_igpu_churn_tecpu_cap2560.json`）:

| 指標 | 値 |
|---|---|
| OOM | **0 / 36**（13.3 min） |
| peak alloc（最大） | 1 755 MiB |
| reserved | 6 リクエスト目で 2 452 MiB に達した後 **30 リクエスト横ばい**（成長なし） |
| GTT 実行中ピーク | **2 854 MiB**（開始前ベースライン 86 MiB → TTS 分 ≈ 2.77 GB、allocator 外 ≈ 400 MiB） |

- allocator 外のオーバーヘッドは一時的なものを含めても約 400 MiB で、stress 後の値と一致する。
- 依然として未検証なのは **VRAM プールに切り替えた後の挙動**（allocator 外の量が同じか、表示側の
  VRAM 使用量の変動でどこまで余白が削れるか）。これは `amdgpu.gttsize=4000` で再起動しないと分からない。

## 14. 再起動後の再開手順（新しいセッション向けの引き継ぎ）

**状況（2026-08-30 午前時点。同日午後に再起動して 15 節で完了）**: 実験 12 は GTT 上での動作点まで完了し、carve-out（UMA 4 GB）を使う
ための準備（TE=cpu、上限 2560 で churn 通過）も済んでいる。残っているのは **カーネル引数
`amdgpu.gttsize=4000` で再起動して、KFD のメモリプールが VRAM（carve-out）に切り替わるかの実測**
だけ。コミット済み（`84568e5` まで、main）。未コミットの作業はない。

### 14.1 前提の確認（再起動直後に 1 回）

```bash
cat /proc/cmdline                      # amdgpu.gttsize=4000 が入っているか
journalctl -k -b | grep -E 'amdgpu.*(VRAM|GTT) memory ready'   # 期待: "4096M of VRAM", "4000M of GTT"
for f in mem_info_vram_used mem_info_gtt_total; do echo "$f=$(( $(cat /sys/class/drm/card2/device/$f)/1048576 )) MiB"; done
# KFD が公開するプール（heap_type 1 のバンクの size が 4 GiB 台になっていれば切り替わっている）
grep -E 'heap_type|size_in_bytes' /sys/class/kfd/kfd/topology/nodes/1/mem_banks/*/properties
```

- `card2` が amdgpu かどうかは `readlink /sys/class/drm/card2/device/driver` で確認（番号は変わり得る）。
- 切り替わっていれば `torch.cuda.mem_get_info()` の total が **13 963 → 4 096 MiB** になる:

```bash
cd ~/dev/Irodori-TTS
HSA_OVERRIDE_GFX_VERSION=9.0.0 .venv-rocm/bin/python -c "import torch; print([v//2**20 for v in torch.cuda.mem_get_info()])"
```

### 14.2 本番相当の確認（切り替わっていた場合）

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0 IRODORI_OPT_CUDA_GRAPH=0 IRODORI_OPT_PREBAKE=0 \
       IRODORI_OPT_TE_DEVICE=cpu IRODORI_OPT_VRAM_LIMIT_MB=2560 IRODORI_OPT_DECODE_CHUNK=96
# 1) 代表入力（RTF と peak を 10 節 / 13.3 節と比較。RTF 1.07〜1.17、peak alloc 1.56〜1.74 GB が目安）
.venv-rocm/bin/python bench/bench_runtime.py --device cuda --precision fp16 --codec-precision fp16 \
  --num-steps 12 --t-schedule-mode sway --inputs short medium long caption_noref --warmup 1 --repeats 2 \
  --tag 12_igpu_vram_final --output docs/experiments/results/12_igpu_vram_final.json
# 2) 宣言上限入力（参照 ≤ 30 s）
.venv-rocm/bin/python bench/stress_vram.py --precision fp16 --codec-precision fp16 --num-steps 12 \
  --t-schedule-mode sway --repeats 1 --graph-fill 0 --cases text_max caption_max caption_max_noref ref15 ref30 worst \
  --tag 12_igpu_vram_stress --output docs/experiments/results/12_igpu_vram_stress.json
# 3) 長時間 churn（sysfs_peak_mib.vram が 3 700 MiB 前後まで上がり、gtt がほぼ動かなければ carve-out を使えている）
.venv-rocm/bin/python bench/churn_igpu.py --rounds 3 --output docs/experiments/results/12_igpu_churn_vram.json
```

判定:
- 3 本とも OOM なし → **carve-out 運用成立**。README の iGPU 節を「carve-out 運用」を既定に書き換え、
  `IRODORI_OPT_TE_DEVICE=cpu IRODORI_OPT_VRAM_LIMIT_MB=2560 IRODORI_OPT_DECODE_CHUNK=96` を推奨構成にする。
  結果を 15 節として追記し、`00-index.md` の 12 行目を更新する。
- OOM が出る → allocator 外のオーバーヘッドか表示側 VRAM の変動が原因。`mem_info_vram_used` を
  TTS 起動前後で控えて差分を出し、上限を 2432 / 2304 に下げて再試行（2304 は GTT 上では caption_max と
  worst が落ちた。13.3 節）。それでも入らなければ decode chunk 64、参照 encode chunk 縮小、speaker
  encoder の CPU 化の順に削る。
- `mem_get_info` の total が 13 963 のまま → 引数が効いていない。`/proc/cmdline` を確認、
  `/etc/default/grub` → `sudo update-grub` → 再起動をやり直す。kernel が `gttsize` を丸めている可能性も
  あるので `journalctl -k -b | grep GTT` の実値を見る（4096 未満であればよい）。

### 14.3 元に戻す

```bash
sudo sed -i 's/ amdgpu.gttsize=4000//' /etc/default/grub && sudo update-grub && sudo reboot
```

GTT 4 GB でも表示には十分なので、切り替えに失敗しても起動・表示に影響は出ない見込み。

### 14.4 このあとの実験 13（速度）

carve-out の件が片付いたら、13.1 節の優先順位（AdaLN 低ランク射影のバッチ化 → Linear 融合 →
長文の GEMM 以外 7.5 s の分解 → codec の GEMM 形状）で RTF < 1（12 step）、長文 16 step ≈ 1.0 を狙う。
すべて出力保持型なので dGPU では音声 hash 一致（bf16 は 02 の理由で FP32 で判定）を要求する。
iGPU の独立 CFG 経路は `det=False` なので hash 比較は使えず、dGPU で判定する。

### 14.5 環境メモ

- `.venv-rocm`（24 GB、gitignore 済）: torch 2.9.1+rocm6.3 / torchaudio 2.9.1 / torchcodec 0.9.1。
  壊れたら 4 節のコマンドで作り直す（rocm7.1 で `uv sync` → 6.3 に差し替え）。
- `.venv`（cu128）は dGPU 用で無変更。dGPU の既定経路は本実験の変更後も音声 hash 一致。
- `HSA_OVERRIDE_GFX_VERSION=9.0.0` を忘れると ROCm が gfx90c を認識せず `cuda.is_available()` が False になる。
- `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory` の警告は無害。
- MIOpen の find-db は `~/.config/miopen/`（初回のみ数分の探索が走る）。

## 15. 再起動後の実測: carve-out (VRAM 4 GiB) 運用の成立

`amdgpu.gttsize=4000` で再起動し（2026-08-30）、14 節の手順をそのまま実行した。

### 15.1 プールの切り替わり

| 確認項目 | 再起動前（GTT） | **再起動後** |
|---|---|---|
| `journalctl -k` | – | `4096M of VRAM memory ready` / `4000M of GTT memory ready`（`GTT size ... but TTM size has been set as 14641475584, this is unusual` の警告は出るが無害） |
| KFD `mem_banks/0`（node 1） | heap_type 1, 13.6 GiB | heap_type 1, **4 GiB** |
| `torch.cuda.mem_get_info()` total | 13 963 MiB | **4 096 MiB**（free 4 038。表示分は差し引かれない） |
| `mem_info_gtt_total` | 13 963 MiB | 4 000 MiB |

確保テスト（`torch.empty` を積む）:

| 累積確保 | `mem_info_vram_used` | `mem_info_gtt_used` |
|---|---|---|
| 0（表示のみ） | 622 MiB | 84 MiB |
| 2 GiB | **2 816** | 85 |
| 3 GiB | 3 840 | 85 |
| 3.7 GiB | 4 081 | **539** |
| プロセス終了後 | **168** | 539 |

- HIP の確保は VRAM（carve-out）に載る。GTT は動かない。
- 4 GiB を超えると OOM ではなく **TTM が古いバッファを GTT に退避**する（表示用の 622 MiB が GTT に
  移り、プロセス終了後もそのまま）。13.2 節の「溢れたら即 OOM」は GTT プール時の話で、VRAM プールでは
  退避が効く。退避後は表示が GTT に住むので、carve-out 4 GiB がほぼ丸ごと TTS に使える。
- プロセス終了直後の `mem_info_vram_used` は数秒〜十数秒遅れて下がる（KFD の解放が遅延）。

### 15.2 本番相当の 3 本（TE=cpu、上限 2560、chunk 96、fp16、sway 12、graph off）

**代表入力**（`results/12_igpu_vram_final.json`、repeats 2）:

| 入力 | 音声長 | wall | **RTF** | sample_rf | decode_latent | peak alloc | VRAM 実使用 max | GTT 時（10 / 13.3 節） |
|---|---|---|---|---|---|---|---|---|
| short | 6.44 s | 6.80 s | **1.056** | 4.78 s | 1.86 s | 1 557 MiB | 2 691 MiB | 7.53 s / 1.17 |
| medium | 10.88 s | 10.68 s | **0.982** | 7.31 s | 3.20 s | 1 620 MiB | 2 693 MiB | 11.80 s / 1.08 |
| long | 28.76 s | 29.48 s | **1.025** | 20.53 s | 8.74 s | 1 743 MiB | 2 697 MiB | 30.84 s / 1.07 |
| caption_noref | 7.32 s | 7.52 s | **1.027** | 5.22 s | 2.06 s | 1 557 MiB | 2 697 MiB | – / 1.13 |

- **GTT 時より 8〜10% 速い**。差はすべて `sample_rf`（short 5.49 → 4.78 s、long 21.8 → 20.5 s）で、
  decode は同じ。carve-out は物理連続でページ変換が軽い（GTT は 4 KiB ページの GPUVM 経由）ため、
  DiT のような細かい GEMM の連打で TLB の効きが違うのだと解釈している（未プロファイル）。
- medium は RTF **0.98** で初めて 1 を切った。peak alloc は GTT 時と同じ（1.56〜1.74 GB）。

**宣言上限入力の stress**（`results/12_igpu_vram_stress.json`、6/6 ok）:

| ケース | peak alloc | reserved | VRAM+GTT 実使用（GTT ≈ 533 は表示分） |
|---|---|---|---|
| text_max | 1 704 MiB | 2 394 MiB | 3 016 MiB |
| caption_max | 2 260 | 2 388 | 3 410 |
| caption_max_noref | 1 959 | 2 388 | 3 410 |
| ref15 / ref30 | 1 739 / 1 806 | 2 406 | 3 424 / 3 428 |
| **worst** | **2 390** | **2 456** | **3 489**（VRAM ≈ 2 956 MiB） |

- worst の VRAM 分 ≈ 2 956 MiB は GTT 時の HIP 実使用 2 952 MiB と一致。プールが変わっても
  allocator 外のオーバーヘッド（≈ 400〜500 MiB）は同じ。
- `bench/stress_vram.py` の ROCm 経路は `mem_info_gtt_used` のみだったので、`vram_used + gtt_used` の
  和を記録するよう変更した（プールがどちらでも「iGPU が保持する総量」になる）。

**長時間 churn**（`results/12_igpu_churn_vram.json`、6 種 × 3 周 = 18 リクエスト、6.3 min）:

| 指標 | 値 |
|---|---|
| OOM | **0 / 18** |
| peak alloc（最大） | 1 753 MiB |
| reserved | 6 リクエスト目で 2 432 MiB に達した後横ばい |
| sysfs ベースライン → 実行中ピーク | VRAM 170 → **2 929 MiB**、GTT 533 → 534 MiB |

- TTS 分 ≈ 2.76 GB がすべて VRAM から取られ、GTT は 1 MiB しか動かない。**carve-out を使えている**。
- 余白: 表示が GTT に退避した状態では 4 096 − 2 929 ≈ 1.1 GB。表示（622 MiB）が VRAM に戻っても
  ≈ 550 MiB 残る。14.2 節で「VRAM が 3 700 前後まで上がる」と予想したのは表示分が VRAM に居続ける
  前提で、実際は退避されるためそれより低い。

### 15.3 採否と推奨構成

**carve-out 運用を採用**。dGPU の 16 GB は VLM に丸ごと渡し、TTS は iGPU の UMA 4 GiB（通常 RAM から
BIOS が既に切り出している分）で動く。通常 RAM の追加消費は ModernBERT (CPU fp32) の約 1.2 GB と
プロセス自体のみ。

| 項目 | 値 |
|---|---|
| カーネル引数 | `amdgpu.gttsize=4000`（`/etc/default/grub`、`update-grub` 済み） |
| 環境変数 | `HSA_OVERRIDE_GFX_VERSION=9.0.0 IRODORI_OPT_CUDA_GRAPH=0 IRODORI_OPT_PREBAKE=0 IRODORI_OPT_TE_DEVICE=cpu IRODORI_OPT_VRAM_LIMIT_MB=2560 IRODORI_OPT_DECODE_CHUNK=96` |
| infer.py | `--model-device cuda --codec-device cuda --precision fp16 --codec-precision fp16 --num-steps 12 --t-schedule-mode sway --max-ref-seconds 30` |
| RTF（sway 12） | short 1.06 / medium 0.98 / long 1.03 / caption 1.03 |
| メモリ | VRAM 実使用 2.7 GB（代表入力）〜 3.0 GB（worst）、余白 ≥ 0.5 GB |

副作用は GTT（iGPU のシステムメモリ側バッファ）がデスクトップ全体で 4 GB に制限されることだけで、
表示は iGPU で通常どおり動いている（表示バッファが GTT に退避しても問題は出ていない）。
戻すときは 14.3 節。

残件は速度側（実験 13、13.1 節の優先順位）。動作点は VRAM プールに変わったので、実験 13 の iGPU 計測は
本節の構成（`12_igpu_vram_final.json`）を基準にする。
