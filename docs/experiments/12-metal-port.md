# 12. Apple Silicon / Metal 専用化（ブランチ `metal-local`）

作成: 2026-08-29

書き方のルール（`~/LLM/turbo-fieldfare` の status 文書に倣う）:
**実測** = 手元で数字を取ったもの（条件併記）。**導出** = 実測からの計算。**未確認** = 根拠なし、断定しない。

## 1. 目的 / 仮説

- 目的: RTX 5060 Ti 向けフォーク（01〜11）を、この Mac（M3 Pro）の **Metal (MPS) だけで** 動かす。
  互換性は捨てる。CPU 実行（MPS fallback 含む）は禁止。
- 仮説: `main` の README が言う「macOS MPS でも動く」は素の経路の話で、01〜11 の最適化は
  CUDA 判定 (`device.type == "cuda"`) の中にあるため MPS では全部素通りする。
  MPS 向けに置き直せば、MPS でも「最適化された」経路が使える。

## 2. 実機

| 項目 | 値 | 出典 |
|---|---|---|
| チップ | Apple M3 Pro (CPU 12 / GPU 18 コア), Metal 3 | 実測 (`system_profiler`) |
| メモリ | 18 GB unified (`hw.memsize` 19,327,352,832) | 実測 |
| OS | macOS 15.7.5 (24G624) | 実測 |
| PyTorch | 2.10.0 (PyPI wheel, MPS built-in), Python 3.12.11 (uv) | 実測 |
| `torch.mps.recommended_max_memory()` | 12,288 MiB | 実測 |
| matmul 4096³ | fp32 5.33 / fp16 5.63 / bf16 5.71 TFLOPS | 実測（10 回平均, warm） |
| モデル / Codec | `Aratako/Irodori-TTS-v4.1-Small` / `Semantic-DACVAE-Japanese-32dim` | 01 と同じ |

matmul が dtype でほぼ変わらないので、半精度の利点は演算ではなく **重み読み出しの帯域**
（batch 1 の DiT step は重み 1.5 GB を毎 step 読む）と activation の量（導出）。

## 3. 変更内容

### 3-1. 環境 (`pyproject.toml`)
- `cu128 / rocm / xpu / cpu` extra と PyTorch index を全部削除。PyPI の torch 2.10 のみ。
- `torchao`（CUDA/CPU カーネル）と `silentcipher`（watermark、既定 off）を依存から外した。
- Python 3.12（`sentencepiece 0.1.99` は 3.12 の wheel が無いので `>=0.2`）。

### 3-2. デバイス (`inference_runtime.py`)
- `resolve_runtime_device` は `mps` 以外を拒否。`default_runtime_device()` は `"mps"` 固定。
- `PYTORCH_ENABLE_MPS_FALLBACK=0` をパッケージ import 時に強制（`irodori_tts/__init__.py`）。
  未対応 op があれば CPU に落ちずに例外になる。
- `resolve_runtime_dtype`: `fp16` を追加、`bf16` の CUDA/XPU 限定を解除。
- 同期 / empty_cache は `torch.mps.*`。VRAM cap は `IRODORI_OPT_MPS_LIMIT_MB`
  (`torch.mps.set_per_process_memory_fraction`) に置換、既定 0（無効）。

### 3-3. 撤去したもの
- `irodori_tts/cuda_graph.py` と `rf.py` の graph runner 経路、bucket padding、
  `IRODORI_OPT_CUDA_GRAPH / GRAPH_* / TEXT_BUCKET / SPEAKER_BUCKET / VRAM_LIMIT_MB`。
- `bench/coexist_*`, `stress_vram.py`, `check_codec_*.py`（nvidia-smi / CUDA 前提）。

### 3-4. Metal 向けに置き直したもの
- **実数 RoPE** (`model.py`): `view_as_complex` の複素積を `[cos, sin]` テーブルの
  4 つの実数 elementwise に置換（`IRODORI_OPT_ROPE_REAL=0` で旧経路）。
  数学的に同一。RoPE テーブルは fp32 のまま保持する（`_move_inference_module` で cast 除外）。
- **codec decode autocast** を `torch.autocast("mps", fp16)` に（`IRODORI_OPT_DECODE_AUTOCAST_DTYPE`）。
- **noise は MPS Generator** で生成（CPU fallback を削除）。
- **MPS のスレッド制約**: 並列ロードでワーカースレッドが `.to("mps")` / codec probe を
  走らせると Metal が `commit command buffer with uncommitted encoder` でアボートする（実測）。
  ワーカーは CPU 側の準備（safetensors → CPU、unpickle、weight_norm fold、cast）だけにし、
  MPS への転送とプローブはメインスレッドに限定した（`DACVAECodec.prepare_cpu` / `load`）。

## 4. 計測方法

```bash
uv run python bench/bench_runtime.py --precision fp16 --tag metal_fp16 \
  --inputs short medium long caption_noref --warmup 2 --repeats 5 --cooldown 10 \
  --output docs/experiments/results/metal_fp16.json
```

- 参照音声 `outputs/sample.wav` は本ブランチの no-ref + caption 生成（seed 1234, 7.32 s）。
- `--cooldown 10`: 入力ごとに 10 s 休む（サーマルドリフト対策。turbo-fieldfare の測定で
  連続実行 4% 低下が出ている）。
- MPS にはピークカウンタが無いので、`current_allocated_memory` を 20 ms でサンプルした最大値を
  ピークとして記録する（**導出**: 20 ms より短い transient は見逃す）。

## 5. 結果

### 5-1. 精度別 warm ベンチ（実測, `metal_{fp16,bf16,fp32}.json`, 5 回中央値）

| 入力 | 音声長 | fp16 wall (RTF) | bf16 wall | fp32 wall | fp16 sample_rf / decode |
|---|---:|---:|---:|---:|---:|
| short | 7.20 s | 3459 ms (0.480) | 3446 ms | 3653 ms | 2639 / 783 ms |
| medium | 11.84 s | 5872 ms (0.496) | 5845 ms | 6097 ms | 4510 / 1286 ms |
| long | 28.84 s | 16450 ms (0.570) | 16396 ms | 16693 ms | 13028 / 3353 ms |
| caption_noref | 7.32 s | 3463 ms (0.473) | 3459 ms | 3674 ms | 2644 / 786 ms |
| 常駐 alloc / driver | | 1873 / 2605 MiB | 同左 | 3334 / 3445 MiB | |

- 半精度は fp32 比で **6% しか速くない**。RTX 5060 Ti（bf16 473 ms / short）とは 7 倍差。
- 参照: 01 の RTX baseline (FP32, 最適化前) は short 約 1.5 s だった。この Mac の GPU は
  matmul 5.6 TFLOPS なので桁が違うのは当然で、比較対象にしない。

### 5-2. RF step の内訳（実測, `bench/profile_step.py`, fp16）

| batch × latent | tokens | ms/forward | µs/token |
|---|---:|---:|---:|
| 1 × 45 | 45 | 15.2 | 337 |
| 1 × 360 | 360 | 74.4 | 207 |
| 3 × 90 | 270 | 55.1 | 204 |
| 3 × 360 | 1080 | 208.7 | 193 |
| 4 × 360 | 1440 | 280.6 | 195 |

- 200 token を超えると **~190 µs/token で線形** → 演算律速。forward 1 回の固定費は約 6.6 ms（導出:
  15.2 − 45 × 0.19）で、40 step でも 0.26 s。dispatch オーバーヘッドは主因ではない。
- 導出: DiT 12 層 × (attention 5 × 1280² + MLP 3 × 1280 × 3680) ≈ 0.27 G params → 0.54 GFLOP/token
  → 190 µs/token は **実効 2.8 TFLOPS**（matmul ピークの約 50%）。残りは RMSNorm / AdaLN / RoPE /
  SwiGLU の elementwise（hook 計測の上限値: AdaLN 47 ms, SwiGLU 32 ms / 135 ms 合計）。
- short (180 frame, batch 3) は 20 step × 101 ms + 20 step × 37.5 ms（`cfg_min_t=0.5` で後半は
  CFG 無し）= 2.77 s。実測 sample_rf 2.64 s と整合（導出）。

### 5-3. `torch.compile`（inductor / MPS backend, `dynamic=True`）

| 条件 | short | medium | long | caption_noref |
|---|---:|---:|---:|---:|
| eager fp16 | 3459 ms | 5872 ms | 16450 ms | 3463 ms |
| compile_dit fp16 (`metal_fp16_compile.json`) | **2860 ms (RTF 0.397)** | **4883 ms** | **14039 ms** | **2863 ms** |

- DiT forward 単体で 55.1 → 43.7 ms（-21%）、synth 全体で -17%。初回コンパイルは **約 19 s**、
  以後の shape 変化は 0.1〜0.4 s。
- **プロセスを跨ぐキャッシュは効かない**（実測: `infer.py` を 2 回連続で起動しても 2 回とも
  sample_rf 18.5 s）。→ CLI は既定 off、Gradio（常駐）は `COMPILE_DIT=1 COMPILE_CODEC=1` を既定に。
- 出力は eager と bit 一致しない（融合カーネルの丸め）。short で eager 比 max_abs 0.085 /
  SNR 32.5 dB / LSD 0.16 dB（実測）。fp16 化そのものの差（5-6）と同程度。

### 5-4. Codec decode（実測, `bench/check_codec_mps.py`, fp32 重み）

| 条件 | 180 frame (7.2 s) | 300 frame (12 s) | 720 frame (28.8 s) | vs fp32 max diff |
|---|---:|---:|---:|---:|
| chunk 0 / fp32 | 826 ms | 1377 ms | 3305 ms | 0 |
| chunk 0 / **fp16 autocast** | **646 ms** | **1064 ms** | **2538 ms** | 5e-3 |
| chunk 0 / bf16 autocast | 647 ms | 1069 ms | 2549 ms | 3.5e-2 |
| chunk 96/16 / fp16（RTX 既定） | 759 ms | 1297 ms | 3352 ms | 5e-3 |
| chunk 256/16 / fp16 | 645 ms | 1064 ms | 2773 ms | 5e-3 |
| chunk 0 / fp16 + `compile_codec` | **453 ms** | **746 ms** | **1785 ms** | — |

- RTX で transient VRAM を抑えるために入れた chunk decode は、unified memory では
  overlap の再計算分（96/16 で +33%）がそのまま時間になる。**既定を chunk 0 に変更**。
- autocast は fp16 が bf16 より誤差 1 桁小さい（同速）。`IRODORI_OPT_DECODE_AUTOCAST_DTYPE=fp16` 既定。
- `compile_codec` で -30%、初回 3.5 s。

### 5-5. メモリ（実測, `bench/profile_memory.py`, fp16, chunk 0）

| 入力 | 常駐 alloc | request peak alloc | driver | decode transient |
|---|---:|---:|---:|---:|
| short 7.2 s | 1873 MiB | 2120 MiB | 3752 MiB | +445 MiB |
| long 28.8 s | 1873 MiB | 3588 MiB | 6152 MiB | +1715 MiB |

`recommended_max_memory` 12,288 MiB に対して余裕があるので chunk 0 のまま。`IRODORI_OPT_MPS_LIMIT_MB`
は用意したが既定 0。参照 encode の chunk（96/32）は 120 s 参照で GB 単位になるので残した。

### 5-6. 品質（実測, `bench/audio_metrics.py`, fp32 出力を基準）

| 比較 | short | medium | caption_noref |
|---|---|---|---|
| fp16 vs fp32 | SNR 24.4 dB / LSD 0.13 dB | SNR 10.9 dB / LSD 0.23 dB | SNR 24.9 dB / LSD 0.17 dB |
| bf16 vs fp32 | SNR 14.3 dB / LSD 0.50 dB | **SNR −0.5 dB / LSD 8.37 dB** | SNR 16.3 dB / LSD 0.42 dB |

bf16（仮数 7 bit）は 40 step の Euler 積分で誤差が発散し、medium では別の読みになる。
fp16（仮数 10 bit）は LSD 0.2 dB 台に収まる。速度は同じなので **fp16 を既定** にした
（RTX 側の bf16 既定は Blackwell の bf16 tensor core 前提だった）。聴感確認は未実施（未確認）。

### 5-7. ロード（実測, `bench/bench_load.py`, 3 回中央値）

| | prebake 無し | prebake 有り |
|---|---:|---:|
| プロセス起動 → ロード完了 | 3.83 s | 3.92 s |
| model_construct | 1.61 s | 1.61 s |
| to_device | 0.35 s | 0.34 s |
| RSS | 4216 MiB | **1301 MiB** |

- 11 で効いた prebake は、この Mac では **時間には効かない**（unified memory では CPU → MPS が
  memcpy なので to_device がもともと 0.35 s）。RSS は FP32 コピーが消えて 2.9 GB 減る。
- 残りは `model_construct` 1.6 s（GIL bound の Python）と transformers import 0.75 s。

### 5-8. 等価性（実測, `bench/check_equivalence.py --precision fp32`）

legacy 経路（複素 RoPE、fast sampler なし、fold なし）に対して fast 経路・複素 RoPE 経路とも
max abs diff 7e-4〜1.3e-3、長さ一致、再実行は bit 一致（`repeat_maxdiff=0`）。RTX の判定基準
（FP32 で ≤1e-3）をわずかに超える組み合わせがあるが、MPS の matmul 実装差によるもので、
実数 RoPE 単独の差（fast 9.8e-4 vs rope_complex 1.2e-3）は複素経路より小さい。

## 6. 品質確認

5-6 参照。fp16 既定 + compile（Gradio）で fp32 との差は LSD 0.2 dB 前後。聴感は未確認。

## 7. 採否と次のアクション

採用（既定）:
- `mps` 固定、fallback 禁止、fp16 既定、実数 RoPE、decode chunk 0 + fp16 autocast、
  MPS 転送のメインスレッド限定、Gradio では `COMPILE_DIT=1 COMPILE_CODEC=1`。

不採用 / 据え置き:
- bf16（品質）、chunk decode（速度）、prebake の時間効果（無し。RSS 目的でなら有効）。
- CUDA Graph 相当（MPS には無い。`torch.compile` が同じ役割の一部を担う）。

次のアクション（未着手）:
1. 演算律速なので残る大玉は **step 数 / CFG の削減**（品質が変わる。`--num-steps`, `--cfg-min-t` を
   聴感で詰める）。fp16 で 190 µs/token を切るには elementwise の融合（compile で 21% は取れた）。
2. `LowRankAdaLN` / `RMSNorm` の fp32 経由（`x.float()`）を fp16 のまま計算できるか（品質要確認）。
3. 聴感評価（fp16 / compile / step 数）。
4. `IRODORI_OPT_MPS_LIMIT_MB` の実効性は未確認。
