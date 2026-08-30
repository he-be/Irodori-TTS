# 実験インデックス

| # | ファイル | 内容 | 状態 |
|---|---|---|---|
| 01 | [01-baseline.md](01-baseline.md) | 環境プローブ、推論経路の調査、FP32 / BF16 ベースライン計測 | 完了 |
| 02 | [02-output-preserving.md](02-output-preserving.md) | condition 再利用、text/caption crop、同期除去、mask 事前計算、ロード順序（出力保持型） | 完了 |
| 03 | [03-cuda-graph.md](03-cuda-graph.md) | RF step の CUDA Graph 化（shape bucketing、mutable KV 対応）、ablation | 完了 |
| 04 | [04-codec-and-watermark.md](04-codec-and-watermark.md) | codec weight_norm fold、codec BF16（不採用）、watermark 無効化、compile（不採用） | 完了 |
| 05 | [05-reference-cache.md](05-reference-cache.md) | 参照音声 L1（latent）/ L2（speaker state）キャッシュ | 完了 |
| 06 | [06-memory.md](06-memory.md) | VRAM プロファイル（ピーク = codec decode）、overlap chunk decode、ハード上限 | 完了 |
| 07 | [07-compile-and-quality.md](07-compile-and-quality.md) | DiT compile、最終ベンチ（nvidia-smi 3.1 GB）、BF16 品質指標と聴感、decode-only BF16 採用 | 完了 |
| 08 | [08-vram-cap-floor.md](08-vram-cap-floor.md) | VRAM ハード上限の OOM 試験: 代表入力での下限 3072 MB（当時は既定 3584 据え置き）、2816 以下は OOM | 完了（09 が更新） |
| 09 | [09-vram-safe-operating-point.md](09-vram-safe-operating-point.md) | 宣言上限入力での stress。参照 encode の chunk 化、CUDA Graph の static/pool 上限化 → **既定を 3072 MB に変更** | 完了（10 が更新） |
| 10 | [10-vlm-coexistence.md](10-vlm-coexistence.md) | llama-swap の VLM との同居。`--n-cpu-moe` による静的な VRAM 配分、同時実行/ロード churn/パイプライン stress、**上限の既定を 3840 MB に修正** | 完了 |
| 11 | [11-load-time.md](11-load-time.md) | ロード時間の分解。捨てる乱数初期化の除去、事前計算バンドル（prebake）、import 裏での並列ロード → **9.55 → 5.08 s** | 完了 |
| 12 | [12-igpu-offload.md](12-igpu-offload.md) | iGPU (Vega 7 / gfx90c) への TTS オフロード。sway 12 step 前提、CPU ベースライン RTF 1.7〜1.9、ROCm 6.3 wheel + gfx900 偽装、MIOpen の dilated conv 回避 (decode 21.8 → 1.6 s)、iGPU fp16 で RTF 1.02〜1.12 / 3.7 GB (GTT)。carve-out は kernel 仕様で不可視 → `amdgpu.gttsize=4000` で KFD プールを VRAM 4 GiB に切り替え、TE=cpu + 上限 2560 で **RTF 0.98〜1.06 / VRAM 2.7〜3.0 GB**、stress・churn 通過（15 節） | 完了（速度の続きは 13） |
| 13 | [13-igpu-dit-gemm.md](13-igpu-dit-gemm.md) | iGPU の DiT 高速化（出力保持型）。(1) 全 24 AdaLN の低ランク射影を step ごとに bmm 2 回へ（1 728 → 24 launch）、(2) wq/wk/wv/gate と w1/w3 の融合（rocBLAS の得失が M で ±25% 反転するため M 範囲表で分岐、`sweep_linear_fusion.py`）。いずれもメモリ増なし・LoRA 時は自動 fallback。iGPU で **short 6.80 → 6.33 s (RTF 0.98)、medium 0.94、long 1.01、caption 0.96**。dGPU は hash 不変を優先し既定 off（`IRODORI_OPT_ADALN_BATCH` / `IRODORI_OPT_LINEAR_FUSE` = auto → ROCm のみ）。残り候補（長文の非 GEMM、codec GEMM）はユーザー判断で打ち切り | 完了 |
