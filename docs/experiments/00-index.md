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
