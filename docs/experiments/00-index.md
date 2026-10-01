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
| 12 | [12-metal-port.md](12-metal-port.md) | **ブランチ `metal-local`**: Apple M3 Pro / Metal (MPS) 専用化。CUDA Graph・VRAM cap 撤去、実数 RoPE、fp16 既定、MPS スレッド制約、compile、warm ベンチ → **short RTF 0.40（compile）/ 0.48（eager）** | 完了 |
| 13 | [13-ane.md](13-ane.md) | **ANE**: RF step を Core ML で Neural Engine に載せ（torch.export、package 単位の列挙形状、exp 活性化）、GPU に cond 分岐 / 候補 2 を並走 → **short 3459 → 2299 ms（RTF 0.32）**、2 候補 6.40 → 4.16 s | 完了 |
| 14 | [14-step-count.md](14-step-count.md) | **step 数削減**: sway sampling で 40 → 12 step（20 秒以上の出力は自動で 16）。同 step なら sway が linear より良く、8 step は長文で高域ノイズ +4〜5 dB → **short 3459 → 1023 ms（RTF 0.14）、MPS eager 40 step 比 3.38× = step 2.06× × ANE 1.35× × compile 1.23×**。律速は codec decode に移動 | 完了 |
| 15 | [15-decode-ane.md](15-decode-ane.md) | **codec decode の ANE 化と DiT/decode パイプラインの試算**: decoder（789 GFLOP、conv_transpose + Snake）を Core ML 変換 → stride 12/10 の conv_transpose と長さ ≥ 21600 の op が CPU 落ち、全部載る 8 frame でも数値破綻（SNR −22 dB）。単発 request では DiT と decode を重ねられない。長文の自動分割時のみ、セグメント単位のヘテロ振り分けで **1.6× の見込み（導出）** | プローブ完了、実装は未着手 |
| 16 | [16-m1-mini.md](16-m1-mini.md) | **M1 Mac mini を TTS サーバーに**: 同じ `metal-local` を無改造で実測。素は RTF 1.34（実時間より遅い）、最良（ANE 全分岐 + compile）で **short 2687 ms / RTF 0.37**、3 分の原稿 74 s、常駐 1.9 GB、持続 10 連発でドリフト 0.8%。ANE はステップ単体で **3.1×**（M3 Pro は 1.85〜2.5×）、**GPU への cond 分岐は M1 では逆効果**、bf16 は fp16 の半分。**2 ワーカー同時でスループット 1.97×**（MPS のみは 1.12×）。codec decode は M1 でも ANE に載らない（CPU 落ち 120 op、SNR 40.9 dB）。**現行の `full` shape セットは M1 で ANE コンパイルに失敗して CPU 落ち**（ANE なしより 3 倍遅い、失敗はキャッシュされず毎プロセス 20〜75 分やり直す。当初の「キャッシュ容量」説は統合ログで訂正）。要因の分離は 17 | 完了（`full` の失敗要因は 17 へ） |
| 17 | [17-m1-ane-factors.md](17-m1-ane-factors.md) | **M1 の ANE コンパイル失敗を要因に分ける**: 判定を速度ではなく統合ログで取るプローブ（`probe_ane_shapes.py`）。列挙数 23・1×1536 単独・ctx パディング・macOS 27 はどれも単独では障害でない。落ちるのは**小さい bucket と大きい bucket を同じ package に入れ、2 層以上**のとき。**12 層 × batch 1 × 18 形（192〜1536）は M1 の ANE に載る**。M3 Pro の B=3×1536 の失敗は別の機序（コード生成）。`ct.convert` が隠れ ANE コンパイルを走らせている疑い | **作業中**（0 節に引き継ぎ） |
| 18 | [18-m6-prestudy.md](18-m6-prestudy.md) | **M6 Mac mini の事前検討**（mflux / turbo-fieldfare の M6 実測から）: GPU fp16 GEMM 18.6 TFLOP/s（M3 Pro の 3.1×）だが fp32 は 0.95×、ANE 単基は M3 Pro 並みで **GPU の方が速い側に逆転**、ANE 同居で GPU −20〜60%、持続負荷で GPU クロック −8%。単発は GPU 単独が既定になる見込み、最大の不確定は codec decode の conv が NAX に乗るか。測定順序を 5 節に | 机上検討（実測は 19 へ） |
| 19 | [19-m6-mini.md](19-m6-mini.md) | **M6 Mac mini の初回実測**: torch MPS で fp16 18.66 TFLOPS / 帯域 140 GB/s。**GPU 単独 + compile で short 719 ms / RTF 0.10**（M3 Pro 最良 1023 ms の 1.42×）。DiT は 2.14×、decode は 1.37× しか伸びず wall の半分に。ANE は B=1 step では 14.6 vs MPS 18.0 ms と速いが、**batch 2/3 の package が ANE コンパイル失敗（Code=22、7 s）**、使える gpu_branches 2 は short +8% / long −14%。既定は ANE off | GPU 単独を確認、decode 解析・持続は未着手 |
