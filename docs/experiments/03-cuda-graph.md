# 03: RF step の CUDA Graph 化

日付: 2026-08-26

## 1. 目的 / 仮説

BF16 では 1 step ≈ 17.6 ms（short）で、長さにほとんど依存しない（01 参照）。
DiT 12 層 × 数十 kernel × 40 step の launch overhead と、Python 側のオーバーヘッドが支配的。
Euler step 全体（CFG batch の forward + CFG 合成 + Euler 更新）を 1 つの CUDA Graph にして replay する。

## 2. 設計（`irodori_tts/cuda_graph.py`, `irodori_tts/rf.py`）

- capture 対象: `_FastSamplerState.step(x_t, t, dt)` = 「forward（CFG batch）→ CFG 合成 → rescale（任意）→ x + v·dt」
- 入力の分類
  - 動的入力（毎 step コピー）: `x_t (B,S,D)`, `t (B,)`, `dt (1,)` — 合計数十 KB
  - request 定数（request 内では同一オブジェクト）: text/speaker/caption state と mask、
    context K/V cache（12 層 × 最大 6 tensor）、precombined additive attention mask、latent mask
- **const set**: request 定数の shape signature ごとに static buffer を 1 組保持。
  同じ shape の graph（例: CFG あり/なし、S bucket 違い）は const set を共有し、メモリを増やさない。
- **graph entry**: (const set, x/t/dt の shape, use_cfg, alt_index, CFG mode/scales, latent_len) ごとに 1 graph。
  LRU（既定 12 entry）。全 graph が 1 つの memory pool を共有し、出力は replay 直後に clone。
- コピー省略: request 定数は `id()` が前回と同じならコピーしない。request 開始時 (`begin_request`) に
  id テーブルを破棄するので、別 request の tensor が同じ id を再利用しても誤ってスキップしない。
- **mutable KV**: `speaker_kv_scale` は context K/V を in-place で書き換える（開始時と `speaker_kv_min_t` 通過時）。
  sampler が `mark_dirty("context_kv")` を呼び、次の replay 前に static buffer へ強制再コピーする。
- 長さの bucketing（signature の再利用率を上げる）
  - latent 長 S: 32 frame（0.34 s）単位に pad し、pad 位置は `latent_mask=False`
  - text / caption token: 16 単位に pad（mask False）
  - speaker patched token: 64 単位（256 frame = 2.7 s）に pad（mask False）
  - pad 位置は key として mask され、query としての出力は捨てるので real token の値は変わらない
- capture 手順: side stream で 2 回 warmup → `torch.cuda.graph(pool=shared)` で capture → 以降 replay。
  capture 失敗時は runner を無効化して eager にフォールバック。
- model/LoRA variant 変更時: `InferenceRuntime._set_variant()` → `runner.set_variant()` で全 graph を破棄。
- RoPE cache は capture 前に 4096 まで `prewarm_rope` で確保（capture 後の buffer 置換を防ぐ）。
- `get_timestep_embedding` の `torch.tensor(10000.0, device=cuda)` は capture 中に非合法（pageable H2D）
  なので (device, dim) ごとにキャッシュ（値は同じ device op で 1 回だけ計算するので bit 同一）。

## 3. 検証

`bench/check_equivalence.py --precision fp32`（02 参照）: independent / joint / alternating、
ref あり / caption+no-ref、speaker_kv_scale=1.5（ループ途中の in-place 変更）で
legacy との max abs diff ≤ 1.1e-3、graph なしと同等。short→caption→short の順で回しても
2 回目の replay 結果は 1 回目と bit 一致（repeat_maxdiff=0）。

graph stats（fp32 チェック終了時）: 16 capture / 464 replay / 4 evict、12 entries、4 const sets、
static 147 MB。

## 4. 結果

→ 4.1 は全部入り、4.2 は ablation（bf16、watermark なし）。

### 4.1 全最適化 (02_opt_bf16.json) vs BF16 ベースライン (01_baseline_bf16.json, watermark あり)

| 入力 | wall median (01 → 02) | sample_rf (01 → 02) | RTF | GPU util | peak alloc |
|---|---|---|---|---|---|
| short | 915 → **491 ms** | 706 → 387 | 0.141 → 0.076 | 68 → 91 % | 2969 → 2757 MiB |
| medium | 1081 → **675 ms** | 794 → 524 | 0.098 → 0.062 | 79 → 95 % | 3447 → 3231 MiB |
| long | 1935 → **1491 ms** | 1330 → 1144 | 0.067 → 0.052 | 95 → 100 % | 5329 → 5114 MiB |
| caption_noref | 868 → **506 ms** | 698 → 387 | 0.119 → 0.069 | 74 → 100 % | 3058 → 2847 MiB |

warm 後 reserved: 8660 → 3592 MiB。ロード後 allocated 2023 → 1905 MiB（SilentCipher 分）。

### 4.2 Ablation（`bench/run_ablation.sh`、bf16、watermark なし、warmup 3 / repeats 8、util 計測なし）

| 段階 | short wall median | short sample_rf | long wall median | long sample_rf | peak alloc (short / long) |
|---|---|---|---|---|---|
| A: legacy 経路（全スイッチ off） | 866 ms | 694 | 1758 ms | 1325 | 2960 / 5318 MiB |
| B: + condition 再利用 + 同期除去 sampler | 831 | 666 | 1666 | 1258 | 2957 / 5315 |
| C: + text/caption crop | 839 | 673 | 1540 | 1133 | 2956 / 5314 |
| D: + CUDA Graph（bucket 32/16/64） | **554** | 387 | 1552 | 1144 | 3052 / 5408 |
| E: + codec weight_norm fold | 549 | 387 | 1550 | 1144 | 2694 / 5051 |
| F: + reference cache（= 既定） | **491** | 387 | **1490** | 1144 | 2694 / 5051 |

- B（同期除去）は short で -28 ms、long で -67 ms。GPU が空く隙間を潰した分。
- C（crop）は long で -126 ms。caption 空でも 512 token 分の masked key を 40 step × 12 層で参照していた。
  short では S が小さく overhead 支配のため効果が見えない。
- D（Graph）は short で **-285 ms（sample_rf 673 → 387 ms、1 step 9.7 ms）**。long は compute-bound で
  ±0（bucket pad の分わずかに増える）。static buffer で peak alloc +100 MB。
- E（fold）は時間ほぼ不変だが `weight_g/weight_v` の二重保持が消え peak alloc -358 MB。
- F（ref cache）は prepare_reference 54 → 1 ms、predict_duration 34 → 27 ms で -58 ms。

legacy (A) → 既定 (F): short **866 → 491 ms (×1.76)**、long **1758 → 1490 ms (×1.18)**。
01 の FP32 デフォルト（watermark あり）からは short 1316 → 491 ms (×2.68)、long 4495 → 1490 ms (×3.0)。

## 5. 残りのボトルネック（short, 491 ms）

| stage | ms | 割合 |
|---|---|---|
| sample_rf (40 step, graph replay) | 387 | 79 % |
| decode_latent (codec FP32) | 75 | 15 % |
| predict_duration (ModernBERT 25 層 + duration head) | 27 | 5 % |
| その他 | 2 | |

1 step 9.7 ms は S=640 (pad 後), B=3 で約 0.55 TFLOP → 実効 57 TFLOPS。RTX 5060 Ti の BF16 dense
ピークに対しておおむね 60 % 程度で、残りは attention（bool mask のため mem-efficient kernel）と
fp32 で計算している RMSNorm / AdaLN / RoPE の帯域。→ `IRODORI_OPT_COMPILE_DIT`（07 で検証）。
