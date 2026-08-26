# 02: 出力保持型の最適化（condition 再利用 / padding crop / 同期除去 / mask 事前計算 / ロード順序）

日付: 2026-08-26

## 1. 目的 / 仮説

数値経路を（原理的に）変えずに、無駄な計算と GPU 同期を削る。

- `encode_conditions` が duration 予測と sampling で二重に走っている → 1 回にする
- text 256 / caption 512 の固定長 padding が ModernBERT と DiT の context K/V の両方に乗っている
  → 実長に crop（caption 空なら 1 token）
- RF ループ内の `.item()` / `torch.all().item()` / `caption_mask.any().item()` → 事前に host 側で確定
- 毎 layer・毎 step の mask cat / `torch.ones` 生成 → request 単位で additive float mask を 1 回生成
- `get_timestep_embedding` 内の `torch.tensor(10000.0, device=cuda)`（毎 step の同期的 H2D）→ キャッシュ
- `find_flattening_point` の Python ループ（T 回の GPU 同期）→ `unfold` でベクトル化
- FP32 モデル全体を GPU に置いてから cast → param 単位で cast+転送

## 2. 変更内容

| ファイル | 変更 |
|---|---|
| `irodori_tts/opt_config.py` | 新規。`IRODORI_OPT_*` 環境変数で各最適化を on/off |
| `irodori_tts/rf.py` | 旧 sampler を `_sample_euler_rf_cfg_legacy` として保持。`_sample_euler_rf_cfg_fast` と `_FastSamplerState`（1 step を関数化、mask を request 単位でキャッシュ）を追加。`sample_euler_rf_cfg` は dispatcher |
| `irodori_tts/model.py` | `JointAttention.forward` に precombined `attn_mask` fast path、`build_combined_attn_mask`、`prewarm_rope`、timestep freqs キャッシュ |
| `irodori_tts/inference_runtime.py` | text/caption crop、`encoded_conditions` の再利用、`find_flattening_point` ベクトル化、`_move_inference_module` の cast 順序 |

t_schedule は従来どおり device 上で計算してから `.tolist()` で 1 回だけ転送する
（CPU の `linspace` と bit が変わる可能性を避けるため）。

## 3. 検証方法

同一 runtime インスタンス上で legacy 経路と最適化経路を切り替え、同一 seed の音声を比較。

```bash
uv run --no-sync python bench/check_equivalence.py --precision fp32 \
  --inputs short caption_noref --modes independent joint alternating \
  --variants fast_nograph fast_graph
uv run --no-sync python bench/check_equivalence.py --precision bf16 --inputs short \
  --modes independent --variants crop_only reuse_only sampler_only
```

## 4. 等価性の結果

| 変更 | FP32 | BF16 |
|---|---|---|
| condition 再利用のみ | **hash 一致** | **hash 一致** |
| sampler 高速化のみ（同期除去 + mask 事前計算 + timestep cache + tail ベクトル化） | **hash 一致** | **hash 一致** |
| text/caption crop のみ | max abs diff 7.4e-4（音声は ±1 スケール） | max abs diff 0.66 |
| 全部 (graph なし) | 7.4e-4 | 0.66 |

crop は attention の key 長を変えるため、SDPA / ModernBERT の reduction 順序が変わり float 誤差が出る。
FP32 では 1e-3 未満（無音部の量子化 1 LSB = 3e-5 の 20 倍程度、可聴差なし）。
BF16 では 40 step の Euler 積分でこの差がカオス的に増幅され、波形として別物になる
（同じ現象は BF16 で kernel が変わるあらゆる変更で起きる。BF16 同士の bit 比較は意味を持たない）。
→ 出力保持型の判定は **FP32 で max abs diff ≤ 1e-3** を基準にする。

各 CFG mode の FP32 結果（graph なし）: independent 7.4e-4 / joint 3.4e-4 / alternating 5.2e-4（short）、
caption_noref: 5.7e-4 / 6.8e-4 / 1.1e-3。

## 5. 性能

→ CUDA Graph 込みの計測は [03-cuda-graph.md](03-cuda-graph.md) の ablation 表を参照。
