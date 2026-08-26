# 08: VRAM ハード上限の下限探索

日付: 2026-08-26

## 1. 目的

07 の既定（上限 3584 MB）は余裕を持たせた値。現在の構成（bf16 モデル、codec fp32 重み + decode autocast、
chunk 96/16、graph LRU 12）のまま、上限をどこまで下げられるかを OOM 試験で確定する。
「上限を下げて速度が落ちる／落ちない」も同時に見る。

## 2. 方法

`bench/bench_runtime.py --precision bf16 --inputs short medium long caption_noref --warmup 3 --repeats 8`
に `IRODORI_OPT_VRAM_LIMIT_MB` を変えて実行（`results/08_*.json`）。
合わせて「chunk 64 + graph entries 4」で peak が下がるかも確認。

## 3. 結果

| 条件 | 上限 | 結果 | short / medium / long / caption (wall median) | nvidia-smi max |
|---|---|---|---|---|
| 既定 (07) | 3584 | OK | 450 / 646 / 1406 / 481 ms（compile on） | 3079〜3437 MiB |
| 既定 | **3072** | **OK** | 473 / 653 / 1445 / 485 ms（compile off、07 と同等） | **3257〜3305 MiB** |
| 既定 | 2816 | **OOM**（"2.75 GiB allowed"、allocated 2.56 GB + private pool 244 MB + 断片 161 MB） | — | — |
| 既定 | 2560 | OOM | — | — |
| chunk 64 + entries 4 | 3584 | OK だが遅い: 482 / 661 / 1474 / 494、**p95 悪化**（long 1645 ms、再 capture） | peak alloc 2.39〜2.48 GB（ほぼ不変） | 3267〜3667 |
| chunk 64 + entries 4 | 2560 / 2304 | OOM | — | — |

## 4. 分かったこと

- 上限の判定は `max_memory_allocated` だけでなく、**CUDA Graph の private pool（120〜240 MB）と
  reserved-but-unallocated の断片（90〜160 MB）** を含む。実ピーク 2.5 GB に対し上限は
  約 +0.5 GB 必要。
- 現在の構成での **実測下限は 3072 MB**（nvidia-smi ≈ 3.3 GB）。速度への副作用なし。
  2816 以下は OOM。3072 は余白が ~100 MB しかないので、長い caption（512 token）や 120 s の
  参照音声など未計測の入力では thrash / OOM の可能性がある → **既定は 3584 のまま**、
  llama.cpp 側の必要量が厳しいときだけ `IRODORI_OPT_VRAM_LIMIT_MB=3072`。
- decode chunk をこれ以上小さくしても peak は下がらない（decode transient は autocast + chunk 96 で
  既に 300 MB 程度。残りは重み 1.87 GB + graph static/pool + ModernBERT の一時領域）。
  graph entries を減らすと再 capture で p95 が悪化するだけ。

## 5. これ以上下げるには（未実施、品質または速度の検証が必要）

| 手段 | 見込み | 副作用 |
|---|---|---|
| ModernBERT (620 MB bf16) を CPU 常駐にして encode 時だけ転送 | 上限 −0.5 GB | +80〜100 ms/req |
| ModernBERT を int8 weight-only (torchao) | −0.3 GB | テキスト条件付けが変わる（要聴取） |
| codec decoder のみ bf16 重み（encoder は fp32） | −0.2 GB | 04/07 の「こもり」が decode 由来なら NG（要聴取） |
| DiT を int8 weight-only | −0.4 GB | 品質変化、Blackwell で bf16 より遅い可能性、graph との相性未確認 |
| graph を 1 entry に制限 | −0.1〜0.2 GB | 文長が変わるたびに再 capture（+50〜100 ms） |
