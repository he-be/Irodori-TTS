# 17. M1 の ANE コンパイル失敗を要因に分ける（作業中・引き継ぎメモ付き）

作成: 2026-09-22（ブランチ `metal-local`、16 の続き）。**未完**。この文書の 0 節は次のセッションへの引き継ぎで、
完了時に消して 1 節以降を正式な実験記録に整える。

**実測** = 数字を取ったもの。**導出** = 実測からの計算。**未確認** = 根拠なし。

## 0. 引き継ぎ（2026-09-22 08:40 時点、M3 Pro の再起動前）

### 0-1. 新しいセッションで最初にやること

```bash
# 1) mini のジョブ状況（nohup なので MBP の再起動とは無関係に走り続けている）
ssh mh@mhnoMac-mini.local 'for s in set2 set3; do echo "== $s"; cut -c1-235 ~/probe_logs/${s}_summary.txt; done;
  pgrep -fl "probe_ane_shapes.py --tag" | head -1; ps -axo pcpu,etime,comm | grep ANECompilerService | grep -v grep'
# 2) 結果 JSON を手元に回収して一覧
rsync -a mh@mhnoMac-mini.local:dev/Irodori-TTS/docs/experiments/results/ane_probe/ docs/experiments/results/ane_probe/
ssh mh@mhnoMac-mini.local 'cd ~/dev/Irodori-TTS/docs/experiments/results/ane_probe && python3 ~/probe_logs/tab.py'
```

`*_summary.txt` の末尾に `DONE` があればそのセットは完了。set3 は set2 の `DONE` を待って自動で始まる。

### 0-2. mini で走っている / 並んでいるもの

| セット | 本 | 内容 | 状態（08:40）|
|---|---|---|---|
| set2 | `m1_b3_ge192x12_12blk` | **12 層** × batch 3 × 12 形（192〜768）| 変換 1998 s のあと ANE コンパイル中（約 38 分経過）。`--timeout 3000` なので 08:51 頃までに終わらなければ timeout → プローブが `sudo -n pkill` で ANECompilerService を止める（**この経路の初の実地試験**。JSON の `timeout_kill` に kill 前後の状態が残る）|
| set3-1 | `m1_b1_le768x17_2blk_dlast` | 2 ブロック × batch 1 × 17 形（32〜768）、**default = 最大形** | 待ち |
| set3-2 | `m1_b3_dev3_2blk_skipload` | 2 ブロック × batch 3 × dev 3 形、`skip_model_load=True` | 待ち |
| set3-3 | `m1_b1_full23_2blk_dlast` | 2 ブロック × batch 1 × 23 形、default = 最大形 + skip_model_load | 待ち |

mini 側の道具: `~/probe_logs/run_list.sh <list.txt>`（1 行 = `tag|プローブ引数`、順に実行して `<list>_summary.txt` に 1 行ずつ、
最後に `DONE`）、`~/probe_logs/tab.py`（結果 JSON の一覧表）、各本のログは `~/probe_logs/<tag>.log`。
mini の repo には `bench/probe_ane_shapes.py`（未追跡）と `irodori_tts/ane_dit.py`（変更あり）を **scp で置いてある**。
手元で直したら再度 scp すること（mini では commit / pull していない）。環境変数は `HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1`。

### 0-3. 手元（M3 Pro）の未コミットの変更

- `bench/probe_ane_shapes.py`（新規）— 1 回 = 1 候補 package。変換 → compile_model → 子プロセスでロード（= ANE コンパイル）
  → 再ロード → **統合ログで判定**（`ane_ok / compile_failed / timeout`）。`--blocks N` で先頭 N ブロックの代理モデル、
  `--default-index`、`--skip-model-load`、`--ctx-pad-start`、`--cpu-ref`、`--kill-stale`。
  結果は `docs/experiments/results/ane_probe/<tag>.json`。
- `irodori_tts/ane_dit.py` — `export_package()` に `default_index` / `skip_model_load` 引数（既定は従来動作）。
  **ruff format をこのファイルにかけないこと**（既存コードが大量に整形されて差分が汚れる。一度やって戻した）。
- `docs/experiments/16-m1-mini.md` — 5-1 / 5-2 / 6-5 / 8 を統合ログの事実で訂正済み。**ただし 5-2 の
  「13 節 5-1 と同じ署名」は、その後の実測で誤りと分かった**（2-3）。16 節の再訂正が残っている。
- `docs/experiments/results/m1_ane_compiler_log_20260921.txt` — 09-21 夜の `full` ビルドの aned / ANECompilerService ログ。
- `docs/experiments/results/ane_probe/*.json` — ここまでのプローブ結果（M1 13 本 + M3 Pro 2 本）。
- コミットはまだしていない（ユーザーの指示待ち）。`docs/note-mac.md` はこの作業と無関係の未追跡ファイル。

### 0-4. mini の sudo

`/etc/sudoers.d/ane-kill` に `mh ALL=(root) NOPASSWD: /usr/bin/pkill -9 -x ANECompilerService` を設置済み（ユーザーが実施）。
`ssh mh@mhnoMac-mini.local 'sudo -n /usr/bin/pkill -9 -x ANECompilerService'` が通る（exit 1 = 対象なし）。
引数まで完全一致でしか通らない。root の SIGKILL で本当に止まるか、aned が再開しないかは**未確認**（0-2 の timeout で初めて分かる）。
統合ログは sudo 不要: `/usr/bin/log show --predicate 'process == "ANECompilerService" OR process == "aned"'`
（zsh では `log` が builtin なのでフルパス必須）。

### 0-5. 次にやること（順に）

1. set2 の最後と set3 の結果を 2 節の表に足す。読み方:
   - `m1_b3_ge192x12_12blk` が `ane_ok` → 12 層の `full` a_b3 が落ちた原因も「小さい形の混在」で説明がつく。
   - `*_dlast` が `ane_ok` → 原因は `EnumeratedShapes(default=最小形)`。default を最大形にするだけで 32〜1536 の全 bucket が
     M1 に載る可能性。12 層 × batch 1 × 23 形 × default 最大 で本番確認する。
   - `*_dlast` が落ちる → 「1 package 内の形の幅」が効いている。最小 bucket を 64 / 96 / 128 / 160 と上げて
     1536 と同居できる下限を 2 ブロックで二分 → 12 層で確認。あるいは小さい形と大きい形で package を分ける。
   - `*_skipload` の `convert_s` が 58 s → 10 s 前後に落ち、`load_s` が変わらなければ、`ct.convert` の隠れ ANE コンパイル説が確定。
     `export_package` の既定を `skip_model_load=True` にする（ビルド時間がほぼ半分、M3 Pro でも）。
2. M1 用の shape セットを `ane_dit.shape_packages()` に足す（例: `m1` = profile a × batch 1〜3 × 192 以上、batch 3 は ≤768）。
   profile b（ctx 256/256/64）は M1 で**未測定**。要 1 本。
3. 12 層で全 package をビルドし、6 package 同時ロードが M1 の ANE アドレス空間 3.5 GiB に収まるか（2-4）、2 ワーカー時も。
4. e2e（`bench_runtime.py` の 4 入力、16 節 3 節のコマンド）と長文（`bench_longform.py`）を新セットで再測。
   bucket が細かくなるので、dev（192/320/768）でパディングしていた分だけ速くなるはず（導出、未実測）。
5. 安全弁: ロード時に統合ログの `Model load failed` を検出したら MPS に戻す（黙って CPU で 3 倍遅くなるのを防ぐ）。
6. 16 節の再訂正（5-2 の M3 Pro との同一視、6-5、8）と、メモリ `m1-mini-tts-server.md` の更新、コミット。

## 1. 目的 / 仮説

16 節は「`full` shape セットは M1 の ANE に載らない」「OS のコンパイルキャッシュが容量で追い出される」と結論したが、
複数の要因（列挙数 / 最大形 / ctx / 層数 / OS 版 / 背景負荷）を混ぜたまま「M1 だから」に帰していた。要因を 1 つずつ測る。

## 2. ここまでの結果

### 2-1. 16 節の `full` は ANE コンパイルに失敗していた（実測、統合ログ）

6 package すべてで `RegAlloc: failed to allocate tensors with fixed L2 hint`（多数、非致命）→
`ZinCompilerCore::RunRegisterSpilling(): Register spiller failure for ANE 0` → `aned: Compilation failed` → `Model load failed`。
Core ML は黙って CPU で実行。**失敗はキャッシュされない**ので毎プロセス同じ失敗を 757〜4500 秒かけてやり直していた。
「キャッシュ容量」説は誤り。詳細は 16 節 5-2。

### 2-2. 要因の切り分け（実測、M1 mini、`probe_ane_shapes.py`、判定はログ）

| 構成 | 判定 | ANE コンパイル | 再ロード | L2 ミス |
|---|---|---:|---:|---:|
| 1 blk × b1 × 23 形（32〜1536）| 載る | 58 s | 0.05 s | 0 |
| 2 blk × b1 × 23 形（32〜1536）| **失敗** | 114 s | 114 s（失敗）| 12 |
| 2 blk × b1 × 17 形（32〜768）| **失敗** | 61 s | 60 s（失敗）| 6 |
| 2 blk × b1 × 18 形（192〜1536）| 載る | 104 s | 0.08 s | 7 |
| 2 blk × b1 × 5 形（32〜160）| 載る | 11 s | 0.05 s | 0 |
| 2 blk × b1 × 1536 単独 | 載る | 12 s | 0.05 s | 1 |
| 2 blk × b1 × 32 単独 | 載る | 2 s | 0.04 s | 0 |
| 2 blk × b3 × 17 形（32〜768）| 載る | 226 s | 0.08 s | 2 |
| 2 blk × b3 × 17 形（32〜320）| 載る | 126 s | 0.08 s | 0 |
| 2 blk × b3 × dev 3 形 | 載る | 49 s | 0.05 s | 0 |
| 2 blk × b3 × dev 3 形、ctx パディング +14〜16 | 載る | 51 s | 0.05 s | 0 |
| **12 層 × b1 × 18 形（192〜1536）** | **載る** | 715 s | 0.27 s | 7 |
| 参考（16 節）: 12 層 × b1 × 23 形 / b3 × 17 形（`full`）| 失敗 | 763 / 2088 s | 失敗 | 多数 |
| 参考（16 節）: 12 層 × `dev` 3 形 × b1〜3 | 載る | 84〜502 s | 0.2〜1.2 s | — |

ステップ時間（ANE、12 層 × b1）: 1×1536 で 532 ms。1 ブロックでは ANE / CPU = 4.9 / 11.8 ms（1×192）、51.5 / 96.8 ms（1×1536）。

読み取れること:
- **列挙数（23）も、最大形（1×1536）も、ctx パディングも、macOS 27.0 も、単独では障害ではない**（1 ブロックで全部通る）。
  16 節の「M1 の ANE は 23 bucket の列挙を受け付けない」は誤り。
- 小さい形だけ → 通る。大きい形だけ → 通る。**小さい形と大きい形を同じ package に入れ、かつ 2 層以上**で落ちる（batch 1）。
  batch 3 は同じ 17 形でも 2 ブロックなら通る（12 層では落ちた）。テンソルが小さい側で先に落ちる。
- 仮説（**未確認**、set3 で検証中）: `EnumeratedShapes(default=lst[0])` = 最小形を default にしているため、コンパイラが
  default 基準で「L2（オンチップの小メモリ）に固定」するテンソルを決め、大きい形で入りきらず spiller が詰む。
- 実用上は今の時点で、**192 未満の bucket を外せば M1 でも batch 1 は 1536 まで 12 層で載る**（実測）。
- 2 ブロック代理は 12 層の完全な代理ではない（b3 × 17 形は 12 層で失敗、2 ブロックで成功）。最終確認は必ず 12 層で。

### 2-3. M3 Pro の B=3 × 1536 は別の機序（実測、`m3_b3_full23_12blk`）

13 節 5-1 の既知の失敗を再現（ANE コンパイル 316 s → 失敗、再ロードも 315 s、CPU 実行で 3×1536 が 12.9 s/step）。
ただしエラーは `Invalid TD exists` / `Validation for DepMode failed` / `RunCodeGenObjectGen() failed`（直前に
`Latency Splitting ... into 2 parts` が 36 回）で、spiller failure は 0 件。**M1 の失敗（メモリ割り当て）とは別物**で、
外から見える症状（遅いコンパイル・CPU 実行・キャッシュされない）だけが同じ。

### 2-4. ANE のアドレス空間（実測、`ioreg -c AppleARMIODevice -r -w0 | grep -A 25 dart-ane`）

M1: `dart,t8020`、`vm-size` = 0xe0000000 = **3.5 GiB**。M3 Pro: `dart,t8110`、48 bit 級。
package 1 個の重みは 0.68 GB。6 package 同時ロード（4.1 GB）が M1 で成立するかは**未確認**（0-5 の 3）。

### 2-5. 「変換 93 分」の正体（実測 + 推定）

載った package では「変換」時間 ≈ ANE コンパイル時間（223 vs 226 s、123 vs 126 s、98 vs 104 s、12 層 b1 で 569 vs 715 s）。
MIL パス自体は 10 秒。変換中に ANECompilerService が CPU 100%（実測）。→ `ct.convert` が変換後にモデルをロードして
ANE コンパイルを 1 回走らせ、パスが違うので子プロセスのロードでもう 1 回コンパイルしている（**推定**、set3-2 で検証中）。
失敗する package では「変換」が短い（11〜12 s）理由は**未確認**。

### 2-6. 公開情報（非公式、検索結果の要点）

- エラー文言（`Register spiller failure` など）そのものの公開例はゼロ。
- tinygrad の解析・Apple の ANE Transformers 記事: 活性化はオンチップ L2 か DRAM に置かれる。小さいチャンクほど L2 に載る。
- coremltools#1954（M1 16 GB、rank-3 attention の ANE コンパイルが 5〜6 時間）、whisper.cpp#886（M1 Pro で 3〜18 時間）。
- M1 系の ANE は総量約 3 GB で黙って CPU 落ち、M2 以降は落ちない（smpanaro/more-ane-transformers NOTES.md）。
- コンパイルキャッシュは `.mlmodelc` のパス・OS build・ANE 世代に紐づく（Apple forums 762866 / 792336）→ 機械間コピー不可。
  配れるのは変換済み `.mlpackage` まで。
- macOS 14.4 以降 `launchctl kickstart -k` はシステムデーモンに使えない。kill が正攻法。
- macOS 26/27 で ANE コンパイラの失敗・出力破損の報告が複数、15.3.1 でコンパイル 19 s → 440〜560 s の報告
  （argmax-oss-swift#309）。OS 版は実在する交絡だが、mini 1 台では分離できない。
- multifunction model（coremltools 8+）は重みをハッシュで共有し function ごとにロード。固定形状 × function という逃げ道（未検証）。
