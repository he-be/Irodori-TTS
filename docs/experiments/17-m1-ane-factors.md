# 17. M1 の ANE コンパイル失敗を要因に分ける（作業中・引き継ぎメモ付き）

作成: 2026-09-22（ブランチ `metal-local`、16 の続き）。**未完**。この文書の 0 節は次のセッションへの引き継ぎで、
完了時に消して 1 節以降を正式な実験記録に整える。

**実測** = 数字を取ったもの。**導出** = 実測からの計算。**未確認** = 根拠なし。

## 0. 引き継ぎ（2026-09-22 夕、set6 を mini に投入した直後）

### 0-1. 新しいセッションで最初にやること

```bash
# 1) mini のジョブ状況（nohup なので MBP の再起動とは無関係に走り続けている）
ssh mh@mhnoMac-mini.local 'for s in set2 set3; do echo "== $s"; cut -c1-235 ~/probe_logs/${s}_summary.txt; done;
  pgrep -fl "probe_ane_shapes.py --tag" | head -1; ps -axo pcpu,etime,comm | grep ANECompilerService | grep -v grep'
# 2) 結果 JSON を手元に回収して一覧
rsync -a mh@mhnoMac-mini.local:dev/Irodori-TTS/docs/experiments/results/ane_probe/ docs/experiments/results/ane_probe/
ssh mh@mhnoMac-mini.local 'cd ~/dev/Irodori-TTS/docs/experiments/results/ane_probe && python3 ~/probe_logs/tab.py'
```

`*_summary.txt` の末尾に `DONE` があればそのセットは完了。**現在走っているのは set6**（0-2 末尾）。

### 0-2. mini で走らせたもの（set2 / set3 とも完了、09-22 昼）

`set2_summary.txt` / `set3_summary.txt` とも末尾 `DONE`。4 本の結果は 2-2 の表と 2-5 に反映済み。

| セット | 本 | 内容 | 判定 |
|---|---|---|---|
| set2 | `m1_b3_ge192x12_12blk` | **12 層** × batch 3 × 12 形（192〜768）| **失敗**（spiller）|
| set3-1 | `m1_b1_le768x17_2blk_dlast` | 2 ブロック × batch 1 × 17 形（32〜768）、**default = 最大形** | **失敗**（spiller）|
| set3-2 | `m1_b3_dev3_2blk_skipload` | 2 ブロック × batch 3 × dev 3 形、`skip_model_load=True` | 載る |
| set3-3 | `m1_b1_full23_2blk_dlast` | 2 ブロック × batch 1 × 23 形、default = 最大形 + skip_model_load | **失敗**（spiller）|

`m1_b3_ge192x12_12blk` は 1961 s で自力で失敗し `--timeout 3000` に届かなかった。timeout kill 経路は set4 の 1 本目で
試験済み（下表、0-4）。

**set4**（09-22 昼に投入、`~/probe_logs/set4.txt`、全部 `--skip-model-load`、2 本目以降は 12 層）:

| 順 | 本 | 内容 | 目的 | 判定 |
|---|---|---|---|---|
| 1 | `m1_b1_le768x17_2blk_tmo20` | 既知の失敗形（2 blk × b1 × 17 形）を `--timeout 20` で | timeout kill 経路の試験 | **kill 成功**（0-4）|
| 2 | `m1_b3_mid6_12blk` | b3 × 6 形（192,256,320,448,576,768）| b3 の列挙数二分の中点（3 載る / 12 落ちる）| **載る**（1006 s、再ロード 0.14 s、L2 ミス 4）|
| 3 | `m1_b2_ge192x18_12blk` | b2 × 18 形（192〜1536）| batch 2（2 候補生成）は M1 で未測定 | **timeout**（3000 s、失敗ログなし、L2 ミス 6。kill 成功）|
| 4 | `m1_pb_b1_ge192x18_12blk` | profile b × b1 × 18 形 | profile b は M1 で未測定 | **載る**（722 s、再ロード 0.28 s、L2 ミス 7）|
| 5 | `m1_b3_q4_12blk` | b3 × 4 形（192,320,512,768）| 2 が落ちた場合の次点 | **載る**（658 s、再ロード 0.13 s、L2 ミス 2）|
| 6 | `m1_b3_n9_12blk` | b3 × 9 形（192〜768）| 2 が載った場合の次点 | **失敗**（spiller、1220 s、再ロードも 1215 s）|

5 と 6 は片方が冗長になるが、無人で回すので両方入れて b3 の列挙数 3 / 4 / 6 / 9 / 12 の全点を取る。

**set5**（set4 → after_set4 の `DONE` を待って自動開始、`~/probe_logs/set5.txt`、batch 2 だけ）:

| 順 | 本 | 内容 | 目的 |
|---|---|---|---|
| 1 | `m1_b2_mid6_12blk` | b2 × 6 形（192,256,320,448,576,768）| b3 で載った列挙をそのまま b2 で | **載る**（590 s、再ロード 0.14 s、L2 ミス 3）|
| 2 | `m1_b2_ge192x18_12blk_t7200` | b2 × 18 形（192〜1536）、`--timeout 7200` | 3000 s の打ち切りが「遅いだけ」か「終わらない」か | **失敗**（spiller、統合ログ 15:08:06 = 開始から約 4400 s。再ロードも同じ失敗をやり直し）|

**set5 の後に自動で走るもの**（`~/probe_logs/after_set5.sh`、結果は `after_set5_summary.txt`、最後に `DONE`）:
`bench_runtime.py` で (1) short 単独（`m1_conc_m1set_solo`）、(2) short を 2 プロセス同時（`m1_conc_m1set_a` / `_b`、
16 節 4-7 と同じ repeats 5 / cooldown 0。2 × 3 package = 重み 4 GB が ANE アドレス空間 3.5 GiB を超えるかの試験）、
(3) 新設入力 `xlong`（long の本文を約 2 倍に延ばしたもの、latent 約 1440 の見込み）を `m1`（`m1_xlong_m1set`）と
`dev`（`m1_xlong_devset`、768 超は MPS 落ち）で。`bench/bench_runtime.py` に `xlong` を足して scp 済み。

**set6**（09-22 16:54 投入、`~/probe_logs/set6.txt`、全部 12 層 + `--skip-model-load`。前提: 「最長 30 s で十分」と
ユーザーが決めた（09-22）→ latent は 750 で頭打ち、bucket は 768 までしか使われない）:

| 順 | 本 | 内容 | 目的 | 判定 |
|---|---|---|---|---|
| 1 | `m1_pb_b3_mid6_12blk` | profile b × b3 × 6 形（192〜768）| 長い本文（64 トークン超）× CFG を ANE に載せる（4-3 の穴）| **載る**（1014 s、再ロード 0.14 s、L2 ミス 4）|
| 2 | `m1_b1_le768x12_12blk` | profile a × b1 × 12 形（192〜768）| 30 s 上限で使われない 896〜1536 を外す（ビルド短縮、4-2 の 3〜5% の解消）| **載る**（330 s、再ロード 0.19 s、L2 ミス 4）|
| 3 | `m1_pb_b1_le768x12_12blk` | profile b × b1 × 12 形（192〜768）| 同上 | **載る**（332 s、再ロード 0.19 s、L2 ミス 4）|

3 本とも載った（17:26 完了）ので `shape_packages("m1")` を a_b1 / b_b1 × 12 形 + a_b2 / a_b3 / b_b3 × 6 形（5 package）に
書き換え（手元で編集、mini に scp 済み）、`~/probe_logs/after_set6.sh` を 17:27 に投入した: `build_ane.py --shapes m1`
（新規 3 package のコンパイル）→ `bench_runtime.py` 5 入力（4 入力 + `xlong`、出力 `results/m1_ane_m1v2_sway12_compile.json`）→
`bench_longform.py --tag m1_longform_ane_m1v2`。結果は `~/probe_logs/after_set6_summary.txt`、最後に `DONE`。
結果は 4-4（xlong は ANE に載り、RTF 0.634 → 0.511）。

**set4 の後に自動で走るもの（済、4 節）**（`~/probe_logs/after_set4.sh`、`set4_summary.txt` の `DONE` を待って開始、
結果は `~/probe_logs/after_set4_summary.txt` に 1 行ずつ、最後に `DONE`）:
`bench/build_ane.py --shapes m1`（3 package を 1 ワーカーで preload = 同時ロード 2 GB の試験、0-5 の 3）→
`bench_runtime.py` 4 入力（16 節 3 節と同じ引数、`IRODORI_OPT_ANE_SHAPES=m1`、出力
`results/m1_ane_m1set_sway12_compile.json`）→ `bench_longform.py --tag m1_longform_ane_m1set`。
ログは `~/probe_logs/build_m1.log` / `bench_m1.log` / `longform_m1.log`。この時点の `m1` は **a_b1 × 18 形 + b_b1 × 18 形 +
a_b3 × 6 形**（b2 なし、b_b3 なし）。3 package の同時ロードは重み 2.0 GB（導出）。set5-1 の後に a_b2 × 6 形を足した
（4 package。after_set5 のベンチは 1 候補なので a_b2 はロードされない）。
mini には手元の `ane_dit.py` / `ane_worker.py` / `opt_config.py` / `build_ane.py` / `check_ane.py` を scp 済み。
所要は合計 2.5〜3 時間の見込み（導出）。`run_list.sh` は `--timeout 3000 --kill-stale` を**引数の前**に置くように直した
（行ごとの `--timeout` で上書きできる。argparse は後勝ち）。リストは**絶対パス**で渡す（スクリプト内で cd するため）。

mini 側の道具: `~/probe_logs/run_list.sh <list.txt>`（1 行 = `tag|プローブ引数`、順に実行して `<list>_summary.txt` に 1 行ずつ、
最後に `DONE`）、`~/probe_logs/tab.py`（結果 JSON の一覧表）、各本のログは `~/probe_logs/<tag>.log`。
mini の repo には `bench/probe_ane_shapes.py`（未追跡）と `irodori_tts/ane_dit.py`（変更あり）を **scp で置いてある**。
手元で直したら再度 scp すること（mini では commit / pull していない）。環境変数は `HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1`。

### 0-3. 手元（M3 Pro）の未コミットの変更

- `bench/probe_ane_shapes.py`（新規）— 1 回 = 1 候補 package。変換 → compile_model → 子プロセスでロード（= ANE コンパイル）
  → 再ロード → **統合ログで判定**（`ane_ok / compile_failed / timeout`）。`--blocks N` で先頭 N ブロックの代理モデル、
  `--default-index`、`--skip-model-load`、`--ctx-pad-start`、`--cpu-ref`、`--kill-stale`。
  結果は `docs/experiments/results/ane_probe/<tag>.json`。
- `irodori_tts/ane_dit.py` — `export_package()` に `default_index` / `skip_model_load` 引数。**`skip_model_load` の既定を
  `True` にした**（2-5）。加えて安全弁（3 節）: `ensure_packages()` は `<stem>.ane_failed.json` マーカーのある package を
  外す（OS 版が同じ場合のみ）、`AneStepRunner._after_load()` がワーカーの `ane_failed` を受けて package を外し
  マーカーを書く、`make_context()` はその場合 None を返す（呼び出し側は既に MPS に戻る実装）。
  **ruff format をこのファイルにかけないこと**（既存コードが大量に整形されて差分が汚れる。一度やって戻した）。
- `irodori_tts/ane_worker.py` — `_load()` がロード後（2 秒超のとき = コンパイルが走ったとき）に統合ログ（`process == "aned"`）を
  見て `Model load failed` / `Compilation failed` があればモデルを捨て `ane_failed` を返す（3 節）。
- `docs/experiments/16-m1-mini.md` — 5-1 / 5-2 / 6-5 / 8 を統合ログの事実で訂正済み。**ただし 5-2 の
  「13 節 5-1 と同じ署名」は、その後の実測で誤りと分かった**（2-3）。16 節の再訂正が残っている。
- `docs/experiments/results/m1_ane_compiler_log_20260921.txt` — 09-21 夜の `full` ビルドの aned / ANECompilerService ログ。
- `docs/experiments/results/ane_probe/*.json` — ここまでのプローブ結果（M1 13 本 + M3 Pro 2 本）。
- 上記はすべて 5e0b2f8 でコミット済み（09-22 16:50）。`docs/note-mac.md` はこの作業と無関係の未追跡ファイル。

### 0-4. mini の sudo

`/etc/sudoers.d/ane-kill` に `mh ALL=(root) NOPASSWD: /usr/bin/pkill -9 -x ANECompilerService` を設置済み（ユーザーが実施）。
`ssh mh@mhnoMac-mini.local 'sudo -n /usr/bin/pkill -9 -x ANECompilerService'` が通る（exit 1 = 対象なし）。
引数まで完全一致でしか通らない。**実測（set4-1）**: 20 秒で子プロセスを打ち切り → `pkill_rc` 0 → ANECompilerService は
2 秒後も 22 秒後も不在、次のジョブは新しい ANECompilerService で始まった。root の SIGKILL で止まり、aned は勝手に再開しない。
統合ログは sudo 不要: `/usr/bin/log show --predicate 'process == "ANECompilerService" OR process == "aned"'`
（zsh では `log` が builtin なのでフルパス必須）。

### 0-5. 次にやること（順に）

1. （済）set2 / set3 の結果を 2 節に反映した。分岐の結果は 2-2 / 2-5 のとおり。
   - batch 1 の 192 未満の下限探索は**やらない**（short 入力 latent 180 は 192 に丸まるだけで損が小さく、2 ブロック代理が
     使えないので 12 層で数本要る。費用対効果が低い）。
   - （済）batch 3 の列挙数: 6 形は載り 9 形は落ちた → `S_BUCKETS_M1_B3` は 6 形で確定（7 / 8 形は未測定、価値が低い）。
   - （済）`export_package` の既定を `skip_model_load=True` にした。
2. （済）M1 用の shape セット `m1` を `ane_dit.shape_packages()` に足した: a_b1 / b_b1 = 192〜1536 の 18 形、
   a_b2 / a_b3 = 192〜768 の 6 形（`S_BUCKETS_M1_B23`）、b_b2 / b_b3 なし（未測定。長リファレンス × 2 候補 / CFG 3 分岐は MPS）。`opt_config.py` と `bench/build_ane.py` / `check_ane.py` の choices にも `m1` を足した。
3. （済）`m1` の 3 package を 12 層でビルド: 変換 3 × 30 s（`skip_model_load` 既定化後）、ワーカーでの ANE コンパイル
   a_b1 712 s / b_b1 721 s / a_b3 995 s、合計 2564 s（43 分）。**3 package（重み 2.0 GB）の同時ロードは失敗なし**
   （`ane_failed` なし、実測 12:53〜13:36）。2 ワーカー時（4 GB > 3.5 GiB）は after_set5 で自動測定。
4. （済、4 節）e2e と長文を `m1` セットで再測 → dev と同速。「細かい bucket で速くなる」は成り立たなかった（4 入力は同じ
   bucket に丸まる。長文はセグメントごとに相殺）。`m1` の利点は守備範囲（batch 1 の 768 超、profile b）で、
   768 超は after_set5 の `xlong` で測る。
5. （済、3 節）安全弁。
6. （16 節の再訂正は済）メモリ `m1-mini-tts-server.md` の更新、コミット。
7. （済、0-2 の set6 と 4-4）b_b3 × 6 形と b1 / b_b1 の 768 までの 12 形が載り、`m1` を 5 package に書き換えて再測した。
   (a) 再起動後の 2 ワーカー再測も済（4-2-1）: 16 節 4-7 の 1.97× は測定窓のずれで、実際は 1.14×。
8. （済、09-22 夜、ユーザーが「auto」を選択）`opt_config.py` に `chip_name()` / `is_m1_chip()` /
   `default_ane_shapes()` / `default_ane_gpu_branches()` を足し、`sysctl machdep.cpu.brand_string` が **無印 `Apple M1`
   に完全一致**するときだけ shape セット `m1`・GPU 分岐 0 を既定にした。3 つの `gradio_app*.py` の `setdefault` も
   この関数を使う。環境変数を明示すればそちらが勝つ。M1 Pro / Max / Ultra・M2 系は未測定なので M3 Pro の既定のまま
   （`full` が載らなければ安全弁が効く）。実機確認: M3 Pro → `full` / 1、mini（Apple M1）→ `m1` / 0、環境変数なしの
   short ベンチで `runner ready (m1, ne, 5 packages)`・`gpu_branches=0`・2630 ms（RTF 0.365）。

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
| 2 blk × b3 × dev 3 形、`skip_model_load` | 載る | 51 s | 0.05 s | 0 |
| 2 blk × b3 × dev 3 形、ctx パディング +14〜16 | 載る | 51 s | 0.05 s | 0 |
| 2 blk × b1 × 17 形（32〜768）、**default = 最大形** | **失敗** | 61 s | 60 s（失敗）| 6 |
| 2 blk × b1 × 23 形（32〜1536）、**default = 最大形** | **失敗** | 114 s | 114 s（失敗）| 12 |
| **12 層 × b1 × 18 形（192〜1536）** | **載る** | 715 s | 0.27 s | 7 |
| **12 層 × b3 × 12 形（192〜768）** | **失敗** | 1961 s | 1944 s（失敗）| 14 |
| **12 層 × b3 × 6 形（192,256,320,448,576,768）** | **載る** | 1006 s | 0.14 s | 4 |
| 12 層 × b3 × 4 形（192,320,512,768） | 載る | 658 s | 0.13 s | 2 |
| **12 層 × b3 × 9 形（192〜768）** | **失敗** | 1220 s | 1215 s（失敗）| 8 |
| **12 層 × b2 × 18 形（192〜1536）** | **失敗** | 4430 s（timeout 3000 s では打ち切り）| 4401 s（失敗）| 16 |
| **12 層 × b1 × profile b × 18 形（192〜1536）** | **載る** | 722 s | 0.28 s | 7 |
| 12 層 × b2 × 6 形（192〜768） | 載る | 590 s | 0.14 s | 3 |
| **12 層 × b3 × profile b × 6 形（192〜768）** | **載る** | 1014 s | 0.14 s | 4 |
| 12 層 × b1 × 12 形（192〜768） | 載る | 330 s | 0.19 s | 4 |
| 12 層 × b1 × profile b × 12 形（192〜768） | 載る | 332 s | 0.19 s | 4 |
| 参考（16 節）: 12 層 × b1 × 23 形 / b3 × 17 形（`full`）| 失敗 | 763 / 2088 s | 失敗 | 多数 |
| 参考（16 節）: 12 層 × `dev` 3 形 × b1〜3 | 載る | 84〜502 s | 0.2〜1.2 s | — |

ステップ時間（ANE、12 層 × b1）: 1×1536 で 532 ms。12 層 × b3（6 形 package）: 3×192 で 105 ms、3×320 で 198 ms、
3×768 で 602 ms（16 節 4-3 の dev 3 形での 3×180 = 102 ms と同水準、bucket を増やしても形あたりの速度は変わらない）。
12 層 × b1 × profile b（ctx 256/256/64）: 1×192 で 49 ms、1×576 で 159 ms、1×1408 で 520 ms。
12 層 × b2（6 形）: 2×192 で 67 ms、2×768 で 387 ms。12 層 × b3 × profile b（6 形）: 3×192 で 139 ms、3×768 で 720 ms
（profile a の 105 / 602 ms より ctx 分だけ重い）。12 層 × b1 × 12 形（≤768）: 1×192 で 37 ms、1×768 で 197 ms、
profile b では 49 / 228 ms（18 形 package の 1×192 = 49 ms と同じ。列挙数を 18 → 12 に減らしても形あたりの速度は変わらない）。1 ブロックでは ANE / CPU = 4.9 / 11.8 ms（1×192）、51.5 / 96.8 ms（1×1536）。

読み取れること:
- **列挙数（23）も、最大形（1×1536）も、ctx パディングも、macOS 27.0 も、単独では障害ではない**（1 ブロックで全部通る）。
  16 節の「M1 の ANE は 23 bucket の列挙を受け付けない」は誤り。
- 小さい形だけ → 通る。大きい形だけ → 通る。**小さい形と大きい形を同じ package に入れ、かつ 2 層以上**で落ちる（batch 1）。
  batch 3 は同じ 17 形でも 2 ブロックなら通る（12 層では落ちた）。テンソルが小さい側で先に落ちる。
- **`EnumeratedShapes` の default を最小形から最大形に変えても何も変わらない**（実測）。17 形で 60.73 → 60.88 s、
  23 形で 113.85 → 114.21 s、L2 ミスも 6 → 6 / 12 → 12 で同数、どちらも spiller failure 2 件で失敗。
  「default 基準で L2 に固定するテンソルを決めているから」という仮説（前版）は**棄却**。
- **batch 3 の 12 層は、小さい形を全部外しても落ちる**（実測、192〜768 の 12 形）。同じ 12 形の下限 192 は batch 1 の
  18 形（192〜1536）が 12 層で載る条件と同じなので、batch 3 の失敗は「小さい形の混在」では説明できない。
  batch 3 で載った実測は 12 層 × dev 3 形（16 節）、**12 層 × 6 形（set4-2）**、2 ブロック × 17 形。層数と列挙数の両方が効いている。
  **12 層 × batch 3 の上限は 6 形と 9 形の間**（3 / 4 / 6 載る、9 / 12 落ちる。7 / 8 は未測定、`m1` は 6 形で確定）。
  失敗して CPU に落ちた package の step は 3×192 で 933 ms、3×768 で 5349 ms（ANE の 105 / 602 ms の 9 倍、実測）。
- **batch 2 × 18 形も同じ spiller failure で落ちる。ただし失敗に届くまでが遅い**（約 4400 s。統合ログ 13:54 開始 → 15:06
  `Register spiller failure` → 15:08 `Model load failed`）。timeout 3000 s の 1 本目はこの失敗に届く前に切っていただけで、
  「batch 2 はコンパイルが終わらない」という前版の読みは誤り。16 節 5-1 の `full` a_b2 = 4474 s も同じ失敗。
  batch 1 の同じ 18 形が 715 s で載るのに対し、batch 2 は落ちるまでに 6 倍かかる（なぜ遅いかは**未確認**）。
  同じ 6 形（192〜768）なら batch 2 は 590 s で載る（set5-1）→ **`m1` の a_b2 は 6 形**。6 と 18 の間は未測定
  （2 候補生成で 768 超が要る場面は少なく、1 本 1 時間超なので後回し）。
  採用の判断は「載るか」で決めており、コンパイル時間は導入時の一回きりのコスト（載れば再ロード 0.1〜0.3 s）として別に扱う。
- **profile b（ctx 256/256/64）は batch 1 なら 18 形が 12 層で載る**（実測、722 s。profile a の 715 s と同じ）。ctx の幅は
  障害ではない。profile b × batch 3 は未測定。
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
ANE コンパイルを 1 回走らせ、パスが違うので子プロセスのロードでもう 1 回コンパイルしている。

`skip_model_load=True` で**確定**（実測）: 2 blk × b3 × dev 3 形で `convert_s` 58.4 → **7.4 s**、`load_s` は 49.2 → 51.2 s で不変、
判定も `ane_ok` のまま。ピーク RSS も 1881 → 1618 MiB。失敗する package でも同じで、2 blk × b1 × 23 形は
`convert_s` 12.4 → 7.1 s、`load_s` 113.9 → 114.2 s、判定は `compile_failed` のまま。
→ 変換時間の大半は `ct.convert` 内の ANE コンパイルであり、これは捨ててよい（子プロセスのロードで同じものをやり直すため）。
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

## 3. 安全弁: ANE コンパイル失敗を検出して MPS に戻す（実装、09-22）

16 節 5 節の被害は「失敗が黙って CPU 実行になり（MPS の 3 倍遅い）、失敗はキャッシュされないので次のプロセスも同じ
60〜4500 秒を払う」こと。Core ML は API 上はエラーを返さないので、検出は統合ログしかない。

- `ane_worker.py` の `_load()`: `CompiledMLModel` のロードが **2 秒を超えたら**（キャッシュ済みのロードは M1 の 12 層でも
  ≤ 1.2 秒、16 節）、ロード開始〜終了の窓で `/usr/bin/log show --predicate 'process == "aned"'` を引き、
  `Model load failed` / `Compilation failed` を数える。あればモデルを捨てて `ane_failed` を返す。`ctx` の場合は
  コンテキストも保存しない。`compute_units` が `ne` / `all` のときだけ。
- `ane_dit.py` の `AneStepRunner._after_load()`: `ane_failed` を受けたら package を `self.packages` から外し
  （`find_shape` がその形を返さなくなる → `make_context` が None → サンプラーは既存の実装で MPS に戻る）、
  `<stem>.ane_failed.json` を書く（OS 版、ロード秒、ログの件数）。`ensure_packages()` は次のプロセスでこのマーカーを見て
  package を外す（OS 版が同じ場合のみ。OS 更新でコンパイラが変わる可能性を残す）。
- 限界: 窓ベースなので、2 ワーカーが同時にコンパイルしていると片方の失敗が両方に付く（aned のログはモデル名が
  `<private>`）。`log show` は数秒かかるが、コンパイルが走ったとき（分単位）だけなので無視できる。
- 実機試験: M3 Pro の既知の失敗形（b3 × 23 形 × 1536、12 層、13 節 5-1）を `AneStepRunner` に読ませて、
  1 回目で `ane_failed` + マーカー、2 回目（新プロセス）でスキップになることを確認する（結果は 3-1）。

### 3-1. 実機試験（実測、M3 Pro、09-22）

b3 × 23 形（〜1536）× 12 層を `AneStepRunner(cache_dir=<空ディレクトリ>)` に読ませた（`skip_model_load` 既定で
変換 25 s、`compile_model` 0.3 s）。

| 回 | 結果 |
|---|---|
| 1 回目 | ロード 320 s → aned ログ `Model load failed` 1 / `Compilation failed` 3 → `[ane] WARNING: ANE compile of a_b3 failed ...` → `packages` 空、`a_b3_<hash>.ane_failed.json` 作成 |
| 2 回目（新プロセス）| `[ane] skipping a_b3: ANE compile failed on this machine` → ワーカー起動 2 s、package 0 個 |

初版は 2 回目で落ちた（全 package が外れると共有ブロックのサイズ計算 `max()` が空になる）。ブロックは
フィルタ前の shape セットからサイズを取るように直した（package 0 個でも runner は立ち、全要求が MPS に行く）。
M1 で `full` を指定した場合の挙動は**未実測**だが同じ経路（初回だけ 12〜75 分の失敗コンパイルを払い、以後はスキップ）。

## 4. `m1` セットでの e2e（実測、M1 mini、09-22 13:38、16 節 3 節と同じ引数、`IRODORI_OPT_ANE_SHAPES=m1`）

`m1` = a_b1 × 18 形（192〜1536）+ b_b1 × 18 形 + a_b3 × 6 形（192,256,320,448,576,768）。ANE 全分岐 + compile。
4 入力とも `rf step on ANE + GPU`（ANE で走った、実測）。比較は 16 節 4-1 の「ANE 全分岐 + compile（dev）」。

| 入力 | latent | bucket dev → m1 | dev（16 節）| `m1` | sample_rf dev → m1 |
|---|---:|---|---:|---:|---:|
| short (7.20 s) | 180 | 192 → 192 | 2687 ms (0.373) | 2736 ms (0.380) | 1132 → 1173 ms |
| medium (11.84 s) | 296 | 320 → 320 | 4335 (0.366) | 4528 (0.382) | 1928 → 2104 |
| long (28.84 s) | 721 | 768 → 768 | 13152 (0.456) | 13482 (0.467) | 7539 → 7852 |
| caption_noref (7.32 s) | 183 | 192 → 192 | 2749 (0.376) | 2714 (0.371) | 1138 → 1130 |

**差は無い（+1〜4%、測定ぶれの範囲か僅かに遅い）**。0-5 の 4 で「bucket が細かくなる分だけ速くなるはず（導出）」と
書いたが、この 4 入力の latent 長（180 / 296 / 721 / 183）は dev の 3 bucket でも `m1` の 18 bucket でも**同じ bucket に
丸まる**ので、恩恵を受ける入力が 1 つも無い。細かい bucket の効果は bucket の間に落ちる長さ（例: latent 400 は dev で 768、
`m1` で 448）でしか出ない。長文ベンチ（セグメント長がばらつく）で見る（4-1）。
package が大きい（18 形）ことによる形あたりのペナルティは、ステップ単体（2-2: 3×192 が dev でも 6 形でも 102〜105 ms）
からは見えない。+1〜4% が package サイズ由来かぶれかは**未確認**（repeats 3）。

ロード 5.7 s、warmup 31 s（3 package ともコンパイル済みキャッシュから、実測）。

### 4-1. 長文（実測、`bench_longform.py`、12 セグメント、音声 107.1 s、比較は 16 節 4-6 の `m1_longform_ane` = dev）

| 指標 | dev | `m1` |
|---|---:|---:|
| 通し | 44.30 s（RTF 0.414）| 44.73 s（RTF 0.418）|
| 最初の音まで | 2.05 s | 2.12 s |
| セグメント中央値 | 3951 ms | 3749 ms |

セグメント別（latent は音声秒 × 25 の概算、各 1 回のみ）:

| seg | 音声 s | latent | bucket dev → m1 | dev ms | m1 ms | 比 |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 4.08 | 102 | 192 → 192 | 2046 | 2115 | 1.03 |
| 1 | 9.56 | 239 | 320 → **256** | 4109 | 3915 | 0.95 |
| 2 | 11.00 | 275 | 320 → **288** | 4213 | 4548 | 1.08 |
| 3 | 12.48 | 312 | 320 → 320 | 4559 | 4841 | 1.06 |
| 4 | 11.72 | 293 | 320 → 320 | 4454 | 4540 | 1.02 |
| 5 | 8.40 | 210 | 320 → **224** | 3779 | 3636 | 0.96 |
| 6 | 8.68 | 217 | 320 → **224** | 3848 | 3464 | 0.90 |
| 7 | 5.68 | 142 | 192 → 192 | 2576 | 2817 | 1.09 |
| 8 | 7.52 | 188 | 192 → 192 | 2832 | 2977 | 1.05 |
| 9 | 9.72 | 243 | 320 → **256** | 4053 | 3863 | 0.95 |
| 10 | 10.28 | 257 | 320 → **288** | 4157 | 4510 | 1.08 |
| 11 | 7.96 | 199 | 320 → **224** | 3673 | 3500 | 0.95 |

読み取れること:
- 通しでは**差が無い**（+1%）。bucket が 320 → 224 / 256 に下がったセグメントは 4〜10% 速いが、同じ bucket のセグメントは
  2〜9% 遅く、相殺している。288 に下がった 2 本（seg 2 / 10）が 8% 遅いのは説明がつかない（各 1 回なので**ぶれの可能性**、
  未確認）。
- セグメントの壁時間は decode（16 節 4-2: short で 1.4 s / 2.7 s）が半分を占め、RF step の bucket 差はその残り半分の
  さらに一部にしか効かない（導出）。bucket を細かくしても M1 の長文は速くならない。

### 4-2. 2 ワーカー同時実行（実測、short、repeats 5、cooldown 0、16 節 4-7 と同じ手順、09-22 16:34〜16:41）

| 構成 | 単独 | 2 プロセス同時（各中央値） | predict（ANE step 12 回） | decode |
|---|---:|---|---:|---:|
| dev、09-21（16 節 4-7、再起動直後） | 2672 ms | **2649 / 2772 ms** | 1046 → 1054 / 1097 ms | 1390 → 1390 / 1425 ms |
| `m1`、09-22 16:34 | 2743 ms | **5932 / 5882 ms** | 1099 → 2958 / 2955 ms | 1424 → 2704 / 2742 ms |
| dev、09-22 16:39（対照） | 2683 ms | **5094 / 4445 ms** | 1064 → 2502 / 2462 ms | 1406 → 2424 / 2333 ms |
| `m1`、09-22 16:41（再測） | — | **5980 / 6011 ms** | 2999 / 2961 ms | 2759 / 2772 ms |

- **今日は dev でも 2 ワーカーで各リクエストが約 2 倍遅い**（対照）。16 節 4-7 の「レイテンシを落とさずスループット 2 倍」は
  今日の機械状態では dev でも再現しない。したがって `m1` 固有の問題ではない。ANE の step も GPU の decode も両方遅くなっている
  （演算器の取り合いではなく、機械全体が遅い）。
- `m1` は dev よりさらに遅い（predict 2960 vs 2480 ms）。単独でも `m1` の step は dev より 3〜5% 遅い（1099 vs 1046〜1064 ms、
  3 回とも）。18 形 package の形あたりのコストは小さいが実在する（実測）。
- 昨日と今日の違い（実測、16:42）: 稼働 1 日 1 時間（16 節は再起動直後）、**swap 使用 1.8 GB**（16 節の 2 ワーカー時は swap 0）、
  pageouts 96883、`WallpaperAerialsExtension` + `VideoToolbox` XPC が 19 時間動き続けている（ヘッドレスなのに空撮壁紙の
  動画デコード、CPU 8% + 4%、GPU 使用は**未確認**）、`aned` は 1 日稼働。どれが効いているかは**未確認**。
  10 時間にわたる ANE コンパイルの後遺症（aned / メモリ）か、壁紙の GPU 使用かを分けるには**再起動して dev / `m1` の
  2 ワーカーを取り直す**のが最短（`~/probe_logs/after_conc.sh` をそのまま再実行できる）。sudo は pkill 限定なので再起動は
  ユーザー作業。→ 再起動後に取り直した（4-2-1）。

#### 4-2-1. 再起動直後の再測（実測、09-22 19:03〜19:05、`after_conc3.sh`、swap 0、稼働 2 分）

| 構成 | 単独 | 2 プロセス同時（各中央値） | predict（ANE step 12 回） | decode |
|---|---:|---|---:|---:|
| dev | 2618 ms | **4790 / 4404 ms** | 1023 → 2427 / 2361 ms | 1381 → 2526 / 2583 ms |
| `m1`（第 2 版、5 package） | 2629 ms | **5318 / 5185 ms** | 1023 → 2422 / 2285 ms | 1387 → 2456 / 2695 ms |

- **再起動しても 2 ワーカーは各リクエストが 1.7〜2.0 倍遅い**（dev でも）。機械状態の説は棄却。ANE step と GPU decode の
  両方が 2 倍近く伸びており、2 プロセスが同じ演算器を同じ時刻に取り合っている（ロックステップ）。
- **16 節 4-7 の「1.97×」は測定窓のずれ**（結果 JSON のタイムスタンプから導出）: 09-21 の a は load 7.6 s + warmup 19.3 s、
  b は 5.3 + 9.1 s で、b の 5 回（〜19:19:07）は a の warmup と重なり、a の 5 回（19:19:06〜19:19:19）は b の終了後に
  単独で走っていた。各プロセスの中央値が単独と同じだったのは重なっていなかったから。同日の p95（3568 / 3340 ms）に
  重なった分だけ出ている。今日の a / b は load 4.2 + warmup 10 s で揃い、同じ秒に終わっている（完全に重なった）。
- スループットは dev で 2 × 2618 / 4597 = **1.14×**、`m1` で 1.00×（導出）。16 節 4-7 の MPS のみ 1.12× と同じ水準で、
  **ANE を使っても 2 ワーカーでスループットは増えない**（同じ位相で投げた場合。到着がずれた場合の重なりは**未確認**）。
- `m1` の単独は dev と**同じ**（2629 vs 2618 ms、predict とも 1023 ms）。4-2 の「`m1` は step が 3〜5% 遅い」は
  18 形 package のときの値で、12 形にした第 2 版では消えた（package サイズか機械状態かは分離できていない）。
  2 ワーカーでの `m1` と dev の差（5318 / 5185 vs 4790 / 4404）は a / b 間のばらつき（400 ms）と同程度で、
  5 package × 2 ワーカー（重み 6.8 GB、導出）が ANE のアドレス空間 3.5 GiB を超える影響は**この数字からは見えない**。

### 4-3. `xlong`（latent 768 超のつもりだった入力、実測）

`bench_runtime.py` に足した `xlong`（269 字、109 トークン）は 2 つの理由で狙いを外した:

1. **1 リクエストは既定 `max_seconds=30.0` で 750 フレームに頭打ち**（`inference_runtime.py`、予測 1008 → 750）。
   呼び出し側が `max_seconds` を上げない限り latent 768 超は発生せず、bucket 896〜1536 は使われない。
2. **109 トークンは profile a の text 64 を超える**ので profile b が要るが、CFG は batch 3 で `m1` に b_b3 は無い（dev には
   profile b 自体が無い）。両セットとも `rf step on MPS (ANE fallback)`、RTF 0.634（sample_rf 13.2 s、decode 5.6 s）。

つまり **長い本文（64 トークン超、およそ 150 字超）は M1 では今も MPS 落ち**で、それを ANE に載せるには b_b3（profile b ×
batch 3 × 6 形）の測定が要る（未測定、1 本 20〜30 分）。長文アプリ（`gradio_app.py`）はセグメントに分けるので、セグメントが
150 字以内なら profile a で足りる（16 節 4-6 の長文ベンチは 12 セグメントとも ANE、実測）。

### 4-4. `m1` 第 2 版（5 package）での再測（実測、M1 mini、09-22 17:27〜18:12、`after_set6.sh`）

`m1` = a_b1 / b_b1 × 12 形（192〜768）+ a_b2 / a_b3 / b_b3 × 6 形。ビルド（変換 3 × 30 s + ワーカーでの ANE コンパイル
a_b1 330 s / b_b1 333 s / a_b2 595 s / a_b3 0.1 s（キャッシュ済み）/ b_b3 1014 s）は合計 2409 s = 40 分。全部を初回から
焼くと a_b3 の 1006 s を足して約 57 分（導出）。5 package の同時ロード（重み 3.4 GB、導出）は失敗なし。
ロード 5.7 s、warmup 47 s（3 package のときの 31 s から package 数分だけ増加）。

| 入力 | latent | 経路 | 第 1 版（4 節）| 第 2 版 | sample_rf |
|---|---:|---|---:|---:|---:|
| short | 180 | ANE | 2736 ms (0.380) | 2720 ms (0.378) | 1173 → 1158 ms |
| medium | 296 | ANE | 4528 (0.382) | 4484 (0.379) | 2104 → 2022 |
| long | 721 | ANE | 13482 (0.467) | 13483 (0.468) | 7852 → 7842 |
| caption_noref | 183 | ANE | 2714 (0.371) | 2734 (0.374) | 1130 → 1139 |
| **xlong**（109 トークン）| 750（1008 を頭打ち）| **MPS → ANE（b_b3）** | 19024 (0.634) | **15319 (0.511)** | 13228 → **9419** |

- 4 入力は第 1 版と**同じ**（±1%）。18 形 → 12 形で形あたりのペナルティは変わらない（ステップ単体でも同じ、2-2）。
  bucket を 768 で切った利点はビルド時間（b1 で 715 → 330 s）だけ。
- **xlong は b_b3 で ANE に載った**（`rf step on ANE + GPU`、steps 16、predict 9291 ms）。MPS 落ちの第 1 版より
  **20% 速い**（sample_rf 13.2 → 9.4 s）。同じ 16 step で long（a_b3、721 → 768）の 7842 ms より 20% 重いのは、
  profile b の ctx 分（3×768 で 720 vs 602 ms/step、2-2）そのもの。4-3 の穴は塞がった。
- 長文（12 セグメント、107.1 s）: 通し 43.86 s（RTF 0.410）、最初の音まで 2.16 s、セグメント中央値 3646 ms。第 1 版の
  44.73 s（0.418）、dev の 44.30 s（0.414）と**同じ**。セグメントは全部 56 字以下なので profile a のままで、b_b3 は使われない。

## 5. 結論（09-22 時点）

1. **16 節の「M1 の ANE には `full` が載らない」の実体**: (a) batch 1 は 192 未満の小さい bucket と大きい bucket を同じ
   package に入れると 2 層以上で spiller が落ちる（192 以上なら 1536 まで 18 形が載る）、(b) batch 3 は列挙数の上限が
   6 と 9 の間にある、(c) batch 2 も 6 形は載り 18 形は落ちる（同じ spiller だが失敗まで 74 分かかる）。列挙数 23 も
   最大形も ctx 幅（profile b）も OS 版も、単独では障害ではない。
2. **`m1` セット**（第 2 版: a_b1 / b_b1 × 12 形（192〜768）、a_b2 / a_b3 / b_b3 × 6 形。「最長 30 s で十分」をユーザーが
   決めたので 768 超の bucket は外した）は M1 の ANE に載り（5 package 同時ロードを実測、4-4）、ビルドは初回約 57 分
   （導出）。4 入力の e2e も長文も dev と**同速**。`m1` が dev に対して買うのは守備範囲で、その実利は
   (a) **長い本文（64 トークン超、約 150 字超）× CFG が b_b3 で ANE に載る**（xlong で RTF 0.634 → 0.511、4-4）、
   (b) 2 候補生成（a_b2）、(c) 長いリファレンス × CFG なし（b_b1）。b_b2（長い本文 × 2 候補）だけが未測定で MPS 落ち。
3. **安全弁**（3 節）で、載らない package は初回の失敗コンパイル 1 回だけ払って以後 MPS に固定される。
   黙って CPU で 9 倍遅くなる（2-2 の 933 ms/step）ことは無くなった。
4. M1 での推奨: 短い本文（150 字以内）だけなら `dev`（ビルド 35 分、16 節）で足りる。150 字を超える本文を 1 リクエストで
   投げるなら `m1`（5 package、約 57 分）。`full` は指定しないこと（6 package × 12〜75 分の失敗コンパイルを初回に払う）。
5. **16 節 4-7 の「2 ワーカーでスループット 2 倍」は誤り**（4-2-1）。再起動直後でも dev で各リクエスト 1.7〜1.8 倍遅く、
   スループットは 1.14×（MPS のみの 1.12× と同じ）。09-21 の値は 2 プロセスの測定窓が重なっていなかった（a の長い warmup
   の間に b が走り終えていた）。M1 で ANE を使う理由は単独のレイテンシ（RTF 0.37〜0.47）であって、並列スループットではない。
