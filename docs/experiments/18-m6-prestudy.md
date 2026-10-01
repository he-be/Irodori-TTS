# 18. M6 Mac mini への適用性の事前検討（mflux / turbo-fieldfare の M6 実測から）

状態: **机上検討**（M6 での TTS 実測はまだ無い。M6 の GPU が別作業で埋まっているため）。
凡例: **実測** = 出典リポジトリに証拠ディレクトリ / measurements がある値、**導出** = そこから計算した値、
**未確認** = どちらにも無い値。M6 の値の出典は `~/dev/mflux/docs/16gb/`（以下 mflux）と
`~/LLM/turbo-fieldfare/docs/m6-prefill/`, `docs/investigations/`, `bench/m6/results/`（以下 tf）。

## 1. 目的

M1（16 節）と M3 Pro（12〜15 節）で最適化した `metal-local` を、M6 Mac mini（16 GB）で比較する。
その前に、先行 2 プロジェクトが M6 で測った事実を集め、**何がそのまま効き、何が逆転し、何を先に測るべきか**を決める。

## 2. M6 mini の素性（実測、tf hostinfo / mflux m7）

| 項目 | M6 mini | M1 mini（16 節） | M3 Pro（12 節） |
|---|---|---|---|
| hw.model / chip | Mac18,5 / "Apple M6" | Apple M1 | Mac15,6 / Apple M3 Pro |
| メモリ | 16 GB（`recommendedMaxWorkingSetSize` 12,124 MiB） | 16 GB（`recommended_max_memory` 12,124 MiB） | 18 GB（12,288 MiB） |
| OS | macOS 27.0 (26A428) | macOS 27.0 (26A428) | macOS 15.7.5 |
| GPU | 12 コア、`applegpu_g18g`、MTLGPUFamily apple11 / **metal4 true** | 8 コア | 18 コア、`g15s`、apple9、metal4 false |
| ANE | Core ML compute plan が **`ane_cores: 32`** を報告（16 コア × 2 基の解釈、mflux m11a:5） | 16 コア | 16 コア |
| SSD | 228 GB、読み **3.34 GB/s**（M3 Pro 6.7 の半分） | — | 1 TB、6.7 GB/s |
| ツールチェーン | CLT のみ（Xcode 無し）、システム Python 3.9.6、uv は `~/.local/bin/uv` | uv は `~/.local/bin/uv` | Xcode 26.6 |
| メモリ帯域 | **未確認**（両リポジトリとも実測なし。公称 153 GB/s @16GB） | 実測 61〜63 GB/s | 理論 150 GB/s |
| 2026-09-24 時点 | ディスク 118 GB 空き、`~/dev` に ComfyUI / mflux / tsugumi、**Irodori-TTS は未配置** | | |

ssh は `~/.ssh/config` の `m6`（Tailscale）/ `m6-lan`（192.168.0.64、1GbE）/ `m6-tb`（TB4 直結、常設しない）。

## 3. TTS に直結する M6 の実測

### 3-1. GPU の行列積は fp16 だけが速い（実測、tf gpu_peak_probe / mflux m7, m9）

| 4096³ GEMM | M6 | M3 Pro | 比 |
|---|---:|---:|---:|
| fp16（MPSMatrixMultiplication） | **18.61 TFLOP/s** | 5.94 | **3.1×** |
| fp32（同） | **4.85** | 5.13 | **0.95×** |
| fp16 / bf16（PyTorch MPS、mflux m9） | 18.72 / 18.73 | — | |
| fp16 / bf16（MLX） | 19.10 / 19.19 | | |

- GPU 各コアの **Neural Accelerator（tf では NAX と呼ぶ。ANE とは別物）** が fp16/bf16 の行列積だけを受ける。
  fp32 は従来経路のままで、**M3 Pro より 5% 遅い**。
- PyTorch MPS は何もしなくても fp16 で 18.7 が出る（mflux m9:18-45、実測）。**MPS の bf16 は fp16 と同速**。
- 形が小さいと落ちる: MLX は 2048³ で fp16 13.1 / bf16 8.0（mflux m9）。MPS 側の小形の値は未確認。
- 合成ベンチ（5120×6144 @ 6144²、mflux m7:180-187）: fp16 matmul 2.12×、sdpa 3.16×、q4 3.59×（M3 Pro 比）。
- 本番ブロック（Krea2 DiT、mflux）: fp32 活性のまま M3 Pro 比 1.48〜1.61×、bf16 化で更に 1.84×。

### 3-2. ANE は M3 Pro 並み、GPU の方が速い側に逆転する（実測、mflux m11a）

| ANE 単基、fp16 重み | TOPS |
|---|---:|
| 6144→6144（M=4126） | 6.55 |
| 6144→16384 | 7.37 |
| 16384→6144（K が大きい down） | 3.68 |
| int8 行スケール重み（gate） | 13.07 |
| W8A8（mflux m11a3） | 20.8 |

- 13 節の M3 Pro ANE 実測は fp16 で 7.3〜8.7 TFLOPS。**M6 の ANE 単基は fp16 重みでは M3 Pro とほぼ同じ**。
  一方 GPU は 5.6 → 18.6 になったので、**M3 Pro で「ANE 2×速い」だった関係が M6 では「GPU 2.5×速い」に反転する**（導出）。
- 2 基目の ANE は **別プロセスからしか使えない**（2 プロセスで +34%、1 プロセス 2 スレッドは +2%、mflux m11a:141-148）。
  TTS の worker は既に別プロセスなので、worker を 2 つ立てれば届く（未確認）。
- **ANE を GPU と同時に回すと GPU が遅くなる**（実測、mflux m11a3）: 同居する ANE 演算の種類で GPU −20%（gate/up）〜 **−60%**（down 込み）。
  演算なしの DMA だけでも −11%、CPU だけの同居でも −16%。原因はメモリ系のトラフィックで、損はデューティに比例し閾値なし。
  ANE 1 TOPS あたり GPU が失う TFLOPS は 0.27〜1.07。
- ANE の初回コンパイルは 63〜251 s / モデル、2 回目ロード 0.02 s。unified log にコンパイル失敗は無し（Krea2 の形で）。
  **TTS の `full` セットが M6 の ANE に載るかは未確認**（M1 では register spiller で失敗、17 節）。macOS は M1 mini と同じ 27.0。
- coremltools 9.0 は **Python 3.14 だとネイティブ拡張が載らない**（`_MLModelProxy` 失敗、mflux m11a:222-232）。
  TTS は `requires-python >=3.12,<3.13` なので uv が 3.12 を取り、この罠は踏まない。

### 3-3. 持続負荷で GPU クロックが 8% 落ちる（実測、mflux residue-fixes:117-139）

| 連続実行 | ms/ブロック | GPU クロック | ダイ温度 最大/平均 |
|---|---:|---:|---|
| 1 回目（15 分休止後） | 278.0 | 1734 MHz | 97.0 / 77.0 °C |
| 2 回目 | 282.8 | 1698 | 107.9 / 96.5 |
| 4 回目以降 | 300.1 | 1594 | 108.3 / 102.3 |

- 開始 59 s、ダイ 102.5 °C で最高クロックを割る。thermal pressure は `Nominal` のまま。15 分休むと戻る。
- M1 mini の 10 連発（16 節 4-6）はドリフト 0.8% だった。**M6 は 3 分原稿の連結生成（M3 Pro で 35 s）の途中でクロックが落ちる**可能性が高い（導出）。
- 温度は root 無しで読める（mflux `hid_temps.py`）。クロックと電力は `sudo powermetrics`（mflux 側で NOPASSWD 設定済み）。

### 3-4. その他

- `audiomxd` 72% + `configd` 42% が常時動いている（mflux m11a3:267）。M1 mini と同じ現象（16 節 7）。
- Metal System Trace / GPU キャプチャは Xcode が無いので M6 では取れない。
- 16 GB なので `sudo purge` が使えず、ページキャッシュを冷やせない。TTS の prebake バンドル（1.9 GB）や ANE cache（10 GB）の初回ロード時間を測るなら再起動直後で取る。
- 出力は同一機なら再現するが、M3 Pro とはビット一致しない（累積順、mflux m7:205-216）。TTS の品質比較は 14 節と同じく LSD ではなく聴感で。

## 4. TTS への適用性（導出）

### 4-1. 効くもの

1. **DiT step の GPU 実行がそのまま速くなる**。12 節で DiT step は compute-bound（190 µs/token、実効 2.8 TFLOPS = ピークの 50%）。
   `metal-local` は fp16 既定で Linear は全部 fp16 なので、コード変更なしで NAX の恩恵を受ける。
   期待値は mflux の本番ブロックの実績から **2〜3×**（fp16 matmul 2.1×、sdpa 3.2×、ブロック全体 1.5〜1.6×+bf16 1.84×）。
   TTS の行列は小さい（batch 3 × 180〜750 token × hidden）ので下限側に寄る可能性がある（未確認）。
2. **codec decode**（14 節以降の律速、M3 Pro で 463 ms、789 GFLOP、実効 1.7 TFLOPS）。conv_transpose / conv1d が
   MPS で NAX 経路に乗るかは **未確認**。乗れば DiT と同程度、乗らなければ Snake（elementwise）と同じく帯域律速で **ほぼ据え置き**。
   ここが M6 での最大の不確定要素で、最初に単体で測る。
3. **bf16** は M6 の MPS では fp16 と同速。ただし 12 節の理由（40 Euler step で fp32 から乖離）で fp16 のまま。速度上の動機は無い。

### 4-2. 逆転するもの

1. **ANE の役割**。M3 Pro では「DiT を ANE、cond 分岐 1 本を GPU」で 1.35×。M6 では GPU 単独の DiT が ANE より速く、
   さらに ANE 同居で GPU が −20〜60% 遅くなるので、**単発 request では ANE を使わない構成が既定になる見込み**。
   `opt_config.py` は chip 名の完全一致で M1 だけ分岐しており、**"Apple M6" は M3 Pro 既定（`full` + GPU 分岐 1）を受ける**。
   M6 実測後に既定を決める（ANE off、または gpu_branches 2 = ANE は 1 分岐だけ）。
2. **ANE の残る使い道**は 13 節の整理どおり「2 台目の演算器」だが、M6 では GPU への干渉を差し引く必要がある。
   候補: (a) プロセス起動の速さ（0.2 s ロード、常駐サーバーでは無意味）、(b) 15 節のセグメント単位パイプライン（DiT を ANE、decode を GPU）。
   (b) は mflux の実績（理想実装でも −6〜−8% 止まり、m11a3:226-247）から見て、TTS でも 15 節の 1.6× は出ない可能性が高い。
3. **fp32 の経路が相対的に 4× 高くつく**。M3 Pro では fp32/fp16 同速だったので放置できた fp32 の行列積が、M6 では NAX に乗らない。
   現状 `model.py` の fp32 は RMSNorm / AdaLN / RoPE の elementwise（帯域律速で影響小）だが、attention の実 dtype と
   `codec.py` の autocast 外の conv を監査する。

### 4-3. 見込みの数字（導出、短文 7.2 s、sway 12 step）

| | M3 Pro 実測（14 節） | M6 見込み |
|---|---:|---:|
| sample_rf | 523 ms（ANE + GPU cond） | 250〜375（GPU 単独、2〜3×） |
| decode_latent | 463 ms（compile 済み MPS） | 230〜460（conv が NAX に乗るかで二択） |
| wall | 1023 ms（RTF 0.14） | **500〜850 ms（RTF 0.07〜0.12）** |

5060 Ti の 450 ms（RTF 0.06）に並ぶのは decode も 2× になった場合だけ。

## 5. M6 で測る順番（GPU が空いてから）

前提: 他のモデルプロセスが動いていないことを `pgrep` で確認（tf `run_probe.sh` と同じ規則）。1 行の hostinfo を残す。

0. 配置: `git clone` → `uv sync`（Python 3.12 が取られることを確認）→ `prebake_runtime.py`。`HF_HUB_DISABLE_XET=1` は M1 mini と同じ網なら必要。
1. **GPU ピーク**（16 節 2-1 と同じ `4096³` fp32/fp16/bf16 + 512 MiB コピー）: torch 2.10 で fp16 18.6 が出るかと、
   **メモリ帯域の初めての実測**（両リポジトリに無い）。
2. **ベースライン**: `bench/bench_runtime.py --precision fp16 --cooldown 10` を MPS eager / compile、ANE off で。
   3-3 のクロック低下があるので ABAB 交互 + `hid_temps.py` でダイ温度を並記。M1 / M3 Pro / M6 の 3 機表を作る。
3. **decode 単体**: `decode_latent` の時間と実効 TFLOPS。1.7 から動かなければ conv は NAX に乗っていない → decode の次の手は M6 でも別途。
4. **ANE step 単体**: `check_ane.py --shapes dev` で ANE step vs GPU step（M1 の 4-3 と同じ表）。ここで ANE が遅い側なら 5 は簡略化。
5. **ANE 同居の干渉**: ANE worker が動いている間の GPU step 時間（mflux の −11〜60% が TTS の形でも出るか）。
6. **`full` の可否**: 30〜57 分のビルド前に `probe_ane_shapes.py` で unified log 判定（17 節の手順）。
7. 既定の決定 → `opt_config.py` に "Apple M6" の分岐を追加。
8. 持続: 3 分原稿の連結生成と 2 ワーカー同時（2 基目の ANE に届くか）。クロックと温度を並記。

## 6. 出典

- tf: `docs/m6-prefill/03-FACTS.md`, `17-NAX-NOT-USED.md`, `10-G0-BASELINE.md`, `05-RISKS.md`, `docs/investigations/M6_MAC_MINI_TARGETING.md`,
  `M6_SSD_BANDWIDTH.md`, `docs/TWO_MACHINE_DEV.md`, `bench/m6/results/2026-09-22-Mac18-5-gpu_peak_probe/`, `...-gpu_family_probe/`
- mflux: `docs/16gb/measurements/2026-09-22-m7-m6-mac-mini.md`, `m9-ceiling-probe.md`, `m8-block-profile.md`, `m11a-ane-probe.md`,
  `2026-09-23-m11a3-ane-coscan.md`, `2026-09-23-comfyui-residue-fixes.md`, `HANDOFF.md`, `runs/m6mini/*.json`
- 注意: tf `docs/mtp/26-M6-RESULTS.md` の「M6」はマイルストーン名で、M6 機とは無関係。
