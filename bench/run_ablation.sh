#!/usr/bin/env bash
# Sequential ablation of the local optimizations on the Metal build (fp16, watermark off).
# Usage: bench/run_ablation.sh [inputs...]
set -euo pipefail
cd "$(dirname "$0")/.."
INPUTS=("${@:-short long}")
if [ $# -eq 0 ]; then INPUTS=(short long); fi
OUT=docs/experiments/results
PY=.venv/bin/python
COMMON=(--precision fp16 --inputs "${INPUTS[@]}" --warmup 2 --repeats 6 --cooldown 10)

run() {
  local tag=$1; shift
  echo "=== $tag"
  $PY bench/bench_runtime.py "${COMMON[@]}" --tag "$tag" --output "$OUT/$tag.json" "$@" 2>&1 \
    | grep -v -i "warning\|Fetching\|WeightNorm\|dacvae:" || true
}

# A: legacy code path (all switches off), watermark off
run metal_ablation_A_legacy --env IRODORI_OPT_REUSE_COND=0 --env IRODORI_OPT_CROP_TEXT=0 \
  --env IRODORI_OPT_FAST_SAMPLER=0 --env IRODORI_OPT_CODEC_FOLD_WN=0 \
  --env IRODORI_OPT_REF_CACHE=0 --env IRODORI_OPT_CPU_CAST=0 --env IRODORI_OPT_ROPE_REAL=0 \
  --env IRODORI_OPT_DECODE_AUTOCAST=0
# B: + condition reuse + sync-free sampler
run metal_ablation_B_reuse_sampler --env IRODORI_OPT_CROP_TEXT=0 --env IRODORI_OPT_CODEC_FOLD_WN=0 \
  --env IRODORI_OPT_REF_CACHE=0 --env IRODORI_OPT_ROPE_REAL=0 --env IRODORI_OPT_DECODE_AUTOCAST=0
# C: + text/caption crop
run metal_ablation_C_crop --env IRODORI_OPT_CODEC_FOLD_WN=0 --env IRODORI_OPT_REF_CACHE=0 \
  --env IRODORI_OPT_ROPE_REAL=0 --env IRODORI_OPT_DECODE_AUTOCAST=0
# D: + real-valued RoPE
run metal_ablation_D_rope_real --env IRODORI_OPT_CODEC_FOLD_WN=0 --env IRODORI_OPT_REF_CACHE=0 \
  --env IRODORI_OPT_DECODE_AUTOCAST=0
# E: + codec weight-norm fold + decode autocast
run metal_ablation_E_codec --env IRODORI_OPT_REF_CACHE=0
# F: + reference cache (== all defaults)
run metal_ablation_F_all
