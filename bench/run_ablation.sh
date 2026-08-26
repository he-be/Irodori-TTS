#!/usr/bin/env bash
# Sequential ablation of the local optimizations (bf16, watermark off everywhere).
# Usage: bench/run_ablation.sh [inputs...]
set -euo pipefail
cd "$(dirname "$0")/.."
INPUTS=("${@:-short long}")
if [ $# -eq 0 ]; then INPUTS=(short long); fi
OUT=docs/experiments/results
PY=.venv/bin/python
COMMON=(--precision bf16 --inputs "${INPUTS[@]}" --warmup 3 --repeats 8 --no-util)

run() {
  local tag=$1; shift
  echo "=== $tag"
  $PY bench/bench_runtime.py "${COMMON[@]}" --tag "$tag" --output "$OUT/$tag.json" "$@" 2>&1 \
    | grep -v -i "warning\|Fetching\|WeightNorm\|SDR\|dacvae:" || true
}

# A: legacy code path (all switches off), watermark off
run ablation_A_legacy --env IRODORI_OPT_REUSE_COND=0 --env IRODORI_OPT_CROP_TEXT=0 \
  --env IRODORI_OPT_FAST_SAMPLER=0 --env IRODORI_OPT_CUDA_GRAPH=0 \
  --env IRODORI_OPT_CODEC_FOLD_WN=0 --env IRODORI_OPT_REF_CACHE=0 --env IRODORI_OPT_CPU_CAST=0
# B: + condition reuse + sync-free sampler (no crop, no graph)
run ablation_B_reuse_sampler --env IRODORI_OPT_CROP_TEXT=0 --env IRODORI_OPT_CUDA_GRAPH=0 \
  --env IRODORI_OPT_CODEC_FOLD_WN=0 --env IRODORI_OPT_REF_CACHE=0
# C: + text/caption crop (no graph)
run ablation_C_crop --env IRODORI_OPT_CUDA_GRAPH=0 --env IRODORI_OPT_CODEC_FOLD_WN=0 \
  --env IRODORI_OPT_REF_CACHE=0
# D: + CUDA graph
run ablation_D_graph --env IRODORI_OPT_CODEC_FOLD_WN=0 --env IRODORI_OPT_REF_CACHE=0
# E: + codec weight-norm fold
run ablation_E_codec_fold --env IRODORI_OPT_REF_CACHE=0
# F: + reference cache (== all defaults)
run ablation_F_all
