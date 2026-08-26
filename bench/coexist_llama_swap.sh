#!/usr/bin/env bash
# Start llama-swap for the co-existence experiment (experiment 10).
#
# config.yaml is used verbatim; the VRAM policy is injected through llama.cpp's
# environment variables so that the served model/settings stay untouched:
#
#   LLAMA_ARG_N_GPU_LAYERS=99   pins every layer to the GPU, which makes --fit
#                               bail out ("n_gpu_layers already set by user")
#   LLAMA_ARG_N_CPU_MOE=N       keeps the MoE experts of N layers on the CPU
#
# Without this, -fit on sizes the model against *free* VRAM at load time, so the
# VLM grabs whatever the TTS is not holding at that moment (see docs/experiments/10).
set -eu
: "${NCMOE:=11}"
export LLAMA_ARG_N_GPU_LAYERS=99
export LLAMA_ARG_N_CPU_MOE="$NCMOE"
cd /home/mh/LLM
exec ./llama-swap --config config.yaml "$@"
