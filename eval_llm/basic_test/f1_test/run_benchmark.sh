#!/usr/bin/env bash


#set -euo pipefail

#export VLLM_MODEL_NAME=MiniCPM-o-4.5
#export BENCHMARK_OUTPUT_MODEL_DIR=MiniCPM-o-4.5-text-chat-compatible
#export VLLM_EXPERIMENT_ID=vllm_text_chat_no_flashinfer_sampler
#export VLLM_BASE_URL=http://127.0.0.1:8000
#export VLLM_EXPERIMENT_ARGS='VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve MiniCPM-o-4.5 from ModelScope text chat compatible mode'
#export BENCHMARK_REQUEST_TIMEOUT_S=240

#python run_benchmark.py

set -euo pipefail
export VLLM_MODEL_NAME=MiniCPM-o-4.5
export BENCHMARK_OUTPUT_MODEL_DIR=MiniCPM-o-4.5-text-chat-compatible
export VLLM_EXPERIMENT_ID=vllm_text_chat_no_flashinfer_sampler
export VLLM_BASE_URL=http://127.0.0.1:8000
export VLLM_EXPERIMENT_ARGS='VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve MiniCPM-o-4.5 from ModelScope text chat compatible mode'
export BENCHMARK_REQUEST_TIMEOUT_S=240

python run_benchmark.py

