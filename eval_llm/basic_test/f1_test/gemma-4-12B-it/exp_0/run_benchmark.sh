#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/asr_llm_agent/eval_llm/basic_test/f1_test
export VLLM_MODEL_NAME=gemma-4-12B-it
export BENCHMARK_OUTPUT_MODEL_DIR=gemma-4-12B-it
export VLLM_EXPERIMENT_ID=0
export VLLM_BASE_URL=http://127.0.0.1:8000
export VLLM_EXPERIMENT_ARGS=''
export BENCHMARK_REQUEST_TIMEOUT_S=240

python /root/projects/asr_llm_agent/eval_llm/basic_test/f1_test/run_benchmark.py