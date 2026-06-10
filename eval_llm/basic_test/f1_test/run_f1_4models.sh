#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${ROOT_DIR}/run_benchmark.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
VLLM_METRICS_URL="${VLLM_METRICS_URL:-${VLLM_BASE_URL}/metrics}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${ROOT_DIR}/logs}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-600}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

mkdir -p "${SERVER_LOG_DIR}"

VLLM_PID=""

cleanup_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  VLLM_PID=""
}

wait_vllm_ready() {
  local timeout_s="$1"
  local elapsed=0
  while true; do
    if curl -sSf "${VLLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    if (( elapsed >= timeout_s )); then
      return 1
    fi
  done
}

run_single_model() {
  local model_name="$1"
  local exp_id="$2"
  local extra_args=""
  local result_model_dir="${model_name##*/}"
  local log_file="${SERVER_LOG_DIR}/f1_${exp_id}_${result_model_dir}.log"
  local -a cmd=(
    vllm serve "${model_name}"
    --port "${VLLM_PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --reasoning-parser qwen3
    --language-model-only
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --enable-prefix-caching
  )
  echo "========== F1 MODEL ${exp_id} =========="
  echo "model=${model_name}"
  echo "extra_args=${extra_args:-<default>}"
  echo "output_model_dir=${result_model_dir}"
  cleanup_vllm

  VLLM_USE_MODELSCOPE=true "${cmd[@]}" >"${log_file}" 2>&1 &
  VLLM_PID=$!
  echo "vLLM started pid=${VLLM_PID}, log=${log_file}"

  if ! wait_vllm_ready "${SERVER_READY_TIMEOUT_S}"; then
    echo "vLLM ready timeout for model=${model_name}, log=${log_file}" >&2
    cleanup_vllm
    return 1
  fi

  export VLLM_MODEL_NAME="${model_name}"
  export VLLM_EXPERIMENT_ARGS="${extra_args}"
  export VLLM_EXPERIMENT_ID="${exp_id}"
  export BENCHMARK_OUTPUT_MODEL_DIR="${result_model_dir}"
  export VLLM_BASE_URL="${VLLM_BASE_URL}"
  export VLLM_METRICS_URL="${VLLM_METRICS_URL}"

  "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}"
  cleanup_vllm
}

trap cleanup_vllm EXIT

#MODEL_0="${MODEL_0:-Qwen/Qwen3.5-27B-FP8}"
MODEL_1="${MODEL_1:-/root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4}"
MODEL_2="${MODEL_2:-/root/autodl-tmp/models/models/Qwen/Qwen3.5-35B-A3B-FP8}"
MODEL_3="${MODEL_3:-/root/autodl-tmp/models/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"

#run_single_model "${MODEL_0}" "0"
run_single_model "${MODEL_1}" "1"
run_single_model "${MODEL_2}" "2"
run_single_model "${MODEL_3}" "3"

echo "all 4 model f1 benchmarks completed"
