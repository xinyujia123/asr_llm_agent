#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/asr_llm_agent-main/eval_llm/basic_test/hardware_test"
STEADY_SCRIPT="${ROOT_DIR}/run_hardware_benchmark_steady_rps.py"
BURST_SCRIPT="${ROOT_DIR}/run_hardware_benchmark_burst_concurrency.py"

MODEL_NAME="${VLLM_MODEL_NAME:-/root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4}"
RESULT_MODEL_DIR="${MODEL_NAME##*/}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${ROOT_DIR}/test_script/logs}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-600}"
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

run_single_experiment() {
  local exp_index="$1"
  local extra_args="$2"
  local log_file="${SERVER_LOG_DIR}/vllm_exp_${exp_index}.log"
  local -a cmd=(
    vllm serve "${MODEL_NAME}"
    --port "${VLLM_PORT}"
    --max-model-len 8192
    --reasoning-parser qwen3
    --language-model-only
    --gpu-memory-utilization 0.7
    --enable-prefix-caching
  )
  if [[ -n "${extra_args}" ]]; then
    read -r -a parsed_extra <<< "${extra_args}"
    cmd+=("${parsed_extra[@]}")
  fi

  echo "========== EXP ${exp_index} =========="
  echo "model=${MODEL_NAME}"
  echo "result_dir_model=${RESULT_MODEL_DIR}"
  echo "extra_args=${extra_args:-<default>}"
  cleanup_vllm

  VLLM_USE_MODELSCOPE=true "${cmd[@]}" >"${log_file}" 2>&1 &
  VLLM_PID=$!
  echo "vLLM started pid=${VLLM_PID}, log=${log_file}"

  if ! wait_vllm_ready "${SERVER_READY_TIMEOUT_S}"; then
    echo "vLLM ready timeout in exp ${exp_index}, log=${log_file}" >&2
    cleanup_vllm
    return 1
  fi

  export VLLM_MODEL_NAME="${MODEL_NAME}"
  export VLLM_EXPERIMENT_ARGS="${extra_args}"
  export BENCHMARK_OUTPUT_MODEL_DIR="${RESULT_MODEL_DIR}"
  export EXP_INDEX="${exp_index}"
  export VLLM_BASE_URL="${VLLM_BASE_URL}"
  export VLLM_METRICS_URL="${VLLM_BASE_URL}/metrics"

  python "${STEADY_SCRIPT}"
  python "${BURST_SCRIPT}"

  cleanup_vllm
}

trap cleanup_vllm EXIT

EXTRA_ARGS_0=""
EXTRA_ARGS_1="--performance-mode throughput"
EXTRA_ARGS_2="--kv-cache-dtype fp8"
EXTRA_ARGS_3="--async-scheduling"
EXTRA_ARGS_4="--max-num-seqs 512 --max-num-batched-tokens 8192"
EXTRA_ARGS_5="--quantization gptq_marlin"
EXTRA_ARGS_6="--quantization moe_wna16"

run_single_experiment 0 "${EXTRA_ARGS_0}"
run_single_experiment 1 "${EXTRA_ARGS_1}"
run_single_experiment 2 "${EXTRA_ARGS_2}"
run_single_experiment 3 "${EXTRA_ARGS_3}"
run_single_experiment 4 "${EXTRA_ARGS_4}"
run_single_experiment 5 "${EXTRA_ARGS_5}"
run_single_experiment 6 "${EXTRA_ARGS_6}"

echo "all experiments completed"
