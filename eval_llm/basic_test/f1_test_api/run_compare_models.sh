#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export BENCHMARK_EXP_DIR="${BENCHMARK_EXP_DIR:-exp_1}"
export BENCHMARK_WRITE_LEGACY_OUTPUTS="${BENCHMARK_WRITE_LEGACY_OUTPUTS:-false}"
export BENCHMARK_PREFLIGHT="${BENCHMARK_PREFLIGHT:-true}"
export BENCHMARK_LOG_REQUESTS="${BENCHMARK_LOG_REQUESTS:-true}"
export BENCHMARK_CONCURRENCY=1
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-8192}"
export LLM_REQUEST_TIMEOUT_S="${LLM_REQUEST_TIMEOUT_S:-300}"

LOG_DIR="${SCRIPT_DIR}/${BENCHMARK_EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"

BAICHUAN_M3_MODEL="${BAICHUAN_M3_MODEL:-Baichuan-M3}"
BAICHUAN_M3_PLUS_MODEL="${BAICHUAN_M3_PLUS_MODEL:-Baichuan-M3-Plus}"
QWEN36_FLASH_MODEL="${QWEN36_FLASH_MODEL:-qwen3.6-flash}"
QWEN37_PLUS_MODEL="${QWEN37_PLUS_MODEL:-qwen3.7-plus}"
QWEN_BASE_URL="${QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

run_case() {
  local provider="$1"
  local model="$2"
  local prefix="$3"

  export LLM_PROVIDER="${provider}"
  export LLM_MODEL_NAME="${model}"
  export BENCHMARK_OUTPUT_PREFIX="${prefix}"

  if [[ "${provider}" == "baichuan" ]]; then
    export BAICHUAN_MODEL_NAME="${model}"
    if [[ -n "${BAICHUAN_API_KEY:-}" ]]; then
      export LLM_API_KEY="${BAICHUAN_API_KEY}"
    fi
  else
    unset BAICHUAN_MODEL_NAME
    export LLM_BASE_URL="${QWEN_BASE_URL}"
    if [[ -n "${QWEN_API_KEY:-}" ]]; then
      export LLM_API_KEY="${QWEN_API_KEY}"
    fi
  fi

  echo "========== ${prefix} (${model}) =========="
  echo "provider=${provider}"
  echo "exp_dir=${BENCHMARK_EXP_DIR}"
  echo "max_tokens=${LLM_MAX_TOKENS}"
  echo "timeout=${LLM_REQUEST_TIMEOUT_S}s"

  python run_benchmark_api.py 2>&1 | tee "${LOG_DIR}/${prefix}.log"
}

run_case "baichuan" "${BAICHUAN_M3_MODEL}" "baichuan-m3"
run_case "baichuan" "${BAICHUAN_M3_PLUS_MODEL}" "baichuan-m3-plus"
run_case "qwen" "${QWEN36_FLASH_MODEL}" "qwen3.6-flash"
run_case "qwen" "${QWEN37_PLUS_MODEL}" "qwen3.7-plus"

echo "All compare benchmarks completed. Results: ${SCRIPT_DIR}/${BENCHMARK_EXP_DIR}"
