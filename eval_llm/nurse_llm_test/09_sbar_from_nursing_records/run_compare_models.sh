#!/usr/bin/env bash
set -euo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCENARIO_DIR}/.." && pwd)"
RUNNER="${RUNNER:-${ROOT_DIR}/run_nurse_scenario_api.py}"

export BENCHMARK_PREFLIGHT="${BENCHMARK_PREFLIGHT:-true}"
export BENCHMARK_LOG_REQUESTS="${BENCHMARK_LOG_REQUESTS:-true}"
export BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-10}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"
export LLM_REQUEST_TIMEOUT_S="${LLM_REQUEST_TIMEOUT_S:-300}"
export BAICHUAN_WITH_SEARCH_ENHANCE="${BAICHUAN_WITH_SEARCH_ENHANCE:-false}"

BAICHUAN_M3_MODEL="${BAICHUAN_M3_MODEL:-Baichuan-M3}"
BAICHUAN_M3_PLUS_MODEL="${BAICHUAN_M3_PLUS_MODEL:-Baichuan-M3-Plus}"
QWEN36_FLASH_MODEL="${QWEN36_FLASH_MODEL:-qwen3.6-flash}"
QWEN37_PLUS_MODEL="${QWEN37_PLUS_MODEL:-qwen3.7-plus}"
QWEN_BASE_URL="${QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

run_case() {
  local prompt_name="$1"
  local prompt_file="$2"
  local provider="$3"
  local model="$4"
  local prefix="$5"

  local output_dir="${SCENARIO_DIR}/results/${prompt_name}"
  local log_dir="${SCENARIO_DIR}/logs/${prompt_name}"
  mkdir -p "${output_dir}" "${log_dir}"

  export LLM_PROVIDER="${provider}"
  export LLM_MODEL_NAME="${model}"

  if [[ "${provider}" == "baichuan" ]]; then
    export BAICHUAN_MODEL_NAME="${model}"
    unset LLM_BASE_URL || true
    if [[ -n "${BAICHUAN_API_KEY:-}" ]]; then
      export LLM_API_KEY="${BAICHUAN_API_KEY}"
    fi
  else
    unset BAICHUAN_MODEL_NAME || true
    export LLM_BASE_URL="${QWEN_BASE_URL}"
    if [[ -n "${QWEN_API_KEY:-}" ]]; then
      export LLM_API_KEY="${QWEN_API_KEY}"
    fi
  fi

  echo "========== 09_sbar_from_nursing_records / ${prompt_name} / ${prefix} (${model}) =========="
  echo "provider=${provider}"
  echo "prompt=${prompt_file}"
  echo "runner=${RUNNER}"
  echo "concurrency=${BENCHMARK_CONCURRENCY}"
  echo "max_tokens=${LLM_MAX_TOKENS}"
  echo "timeout=${LLM_REQUEST_TIMEOUT_S}s"
  echo "baichuan_with_search_enhance=${BAICHUAN_WITH_SEARCH_ENHANCE}"

  python "${RUNNER}" \
    --scenario-dir "${SCENARIO_DIR}" \
    --prompt "${prompt_file}" \
    --output-dir "${output_dir}" \
    --output-prefix "${prefix}" \
    2>&1 | tee "${log_dir}/${prefix}.log"
}

run_prompt_group() {
  local prompt_name="$1"
  local prompt_file="$2"

  run_case "${prompt_name}" "${prompt_file}" "baichuan" "${BAICHUAN_M3_MODEL}" "baichuan-m3"
  run_case "${prompt_name}" "${prompt_file}" "baichuan" "${BAICHUAN_M3_PLUS_MODEL}" "baichuan-m3-plus"
  run_case "${prompt_name}" "${prompt_file}" "qwen" "${QWEN36_FLASH_MODEL}" "qwen3.6-flash"
  run_case "${prompt_name}" "${prompt_file}" "qwen" "${QWEN37_PLUS_MODEL}" "qwen3.7-plus"
}

run_prompt_group "professional_guided" "${SCENARIO_DIR}/prompt_professional.md"
run_prompt_group "simple_guided" "${SCENARIO_DIR}/prompt_simple.md"

echo "SBAR nursing-record experiment completed. Results: ${SCENARIO_DIR}/results"
