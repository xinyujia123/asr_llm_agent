#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-${ROOT_DIR}/run_nurse_scenario_api.py}"

export BENCHMARK_WRITE_LEGACY_OUTPUTS="${BENCHMARK_WRITE_LEGACY_OUTPUTS:-false}"
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
  local scenario_dir="$1"
  local provider="$2"
  local model="$3"
  local prefix="$4"

  local log_dir="${scenario_dir}/logs"
  local output_dir="${scenario_dir}/results"
  mkdir -p "${log_dir}" "${output_dir}"

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

  echo "========== $(basename "${scenario_dir}") / ${prefix} (${model}) =========="
  echo "provider=${provider}"
  echo "runner=${RUNNER}"
  echo "concurrency=${BENCHMARK_CONCURRENCY}"
  echo "max_tokens=${LLM_MAX_TOKENS}"
  echo "timeout=${LLM_REQUEST_TIMEOUT_S}s"
  echo "baichuan_with_search_enhance=${BAICHUAN_WITH_SEARCH_ENHANCE}"

  python "${RUNNER}" \
    --scenario-dir "${scenario_dir}" \
    --output-prefix "${prefix}" \
    2>&1 | tee "${log_dir}/${prefix}.log"
}

run_scenario() {
  local scenario_dir="$1"

  if [[ ! -f "${scenario_dir}/prompt.md" && -f "${scenario_dir}/prompt_professional.md" && -f "${scenario_dir}/prompt_simple.md" && -x "${scenario_dir}/run_compare_models.sh" ]]; then
    echo "Run custom prompt-variant scenario: ${scenario_dir}"
    "${scenario_dir}/run_compare_models.sh"
    return 0
  fi

  if [[ ! -f "${scenario_dir}/dataset.jsonl" || ! -f "${scenario_dir}/prompt.md" ]]; then
    echo "Skip ${scenario_dir}: missing dataset.jsonl or prompt.md"
    return 0
  fi

  run_case "${scenario_dir}" "baichuan" "${BAICHUAN_M3_MODEL}" "baichuan-m3"
  run_case "${scenario_dir}" "baichuan" "${BAICHUAN_M3_PLUS_MODEL}" "baichuan-m3-plus"
  run_case "${scenario_dir}" "qwen" "${QWEN36_FLASH_MODEL}" "qwen3.6-flash"
  run_case "${scenario_dir}" "qwen" "${QWEN37_PLUS_MODEL}" "qwen3.7-plus"
}

if [[ $# -gt 0 ]]; then
  run_scenario "$(cd "$1" && pwd)"
else
  for scenario_dir in "${ROOT_DIR}"/[0-9][0-9]_*; do
    [[ -d "${scenario_dir}" ]] || continue
    run_scenario "${scenario_dir}"
  done
fi

echo "All nurse scenario benchmarks completed."
