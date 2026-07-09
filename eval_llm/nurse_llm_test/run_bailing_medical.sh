#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-${ROOT_DIR}/run_nurse_scenario_api.py}"

export LLM_PROVIDER="bailing"
export LLM_BASE_URL="${BAILING_BASE_URL:-${ANT_LING_BASE_URL:-https://api.ant-ling.com/v1/}}"
export LLM_MODEL_NAME="${BAILING_MODEL:-${BAILING_MODEL_NAME:-${ANT_LING_MODEL_NAME:-${LING_MODEL_NAME:-AntAngelMed}}}}"

if [[ -n "${BAILING_API_KEY:-}" ]]; then
  export LLM_API_KEY="${BAILING_API_KEY}"
elif [[ -n "${ANT_LING_API_KEY:-}" ]]; then
  export LLM_API_KEY="${ANT_LING_API_KEY}"
elif [[ -n "${LING_API_KEY:-}" ]]; then
  export LLM_API_KEY="${LING_API_KEY}"
fi

export BENCHMARK_WRITE_LEGACY_OUTPUTS="${BENCHMARK_WRITE_LEGACY_OUTPUTS:-false}"
export BENCHMARK_PREFLIGHT="${BENCHMARK_PREFLIGHT:-true}"
export BENCHMARK_LOG_REQUESTS="${BENCHMARK_LOG_REQUESTS:-true}"
export BENCHMARK_CONCURRENCY="1"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"
export LLM_REQUEST_TIMEOUT_S="${LLM_REQUEST_TIMEOUT_S:-300}"
export BAICHUAN_WITH_SEARCH_ENHANCE="${BAICHUAN_WITH_SEARCH_ENHANCE:-false}"

BAILING_OUTPUT_PREFIX="${BAILING_OUTPUT_PREFIX:-bailing-medical}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

run_case() {
  local scenario_dir="$1"
  local prompt_file="$2"
  local output_dir="$3"
  local log_dir="$4"

  mkdir -p "${output_dir}" "${log_dir}"

  echo "========== $(basename "${scenario_dir}") / ${BAILING_OUTPUT_PREFIX} (${LLM_MODEL_NAME}) =========="
  echo "provider=${LLM_PROVIDER}"
  echo "base_url=${LLM_BASE_URL}"
  echo "prompt=${prompt_file}"
  echo "runner=${RUNNER}"
  echo "python=${PYTHON_BIN}"
  echo "execution=sequential"
  echo "concurrency=${BENCHMARK_CONCURRENCY}"
  echo "max_tokens=${LLM_MAX_TOKENS}"
  echo "timeout=${LLM_REQUEST_TIMEOUT_S}s"
  echo "output_dir=${output_dir}"

  "${PYTHON_BIN}" "${RUNNER}" \
    --scenario-dir "${scenario_dir}" \
    --prompt "${prompt_file}" \
    --output-dir "${output_dir}" \
    --output-prefix "${BAILING_OUTPUT_PREFIX}" \
    2>&1 | tee "${log_dir}/${BAILING_OUTPUT_PREFIX}.log"
}

run_scenario() {
  local scenario_dir="$1"

  if [[ -f "${scenario_dir}/prompt_professional.md" && -f "${scenario_dir}/prompt_simple.md" ]]; then
    run_case \
      "${scenario_dir}" \
      "${scenario_dir}/prompt_professional.md" \
      "${scenario_dir}/results/professional_guided" \
      "${scenario_dir}/logs/professional_guided"
    run_case \
      "${scenario_dir}" \
      "${scenario_dir}/prompt_simple.md" \
      "${scenario_dir}/results/simple_guided" \
      "${scenario_dir}/logs/simple_guided"
    return 0
  fi

  if [[ ! -f "${scenario_dir}/dataset.jsonl" || ! -f "${scenario_dir}/prompt.md" ]]; then
    echo "Skip ${scenario_dir}: missing dataset.jsonl or prompt.md"
    return 0
  fi

  run_case \
    "${scenario_dir}" \
    "${scenario_dir}/prompt.md" \
    "${scenario_dir}/results" \
    "${scenario_dir}/logs"
}

if [[ $# -gt 0 ]]; then
  run_scenario "$(cd "$1" && pwd)"
else
  for scenario_dir in "${ROOT_DIR}"/[0-9][0-9]_*; do
    [[ -d "${scenario_dir}" ]] || continue
    run_scenario "${scenario_dir}"
  done
fi

echo "Bailing medical nurse scenario benchmark completed."
