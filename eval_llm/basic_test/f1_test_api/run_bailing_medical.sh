#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export LLM_PROVIDER="bailing"
export LLM_BASE_URL="${BAILING_BASE_URL:-${ANT_LING_BASE_URL:-https://api.ant-ling.com/v1/}}"
export LLM_MODEL_NAME="${BAILING_MODEL_NAME:-${ANT_LING_MODEL_NAME:-${LING_MODEL_NAME:-AntAngelMed}}}"

if [[ -n "${BAILING_API_KEY:-}" ]]; then
  export LLM_API_KEY="${BAILING_API_KEY}"
elif [[ -n "${ANT_LING_API_KEY:-}" ]]; then
  export LLM_API_KEY="${ANT_LING_API_KEY}"
elif [[ -n "${LING_API_KEY:-}" ]]; then
  export LLM_API_KEY="${LING_API_KEY}"
fi

export BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${SCRIPT_DIR}}"
export BENCHMARK_WRITE_LEGACY_OUTPUTS="${BENCHMARK_WRITE_LEGACY_OUTPUTS:-false}"
export BENCHMARK_PREFLIGHT="${BENCHMARK_PREFLIGHT:-true}"
export BENCHMARK_LOG_REQUESTS="${BENCHMARK_LOG_REQUESTS:-true}"
export BENCHMARK_CONCURRENCY="1"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-8192}"
export LLM_REQUEST_TIMEOUT_S="${LLM_REQUEST_TIMEOUT_S:-300}"
BENCHMARK_RUNNER="${BENCHMARK_RUNNER:-${SCRIPT_DIR}/run_benchmark_api.py}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

BAILING_OUTPUT_PREFIX="${BAILING_OUTPUT_PREFIX:-bailing-medical}"
RUN_EXP_1="${RUN_EXP_1:-true}"
RUN_EXP_2="${RUN_EXP_2:-true}"

run_case() {
  local exp_dir="$1"
  local prompt_name="$2"
  local prefix="$3"

  export BENCHMARK_EXP_DIR="${exp_dir}"
  export BENCHMARK_PROMPT_NAME="${prompt_name}"
  export BENCHMARK_OUTPUT_PREFIX="${prefix}"

  local output_dir="${BENCHMARK_OUTPUT_ROOT}/${BENCHMARK_EXP_DIR}"
  local log_dir="${output_dir}/logs"
  mkdir -p "${log_dir}"

  echo "========== ${prefix} (${LLM_MODEL_NAME}) -> ${exp_dir} =========="
  echo "provider=${LLM_PROVIDER}"
  echo "base_url=${LLM_BASE_URL}"
  echo "prompt=${BENCHMARK_PROMPT_NAME}"
  echo "runner=${BENCHMARK_RUNNER}"
  echo "python=${PYTHON_BIN}"
  echo "execution=sequential"
  echo "concurrency=${BENCHMARK_CONCURRENCY}"
  echo "max_tokens=${LLM_MAX_TOKENS}"
  echo "timeout=${LLM_REQUEST_TIMEOUT_S}s"

  "${PYTHON_BIN}" "${BENCHMARK_RUNNER}" 2>&1 | tee "${log_dir}/${prefix}.log"
}

if [[ "${RUN_EXP_1}" == "true" ]]; then
  run_case \
    "exp_1" \
    "MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1" \
    "${BAILING_OUTPUT_PREFIX}"
fi

if [[ "${RUN_EXP_2}" == "true" ]]; then
  run_case \
    "exp_2" \
    "MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_NOHUMAN_V1" \
    "${BAILING_OUTPUT_PREFIX}"
fi

echo "Bailing medical supplement benchmark completed. Results: ${BENCHMARK_OUTPUT_ROOT}/exp_1 and/or ${BENCHMARK_OUTPUT_ROOT}/exp_2"
