#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PARENT_DIR}"

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

export BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${PARENT_DIR}}"
export BENCHMARK_EXP_DIR="${BENCHMARK_EXP_DIR:-$(basename "${SCRIPT_DIR}")}"
export BENCHMARK_OUTPUT_PREFIX="${BENCHMARK_OUTPUT_PREFIX:-bailing-medical}"
export BENCHMARK_PROMPT_NAME="${BENCHMARK_PROMPT_NAME:-MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1}"
export BENCHMARK_WRITE_LEGACY_OUTPUTS="${BENCHMARK_WRITE_LEGACY_OUTPUTS:-false}"
export BENCHMARK_PREFLIGHT="${BENCHMARK_PREFLIGHT:-true}"
export BENCHMARK_LOG_REQUESTS="${BENCHMARK_LOG_REQUESTS:-true}"
export BENCHMARK_CONCURRENCY="1"
export LLM_STREAM="${LLM_STREAM:-true}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-8192}"
export LLM_REQUEST_TIMEOUT_S="${LLM_REQUEST_TIMEOUT_S:-300}"
BENCHMARK_RUNNER="${BENCHMARK_RUNNER:-${PARENT_DIR}/run_benchmark_api.py}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

OUTPUT_DIR="${BENCHMARK_OUTPUT_ROOT}/${BENCHMARK_EXP_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "========== ${BENCHMARK_OUTPUT_PREFIX} (${LLM_MODEL_NAME}) -> ${BENCHMARK_EXP_DIR} =========="
echo "provider=${LLM_PROVIDER}"
echo "base_url=${LLM_BASE_URL}"
echo "prompt=${BENCHMARK_PROMPT_NAME}"
echo "runner=${BENCHMARK_RUNNER}"
echo "python=${PYTHON_BIN}"
echo "execution=sequential"
echo "concurrency=${BENCHMARK_CONCURRENCY}"
echo "stream=${LLM_STREAM}"
echo "max_tokens=${LLM_MAX_TOKENS}"
echo "timeout=${LLM_REQUEST_TIMEOUT_S}s"
echo "output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" "${BENCHMARK_RUNNER}" 2>&1 | tee "${LOG_DIR}/${BENCHMARK_OUTPUT_PREFIX}.log"

echo "Bailing medical exp_1 supplement benchmark completed. Results: ${OUTPUT_DIR}"
