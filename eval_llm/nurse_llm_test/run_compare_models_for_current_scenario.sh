#!/usr/bin/env bash
set -euo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCENARIO_DIR}/.." && pwd)"

exec "${ROOT_DIR}/run_compare_models.sh" "${SCENARIO_DIR}"
