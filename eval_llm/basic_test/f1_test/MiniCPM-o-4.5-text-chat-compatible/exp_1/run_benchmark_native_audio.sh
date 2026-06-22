#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_0.22.1

cd "$SCRIPT_DIR"
python run_benchmark_native_audio.py "$@"
