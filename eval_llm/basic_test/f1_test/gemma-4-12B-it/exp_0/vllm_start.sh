#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=/root/autodl-tmp/models/models/google/gemma-4-12B-it

VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "/root/autodl-tmp/models/models/google/gemma-4-12B-it" \
--host 0.0.0.0 \
--port 8000 \
--trust-remote-code \
--language-model-only \
--default-chat-template-kwargs '{"enable_thinking": false}'