#!/bin/bash

MODEL_NAME=/root/autodl-tmp/models/OpenBMB/MiniCPM-V-4___6

VLLM_USE_FLASHINFER_SAMPLER=0 nohup vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name MiniCPM-V-4.6 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > /root/autodl-tmp/logs/minicpm-v-4.6-vllm-no-flashinfer.log 2>&1 &