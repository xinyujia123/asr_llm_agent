原始实验
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.65 