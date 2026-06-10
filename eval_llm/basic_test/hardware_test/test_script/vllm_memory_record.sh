#!/bin/bash

set -euo pipefail

LOG_FILE="vllm.log"
BASE_OUTPUT_DIR="/root/asr_llm_agent-main/eval_llm/basic_test/hardware_test"

models=(
  "/root/autodl-tmp/models/models/cyankiwi/gemma-4-31B-it-AWQ-4bit",
  "/root/autodl-tmp/models/models/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
)

PORT=8000
TIMEOUT=300

for model_name in "${models[@]}"; do
    echo "=================================================="
    echo "🚀 正在测试模型: $model_name"

    short_model_name="${model_name##*/}"
    MODEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${short_model_name}"
    OUTPUT_FILE="${MODEL_OUTPUT_DIR}/stats.txt"

    mkdir -p "$MODEL_OUTPUT_DIR"
    rm -f "$LOG_FILE" "$OUTPUT_FILE"

    echo "⏳ 正在启动 vLLM..."
    VLLM_USE_MODELSCOPE=true vllm serve "$model_name" \
        --port "$PORT" \
        --max-model-len 8192 \
        --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
        --gpu-memory-utilization 0.6 \
        --enable-prefix-caching > "$LOG_FILE" 2>&1 &
#        --reasoning-parser qwen3 \
#        --language-model-only \

    VLLM_PID=$!

    echo "⏳ 等待模型加载并捕获 CUDA Graph ..."
    ELAPSED=0
    while ! grep -q "Graph capturing finished" "$LOG_FILE"; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))

        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "❌ vLLM 进程已退出，请检查日志: $LOG_FILE"
            tail -n 50 "$LOG_FILE" || true
            break
        fi

        if [ "$ELAPSED" -gt "$TIMEOUT" ]; then
            echo "❌ 启动超时，请检查日志: $LOG_FILE"
            kill "$VLLM_PID" 2>/dev/null || true
            wait "$VLLM_PID" 2>/dev/null || true
            break
        fi

        printf "."
    done

    if grep -q "Graph capturing finished" "$LOG_FILE"; then
        echo
        echo "✅ 加载完成，正在提取数据..."

        MODEL_MEM=$(grep "Model loading took" "$LOG_FILE" | grep -oP '\d+\.\d+ [GM]iB(?= memory)' | tail -n 1 || true)
        KV_MEM=$(grep "Available KV cache memory:" "$LOG_FILE" | grep -oP '\d+\.\d+ [GM]iB' | tail -n 1 || true)
        GRAPH_MEM=$(grep "Graph capturing finished" "$LOG_FILE" | grep -oP '(?<=took )\d+\.\d+ [GM]iB' | tail -n 1 || true)

        {
            echo "=== vLLM 资源占用报告 ($(date)) ==="
            echo "模型名称: $model_name"
            echo "模型短名: $short_model_name"
            echo "权重实际占用: ${MODEL_MEM:-未提取到}"
            echo "KV Cache 预分配: ${KV_MEM:-未提取到}"
            echo "Graph Capturing 占用: ${GRAPH_MEM:-未提取到}"
            echo "日志文件: $(pwd)/$LOG_FILE"
        } > "$OUTPUT_FILE"

        cat "$OUTPUT_FILE"
        echo "---"
        echo "数据已保存至: $OUTPUT_FILE"
    fi

    echo "🛑 关闭 vLLM 进程: $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true

    sleep 3
done

echo "✅ 全部模型测试完成"