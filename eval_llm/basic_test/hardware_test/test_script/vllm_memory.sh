#!/bin/bash
LOG_FILE="vllm.log"
OUTPUT_FILE="stats.txt"

# 1. 清理旧的日志和输出文件
rm -f $LOG_FILE $OUTPUT_FILE

echo "🚀 正在启动 vLLM 并加载 Qwen3.5-35B..."

# 2. 启动 vLLM (在后台运行)
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --port 8000 \
    --max-model-len 8192 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --gpu-memory-utilization 0.65 \
    --enable-prefix-caching > $LOG_FILE 2>&1 &

VLLM_PID=$!

echo "⏳ 等待模型加载并捕获 CUDA Graph (预计需要 1-3 分钟)..."

# 3. 循环检查日志，直到看到 "Graph capturing finished"（这是最后一步）
# 如果 5 分钟还没加载完，会自动退出循环防止死循环
TIMEOUT=300
ELAPSED=0
while ! grep -q "Graph capturing finished" "$LOG_FILE"; do
    sleep 5
    ((ELAPSED+=5))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "❌ 启动超时，请检查 vllm.log"
        exit 1
    fi
    printf "."
done

echo -e "\n✅ 加载完成，正在提取数据..."

# 4. 提取信息并写入文件
{
    echo "=== vLLM 资源占用报告 ($(date)) ==="
    
    # 提取 Model Memory
    MODEL_MEM=$(grep "Model loading took" $LOG_FILE | grep -oP '\d+\.\d+ [GM]iB(?= memory)')
    echo "权重实际占用: $MODEL_MEM"
    
    # 提取 KV Cache Memory
    KV_MEM=$(grep "Available KV cache memory:" $LOG_FILE | grep -oP '\d+\.\d+ [GM]iB')
    echo "KV Cache 预分配: $KV_MEM"
    
    # 提取 Graph Capturing Memory
    GRAPH_MEM=$(grep "Graph capturing finished" $LOG_FILE | grep -oP '(?<=took )\d+\.\d+ [GM]iB')
    echo "Graph Capturing 占用: $GRAPH_MEM"
    
} > "$OUTPUT_FILE"

# 5. 打印结果到终端
cat "$OUTPUT_FILE"
echo "---"
echo "数据已保存至: $OUTPUT_FILE"
echo "vLLM 服务正在后台运行 (PID: $VLLM_PID)，如需关闭请运行: kill $VLLM_PID"