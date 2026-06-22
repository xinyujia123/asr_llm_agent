# 定义模型列表
models=('OpenBMB/MiniCPM-o-4.5')

for model_name in "${models[@]}"; do
    echo "========================================"
    echo "正在启动模型: ${model_name}"
    echo "========================================"

    # 1. 在后台启动 vLLM
    # 使用 & 将进程转入后台，并记录进程 ID ($!)
    VLLM_USE_MODELSCOPE=true VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve /root/autodl-tmp/models/models/${model_name} \
        --port 8000 \
        --max-model-len 8192 \
        --language-model-only \
        --gpu-memory-utilization 0.5 \
        --trust-remote-code \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --enable-prefix-caching &
    
    VLLM_PID=$!

    # 2. 等待服务器就绪 (Health Check)
    # vLLM 加载权重需要时间，如果不等待直接跑脚本会报错
    echo "等待 vLLM 加载完成..."
    while ! curl -s http://localhost:8000/health > /dev/null; do
        sleep 5
        echo "检查中... 服务器尚未就绪"
    done
    echo "服务器已启动！开始运行评测..."

    # 3. 运行评测脚本
    VLLM_MODEL_NAME=/root/autodl-tmp/models/models/${model_name} EXP_INDEX=1 ENABLE_SYSTEM_PROMPT_BLOCK_PADDING=true RPS_MULTIPLES=2 \
        python /root/projects/asr_llm_agent/eval_llm/basic_test/hardware_test/steady_rps_chromo.py

    VLLM_MODEL_NAME=/root/autodl-tmp/models/models/${model_name} EXP_INDEX=1 ENABLE_SYSTEM_PROMPT_BLOCK_PADDING=true RPS_MULTIPLES=2 \
        python /root/projects/asr_llm_agent/eval_llm/basic_test/hardware_test/burst_concurrency.py
    # 4. 评测结束，关闭 vLLM 进程
    echo "正在停止模型: ${model_name} (PID: ${VLLM_PID})"
    kill $VLLM_PID
    
    # 等待几秒确保显存完全释放
    sleep 10
done

echo "所有任务已完成！"