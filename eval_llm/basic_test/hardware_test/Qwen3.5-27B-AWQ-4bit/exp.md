exp\_0:
默认vllm参数，开启padding
VLLM\_USE\_MODELSCOPE=true vllm serve cyankiwi/Qwen3.5-27B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching

exp\_1:
默认vllm参数+ --dtype half，开启padding
VLLM\_USE\_MODELSCOPE=true vllm serve cyankiwi/Qwen3.5-27B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching  --dtype half

exp\_2:
默认vllm参数上修改utilization，开启padding
VLLM\_USE\_MODELSCOPE=true vllm serve cyankiwi/Qwen3.5-27B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.9 --enable-prefix-caching

exp\_3:
默认vllm参数上修改utilization，开启padding
VLLM\_USE\_MODELSCOPE=true vllm serve cyankiwi/Qwen3.5-27B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.48 --enable-prefix-caching

<br />

实验结果：exp1和exp2没有明显区别，exp3效果略好

和fp8版本对比：\
在高并发情况下，rps略有提升，但是提升不明显。

在低并发情况下，rps提升很大，不管是相对gptq还是相对fp8
