# 实验记录报告

## 实验环境记录

| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |
| --- | --- | --- |
| exp_0 | `vllm serve hf/Sehyo-Qwen3.5-27B-NVFP4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching` | true |
| exp_1 | `vllm serve hf/Sehyo-Qwen3.5-27B-NVFP4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 64` | true |
| exp_2 | `vllm serve hf/Sehyo-Qwen3.5-27B-NVFP4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-batched-tokens 32768` | true |
| exp_3 | `vllm serve hf/Sehyo-Qwen3.5-27B-NVFP4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-batched-tokens 65536 --max-num-seqs 512` | true |


## 实验结果记录

### 稳态并发测试 (Steady RPS)

#### 目标 RPS: 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 0.9044 | 1.91% | 2.0495 | 2.7685 | 75507.19 | 2302.21 | 67.83 |
| exp_1 | 0.9043 | 1.80% | 2.0744 | 2.7588 | 73805.12 | 2303.14 | 69.09 |
| exp_2 | 0.9045 | 1.97% | 2.0684 | 2.7742 | 74055.12 | 2302.42 | 67.84 |
| exp_3 | 0.9046 | 2.27% | 2.1311 | 2.8261 | 67781.12 | 2303.94 | 69.11 |


#### 目标 RPS: 5

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 4.0060 | 6.51% | 1.9965 | 3.2915 | 75507.19 | 10145.67 | 270.33 |
| exp_1 | 4.0071 | 6.59% | 2.0109 | 3.3298 | 73807.12 | 10150.75 | 272.75 |
| exp_2 | 4.0075 | 6.54% | 1.8793 | 3.2280 | 74055.12 | 10146.19 | 267.12 |
| exp_3 | 4.0089 | 7.72% | 1.9848 | 3.3095 | 67781.12 | 10152.38 | 269.96 |


#### 目标 RPS: 10

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 8.0111 | 19.10% | 2.8323 | 4.7611 | 75507.19 | 20274.67 | 537.28 |
| exp_1 | 8.0092 | 20.23% | 3.1605 | 5.4023 | 73807.12 | 20272.58 | 540.08 |
| exp_2 | 8.0088 | 19.79% | 2.8046 | 4.7126 | 74055.12 | 20268.91 | 537.21 |
| exp_3 | 8.0132 | 22.84% | 2.9804 | 4.8703 | 67783.12 | 20286.74 | 544.39 |


#### 目标 RPS: 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 13.23 | 86.32% | 7.1899 | 10.63 | 75507.19 | 33468.67 | 879.34 |
| exp_1 | 10.96 | 31.41% | 7.4664 | 9.3690 | 73981.12 | 27718.02 | 727.72 |
| exp_2 | 13.24 | 88.08% | 7.1575 | 10.62 | 74055.12 | 33488.60 | 878.95 |
| exp_3 | 12.51 | 98.97% | 7.9320 | 11.42 | 68763.12 | 31661.28 | 834.47 |


#### 目标 RPS: 40

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 13.46 | 99.49% | 18.46 | 21.79 | 75507.19 | 34044.38 | 885.94 |
| exp_1 | 11.72 | 31.41% | 14.97 | 23.74 | 73981.12 | 29633.96 | 769.67 |
| exp_2 | 13.25 | 99.47% | 18.93 | 22.28 | 74055.12 | 33503.74 | 874.85 |
| exp_3 | 12.74 | 99.55% | 18.69 | 22.64 | 68763.12 | 32218.52 | 841.25 |


### 突发并发测试 (Burst Concurrency)

#### 目标并发数 (Concurrency): 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 1.2143 | 0.89% | 0.7964 | 0.7964 | 75507.19 | 3028.57 | 37.64 |
| exp_1 | 1.2176 | 0.84% | 0.7976 | 0.7976 | 73807.12 | 3036.57 | 37.74 |
| exp_2 | 1.0591 | 0.92% | 0.8021 | 0.8021 | 74055.12 | 2641.38 | 32.48 |
| exp_3 | 0.9778 | 1.06% | 0.8192 | 0.8192 | 67779.12 | 2438.65 | 30.31 |


#### 目标并发数 (Concurrency): 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 5.8244 | 10.64% | 2.5540 | 3.3170 | 75507.19 | 14851.83 | 450.13 |
| exp_1 | 5.8284 | 9.58% | 2.5441 | 3.3423 | 73807.12 | 14850.03 | 438.49 |
| exp_2 | 5.8254 | 11.01% | 2.5347 | 3.3266 | 74055.12 | 14853.10 | 448.85 |
| exp_3 | 5.8260 | 12.66% | 2.5490 | 3.3530 | 67779.12 | 14848.34 | 442.77 |


#### 目标并发数 (Concurrency): 50

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 11.27 | 26.01% | 2.9485 | 4.1109 | 75507.19 | 28542.44 | 763.57 |
| exp_1 | 11.26 | 24.57% | 2.9494 | 4.0893 | 74646.46 | 28516.46 | 756.01 |
| exp_2 | 11.26 | 26.97% | 2.9756 | 4.1083 | 74055.12 | 28513.43 | 761.04 |
| exp_3 | 11.27 | 30.71% | 2.9532 | 4.1578 | 67779.12 | 28530.45 | 757.33 |


#### 目标并发数 (Concurrency): 100

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 16.91 | 51.33% | 4.3213 | 5.5296 | 75507.19 | 42807.29 | 1135.76 |
| exp_1 | 12.22 | 31.29% | 5.6199 | 7.5170 | 75113.12 | 30930.87 | 816.49 |
| exp_2 | 17.12 | 53.15% | 4.3308 | 5.5345 | 74055.12 | 43341.68 | 1153.10 |
| exp_3 | 17.11 | 61.26% | 4.3217 | 5.5467 | 68918.46 | 43300.08 | 1150.45 |


#### 目标并发数 (Concurrency): 200

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 20.55 | 99.36% | 7.4350 | 8.9331 | 75509.19 | 51984.27 | 1366.27 |
| exp_1 | 11.98 | 31.41% | 10.43 | 15.69 | 75113.12 | 30297.09 | 789.36 |
| exp_2 | 19.61 | 99.74% | 7.7964 | 9.2791 | 75183.12 | 49615.93 | 1297.59 |
| exp_3 | 17.13 | 99.55% | 8.6557 | 10.51 | 69428.46 | 43317.24 | 1132.65 |


#### 目标并发数 (Concurrency): 300

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 16.75 | 99.49% | 14.08 | 17.05 | 75509.19 | 42382.87 | 1115.51 |
| exp_1 | 11.62 | 31.41% | 14.93 | 24.41 | 75113.12 | 29391.63 | 769.90 |
| exp_2 | 16.62 | 99.74% | 14.04 | 17.12 | 75183.12 | 42048.60 | 1111.03 |
| exp_3 | 15.26 | 99.29% | 15.49 | 18.61 | 69599.12 | 38619.06 | 1018.60 |

