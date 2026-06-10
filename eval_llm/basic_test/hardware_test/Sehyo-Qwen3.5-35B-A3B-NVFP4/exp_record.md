# 实验记录报告

## 实验环境记录

| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |
| --- | --- | --- |
| exp_0 | `vllm serve hf/Sehyo-Qwen3.5-35B-A3B-NVFP4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.6 --enable-prefix-caching` | true |
| exp_1 | `vllm serve hf/Sehyo-Qwen3.5-35B-A3B-NVFP4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.8 --enable-prefix-caching` | true |
| exp_2 | `vllm serve hf/Sehyo-Qwen3.5-35B-A3B-NVFP4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.6 --enable-prefix-caching` | false |


## 实验结果记录

### 稳态并发测试 (Steady RPS)

#### 目标 RPS: 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 1.0354 | 0.44% | 0.6485 | 0.8503 | 65515.12 | 3483.58 | 80.56 |
| exp_1 | 1.0354 | 0.28% | 0.6462 | 0.8489 | 84975.12 | 3483.39 | 80.55 |
| exp_2 | 1.0145 | 0.38% | 0.6784 | 0.8441 | 65515.12 | 2348.48 | 80.35 |


#### 目标 RPS: 5

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 4.6842 | 1.45% | 0.7402 | 1.2166 | 65515.12 | 15691.99 | 322.43 |
| exp_1 | 4.6825 | 0.91% | 0.7497 | 1.2360 | 84975.12 | 15684.07 | 320.13 |
| exp_2 | 4.6862 | 1.39% | 0.7556 | 1.2318 | 65515.12 | 10778.88 | 328.10 |


#### 目标 RPS: 10

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 9.1898 | 3.39% | 0.9530 | 1.6289 | 65515.19 | 30767.39 | 627.23 |
| exp_1 | 9.1927 | 2.07% | 0.9494 | 1.5905 | 84975.12 | 30775.50 | 625.93 |
| exp_2 | 9.1915 | 3.20% | 0.9853 | 1.6159 | 65515.12 | 21120.50 | 635.13 |


#### 目标 RPS: 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 17.63 | 11.94% | 1.8821 | 3.1003 | 65515.19 | 59005.44 | 1185.36 |
| exp_1 | 18.05 | 6.64% | 1.7093 | 2.7749 | 84975.12 | 60392.63 | 1212.96 |
| exp_2 | 18.06 | 10.16% | 1.6560 | 2.6961 | 65515.12 | 41465.17 | 1222.41 |


#### 目标 RPS: 40

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 29.93 | 73.19% | 5.6243 | 9.0576 | 65515.19 | 100168.83 | 2002.80 |
| exp_1 | 29.76 | 45.73% | 5.5667 | 8.9986 | 84975.12 | 99586.66 | 1993.56 |
| exp_2 | 29.99 | 73.24% | 5.5182 | 8.9272 | 65515.12 | 68837.31 | 2017.65 |


#### 目标 RPS: 80

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 29.59 | 99.61% | 16.00 | 19.38 | 65515.19 | 99025.00 | 1985.77 |
| exp_1 | 29.80 | 99.65% | 16.52 | 21.58 | 90769.79 | 99725.55 | 1994.40 |
| exp_2 | 29.61 | 99.55% | 15.94 | 19.32 | 65515.12 | 67970.20 | 1997.53 |


### 突发并发测试 (Burst Concurrency)

#### 目标并发数 (Concurrency): 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 2.3842 | 0.42% | 0.2879 | 0.2879 | 65515.19 | 7891.68 | 73.91 |
| exp_1 | 2.3785 | 0.26% | 0.2830 | 0.2830 | 87577.19 | 7872.83 | 73.73 |
| exp_2 | 2.3814 | 0.37% | 0.2867 | 0.2867 | 65515.12 | 5379.47 | 73.82 |


#### 目标并发数 (Concurrency): 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 12.30 | 5.24% | 1.2973 | 1.5725 | 65515.19 | 41420.36 | 971.66 |
| exp_1 | 12.30 | 3.28% | 1.2408 | 1.5163 | 87577.19 | 41411.74 | 971.46 |
| exp_2 | 11.39 | 5.18% | 1.2672 | 1.5509 | 65515.12 | 26387.09 | 906.44 |


#### 目标并发数 (Concurrency): 50

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 22.42 | 12.82% | 1.4425 | 1.9628 | 65515.19 | 75105.77 | 1547.48 |
| exp_1 | 22.43 | 8.03% | 1.4745 | 1.9956 | 87577.19 | 75149.90 | 1557.47 |
| exp_2 | 22.41 | 12.76% | 1.4307 | 1.9674 | 65515.12 | 51536.20 | 1565.37 |


#### 目标并发数 (Concurrency): 100

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 32.96 | 25.48% | 2.1744 | 2.7471 | 65515.19 | 110374.22 | 2272.84 |
| exp_1 | 32.96 | 15.95% | 2.2179 | 2.8001 | 87577.19 | 110338.14 | 2249.93 |
| exp_2 | 32.97 | 25.39% | 2.1261 | 2.7161 | 65515.12 | 75758.06 | 2277.96 |


#### 目标并发数 (Concurrency): 200

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 44.21 | 50.73% | 3.5123 | 4.2509 | 65515.19 | 147959.73 | 2986.13 |
| exp_1 | 44.30 | 31.78% | 3.4948 | 4.2449 | 87577.19 | 148273.37 | 2988.02 |
| exp_2 | 44.97 | 50.66% | 3.4203 | 4.1546 | 65515.12 | 103250.45 | 3052.36 |


#### 目标并发数 (Concurrency): 300

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 49.70 | 75.99% | 4.8369 | 5.7162 | 65515.19 | 166367.07 | 3379.13 |
| exp_1 | 48.26 | 47.61% | 4.7599 | 5.6654 | 87577.19 | 161558.74 | 3261.09 |
| exp_2 | 51.93 | 75.93% | 4.5823 | 5.4806 | 65515.12 | 119290.15 | 3547.93 |

