# 实验记录报告

## 实验环境记录

| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |
| --- | --- | --- |
| exp_0 | `vllm serve Qwen/Qwen3.5-27B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching` | true |
| exp_1 | `vllm serve Qwen/Qwen3.5-27B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --performance-mode throughput` | true |
| exp_2 | `vllm serve Qwen/Qwen3.5-27B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --kv-cache-dtype fp8` | true |
| exp_3 | `vllm serve Qwen/Qwen3.5-27B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --async-scheduling` | true |
| exp_4 | `vllm serve Qwen/Qwen3.5-27B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 512 --max-num-batched-tokens 8192` | true |


## 实验结果记录

### 稳态并发测试 (Steady RPS)

#### 目标 RPS: 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 0.8575 | 2.46% | 2.7580 | 3.6070 | 74731.12 | 2184.17 | 65.52 |
| exp_1 | 0.8575 | 2.87% | 2.7452 | 3.6011 | 75019.12 | 2183.96 | 65.51 |
| exp_2 | 0.8574 | 2.46% | 2.7569 | 3.6137 | 75623.19 | 2855.96 | 65.50 |
| exp_3 | 0.8574 | 2.18% | 2.7522 | 3.6094 | 74731.12 | 2183.78 | 65.50 |
| exp_4 | 0.8576 | 2.47% | 2.7480 | 3.6048 | 74577.12 | 2184.22 | 65.52 |


#### 目标 RPS: 5

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 3.7659 | 8.59% | 2.4517 | 4.0069 | 74731.12 | 9538.69 | 255.33 |
| exp_1 | 3.7655 | 9.52% | 2.4443 | 3.9930 | 75019.12 | 9537.74 | 255.30 |
| exp_2 | 3.7658 | 8.40% | 2.4527 | 3.9895 | 75623.19 | 12491.62 | 256.22 |
| exp_3 | 3.7655 | 8.59% | 2.4499 | 4.0007 | 74731.12 | 9537.63 | 255.30 |
| exp_4 | 3.7653 | 8.15% | 2.4508 | 3.9919 | 74577.17 | 9537.08 | 255.28 |


#### 目标 RPS: 10

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 7.6396 | 25.39% | 3.4903 | 5.6242 | 74731.12 | 19340.27 | 518.22 |
| exp_1 | 7.6422 | 27.82% | 3.5061 | 5.6458 | 75021.12 | 19346.66 | 518.34 |
| exp_2 | 7.6402 | 24.50% | 3.4223 | 5.5056 | 75625.19 | 25331.62 | 518.31 |
| exp_3 | 7.6420 | 25.51% | 3.4953 | 5.6391 | 74731.12 | 19345.91 | 518.00 |
| exp_4 | 7.6434 | 24.18% | 3.4924 | 5.6274 | 74579.19 | 19349.44 | 518.07 |


#### 目标 RPS: 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 12.73 | 92.60% | 7.4120 | 10.79 | 74731.12 | 32211.99 | 847.96 |
| exp_1 | 12.21 | 99.14% | 8.0348 | 11.46 | 75021.12 | 30902.58 | 813.74 |
| exp_2 | 12.72 | 92.70% | 7.4342 | 10.79 | 75625.19 | 42164.98 | 848.39 |
| exp_3 | 12.74 | 92.31% | 7.4696 | 10.81 | 74731.12 | 32228.99 | 849.26 |
| exp_4 | 12.74 | 88.39% | 7.4129 | 10.79 | 78675.19 | 32240.28 | 849.06 |


#### 目标 RPS: 40

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 12.71 | 99.32% | 19.43 | 23.05 | 74731.12 | 32152.93 | 838.44 |
| exp_1 | 12.53 | 99.46% | 18.76 | 22.72 | 75021.12 | 31685.60 | 828.21 |
| exp_2 | 12.82 | 99.75% | 19.22 | 22.69 | 75625.19 | 42480.72 | 846.70 |
| exp_3 | 12.58 | 99.18% | 19.56 | 23.28 | 74731.12 | 31828.41 | 831.05 |
| exp_4 | 13.00 | 99.54% | 19.11 | 22.41 | 80927.19 | 32877.65 | 858.08 |


### 突发并发测试 (Burst Concurrency)

#### 目标并发数 (Concurrency): 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 0.8176 | 0.95% | 1.0462 | 1.0462 | 74731.12 | 2039.06 | 25.35 |
| exp_1 | 0.8176 | 1.06% | 1.0419 | 1.0419 | 75021.12 | 2038.97 | 25.34 |
| exp_2 | 0.8181 | 0.82% | 1.0772 | 1.0772 | 75623.19 | 2681.72 | 25.36 |
| exp_3 | 0.8183 | 0.95% | 1.0392 | 1.0392 | 74731.12 | 2040.91 | 25.37 |
| exp_4 | 0.8166 | 0.91% | 1.0419 | 1.0419 | 80927.19 | 2036.69 | 25.32 |


#### 目标并发数 (Concurrency): 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 4.7209 | 11.32% | 3.0891 | 4.0337 | 74731.12 | 12039.21 | 366.10 |
| exp_1 | 4.7205 | 12.56% | 3.0867 | 4.0298 | 75021.12 | 12038.21 | 366.07 |
| exp_2 | 4.7222 | 11.19% | 3.1296 | 4.0698 | 75623.19 | 15744.88 | 366.21 |
| exp_3 | 4.7233 | 11.32% | 3.0774 | 4.0216 | 74731.12 | 12045.26 | 366.29 |
| exp_4 | 4.7210 | 10.81% | 3.0825 | 4.0259 | 80927.19 | 12039.45 | 366.11 |


#### 目标并发数 (Concurrency): 50

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 9.5356 | 27.69% | 3.4063 | 4.8290 | 74731.12 | 24153.70 | 647.28 |
| exp_1 | 9.5409 | 30.71% | 3.4077 | 4.8201 | 75021.12 | 24167.20 | 647.64 |
| exp_2 | 9.5390 | 27.56% | 3.4262 | 4.8155 | 75623.19 | 31642.73 | 649.41 |
| exp_3 | 9.5388 | 27.69% | 3.3945 | 4.8193 | 74731.12 | 24162.58 | 648.26 |
| exp_4 | 9.5387 | 26.43% | 3.3955 | 4.8283 | 80927.19 | 24161.47 | 647.49 |


#### 目标并发数 (Concurrency): 100

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 15.33 | 55.18% | 4.7651 | 6.2311 | 74731.12 | 38808.07 | 1036.47 |
| exp_1 | 15.19 | 61.18% | 4.7620 | 6.2208 | 75021.12 | 38461.83 | 1028.10 |
| exp_2 | 14.90 | 54.84% | 4.8942 | 6.3168 | 75623.19 | 49394.87 | 1009.79 |
| exp_3 | 15.04 | 55.16% | 4.7471 | 6.1840 | 74731.12 | 38070.72 | 1017.37 |
| exp_4 | 15.04 | 52.65% | 4.7646 | 6.1917 | 80927.19 | 38073.69 | 1019.01 |


#### 目标并发数 (Concurrency): 200

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 17.62 | 99.51% | 8.9747 | 10.61 | 74731.12 | 44573.09 | 1171.65 |
| exp_1 | 15.93 | 99.55% | 9.0208 | 11.23 | 75021.12 | 40310.79 | 1060.45 |
| exp_2 | 16.27 | 99.88% | 9.0430 | 10.97 | 75623.19 | 53926.37 | 1084.80 |
| exp_3 | 16.69 | 99.45% | 9.0424 | 10.89 | 74731.12 | 42234.13 | 1108.60 |
| exp_4 | 18.09 | 99.51% | 8.3297 | 10.05 | 80927.19 | 45764.70 | 1200.33 |


#### 目标并发数 (Concurrency): 300

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 15.30 | 99.24% | 15.26 | 18.35 | 74731.12 | 38731.06 | 1026.41 |
| exp_1 | 14.91 | 99.39% | 15.56 | 18.97 | 75021.12 | 37739.89 | 997.67 |
| exp_2 | 15.30 | 99.73% | 15.22 | 18.36 | 75623.19 | 50731.06 | 1024.84 |
| exp_3 | 15.63 | 99.37% | 14.85 | 18.12 | 74731.12 | 39546.07 | 1047.92 |
| exp_4 | 15.57 | 99.46% | 14.98 | 18.24 | 80927.19 | 39408.79 | 1042.15 |

