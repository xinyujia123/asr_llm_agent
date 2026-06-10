# 实验记录报告

## 实验环境记录

| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |
| --- | --- | --- |
| exp_0 | `vllm serve cyankiwi/Qwen3.5-27B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching` | true |
| exp_1 | `vllm serve cyankiwi/Qwen3.5-27B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --dtype half` | true |
| exp_2 | `vllm serve cyankiwi/Qwen3.5-27B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.9 --enable-prefix-caching` | true |
| exp_3 | `vllm serve cyankiwi/Qwen3.5-27B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.5 --enable-prefix-caching` | true |


## 实验结果记录

### 稳态并发测试 (Steady RPS)

#### 目标 RPS: 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 0.9749 | 1.17% | 1.2390 | 1.6321 | 75359.19 | 2481.68 | 73.12 |
| exp_1 | 0.9751 | 1.17% | 1.2339 | 1.6223 | 75355.19 | 2482.11 | 73.13 |
| exp_2 | 0.9749 | 0.82% | 1.2394 | 1.6297 | 95057.19 | 2481.70 | 73.12 |
| exp_3 | 0.9750 | 2.03% | 1.2344 | 1.6214 | 55391.12 | 2481.87 | 73.12 |


#### 目标 RPS: 5

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 4.3583 | 3.73% | 1.2864 | 2.1118 | 75359.19 | 11037.10 | 293.32 |
| exp_1 | 4.3589 | 3.73% | 1.2791 | 2.1090 | 75355.19 | 11038.51 | 293.35 |
| exp_2 | 4.3569 | 2.62% | 1.2848 | 2.1278 | 95057.19 | 11033.52 | 293.22 |
| exp_3 | 4.3584 | 6.46% | 1.2519 | 2.0964 | 55391.12 | 11037.36 | 293.32 |


#### 目标 RPS: 10

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 8.7121 | 10.53% | 1.9004 | 3.1532 | 75359.19 | 22054.36 | 589.98 |
| exp_1 | 8.7163 | 10.59% | 1.8956 | 3.1665 | 75355.19 | 22064.54 | 589.92 |
| exp_2 | 8.7100 | 7.47% | 1.8997 | 3.1496 | 95057.19 | 22048.96 | 589.84 |
| exp_3 | 8.7153 | 18.12% | 1.8728 | 3.1073 | 55391.12 | 22061.98 | 589.85 |


#### 目标 RPS: 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 13.07 | 75.51% | 7.9707 | 11.54 | 75359.19 | 33066.61 | 870.10 |
| exp_1 | 13.01 | 75.90% | 8.0600 | 11.60 | 75357.19 | 32919.77 | 866.20 |
| exp_2 | 13.07 | 53.08% | 7.9654 | 11.51 | 95059.19 | 33072.78 | 870.26 |
| exp_3 | 11.34 | 99.95% | 8.8255 | 12.99 | 55393.12 | 28683.74 | 754.75 |


#### 目标 RPS: 40

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 13.10 | 99.66% | 20.53 | 24.73 | 75359.19 | 33142.31 | 866.96 |
| exp_1 | 13.09 | 99.68% | 20.58 | 24.80 | 75357.19 | 33121.23 | 865.59 |
| exp_2 | 14.20 | 99.65% | 19.06 | 24.14 | 97051.19 | 35930.47 | 939.39 |
| exp_3 | 11.86 | 100.00% | 19.07 | 24.94 | 55393.12 | 30006.13 | 784.57 |


### 突发并发测试 (Burst Concurrency)

#### 目标并发数 (Concurrency): 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 1.6104 | 0.75% | 0.5006 | 0.5006 | 75359.19 | 4016.31 | 49.92 |
| exp_1 | 1.6112 | 0.75% | 0.4988 | 0.4988 | 75357.19 | 4018.36 | 49.95 |
| exp_2 | 1.6039 | 0.52% | 0.5002 | 0.5002 | 94301.19 | 4000.08 | 49.72 |
| exp_3 | null | null | null | null | null | null | null |


#### 目标并发数 (Concurrency): 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 7.6064 | 8.97% | 2.0207 | 2.5202 | 75359.19 | 19388.43 | 580.37 |
| exp_1 | 7.6058 | 8.97% | 2.0268 | 2.5172 | 75357.19 | 19386.83 | 580.32 |
| exp_2 | 7.6061 | 6.30% | 2.0397 | 2.5359 | 94301.85 | 19387.66 | 580.35 |
| exp_3 | null | null | null | null | null | null | null |


#### 目标并发数 (Concurrency): 50

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 12.82 | 21.65% | 2.9163 | 3.6897 | 75359.19 | 32470.99 | 862.43 |
| exp_1 | 12.38 | 21.64% | 2.9671 | 3.7382 | 75357.19 | 31363.74 | 833.50 |
| exp_2 | 12.82 | 15.21% | 2.9512 | 3.7123 | 94303.19 | 32470.66 | 864.41 |
| exp_3 | null | null | null | null | null | null | null |


#### 目标并发数 (Concurrency): 100

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 15.68 | 43.08% | 5.1772 | 6.0121 | 75359.19 | 39698.31 | 1061.14 |
| exp_1 | 15.52 | 43.08% | 5.2373 | 6.0801 | 75357.19 | 39288.60 | 1050.42 |
| exp_2 | 15.52 | 30.24% | 5.1969 | 6.0382 | 94303.19 | 39277.13 | 1050.11 |
| exp_3 | null | null | null | null | null | null | null |


#### 目标并发数 (Concurrency): 200

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 18.49 | 85.52% | 9.1690 | 10.41 | 75359.19 | 46789.23 | 1231.10 |
| exp_1 | 18.39 | 85.52% | 9.3220 | 10.54 | 75357.19 | 46535.06 | 1224.50 |
| exp_2 | 18.39 | 60.10% | 9.1889 | 10.43 | 94303.19 | 46524.92 | 1224.23 |
| exp_3 | null | null | null | null | null | null | null |


#### 目标并发数 (Concurrency): 300

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 15.73 | 99.57% | 15.94 | 18.29 | 75359.19 | 39816.49 | 1054.77 |
| exp_1 | 15.60 | 99.77% | 16.09 | 18.42 | 75357.19 | 39485.21 | 1045.95 |
| exp_2 | 19.51 | 90.04% | 13.49 | 15.02 | 95057.19 | 49379.93 | 1308.45 |
| exp_3 | null | null | null | null | null | null | null |

