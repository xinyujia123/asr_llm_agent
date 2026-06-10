# 实验记录报告

## 实验环境记录

| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |
| --- | --- | --- |
| exp_0 | `vllm serve /root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching` | true |
| exp_1 | `vllm serve /root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --performance-mode throughput` | true |
| exp_2 | `vllm serve /root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --kv-cache-dtype fp8` | true |
| exp_3 | `vllm serve /root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --async-scheduling` | true |
| exp_4 | `vllm serve /root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 512 --max-num-batched-tokens 8192` | true |
| exp_5 | `vllm serve /root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --quantization gptq_marlin` | true |


## 实验结果记录

### 稳态并发测试 (Steady RPS)

#### 目标 RPS: 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 0.9211 | 1.72% | 1.8153 | 2.4259 | 74731.12 | 2346.05 | 70.37 |
| exp_1 | 0.9212 | 1.79% | 1.8226 | 2.4386 | 74919.12 | 2346.20 | 70.38 |
| exp_2 | 0.9382 | 1.50% | 1.7606 | 2.3964 | 75623.12 | 3123.98 | 70.37 |
| exp_3 | 0.9214 | 1.64% | 1.8153 | 2.4253 | 74731.12 | 2346.89 | 70.40 |
| exp_4 | 0.9324 | 1.81% | 1.8189 | 2.4261 | 74575.12 | 2374.92 | 71.23 |
| exp_5 | 0.9325 | 1.72% | 1.8150 | 2.4298 | 74731.12 | 2375.00 | 71.24 |


#### 目标 RPS: 5

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 4.0734 | 6.76% | 1.9089 | 3.0867 | 74731.12 | 10318.95 | 277.64 |
| exp_1 | 4.0728 | 7.48% | 1.9118 | 3.0960 | 74919.12 | 10316.85 | 276.84 |
| exp_2 | 4.0730 | 6.10% | 1.8220 | 3.0141 | 75623.12 | 13509.20 | 275.50 |
| exp_3 | 4.0726 | 6.76% | 1.9105 | 3.0812 | 74731.12 | 10317.03 | 277.59 |
| exp_4 | 4.0716 | 6.47% | 1.9094 | 3.1026 | 74575.12 | 10314.54 | 277.52 |
| exp_5 | 4.0718 | 6.76% | 1.9123 | 3.0780 | 74731.12 | 10315.04 | 277.53 |


#### 目标 RPS: 10

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 8.0101 | 25.08% | 3.4090 | 5.5755 | 74731.12 | 20279.42 | 544.61 |
| exp_1 | 8.0106 | 27.92% | 3.4273 | 5.5920 | 74919.12 | 20280.30 | 544.27 |
| exp_2 | 8.0102 | 23.51% | 3.2446 | 5.3544 | 75623.12 | 26557.51 | 542.53 |
| exp_3 | 8.0034 | 25.10% | 3.4048 | 5.5686 | 74731.12 | 20262.40 | 544.15 |
| exp_4 | 8.0110 | 24.67% | 3.4569 | 5.6285 | 74575.12 | 20281.37 | 544.35 |
| exp_5 | 8.0056 | 25.38% | 3.4204 | 5.5894 | 74731.12 | 20267.96 | 544.30 |


#### 目标 RPS: 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 12.32 | 97.73% | 8.7196 | 12.23 | 74731.12 | 31167.43 | 823.10 |
| exp_1 | 11.56 | 100.00% | 9.6165 | 13.34 | 74919.12 | 29243.52 | 772.43 |
| exp_2 | 12.26 | 97.77% | 8.8193 | 12.32 | 75623.12 | 40620.85 | 817.32 |
| exp_3 | 12.31 | 97.21% | 8.7563 | 12.29 | 74731.12 | 31141.55 | 822.85 |
| exp_4 | 12.47 | 92.54% | 8.6207 | 12.09 | 79935.12 | 31541.08 | 833.45 |
| exp_5 | 12.31 | 98.21% | 8.7583 | 12.30 | 74731.12 | 31152.91 | 823.63 |


#### 目标 RPS: 40

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 11.83 | 99.87% | 22.19 | 25.86 | 74731.17 | 29929.77 | 786.01 |
| exp_1 | 11.62 | 99.85% | 21.94 | 25.86 | 74919.12 | 29405.28 | 771.75 |
| exp_2 | 11.86 | 99.61% | 22.21 | 25.86 | 75623.12 | 39297.54 | 785.68 |
| exp_3 | 11.74 | 99.73% | 22.51 | 26.19 | 74731.12 | 29704.47 | 779.66 |
| exp_4 | 11.92 | 99.62% | 22.34 | 25.77 | 82267.12 | 30163.96 | 792.06 |
| exp_5 | 11.76 | 99.85% | 22.45 | 26.11 | 74731.12 | 29754.64 | 780.88 |


### 突发并发测试 (Burst Concurrency)

#### 目标并发数 (Concurrency): 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 1.2172 | 0.93% | 0.6888 | 0.6888 | 74731.19 | 3035.81 | 37.73 |
| exp_1 | 1.2185 | 1.03% | 0.6870 | 0.6870 | 74919.12 | 3039.00 | 37.77 |
| exp_2 | 1.2171 | 0.80% | 0.6842 | 0.6842 | 75623.12 | 3989.71 | 37.73 |
| exp_3 | 1.2170 | 0.93% | 0.6883 | 0.6883 | 74731.12 | 3035.19 | 37.73 |
| exp_4 | 1.2172 | 0.89% | 0.6899 | 0.6899 | 82267.12 | 3035.70 | 37.73 |
| exp_5 | 1.2171 | 0.93% | 0.6836 | 0.6836 | 74731.12 | 3035.41 | 37.73 |


#### 目标并发数 (Concurrency): 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 5.8253 | 11.09% | 2.6271 | 3.3255 | 74731.19 | 14851.66 | 447.68 |
| exp_1 | 5.8263 | 12.26% | 2.6297 | 3.3272 | 74919.12 | 14850.77 | 444.26 |
| exp_2 | 5.8255 | 10.95% | 2.6285 | 3.3226 | 75623.12 | 19419.34 | 447.69 |
| exp_3 | 5.8242 | 11.09% | 2.6382 | 3.3207 | 74731.12 | 14850.17 | 448.95 |
| exp_4 | 5.8218 | 10.62% | 2.6206 | 3.3208 | 82267.12 | 14842.65 | 447.40 |
| exp_5 | 5.8235 | 11.09% | 2.6171 | 3.3177 | 74731.12 | 14846.90 | 447.53 |


#### 目标并发数 (Concurrency): 50

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 10.63 | 27.18% | 3.4015 | 4.4548 | 74731.19 | 26926.14 | 723.21 |
| exp_1 | 10.63 | 30.04% | 3.4174 | 4.4798 | 74919.12 | 26922.35 | 721.31 |
| exp_2 | 10.64 | 26.92% | 3.4000 | 4.4668 | 75623.12 | 35274.76 | 719.91 |
| exp_3 | 10.63 | 27.18% | 3.3996 | 4.4625 | 74731.12 | 26919.70 | 722.21 |
| exp_4 | 10.48 | 26.00% | 3.4316 | 4.4838 | 82267.12 | 26536.12 | 712.75 |
| exp_5 | 10.63 | 27.18% | 3.4138 | 4.4739 | 74731.12 | 26921.11 | 722.26 |


#### 目标并发数 (Concurrency): 100

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 14.19 | 53.54% | 5.5586 | 6.7011 | 74731.19 | 35914.85 | 963.30 |
| exp_1 | 14.18 | 59.27% | 5.5784 | 6.7061 | 74919.12 | 35909.64 | 963.07 |
| exp_2 | 14.19 | 53.45% | 5.5529 | 6.6945 | 75623.12 | 47042.29 | 961.56 |
| exp_3 | 14.19 | 53.59% | 5.5683 | 6.7012 | 74731.12 | 35915.04 | 963.86 |
| exp_4 | 14.18 | 51.28% | 5.5773 | 6.7200 | 82267.12 | 35906.95 | 964.29 |
| exp_5 | 14.19 | 53.59% | 5.5814 | 6.7136 | 74731.12 | 35924.20 | 964.75 |


#### 目标并发数 (Concurrency): 200

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 15.69 | 99.60% | 10.24 | 11.78 | 74731.19 | 39694.42 | 1048.81 |
| exp_1 | 15.05 | 99.71% | 10.71 | 12.42 | 74919.12 | 38077.42 | 1004.92 |
| exp_2 | 16.27 | 99.47% | 10.23 | 11.72 | 75623.12 | 53919.11 | 1083.99 |
| exp_3 | 16.56 | 99.60% | 10.18 | 11.66 | 74731.12 | 41898.22 | 1107.06 |
| exp_4 | 16.72 | 99.37% | 9.7280 | 11.28 | 82267.12 | 42292.76 | 1116.72 |
| exp_5 | 16.11 | 99.60% | 10.29 | 11.84 | 74731.15 | 40767.47 | 1074.92 |


#### 目标并发数 (Concurrency): 300

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 13.88 | 99.60% | 17.64 | 20.68 | 74731.19 | 35128.90 | 933.67 |
| exp_1 | 13.47 | 99.68% | 18.00 | 21.39 | 74919.12 | 34093.50 | 906.18 |
| exp_2 | 13.85 | 99.60% | 17.68 | 20.90 | 75623.12 | 45914.38 | 930.20 |
| exp_3 | 13.89 | 99.85% | 17.67 | 20.79 | 74731.12 | 35151.04 | 934.87 |
| exp_4 | 14.01 | 99.37% | 17.48 | 20.57 | 82267.12 | 35462.20 | 942.82 |
| exp_5 | 13.75 | 99.60% | 17.71 | 20.83 | 74731.19 | 34812.19 | 926.22 |

