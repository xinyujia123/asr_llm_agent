# 实验记录报告

## 实验环境记录

| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |
| --- | --- | --- |
| exp_0 | `vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching` | true |
| exp_1 | `vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --performance-mode throughput` | false |
| exp_2 | `vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --kv-cache-dtype fp8` | false |
| exp_3 | `vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --async-scheduling` | false |
| exp_4 | `vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 512 --max-num-batched-tokens 8192` | false |
| exp_5 | `root      68121 19.0  0.1 25940380 1525748 pts/1 Sl+ 10:12   1:05 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --performance-mode throughput` | false |
| exp_6 | `root      80319 18.4  0.1 25941412 1526460 pts/1 Sl+ 10:20   1:04 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --kv-cache-dtype fp8` | false |
| exp_7 | `root      89235 18.6  0.1 25941412 1526472 pts/1 Sl+ 10:27   1:04 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --async-scheduling` | false |
| exp_8 | `root     116983 18.7  0.1 25941540 1526636 pts/1 Sl+ 10:34   1:04 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 512 --max-num-batched-tokens 8192` | false |


## 实验结果记录

### 稳态并发测试 (Steady RPS)

#### 目标 RPS: 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 1.0359 | 0.33% | 0.5059 | 0.6297 | 74733.19 | 3484.55 | 79.90 |
| exp_1 | 1.0356 | 0.31% | 0.4880 | 0.6197 | 74909.12 | 2393.80 | 78.64 |
| exp_2 | 1.0579 | 0.23% | 0.4623 | 0.5944 | 75605.12 | 2444.28 | 79.13 |
| exp_3 | 1.0358 | 0.28% | 0.4920 | 0.6196 | 74733.12 | 2394.47 | 78.92 |
| exp_4 | 1.0355 | 0.27% | 0.4877 | 0.6144 | 74739.12 | 2392.55 | 77.53 |
| exp_5 | 1.0357 | 0.31% | 0.4931 | 0.6170 | 74909.12 | 2394.41 | 78.92 |
| exp_6 | 1.0430 | 0.23% | 0.4809 | 0.5991 | 75605.12 | 2411.39 | 79.74 |
| exp_7 | 1.0358 | 0.28% | 0.4806 | 0.6199 | 74733.12 | 2393.12 | 77.48 |
| exp_8 | 1.0363 | 0.27% | 0.4867 | 0.6184 | 74739.12 | 2394.85 | 78.00 |


#### 目标 RPS: 5

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 4.7749 | 0.90% | 0.5989 | 0.9805 | 74733.19 | 15994.59 | 327.53 |
| exp_1 | 4.7749 | 0.92% | 0.6023 | 0.9852 | 74909.12 | 10975.38 | 326.70 |
| exp_2 | 4.7755 | 0.85% | 0.5961 | 0.9655 | 75605.12 | 10975.39 | 325.56 |
| exp_3 | 4.7746 | 0.85% | 0.5967 | 0.9829 | 74733.12 | 10974.81 | 326.90 |
| exp_4 | 4.7748 | 0.82% | 0.5999 | 0.9785 | 74739.12 | 10974.83 | 326.41 |
| exp_5 | 4.7754 | 0.92% | 0.6026 | 0.9852 | 74909.12 | 10976.31 | 326.67 |
| exp_6 | 4.7763 | 0.80% | 0.5955 | 0.9757 | 75605.12 | 10976.58 | 324.88 |
| exp_7 | 4.7743 | 0.85% | 0.6043 | 0.9840 | 74733.12 | 10974.21 | 327.01 |
| exp_8 | 4.7756 | 0.82% | 0.6017 | 0.9836 | 74739.12 | 10976.76 | 326.68 |


#### 目标 RPS: 10

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 9.3702 | 2.18% | 0.8131 | 1.4447 | 74733.19 | 31366.07 | 634.12 |
| exp_1 | 9.3718 | 2.36% | 0.8095 | 1.4644 | 74909.12 | 21522.16 | 634.75 |
| exp_2 | 9.3672 | 2.02% | 0.8014 | 1.4092 | 75605.12 | 21511.84 | 634.81 |
| exp_3 | 9.1451 | 2.18% | 0.8232 | 1.4718 | 74733.12 | 21010.75 | 628.56 |
| exp_4 | 9.3672 | 2.09% | 0.8092 | 1.4697 | 74739.12 | 21509.12 | 632.13 |
| exp_5 | 9.3696 | 2.36% | 0.8137 | 1.4721 | 74909.12 | 21515.84 | 633.54 |
| exp_6 | 9.3675 | 2.11% | 0.8166 | 1.4469 | 75605.12 | 21512.63 | 634.93 |
| exp_7 | 9.2534 | 2.19% | 0.8011 | 1.4633 | 74733.12 | 21253.97 | 630.49 |
| exp_8 | 9.2584 | 2.09% | 0.8092 | 1.4721 | 74739.12 | 21265.25 | 630.52 |


#### 目标 RPS: 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 18.40 | 6.31% | 1.3657 | 2.2481 | 74733.19 | 61564.92 | 1226.53 |
| exp_1 | 18.40 | 6.90% | 1.3714 | 2.2779 | 74909.12 | 42224.24 | 1225.29 |
| exp_2 | 18.40 | 6.13% | 1.3148 | 2.1983 | 75605.12 | 42223.95 | 1225.58 |
| exp_3 | 18.39 | 6.51% | 1.3858 | 2.2951 | 74733.12 | 42216.76 | 1230.34 |
| exp_4 | 18.40 | 6.24% | 1.3803 | 2.3142 | 74739.12 | 42216.14 | 1220.41 |
| exp_5 | 18.40 | 7.10% | 1.4184 | 2.3655 | 74909.12 | 42223.20 | 1226.78 |
| exp_6 | 18.40 | 6.27% | 1.4161 | 2.3057 | 75605.12 | 42238.52 | 1231.96 |
| exp_7 | 18.40 | 6.40% | 1.3767 | 2.2810 | 74733.12 | 42226.42 | 1226.48 |
| exp_8 | 18.40 | 6.25% | 1.4027 | 2.3180 | 74739.12 | 42220.63 | 1226.61 |


#### 目标 RPS: 40

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 30.85 | 51.87% | 5.1049 | 8.3783 | 74733.19 | 103205.47 | 2041.77 |
| exp_1 | 29.87 | 56.44% | 5.0963 | 8.3881 | 74909.12 | 68552.02 | 1992.57 |
| exp_2 | 31.01 | 52.00% | 5.0968 | 8.3722 | 75605.12 | 71164.72 | 2055.74 |
| exp_3 | 30.68 | 51.94% | 5.1530 | 8.4455 | 74733.12 | 70402.55 | 2041.47 |
| exp_4 | 30.87 | 50.87% | 5.1682 | 8.4459 | 88324.46 | 70827.14 | 2037.33 |
| exp_5 | 30.90 | 57.25% | 5.1630 | 8.4838 | 74909.12 | 70912.84 | 2045.20 |
| exp_6 | 30.11 | 52.12% | 5.1422 | 8.4294 | 75605.12 | 69105.20 | 2016.61 |
| exp_7 | 30.35 | 51.96% | 5.0999 | 8.4148 | 74733.12 | 69636.59 | 2015.80 |
| exp_8 | 30.76 | 49.69% | 5.0701 | 8.3933 | 86792.46 | 70567.80 | 2032.31 |


### 突发并发测试 (Burst Concurrency)

#### 目标并发数 (Concurrency): 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 2.3803 | 0.18% | 0.2208 | 0.2208 | 80885.12 | 5377.07 | 73.79 |
| exp_1 | 2.3786 | 0.29% | 0.2223 | 0.2223 | 74909.12 | 5373.30 | 73.74 |
| exp_2 | 2.3450 | 0.07% | 0.2161 | 0.2161 | 81765.12 | 5297.35 | 72.69 |
| exp_3 | 2.3767 | 0.18% | 0.2225 | 0.2225 | 78873.12 | 5368.96 | 73.68 |
| exp_4 | 2.3749 | 0.26% | 0.2238 | 0.2238 | 84763.12 | 5364.93 | 73.62 |
| exp_5 | 2.3763 | 0.00% | 0.2239 | 0.2239 | 74909.12 | 7865.41 | 73.66 |
| exp_6 | 2.3770 | 0.00% | 0.2217 | 0.2217 | 81749.12 | 10302.05 | 73.69 |
| exp_7 | 2.3814 | 0.00% | 0.2255 | 0.2255 | 80937.12 | 7882.29 | 73.82 |
| exp_8 | 2.3762 | 0.00% | 0.2236 | 0.2236 | 84997.12 | 7865.31 | 73.66 |


#### 目标并发数 (Concurrency): 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 14.04 | 3.88% | 1.0585 | 1.3099 | 80885.12 | 32516.26 | 1100.93 |
| exp_1 | 14.04 | 4.21% | 1.0640 | 1.3208 | 74909.12 | 32509.35 | 1103.86 |
| exp_2 | 14.00 | 3.80% | 1.0443 | 1.3021 | 81765.12 | 32420.37 | 1084.59 |
| exp_3 | 14.04 | 3.88% | 1.0581 | 1.3184 | 78873.12 | 32508.13 | 1100.65 |
| exp_4 | 14.04 | 3.73% | 1.0456 | 1.3128 | 84763.12 | 32515.16 | 1095.46 |
| exp_5 | 12.29 | 4.29% | 1.1314 | 1.3786 | 74909.12 | 41407.80 | 980.37 |
| exp_6 | 12.29 | 3.94% | 1.1119 | 1.3753 | 81749.12 | 53976.07 | 955.06 |
| exp_7 | 13.46 | 3.93% | 1.0862 | 1.3507 | 80937.12 | 45310.90 | 1056.42 |
| exp_8 | 13.45 | 3.78% | 1.0933 | 1.3477 | 84997.12 | 45286.19 | 1055.85 |


#### 目标并发数 (Concurrency): 50

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 25.54 | 9.56% | 1.3072 | 1.7486 | 80885.12 | 58688.86 | 1733.16 |
| exp_1 | 25.56 | 10.37% | 1.3015 | 1.7515 | 74909.12 | 58737.44 | 1741.67 |
| exp_2 | 25.54 | 9.44% | 1.3240 | 1.7531 | 81765.12 | 58709.93 | 1742.57 |
| exp_3 | 24.66 | 9.56% | 1.3124 | 1.7561 | 78873.12 | 56683.86 | 1687.93 |
| exp_4 | 25.57 | 9.19% | 1.3138 | 1.7471 | 84763.12 | 58777.27 | 1741.04 |
| exp_5 | 24.65 | 10.42% | 1.3710 | 1.8110 | 74909.12 | 82572.87 | 1697.64 |
| exp_6 | 24.64 | 9.36% | 1.4213 | 1.8645 | 81749.12 | 107779.15 | 1680.61 |
| exp_7 | 24.64 | 9.60% | 1.3637 | 1.8045 | 80937.12 | 82547.38 | 1693.73 |
| exp_8 | 24.65 | 9.24% | 1.3724 | 1.8173 | 84997.12 | 82583.38 | 1693.18 |


#### 目标并发数 (Concurrency): 100

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 30.94 | 19.02% | 1.9185 | 2.4328 | 80885.12 | 71100.52 | 2140.12 |
| exp_1 | 35.64 | 20.64% | 1.9238 | 2.4387 | 74909.12 | 81861.92 | 2409.03 |
| exp_2 | 38.00 | 18.83% | 1.9135 | 2.4180 | 81765.12 | 87252.59 | 2569.66 |
| exp_3 | 37.99 | 19.02% | 1.9338 | 2.4377 | 78873.12 | 87239.74 | 2574.69 |
| exp_4 | 38.01 | 18.30% | 1.9192 | 2.4258 | 84763.12 | 87289.82 | 2567.19 |
| exp_5 | 33.23 | 20.73% | 2.0343 | 2.5304 | 74909.12 | 111251.24 | 2252.77 |
| exp_6 | 35.29 | 18.89% | 2.1253 | 2.6282 | 81749.12 | 154282.16 | 2382.75 |
| exp_7 | 33.25 | 19.11% | 2.0187 | 2.5340 | 80937.12 | 111313.16 | 2259.03 |
| exp_8 | 36.18 | 18.38% | 2.0194 | 2.5166 | 84997.17 | 121109.71 | 2454.98 |


#### 目标并发数 (Concurrency): 200

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 45.09 | 37.94% | 3.1714 | 3.8239 | 80885.12 | 103501.52 | 3014.81 |
| exp_1 | 47.29 | 41.18% | 3.1681 | 3.8310 | 74909.12 | 108552.27 | 3154.63 |
| exp_2 | 49.39 | 37.61% | 3.2184 | 3.8516 | 81765.12 | 113358.31 | 3298.53 |
| exp_3 | 49.25 | 37.95% | 3.1761 | 3.8314 | 78873.12 | 113044.31 | 3295.27 |
| exp_4 | 49.29 | 36.52% | 3.1644 | 3.8167 | 84763.12 | 113122.25 | 3277.76 |
| exp_5 | 45.07 | 41.23% | 3.2956 | 3.9817 | 74909.12 | 150811.65 | 3016.43 |
| exp_6 | 44.94 | 37.65% | 3.4575 | 4.1586 | 81749.12 | 196385.36 | 3000.17 |
| exp_7 | 45.07 | 37.99% | 3.2964 | 3.9908 | 80937.12 | 150822.57 | 3014.81 |
| exp_8 | 47.04 | 36.55% | 3.3075 | 3.9780 | 84997.19 | 157404.12 | 3145.79 |


#### 目标并发数 (Concurrency): 300

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 49.37 | 56.86% | 4.3113 | 5.1625 | 80885.12 | 113385.21 | 3355.65 |
| exp_1 | 51.87 | 61.70% | 4.2730 | 5.1479 | 74909.12 | 119100.79 | 3498.27 |
| exp_2 | 54.79 | 56.38% | 4.4025 | 5.2354 | 81765.12 | 125789.28 | 3686.36 |
| exp_3 | 53.01 | 56.86% | 4.3223 | 5.1798 | 78873.12 | 121728.20 | 3578.72 |
| exp_4 | 55.70 | 54.71% | 4.2735 | 5.1000 | 84763.12 | 127879.58 | 3741.01 |
| exp_5 | 52.22 | 61.76% | 4.5248 | 5.4167 | 74909.12 | 174783.04 | 3516.74 |
| exp_6 | 49.41 | 56.43% | 4.9104 | 5.7711 | 81749.12 | 215978.64 | 3327.97 |
| exp_7 | 51.55 | 56.91% | 4.6180 | 5.4680 | 80937.12 | 172545.50 | 3474.79 |
| exp_8 | 52.30 | 54.76% | 4.5329 | 5.3839 | 84997.19 | 175042.59 | 3522.37 |

