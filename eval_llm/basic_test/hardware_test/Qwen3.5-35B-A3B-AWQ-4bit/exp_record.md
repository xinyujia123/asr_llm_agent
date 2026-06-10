# 实验记录报告

## 实验环境记录

| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |
| --- | --- | --- |
| exp_0 | `vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.5 --enable-prefix-caching` | true |
| exp_1 | `vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching` | true |
| exp_2 | `vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 64` | true |
| exp_3 | `vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 128` | true |
| exp_4 | `vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 256` | true |
| exp_5 | `vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.45 --enable-prefix-caching` | false |
| exp_6 | `vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-batched-tokens 65536` | true |


## 实验结果记录

### 稳态并发测试 (Steady RPS)

#### 目标 RPS: 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 1.0575 | 0.63% | 0.4695 | 0.5940 | 55337.19 | 3554.40 | 78.68 |
| exp_1 | 1.0361 | 0.34% | 0.4896 | 0.6205 | 78833.19 | 3482.20 | 77.08 |
| exp_2 | 1.0357 | 0.32% | 0.4845 | 0.6139 | 74059.12 | 3480.93 | 77.05 |
| exp_3 | 1.0359 | 0.32% | 0.4878 | 0.6203 | 74235.19 | 3481.66 | 77.07 |
| exp_4 | 1.0358 | 0.32% | 0.4851 | 0.6179 | 74935.12 | 3481.24 | 77.06 |
| exp_5 | 1.0357 | 0.69% | 0.4869 | 0.6186 | 50497.15 | 2392.46 | 77.06 |
| exp_6 | 1.0358 | 0.37% | 0.4811 | 0.6050 | 72385.12 | 3481.30 | 77.06 |


#### 目标 RPS: 5

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 4.7751 | 1.72% | 0.5911 | 0.9611 | 55337.19 | 15992.63 | 325.02 |
| exp_1 | 4.7466 | 0.99% | 0.6306 | 1.0004 | 78833.19 | 15898.11 | 323.85 |
| exp_2 | 4.7741 | 0.87% | 0.6104 | 0.9949 | 74061.12 | 15988.36 | 323.94 |
| exp_3 | 4.7760 | 0.87% | 0.6386 | 1.0008 | 74235.19 | 15994.45 | 323.75 |
| exp_4 | 4.7750 | 0.88% | 0.5977 | 0.9841 | 74935.12 | 15990.96 | 323.56 |
| exp_5 | 4.7746 | 2.13% | 0.6054 | 0.9881 | 50497.19 | 10970.85 | 322.86 |
| exp_6 | 4.7441 | 1.04% | 0.6262 | 1.0323 | 72385.12 | 15886.62 | 320.52 |


#### 目标 RPS: 10

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 9.3695 | 4.14% | 0.7976 | 1.4078 | 55337.19 | 31358.36 | 628.76 |
| exp_1 | 9.3706 | 2.33% | 0.8316 | 1.4811 | 78833.19 | 31363.69 | 630.61 |
| exp_2 | 9.3693 | 2.53% | 0.8357 | 1.6366 | 74061.12 | 31354.08 | 625.21 |
| exp_3 | 9.3647 | 2.16% | 0.8245 | 1.4594 | 74235.19 | 31341.66 | 627.93 |
| exp_4 | 9.3709 | 2.16% | 0.8167 | 1.4508 | 74935.12 | 31362.16 | 627.88 |
| exp_5 | 9.3663 | 5.29% | 0.8260 | 1.4674 | 50497.19 | 21502.36 | 627.38 |
| exp_6 | 9.3686 | 2.64% | 0.8703 | 1.5220 | 72385.15 | 31355.81 | 629.32 |


#### 目标 RPS: 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 18.41 | 12.27% | 1.3643 | 2.2508 | 55337.19 | 61575.12 | 1217.29 |
| exp_1 | 18.40 | 6.74% | 1.4098 | 2.2751 | 78833.19 | 61560.72 | 1218.84 |
| exp_2 | 17.34 | 11.85% | 3.0023 | 4.8791 | 74061.19 | 57993.62 | 1143.07 |
| exp_3 | 18.39 | 6.27% | 1.3977 | 2.2966 | 74235.19 | 61533.21 | 1215.56 |
| exp_4 | 18.40 | 6.21% | 1.3571 | 2.2442 | 74935.12 | 61544.50 | 1211.15 |
| exp_5 | 18.40 | 15.40% | 1.4089 | 2.3469 | 50497.19 | 42212.56 | 1211.10 |
| exp_6 | 18.27 | 8.90% | 1.6306 | 2.7269 | 72385.19 | 61107.70 | 1208.04 |


#### 目标 RPS: 40

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 30.19 | 98.46% | 5.2154 | 8.5700 | 55337.19 | 100993.78 | 1975.72 |
| exp_1 | 18.05 | 50.17% | 9.1337 | 25.12 | 78833.19 | 60375.66 | 850.91 |
| exp_2 | 19.85 | 11.88% | 7.2282 | 10.65 | 74061.19 | 66395.49 | 1298.87 |
| exp_3 | 25.55 | 23.78% | 5.7810 | 7.9522 | 78715.19 | 85465.54 | 1672.23 |
| exp_4 | 22.39 | 42.30% | 11.22 | 15.09 | 87630.46 | 74905.60 | 1309.98 |
| exp_5 | 28.23 | 99.21% | 5.6534 | 8.9593 | 50497.19 | 64745.75 | 1845.87 |
| exp_6 | 30.77 | 59.26% | 5.2219 | 8.5267 | 72385.19 | 102907.15 | 2012.10 |


### 突发并发测试 (Burst Concurrency)

#### 目标并发数 (Concurrency): 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 2.3778 | 0.00% | 0.2214 | 0.2214 | 55335.19 | 7870.46 | 73.71 |
| exp_1 | 2.3870 | 0.00% | 0.2299 | 0.2299 | 74777.12 | 7901.12 | 74.00 |
| exp_2 | 2.3745 | 0.00% | 0.2276 | 0.2276 | 74059.12 | 7859.46 | 73.61 |
| exp_3 | 2.3758 | 0.00% | 0.2309 | 0.2309 | 74235.12 | 7863.94 | 73.65 |
| exp_4 | 2.3825 | 0.00% | 0.2435 | 0.2435 | 74937.19 | 7885.99 | 73.86 |
| exp_5 | 2.3800 | 0.00% | 0.2325 | 0.2325 | 50497.12 | 7877.94 | 73.78 |
| exp_6 | 2.3529 | 0.12% | 0.2287 | 0.2287 | 72385.19 | 7788.00 | 72.93 |


#### 目标并发数 (Concurrency): 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 14.03 | 7.64% | 1.1030 | 1.3643 | 55337.19 | 47238.15 | 1089.18 |
| exp_1 | 12.87 | 4.05% | 1.1008 | 1.3735 | 74777.12 | 43327.88 | 990.78 |
| exp_2 | 14.04 | 3.81% | 1.1109 | 1.3797 | 74059.12 | 47255.15 | 1078.83 |
| exp_3 | 12.31 | 3.87% | 1.1400 | 1.4130 | 74235.12 | 41422.15 | 943.65 |
| exp_4 | 12.30 | 3.78% | 1.1477 | 1.4186 | 74937.19 | 41383.24 | 943.37 |
| exp_5 | 14.02 | 9.55% | 1.1134 | 1.3876 | 50497.12 | 47164.91 | 1069.23 |
| exp_6 | 13.46 | 4.41% | 1.1250 | 1.3958 | 72385.19 | 45285.27 | 1028.29 |


#### 目标并发数 (Concurrency): 50

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 24.66 | 18.39% | 1.3751 | 1.8244 | 55337.19 | 82575.06 | 1677.72 |
| exp_1 | 24.62 | 9.92% | 1.3905 | 1.8364 | 74777.12 | 82470.26 | 1675.10 |
| exp_2 | 24.66 | 9.32% | 1.3805 | 1.8303 | 74059.12 | 82573.06 | 1673.17 |
| exp_3 | 24.66 | 9.32% | 1.4175 | 1.8704 | 74235.12 | 82594.58 | 1685.36 |
| exp_4 | 24.61 | 9.35% | 1.4178 | 1.8669 | 74937.19 | 82413.98 | 1675.56 |
| exp_5 | 24.62 | 23.36% | 1.3977 | 1.8392 | 50497.12 | 82454.86 | 1677.84 |
| exp_6 | 24.63 | 10.78% | 1.3908 | 1.8575 | 72385.19 | 82480.65 | 1670.33 |


#### 目标并发数 (Concurrency): 100

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 37.10 | 36.54% | 2.0350 | 2.5351 | 55337.19 | 124177.22 | 2487.51 |
| exp_1 | 36.24 | 19.69% | 2.0455 | 2.5613 | 74777.12 | 121283.06 | 2421.06 |
| exp_2 | 25.62 | 11.90% | 2.8426 | 3.6448 | 74189.12 | 85748.89 | 1719.99 |
| exp_3 | 36.25 | 18.55% | 2.0535 | 2.5737 | 74765.79 | 121323.91 | 2426.85 |
| exp_4 | 35.31 | 18.58% | 2.0823 | 2.6022 | 75129.19 | 118178.99 | 2369.57 |
| exp_5 | 37.08 | 46.52% | 2.0291 | 2.5513 | 50497.12 | 124102.96 | 2474.68 |
| exp_6 | 37.10 | 21.42% | 2.0807 | 2.5672 | 72385.19 | 124177.98 | 2481.74 |


#### 目标并发数 (Concurrency): 200

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 47.06 | 72.74% | 3.2650 | 3.9485 | 55337.19 | 157415.97 | 3103.14 |
| exp_1 | 47.05 | 39.23% | 3.2897 | 3.9723 | 74777.12 | 157389.32 | 3098.84 |
| exp_2 | 22.24 | 11.89% | 5.7816 | 8.5184 | 74189.12 | 74389.63 | 1460.60 |
| exp_3 | 34.48 | 23.72% | 4.4238 | 5.4917 | 75155.12 | 115340.56 | 2275.87 |
| exp_4 | 47.05 | 36.99% | 3.3375 | 4.0209 | 76825.19 | 157419.08 | 3115.19 |
| exp_5 | 47.08 | 92.42% | 3.3153 | 3.9987 | 50497.12 | 157498.82 | 3104.92 |
| exp_6 | 47.02 | 42.65% | 3.3063 | 3.9875 | 72385.19 | 157307.45 | 3101.92 |


#### 目标并发数 (Concurrency): 300

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 41.91 | 99.70% | 5.8609 | 6.7532 | 55337.19 | 140259.16 | 2789.93 |
| exp_1 | 52.72 | 58.77% | 4.5563 | 5.3787 | 74777.12 | 176394.35 | 3502.08 |
| exp_2 | 20.87 | 11.89% | 8.6343 | 13.80 | 74189.12 | 69819.05 | 1387.11 |
| exp_3 | 29.90 | 23.74% | 7.2569 | 9.5014 | 75155.12 | 100054.77 | 1988.10 |
| exp_4 | 39.90 | 47.44% | 6.0433 | 7.0593 | 78745.19 | 133505.68 | 2654.75 |
| exp_5 | 38.44 | 99.42% | 6.1074 | 7.3737 | 50497.12 | 128622.56 | 2553.05 |
| exp_6 | 52.23 | 63.89% | 4.5955 | 5.4298 | 72385.19 | 174790.34 | 3477.91 |

## 实验分析
exp_1的断崖式下跌