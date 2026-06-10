# 实验记录报告

## 实验环境记录

| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |
| --- | --- | --- |
| exp_0 | `root     185280 18.4  0.1 25935028 1519292 pts/1 Sl+ 11:13   1:04 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching` | false |
| exp_1 | `root     192866 17.7  0.1 25932988 1516132 pts/1 Sl+ 11:20   1:03 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --performance-mode throughput` | false |
| exp_2 | `root     209105 17.8  0.1 25936048 1518612 pts/1 Sl+ 11:28   1:03 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --kv-cache-dtype fp8` | false |
| exp_3 | `root     224493 18.9  0.1 25935032 1519140 pts/1 Sl+ 11:35   1:06 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --async-scheduling` | false |
| exp_4 | `root     228084 17.8  0.1 25937208 1519832 pts/1 Sl+ 11:43   1:02 /root/miniconda3/bin/python /root/miniconda3/bin/vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8000 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.7 --enable-prefix-caching --max-num-seqs 512 --max-num-batched-tokens 8192` | false |


## 实验结果记录

### 稳态并发测试 (Steady RPS)

#### 目标 RPS: 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 1.0360 | 0.42% | 0.5634 | 0.7320 | 74605.19 | 2393.75 | 77.70 |
| exp_1 | 1.0362 | 0.48% | 0.5738 | 0.7275 | 74693.12 | 2395.64 | 79.16 |
| exp_2 | 1.0364 | 0.35% | 0.5594 | 0.6972 | 75471.12 | 2395.49 | 78.56 |
| exp_3 | 1.0360 | 0.42% | 0.5626 | 0.7259 | 74605.12 | 2393.80 | 77.70 |
| exp_4 | 1.0355 | 0.40% | 0.5600 | 0.7305 | 74531.12 | 2392.69 | 77.66 |


#### 目标 RPS: 5

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 4.6860 | 1.53% | 0.7480 | 1.2418 | 74605.19 | 10772.96 | 322.68 |
| exp_1 | 4.6871 | 1.77% | 0.7493 | 1.2411 | 74693.12 | 10776.40 | 323.69 |
| exp_2 | 4.6861 | 1.45% | 0.7436 | 1.2398 | 75475.12 | 10773.68 | 323.15 |
| exp_3 | 4.6845 | 1.53% | 0.7433 | 1.2327 | 74605.12 | 10770.10 | 323.14 |
| exp_4 | 4.6853 | 1.45% | 0.7379 | 1.2225 | 74531.12 | 10772.40 | 323.63 |


#### 目标 RPS: 10

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 9.1989 | 4.16% | 1.1378 | 1.8533 | 74605.19 | 21132.56 | 630.59 |
| exp_1 | 9.1948 | 4.70% | 1.1529 | 1.8541 | 74693.12 | 21125.06 | 632.15 |
| exp_2 | 9.1983 | 3.94% | 1.1445 | 1.8541 | 75475.12 | 21131.48 | 630.76 |
| exp_3 | 9.1948 | 4.16% | 1.1378 | 1.8506 | 74605.12 | 21123.17 | 630.36 |
| exp_4 | 9.1974 | 3.86% | 1.1337 | 1.8501 | 74531.12 | 21128.95 | 630.45 |


#### 目标 RPS: 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 17.13 | 19.57% | 2.7011 | 4.6010 | 74605.19 | 39314.61 | 1154.02 |
| exp_1 | 16.93 | 24.15% | 2.7952 | 4.7951 | 74693.12 | 38857.51 | 1141.71 |
| exp_2 | 17.13 | 19.42% | 2.6535 | 4.5156 | 75475.12 | 39335.11 | 1160.94 |
| exp_3 | 17.02 | 19.96% | 2.7159 | 4.6448 | 74605.12 | 39082.43 | 1148.30 |
| exp_4 | 17.03 | 18.63% | 2.6996 | 4.6391 | 74531.12 | 39088.35 | 1145.71 |


#### 目标 RPS: 40

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 29.47 | 81.24% | 5.7252 | 9.0901 | 74605.19 | 67644.60 | 1973.12 |
| exp_1 | 29.02 | 96.44% | 5.9340 | 9.3437 | 74693.12 | 66620.76 | 1942.27 |
| exp_2 | 29.13 | 81.14% | 5.7525 | 9.1120 | 75475.12 | 66861.73 | 1958.17 |
| exp_3 | 29.41 | 81.14% | 5.7541 | 9.1046 | 74605.12 | 67504.04 | 1969.36 |
| exp_4 | 29.27 | 77.41% | 5.7618 | 9.1359 | 94175.12 | 67177.41 | 1959.63 |


#### 目标 RPS: 80

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 29.52 | 99.48% | 15.99 | 19.09 | 74605.19 | 67766.33 | 1981.30 |
| exp_1 | 29.11 | 99.36% | 15.85 | 19.19 | 74693.12 | 66814.49 | 1952.40 |
| exp_2 | 29.48 | 99.34% | 16.01 | 19.19 | 75475.12 | 67676.91 | 1980.90 |
| exp_3 | 29.73 | 99.44% | 15.93 | 19.09 | 74605.12 | 68238.21 | 1995.17 |
| exp_4 | 29.53 | 99.60% | 15.92 | 19.17 | 84360.46 | 67783.81 | 1983.53 |


### 突发并发测试 (Burst Concurrency)

#### 目标并发数 (Concurrency): 1

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 2.3757 | 0.40% | 0.2511 | 0.2511 | 74605.19 | 5366.81 | 73.65 |
| exp_1 | 2.3773 | 0.31% | 0.2508 | 0.2508 | 74693.12 | 5370.24 | 73.70 |
| exp_2 | 2.3751 | 0.33% | 0.2468 | 0.2468 | 75475.12 | 5365.26 | 73.63 |
| exp_3 | 2.3790 | 0.40% | 0.2559 | 0.2559 | 74605.12 | 5374.05 | 73.75 |
| exp_4 | 2.3749 | 0.38% | 0.2540 | 0.2540 | 77947.12 | 5364.79 | 73.62 |


#### 目标并发数 (Concurrency): 20

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 10.94 | 5.71% | 1.2925 | 1.6126 | 74605.19 | 25349.31 | 858.62 |
| exp_1 | 10.95 | 6.59% | 1.3374 | 1.6399 | 74693.12 | 25364.53 | 862.31 |
| exp_2 | 10.96 | 5.60% | 1.3232 | 1.6267 | 75475.12 | 25389.84 | 874.81 |
| exp_3 | 10.93 | 5.71% | 1.3138 | 1.6319 | 74605.12 | 25314.43 | 857.44 |
| exp_4 | 10.95 | 5.40% | 1.3014 | 1.6188 | 77947.12 | 25361.53 | 859.03 |


#### 目标并发数 (Concurrency): 50

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 22.43 | 14.07% | 1.5139 | 2.0645 | 74605.19 | 51581.31 | 1552.53 |
| exp_1 | 22.44 | 16.24% | 1.5208 | 2.0805 | 74693.12 | 51583.07 | 1548.23 |
| exp_2 | 22.43 | 13.89% | 1.5060 | 2.0577 | 75475.12 | 51585.16 | 1554.38 |
| exp_3 | 21.80 | 14.07% | 1.5273 | 2.0752 | 74605.12 | 50122.84 | 1506.06 |
| exp_4 | 22.40 | 13.31% | 1.5185 | 2.0840 | 77947.12 | 51485.00 | 1540.36 |


#### 目标并发数 (Concurrency): 100

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 32.95 | 28.07% | 2.1275 | 2.7836 | 74605.19 | 75682.72 | 2250.66 |
| exp_1 | 32.97 | 32.37% | 2.1240 | 2.7921 | 74693.12 | 75744.64 | 2252.61 |
| exp_2 | 32.97 | 27.71% | 2.1070 | 2.7543 | 75475.12 | 75749.19 | 2266.18 |
| exp_3 | 32.92 | 28.07% | 2.1323 | 2.7999 | 74605.12 | 75624.88 | 2260.75 |
| exp_4 | 32.92 | 26.55% | 2.1211 | 2.7771 | 77947.12 | 75636.03 | 2257.47 |


#### 目标并发数 (Concurrency): 200

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 47.05 | 55.85% | 3.2061 | 4.0355 | 74605.19 | 108014.68 | 3169.99 |
| exp_1 | 45.56 | 64.47% | 3.2139 | 4.0507 | 74693.12 | 104604.62 | 3076.50 |
| exp_2 | 46.17 | 55.36% | 3.2326 | 4.0582 | 75475.12 | 106010.14 | 3117.37 |
| exp_3 | 46.76 | 55.85% | 3.2186 | 4.0410 | 74605.12 | 107344.09 | 3148.56 |
| exp_4 | 46.85 | 52.84% | 3.2036 | 4.0362 | 77947.12 | 107545.62 | 3158.17 |


#### 目标并发数 (Concurrency): 300

| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp_0 | 51.23 | 83.70% | 4.5539 | 5.5279 | 74605.19 | 117652.45 | 3484.13 |
| exp_1 | 52.27 | 96.62% | 4.4458 | 5.4204 | 74693.12 | 120039.62 | 3544.23 |
| exp_2 | 51.63 | 83.00% | 4.5341 | 5.5191 | 75475.12 | 118586.34 | 3514.28 |
| exp_3 | 51.26 | 83.70% | 4.5040 | 5.4772 | 74605.12 | 117705.49 | 3470.33 |
| exp_4 | 51.61 | 79.18% | 4.5365 | 5.5111 | 77947.12 | 118531.96 | 3505.97 |

