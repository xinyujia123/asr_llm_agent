# Qwen3.5-27B-GPTQ-Int4 硬件压测实验报告（exp_0 ~ exp_5）

## 1. 实验目的

本次实验针对 `Qwen3.5-27B-GPTQ-Int4` 在本地 vLLM 部署下的并发服务能力进行测试，比较不同启动参数对以下指标的影响：

- 稳态吞吐能力（steady_rps）
- 突发并发能力（burst_concurrency）
- 请求延迟（P50 / P95）
- GPU 计算利用率
- KV Cache 使用率
- 显存占用

本实验同时考虑生产环境需求：**system prompt block padding 必须开启**，以尽可能提高前缀 KV Cache 命中率。因此，本次结果以“开启 padding 的真实业务场景”为前提进行分析。

---

## 2. 实验配置

### 2.1 基础启动参数

所有实验均基于以下公共参数：

```bash
vllm serve <MODEL_NAME> \
  --port 8000 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --gpu-memory-utilization 0.7 \
  --enable-prefix-caching
````

### 2.2 各实验变量

| 实验    | 额外参数                                               |
| ----- | -------------------------------------------------- |
| exp_0 | 默认，无额外参数                                           |
| exp_1 | `--performance-mode throughput`                    |
| exp_2 | `--kv-cache-dtype fp8`                             |
| exp_3 | `--async-scheduling`                               |
| exp_4 | `--max-num-seqs 512 --max-num-batched-tokens 8192` |
| exp_5 | `--quantization gptq_marlin`                       |

---

## 3. 重要说明：为什么 exp_2 的 token 吞吐量看起来更高

本次 benchmark 代码中，`ENABLE_SYSTEM_PROMPT_BLOCK_PADDING = True`，并且会根据当前 vLLM `/metrics` 中解析出的 `block_size`，对 system prompt 自动补齐到 KV block 边界。

也就是说，**每次请求实际发送的 system prompt 长度并不是严格固定的**，而是会随运行时 block 相关信息动态调整。

这导致：

* exp_2（`--kv-cache-dtype fp8`）下的 `system_padding_suffix` 明显更长
* 每个请求的 `prompt_tokens` 增加
* 因而 `throughput_tokens_per_s` 被动变高

因此：

> **exp_2 的 token throughput 不能直接与其他实验横向比较，尤其不能据此认定其真实吞吐更强。**

更可靠的比较指标应为：

* `actual_success_req_per_s`
* `latency_avg_s`
* `latency_p95_s`

---

## 4. steady_rps 结果对比

### 4.1 核心结果表

下表选取 steady_rps 中最有代表性的两档负载：

* `target_rps = 20`
* `target_rps = 40`

| 实验    | 额外参数                          | 20 RPS 实际成功 RPS | 20 RPS 平均延迟(s) | 20 RPS P95(s) | 40 RPS 实际成功 RPS | 40 RPS 平均延迟(s) | 40 RPS P95(s) |
| ----- | ----------------------------- | --------------: | -------------: | ------------: | --------------: | -------------: | ------------: |
| exp_0 | 默认                            |         12.3185 |         8.5562 |       12.2319 |         11.8308 |        19.5097 |       25.8556 |
| exp_1 | throughput mode               |         11.5581 |         9.5384 |       13.3359 |         11.6236 |        19.2871 |       25.8616 |
| exp_2 | kv-cache fp8                  |         12.2574 |         8.6391 |       12.3250 |         11.8594 |        19.5224 |       25.8608 |
| exp_3 | async scheduling              |         12.3081 |         8.6079 |       12.2880 |         11.7419 |        19.7399 |       26.1900 |
| exp_4 | max-num-seqs / batched-tokens |     **12.4660** |     **8.4185** |   **12.0913** |     **11.9234** |        19.6500 |   **25.7702** |
| exp_5 | gptq_marlin                   |         12.3124 |         8.6197 |       12.2972 |         11.7618 |        19.7072 |       26.1139 |

---

### 4.2 steady_rps 结论

#### 结论 1：系统稳定吞吐上限大约在 12 req/s 左右

从所有实验看：

* 当目标 RPS 提升到 20 或 40 时
* 实际成功 RPS 都集中在 **11.5 ~ 12.5 req/s**

说明当前这套部署的**稳态服务上限约为 12 req/s**。

继续提高目标 RPS，只会带来：

* 请求排队加剧
* 延迟升高
* 但实际成功速率增长有限

---

#### 结论 2：exp_4 是 steady 场景下最优方案

`exp_4`：

```bash
--max-num-seqs 512 --max-num-batched-tokens 8192
```

在 steady 场景中表现最好：

* 20 RPS 下实际成功 RPS 最高：**12.4660**
* 20 RPS 平均延迟最低：**8.4185s**
* 20 RPS P95 最低：**12.0913s**
* 40 RPS 下实际成功 RPS 也最高：**11.9234**

虽然提升幅度不算巨大，但在所有实验中它最稳定、最优。

---

#### 结论 3：throughput mode 没带来正收益

`exp_1` 加了：

```bash
--performance-mode throughput
```

结果反而略差：

* 20 RPS：11.5581，低于 baseline
* P95 延迟也更高

说明对该模型、该任务形态、该负载模式而言，`throughput mode` 没有带来实际收益。

---

#### 结论 4：async scheduling 几乎没有收益

`exp_3` 与 baseline 基本重合，说明：

```bash
--async-scheduling
```

在当前测试条件下，**不是主要优化方向**。

---

#### 结论 5：显式指定 gptq_marlin 也没有明显提升

`exp_5` 基本与 baseline 持平，说明：

* 要么默认后端已经接近该路径
* 要么瓶颈根本不在 quantization backend 名称切换上

---

## 5. burst_concurrency 结果对比

### 5.1 核心结果表

选取最重要的三档：

* 并发 100
* 并发 200
* 并发 300

| 实验    | 额外参数                          | 100 并发成功 req/s | 100 并发平均延迟(s) | 200 并发成功 req/s | 200 并发平均延迟(s) | 300 并发成功 req/s | 300 并发平均延迟(s) |
| ----- | ----------------------------- | -------------: | ------------: | -------------: | ------------: | -------------: | ------------: |
| exp_0 | 默认                            |        14.1864 |        5.5029 |        15.6885 |       10.1119 |        13.8794 |       16.2381 |
| exp_1 | throughput mode               |        14.1844 |        5.5167 |        15.0498 |       10.4659 |        13.4703 |       16.2182 |
| exp_4 | max-num-seqs / batched-tokens |        14.1828 |        5.5174 |    **16.7156** |    **9.6410** |    **14.0110** |       16.2619 |

> 注：burst 这部分重点选取了最关键的 baseline、表现较差的 exp_1、以及最优的 exp_4 进行对比。

---

### 5.2 burst_concurrency 结论

#### 结论 1：baseline 的突发最佳点大约在并发 200

baseline（exp_0）：

* 100 并发：14.19 req/s
* 200 并发：15.69 req/s
* 300 并发：13.88 req/s

说明：

* 并发从 100 增加到 200，吞吐仍在增长
* 继续加到 300 时，吞吐反而回落

因此，当前部署的**突发并发甜点区约在 200 左右**。

---

#### 结论 2：exp_4 明显改善高突发并发表现

在并发 200 下：

* exp_0：15.6885 req/s，平均延迟 10.1119s
* exp_4：**16.7156 req/s**，平均延迟 **9.6410s**

提升非常明确：

* 吞吐增加约 **6.5%**
* 平均延迟下降约 **4.7%**

在并发 300 下，exp_4 仍略优于 baseline。

这说明：

> `--max-num-seqs 512 --max-num-batched-tokens 8192` 确实改善了高并发突发场景下的调度和 batching 效率。

---

#### 结论 3：throughput mode 在 burst 场景下同样无效

exp_1 在并发 200 和 300 下都不如 baseline，因此不建议保留。

---

## 6. 资源利用率观察

### 6.1 GPU 计算利用率

多数实验在高负载下：

* GPU compute usage 长期 **96% ~ 99%**

说明当前瓶颈已经非常接近：

> **GPU 计算瓶颈，而不是 CPU 侧简单调度瓶颈。**

---

### 6.2 KV Cache 使用率

在高负载下，KV cache usage 也会上升到：

* 20 RPS steady 时约 50% ~ 60%
* 40 RPS steady 时约 83% ~ 85%
* 高 burst 时接近 100%

这说明生产环境下开启 padding 是合理的，因为：

* 更长但固定对齐的 system prompt
* 更有利于 prefix cache 命中
* 对长 system prompt 业务场景更贴近真实线上情况

---

### 6.3 显存占用

exp_4 明显比 baseline 占用更多显存，例如：

* baseline steady 40：GPU memory usage 约 **76.34%**
* exp_4 steady 40：GPU memory usage 约 **83.92%**

也就是说：

> exp_4 的提升不是“白来的”，它是用更高的调度容量和更高的显存占用换来的。

---

## 7. 综合结论

### 最终结论 1：在开启 system prompt block padding 的真实生产前提下，最佳方案是 exp_4

推荐参数：

```bash
--max-num-seqs 512 --max-num-batched-tokens 8192
```

原因：

* steady_rps 下综合表现最好
* burst_concurrency 下提升最明显
* 对高并发突发场景最有帮助
* 比其他优化项更接近真实收益

---

### 最终结论 2：不建议继续使用 exp_1 / exp_3 / exp_5 作为主优化方向

这些参数在本轮测试中均未带来明确收益：

* `--performance-mode throughput`
* `--async-scheduling`
* `--quantization gptq_marlin`

可以暂时排除，避免进一步分散测试精力。

---

### 最终结论 3：exp_2 不能按 token throughput 判断优劣

由于 padding 机制导致 exp_2 的 system prompt 更长，其：

* `total_input_tokens`
* `throughput_tokens_per_s`

被动变高，不能直接说明其服务能力更强。

因此 exp_2 当前只能判断为：

* 在 `actual_success_req_per_s` 上与 baseline 接近
* 暂无足够证据证明其更优

---

### 最终结论 4：当前服务稳态上限约 12 req/s，突发甜点区约 200 并发

在当前硬件与模型条件下：

* **steady 稳态能力**：约 **12 req/s**
* **burst 最优区间**：约 **200 并发**

这是后续线上容量规划的核心依据。

---

## 8. 推荐后续实验方向

建议下一轮只围绕 exp_4 做小范围细调，不要再分散测试无效参数：

### 建议网格

* `max-num-seqs`: 256 / 384 / 512
* `max-num-batched-tokens`: 4096 / 8192 / 12288 / 16384

### 重点关注指标

* `actual_success_req_per_s`
* `latency_p95_s`
* `gpu_compute_usage_avg_pct`
* `kv_cache_usage_avg_pct`
* `gpu_memory_usage_avg_pct`

目标不是单纯拉高某个峰值，而是找到：

> **在生产 padding 场景下，steady 10~20 RPS 与 burst 200 并发附近最稳的一组参数。**

---

## 9. 最终推荐

### 当前最推荐的启动方案

```bash
vllm serve /root/autodl-tmp/models/models/Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --port 8000 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --gpu-memory-utilization 0.7 \
  --enable-prefix-caching \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192
```

### 推荐理由

* 与生产环境 padding 策略一致
* 稳态与突发综合最优
* 是本轮实验中唯一有明确正收益的方向

```

如果你要，我可以继续把这份报告再整理成“更正式一点”的版本，比如加上“摘要 / 实验环境 / 风险说明 / 后续计划”四段式结构。
```
