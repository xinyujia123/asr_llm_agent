## 35B-A3B-FP8 与 35B-A3B-GPTQ-Int4（exp_0）简短实验报告

### 实验对象

对比目录：

- `Qwen3.5-35B-A3B-FP8/exp_0`
- `Qwen3.5-35B-A3B-GPTQ-Int4/exp_0`

两组都使用了 `--max-model-len 8192`、`--gpu-memory-utilization 0.7`、`--enable-prefix-caching`。但要注意，这两组 **并非完全同条件**：  
FP8 的 `enable_system_prompt_block_padding=false`，而 GPTQ-Int4 的 steady 测试里是 `true`；同时 steady 测试里的总输入 token 规模也明显不同，所以 steady 结果更适合看总体趋势，不适合做特别死板的一一横向定量比较。:contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

### 总体结论

从 `exp_0` 看：

- **GPTQ-Int4 在低并发和中低负载下更快**
- **FP8 在更高并发下没有明显翻盘，整体上 GPTQ-Int4 仍略占优或接近**
- 两者高压下都没有把 GPU compute 打到极限 99% 常驻，说明这组 35B-A3B 实验里，瓶颈不像 27B 那组那样明显偏“纯算力顶满”
- burst 模式下两者峰值都在 **300 并发附近**，但 GPTQ-Int4 的单请求延迟通常略低、低中并发 req/s 更高。:contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

### Burst 并发压测

| 并发数 | FP8 req/s | GPTQ-Int4 req/s | 更优 |
|---|---:|---:|---|
| 1 | 2.3757 | 2.3803 | 接近 |
| 20 | 10.9448 | 14.0394 | GPTQ |
| 50 | 22.4332 | 25.5375 | GPTQ |
| 100 | 32.9478 | 30.9414 | FP8 |
| 200 | 47.0512 | 45.0898 | FP8 |
| 300 | 51.2255 | 49.3685 | FP8 |

结论：  
burst 下两者都很强，但趋势是 **GPTQ-Int4 在 20/50 并发更优，FP8 在 100 以上略反超**；不过反超幅度不大，整体差距比 27B 那组更小。:contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

### Steady RPS 压测

| 目标 RPS | FP8 实际 req/s | GPTQ-Int4 实际 req/s | 更优 |
|---|---:|---:|---|
| 1 | 1.0360 | 1.0359 | 接近 |
| 5 | 4.6860 | 4.7749 | GPTQ |
| 10 | 9.1989 | 9.3702 | GPTQ |
| 20 | 17.1254 | 18.3997 | GPTQ |
| 40 | 29.4705 | 30.8484 | GPTQ |

结论：  
steady 模式下，**GPTQ-Int4 基本全程略优于 FP8**。但这里要再次强调，steady 结果受 `enable_system_prompt_block_padding` 和输入 token 总量差异影响较大，因此更适合解读为“这套实测配置下 GPTQ-Int4 更占优势”，而不是模型格式本身在所有条件下必然更强。:contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

### 延迟与吞吐

低负载下 GPTQ-Int4 延迟更低：

- burst 20：FP8 `1.2684s`，GPTQ `1.0323s`
- burst 50：FP8 `1.5311s`，GPTQ `1.3149s`
- steady 20：FP8 `2.7880s`，GPTQ `1.4142s`
- steady 40：FP8 `5.7837s`，GPTQ `5.2023s`。:contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14} :contentReference[oaicite:15]{index=15}

吞吐方面，burst 高并发时 FP8 略高：

- burst 300：FP8 `117652 tok/s`，GPTQ `113385 tok/s`。:contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17}

但 steady 下由于两边总输入 token 规模不同，GPTQ-Int4 的 `throughput_tokens_per_s` 更高，不能简单解读为“模型本体绝对更强”，需要结合测试样本规模一起看。:contentReference[oaicite:18]{index=18} :contentReference[oaicite:19]{index=19}

### 资源使用观察

显存占用这里有一个明显现象：

- FP8 burst/steady 的 GPU 占用大约在 `76.22%`
- GPTQ-Int4 burst 中接近 `82.63%`
- GPTQ-Int4 steady 又回到约 `76.35%`

这说明这组实验里，显存表现并不完全一致，可能和具体加载方式、block padding、运行时预留方式有关，不能简单得出“量化一定更省最终占用”的结论。:contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21} :contentReference[oaicite:22]{index=22} :contentReference[oaicite:23]{index=23}

### 简短结论

一句话概括：

> **在这组 35B-A3B 的 exp_0 实验里，GPTQ-Int4 整体表现不差，且在多数低中负载场景下优于 FP8；FP8 只在 burst 高并发段略占上风。**

如果更看重：

- **低延迟、中低并发体验**：优先考虑 `Qwen3.5-35B-A3B-GPTQ-Int4`
- **极限 burst 吞吐**：`Qwen3.5-35B-A3B-FP8` 仍有一点优势。:contentReference[oaicite:24]{index=24} :contentReference[oaicite:25]{index=25} :contentReference[oaicite:26]{index=26} :contentReference[oaicite:27]{index=27}