# 09 SBAR From Nursing Records

实验目的：根据原始护理记录生成交接班 SBAR 摘要。

这个实验使用同一套测试集、两份不同提示词：

- `prompt_professional.md`：详细专业的指导型提示词，明确 SBAR 结构、护理风险、待办事项、升级条件和安全边界。
- `prompt_simple.md`：简单直白的引导型提示词，模拟非医学专业提示词构筑者的写法。

目的是观察医疗/护理大模型相比通用大模型，在提示词不专业时，是否仍能更专业地完成交接班摘要任务。

数据字段：

- `id`：样例编号。
- `scenario`：场景名称。
- `input`：原始护理记录。
- `expected_output`：优秀 SBAR 摘要参考答案。

运行本实验：

```bash
cd /Users/jiaxinyu/ChangHong/projects/nurse_asr_agent/asr_llm_agent/eval_llm/nurse_llm_test/09_sbar_from_nursing_records
./run_compare_models.sh
```

输出位置：

- 专业提示词结果：`results/professional_guided/<model-prefix>_results.json`
- 简单提示词结果：`results/simple_guided/<model-prefix>_results.json`
- 日志：`logs/<prompt-name>/<model-prefix>.log`

默认模型：

- `Baichuan-M3`
- `Baichuan-M3-Plus`
- `qwen3.6-flash`
- `qwen3.7-plus`

默认关闭百川搜索增强：

```bash
export BAICHUAN_WITH_SEARCH_ENHANCE=false
```
