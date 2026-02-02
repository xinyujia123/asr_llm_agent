# Nurse Voice Transcription LLM Benchmark

本项目旨在构建一个针对护士语音转录医疗表单任务的 LLM 评测基准。

## 目录结构

- `config.py`: API 密钥和模型配置。
- `llm_client.py`: 统一的 LLM 接口调用工具。
- `schema.py`: 医疗表单的 JSON 结构定义。
- `utils.py`: 原始文本解析和通用工具函数。
- `generate_benchmark.py`: 调用顶级模型（如 Qwen-Max）生成标准答案（Ground Truth）。
- `evaluate_llm.py`: 运行 Benchmark 并评估待测模型的表现。

## 使用步骤

### 1. 配置 API Key
在 `config.py` 中填入你的 API Key，或者设置对应的环境变量：
- `QWEN_API_KEY`
- `KIMI_API_KEY`
- `DEEPSEEK_API_KEY`

### 2. 生成标准数据集 (Ground Truth)
运行以下脚本，将原始 ASR 文本转换为高质量的 JSON 标准库：
```bash
python3 -m audio_llm_agent.eval_llm.generate_benchmark
```
生成的文件将保存为 `benchmark_standard.jsonl`。

### 3. 运行评估
选择一个模型进行评测：
```python
# 在 evaluate_llm.py 中修改或调用
run_evaluation("kimi")
```
运行评测脚本：
```bash
python3 -m audio_llm_agent.eval_llm.evaluate_llm
```

## 评估指标
- **JSON Format Pass Rate**: 输出是否符合合法的 JSON 格式。
- **Average Semantic Accuracy**: 提取的字段内容与标准答案的匹配程度。
