模型一致性实验结果分析
1. 模型在同一提示词下，输出的稳定性
deepseek v3.2：temperature = 0.0
inference
三个文件的彼此相似度：83.33%, 76.67%, 73.33%
no inference
三个文件的彼此相似度：

qwen3 max thinking: temperature = 0.0
inference
三个文件的彼此相似度：80.00%, 93.33%, 83.33%
no inference
三个文件的彼此相似度：

kimi: temperature = 默认值
inference
三个文件的彼此相似度：80.00%, 93.33%, 86.67%
no inference
三个文件的彼此相似度：


2.不同模型输出的内容对比：
inference:
kimi qwen: 83.33%
kimi deepseek: 80.00%
qwen deepseek: 86.67%

no inference:
