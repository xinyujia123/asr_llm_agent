模型一致性实验结果分析
1. 模型在同一提示词下，输出的稳定性
deepseek v3.2：temperature = 0.0
thinking
三个文件的彼此相似度：76.67% 66.67% 73.33%
no thinking
三个文件的彼此相似度：100.00% 96.67% 96.67%


qwen3 max thinking: temperature = 0.0
thinking ->1
三个文件的彼此相似度：86.67% 83.33% 96.67%(1 2)
no thinking
三个文件的彼此相似度：76.67% 83.33% 80.00%
think 和 no think的彼此相似度：50.00%


kimi: temperature = 默认值
thinking ->1
三个文件的彼此相似度：80.00% 83.33% 80.00%
no thinking
三个文件的彼此相似度：90.00% 66.67% 73.33%
think 和 no think的彼此相似度：73.33% 76.67% 66.67%


glm: temperature = 默认值
thinking
三个文件的彼此相似度: 
no thinking
三个文件的彼此相似度： 93.33% 93.33% 86.67%


### 模型之间的对比
##### think
kimi vs qwen 76.67%
kimi vs deepseek 73.33%
qwen vs deepseek 70.00%


##### no think
kimi vs qwen 50.00%
kimi vs deepseek 86.67%
qwen vs deepseek 46.67%