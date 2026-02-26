import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


# 默认用于生成标准答案的模型
GROUND_TRUTH_MODEL = "qwen"
THINKING = False                 
INFERENCE = True
EXP = 'gold_exp_2'

# 数据路径
DATA_DIR = "/workspace/dataset/asr_llm"
RAW_DATA_PATH = os.path.join(DATA_DIR, "nurse_script_tn.txt")
BENCHMARK_SAVE_DIR= os.path.join("/workspace/audio_llm_agent/eval_llm/benchmark", EXP)


# API 配置
# 建议通过环境变量设置，或者直接在此修改
API_CONFIG = {
    "qwen": {
        "api_key": os.getenv("QWEN_API_KEY", "YOUR_QWEN_API_KEY"),
        "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-max-2026-01-23"
    },
    "kimi": {
        "api_key": os.getenv("KIMI_API_KEY", "YOUR_KIMI_API_KEY"),
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.5"
    },
    "glm": {
        "api_key": os.getenv("GLM_API_KEY", "YOUR_GLM_API_KEY"),
        "model": "glm-4.7"
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/v1",
        "model":  "deepseek-reasoner" if THINKING else "deepseek-chat"
    },
    "baichuan": {
        "api_key": os.getenv("BAICHUAN_API_KEY", "YOUR_BEICHUAN_API_KEY"),
        "base_url": "https://api.baichuan-ai.com/v1",
        "model":  "Baichuan-M3-PLUS"
    }
}


