import os
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Any
from .utils import parse_nurse_scripts
from .llm_client import LLMClient,ZP_LLMClient
from .config import RAW_DATA_PATH, BENCHMARK_SAVE_PATH, GROUND_TRUTH_MODEL
from .schema import EXPECTED_KEYS

# 导入提示词
import sys
sys.path.append('/workspace/audio_llm_agent')
from sak.prompts import MEDICAL_EXTRACTOR_PROMPT

def generate_benchmark():
    # 1. 解析数据
    limit = 30
    thinking = False
    start_time = time.time()
    scripts = parse_nurse_scripts(RAW_DATA_PATH)
    end_time = time.time()
    print(f"Loaded {len(scripts)} scenarios from {RAW_DATA_PATH} in {end_time - start_time:.3f} seconds")

    # 2. 初始化 LLM 客户端
    try:
        if 'glm' in GROUND_TRUTH_MODEL:
            client = ZP_LLMClient(GROUND_TRUTH_MODEL)
        else:
            client = LLMClient(GROUND_TRUTH_MODEL)
    except Exception as e:
        print(f"Failed to initialize LLM client: {e}")
        return

    # 3. 准备生成
    detailed_results = []
    results = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 限制生成数量进行测试，如果需要全量生成请修改此处
    
    for i, script in enumerate(scripts[:limit]):
        print(f"[{i+1}/{limit}] Processing: {script['title']}")
        
        
        now = datetime.now()
        formatted_time= now.strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = MEDICAL_EXTRACTOR_PROMPT.replace("{CURRENT_SYS_TIME}", formatted_time)
        
        # 调用 LLM
        start_time = time.time()
        response_content, reasoning_content = client.chat(system_prompt, script['text'], thinking=thinking)
        end_time = time.time()
        print(f"  LLM response time: {end_time - start_time:.3f} seconds")
        
        keys = EXPECTED_KEYS
        # 尝试解析 JSON
        try:
            start_time = time.time()
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            raw_data = json.loads(match.group())

            missed_keys = []
            appended_keys = []
            for key in keys:
                if key not in raw_data:
                    missed_keys.append(key)
            for key in raw_data:
                if key not in keys:
                    appended_keys.append(key)

            cleaned_data = {}
            for key in keys:            
                val = raw_data.get(key)
                # 2. 更加精细的处理：过滤掉空值或无意义的字符串
                if val is None or str(val).lower() in ["none", "null", "n/a", "unknown", "未知"]:
                    cleaned_data[key] = None
                else:
                    cleaned_data[key] = str(val).strip() # 去掉多余空格
            end_time = time.time()
            print(f"  JSON parsing time: {end_time - start_time:.3f} seconds")

            detailed_results.append({
                "id": script['id'],
                "input_text": script['text'],
                "raw_output": response_content,
                "reasoning_output": reasoning_content if reasoning_content else None,
                "cleaned_json": cleaned_data,
                "model_used": GROUND_TRUTH_MODEL,
                "missed_keys": missed_keys,
                "appended_keys": appended_keys,
            })
        except Exception as e:
            print(f"  Failed to parse JSON for {script['id']}: {e}")
            detailed_results.append({
                "id": script['id'],
                "input_text": script['text'],
                "raw_output": response_content,
                "reasoning_output": reasoning_content if reasoning_content else None,
                "model_used": GROUND_TRUTH_MODEL,
            })
            continue
        
        # 避免请求过快
        time.sleep(2)

    name, ext = os.path.splitext(BENCHMARK_SAVE_PATH)
    new_path = f"{name}_{GROUND_TRUTH_MODEL}_{'think' if thinking else 'no_think'}_{limit}{ext}"
    # 4. 保存结果
    with open(new_path, 'w', encoding='utf-8') as f:
        for entry in detailed_results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully generated benchmark with {len(results)} entries at {new_path}")

if __name__ == "__main__":
    # 注意：运行此脚本需要配置 API Key
    generate_benchmark()
