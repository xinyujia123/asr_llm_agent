import json
import time
from datetime import datetime
from typing import List, Dict, Any
from .llm_client import LLMClient
from .config import BENCHMARK_SAVE_PATH
from .schema import EXPECTED_KEYS

# 导入提示词
import sys
sys.path.append('/workspace/audio_llm_agent')
from sak.prompts import MEDICAL_EXTRACTOR_PROMPT

def calculate_score(gt_json: Dict[str, Any], pred_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    对比标准答案和预测答案，计算得分。
    """
    correct_count = 0
    total_fields = len(EXPECTED_KEYS)
    field_results = {}

    for key in EXPECTED_KEYS:
        gt_val = gt_json.get(key)
        pred_val = pred_json.get(key)
        
        # 统一处理 null 和 None
        if gt_val == "null": gt_val = None
        if pred_val == "null": pred_val = None
        
        # 比较逻辑
        is_correct = str(gt_val).strip().lower() == str(pred_val).strip().lower()
        if is_correct:
            correct_count += 1
        
        field_results[key] = is_correct

    return {
        "accuracy": correct_count / total_fields if total_fields > 0 else 0,
        "correct_fields": correct_count,
        "total_fields": total_fields,
        "details": field_results
    }

def run_evaluation(target_provider: str):
    print(f"Starting evaluation for provider: {target_provider}")
    
    # 1. 加载 Benchmark
    benchmark_data = []
    try:
        with open(BENCHMARK_SAVE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                benchmark_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Benchmark file not found: {BENCHMARK_SAVE_PATH}. Please run generate_benchmark.py first.")
        return

    # 2. 初始化待测模型
    client = LLMClient(target_provider)
    
    # 3. 运行测试
    results = []
    total_accuracy = 0
    format_pass_count = 0
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, item in enumerate(benchmark_data):
        print(f"[{i+1}/{len(benchmark_data)}] Testing scenario: {item['title']}")
        
        prompt = MEDICAL_EXTRACTOR_PROMPT.format(CURRENT_SYS_TIME=current_time)
        response_content = client.chat(prompt, item['input_text'])
        
        # 格式校验
        is_json = False
        pred_json = {}
        try:
            clean_content = response_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            pred_json = json.loads(clean_content)
            is_json = True
            format_pass_count += 1
        except:
            print(f"  Format error for {item['id']}")
            is_json = False

        # 语义校验
        if is_json:
            score_info = calculate_score(item['ground_truth'], pred_json)
            total_accuracy += score_info['accuracy']
            results.append({
                "id": item['id'],
                "accuracy": score_info['accuracy'],
                "format_pass": True
            })
        else:
            results.append({
                "id": item['id'],
                "accuracy": 0,
                "format_pass": False
            })
        
        time.sleep(0.5)

    # 4. 统计结果
    num_items = len(benchmark_data)
    final_format_rate = format_pass_count / num_items if num_items > 0 else 0
    final_avg_accuracy = total_accuracy / num_items if num_items > 0 else 0
    
    print("\n" + "="*30)
    print(f"Evaluation Results for {target_provider}:")
    print(f"Total Scenarios: {num_items}")
    print(f"JSON Format Pass Rate: {final_format_rate:.2%}")
    print(f"Average Semantic Accuracy: {final_avg_accuracy:.2%}")
    print("="*30)

if __name__ == "__main__":
    # 示例：运行 Kimi 的评估
    # run_evaluation("kimi")
    pass
