import sys
import os
import json
import re
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

# Load env vars
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# --- 配置项 ---
LLM_API_KEY = os.getenv("QWEN_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_NAME = "qwen-flash" # Consistent with smart_nursing_agent_0305.py

# Add prompt path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1
except ImportError:
    # Fallback if running from a different directory
    sys.path.append('/workspace/audio_llm_agent/code/eval_llm/basic_test')
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1

# Paths
DATASET_PATH = '/workspace/audio_llm_agent/code/eval_llm/basic_test/dataset.jsonl'
OUTPUT_DIR = '/workspace/audio_llm_agent/code/eval_llm/basic_test/f1_test_api'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def construct_messages(record):
    # Prepare the base prompt with time replacements
    sys_time = record.get('current_sys_time', '')
    day_before = record.get('the_day_before_yesterday', '')
    
    system_content = MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1.replace(
        "{CURRENT_SYS_TIME}", sys_time
    ).replace(
        "{THE_DAY_BEFORE_YESTERDAY}", day_before
    )
    
    # User input
    user_input = record['input_text']
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"分析文本：{user_input}"}
    ]
    return messages

def parse_json(text):
    # Try to find JSON block
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to find just braces
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = text
            
    try:
        return json.loads(json_str)
    except:
        return {}

def normalize_value(val):
    if isinstance(val, list):
        # Sort list and strip strings
        return sorted([str(v).strip() for v in val])
    # Normalize float string if it looks like a number
    s = str(val).strip()
    try:
        f = float(s)
        # Check if integer
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return s

def compare_record(expected, predicted):
    # Returns metrics for this record
    field_stats = {} # {field: {'tp': 0, 'fp': 0, 'fn': 0}}
    
    all_keys = set(expected.keys()) | set(predicted.keys())
    
    for key in all_keys:
        if key not in field_stats:
            field_stats[key] = {'tp': 0, 'fp': 0, 'fn': 0}
            
        exp_val = expected.get(key)
        pred_val = predicted.get(key)
        
        if exp_val is not None and pred_val is not None:
            # Both exist, check value
            if normalize_value(exp_val) == normalize_value(pred_val):
                field_stats[key]['tp'] = 1
            else:
                # Value mismatch counts as FP (wrong prediction) and FN (missed correct value)
                field_stats[key]['fp'] = 1
                field_stats[key]['fn'] = 1
        elif exp_val is not None:
            # Expected but not in predicted
            field_stats[key]['fn'] = 1
        elif pred_val is not None:
            # Predicted but not in expected
            field_stats[key]['fp'] = 1
            
    return field_stats

async def process_record(client, record, semaphore):
    async with semaphore:
        messages = construct_messages(record)
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                temperature=0.1,
                # response_format={"type": "json_object"} 
            )
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            print(f"Error processing record: {e}")
            return ""

async def main():
    print(f"Initializing API client with model {LLM_MODEL_NAME}...")
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    
    print(f"Loading data from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    print(f"Loaded {len(data)} records.")
    
    # Limit concurrency to avoid rate limits
    semaphore = asyncio.Semaphore(10)
    
    print("Generating responses...")
    tasks = [process_record(client, record, semaphore) for record in data]
    generated_texts = await asyncio.gather(*tasks)
    
    results = []
    error_cases = []
    global_stats = {} # {field: {'tp': 0, 'fp': 0, 'fn': 0}}
    
    for i, generated_text in enumerate(generated_texts):
        predicted_json = parse_json(generated_text)
        expected_json = data[i]['expected_output']
        
        record_stats = compare_record(expected_json, predicted_json)
        
        # Check for errors
        has_error = False
        error_details = {}
        for key, stats in record_stats.items():
            if stats['fp'] > 0 or stats['fn'] > 0:
                has_error = True
                error_details[key] = {
                    'expected': expected_json.get(key),
                    'predicted': predicted_json.get(key),
                    'type': 'FP' if stats['fp'] > 0 and stats['fn'] == 0 else ('FN' if stats['fn'] > 0 and stats['fp'] == 0 else 'Mismatch')
                }
        
        if has_error:
            error_cases.append({
                'id': i,
                'input': data[i]['input_text'],
                'expected': expected_json,
                'predicted': predicted_json,
                'error_details': error_details,
                'generated_text': generated_text
            })

        # Aggregate global stats
        for key, stats in record_stats.items():
            if key not in global_stats:
                global_stats[key] = {'tp': 0, 'fp': 0, 'fn': 0}
            global_stats[key]['tp'] += stats['tp']
            global_stats[key]['fp'] += stats['fp']
            global_stats[key]['fn'] += stats['fn']
            
        results.append({
            'input': data[i]['input_text'],
            'expected': expected_json,
            'predicted': predicted_json,
            'generated_text': generated_text,
            'metrics': record_stats
        })
        
    # Calculate F1 per field
    print("\nMetrics per field:")
    final_metrics = {}
    
    # Sort keys for consistent output
    sorted_keys = sorted(global_stats.keys())
    
    for key in sorted_keys:
        stats = global_stats[key]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        final_metrics[key] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        print(f"{key}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f} (TP={tp}, FP={fp}, FN={fn})")
        
    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    error_cases_path = os.path.join(OUTPUT_DIR, 'error_cases.json')
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    with open(error_cases_path, 'w', encoding='utf-8') as f:
        json.dump(error_cases, f, ensure_ascii=False, indent=2)
        
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Found {len(error_cases)} error cases.")

if __name__ == "__main__":
    asyncio.run(main())
