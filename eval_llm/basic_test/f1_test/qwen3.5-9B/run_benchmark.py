import sys
import os
import json
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import (
    MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1,
    DYNAMIC_CONTENT,
    adapt_system_prompt_to_kv_block
)

ENABLE_SYSTEM_PROMPT_BLOCK_PADDING = True
BATCH_SIZE = 10

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.jsonl')
OUTPUT_DIR = os.path.dirname(__file__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3.5-9B")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
CHAT_COMPLETIONS_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"
METRICS_URL = os.getenv("VLLM_METRICS_URL", f"{VLLM_BASE_URL.rstrip('/')}/metrics")

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_padding_suffix():
    if not ENABLE_SYSTEM_PROMPT_BLOCK_PADDING:
        return ""
    benchmark_prompt = MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1
    padded_prompt, meta = adapt_system_prompt_to_kv_block(
        system_prompt=benchmark_prompt,
        enable_padding=ENABLE_SYSTEM_PROMPT_BLOCK_PADDING,
        metrics_url=METRICS_URL,
        model_name=MODEL_NAME,
        vllm_base_url=VLLM_BASE_URL
    )
    padding_units = int(meta.get("padding_units") or 0)
    if padding_units <= 0:
        return ""
    return padded_prompt[-padding_units:]

def construct_messages(record, system_padding_suffix):
    sys_time = record.get('current_sys_time', '')
    day_before = record.get('the_day_before_yesterday', '')
    system_content = MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1 + system_padding_suffix
    
    user_input = DYNAMIC_CONTENT.replace(
        "{CURRENT_SYS_TIME}", sys_time
    ).replace(
        "{THE_DAY_BEFORE_YESTERDAY}", day_before
    ) + record['input_text']
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]

def request_vllm(messages, temperature=0.0, max_tokens=1024, timeout=120):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    request = urllib.request.Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        print(f"HTTP Error when calling vLLM: {e.code} {error_body}")
    except Exception as e:
        print(f"Error when calling vLLM: {e}")
    return ""

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

def process_record(i, record, system_padding_suffix):
    messages = construct_messages(record, system_padding_suffix)
    start_time = time.perf_counter()
    generated_text = request_vllm(messages)
    request_time_seconds = time.perf_counter() - start_time
    predicted_json = parse_json(generated_text)
    expected_json = record['expected_output']
    record_stats = compare_record(expected_json, predicted_json)
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
    error_case = None
    if has_error:
        error_case = {
            'id': i,
            'input': record['input_text'],
            'expected': expected_json,
            'predicted': predicted_json,
            'error_details': error_details,
            'generated_text': generated_text,
            'request_time_seconds': request_time_seconds
        }
    result_item = {
        'input': record['input_text'],
        'expected': expected_json,
        'predicted': predicted_json,
        'generated_text': generated_text,
        'request_time_seconds': request_time_seconds,
        'metrics': record_stats
    }
    return i, result_item, error_case, record_stats, request_time_seconds

def main():
    print(f"Using remote vLLM service: {CHAT_COMPLETIONS_URL}")
    print(f"Using model: {MODEL_NAME}")
    
    print(f"Loading data from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    print(f"Loaded {len(data)} records.")
    
    print("Generating...")
    system_padding_suffix = prepare_padding_suffix()
    
    results = [None] * len(data)
    error_cases = []
    global_stats = {} # {field: {'tp': 0, 'fp': 0, 'fn': 0}}
    total_request_time_seconds = 0.0
    
    for batch_start in range(0, len(data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(data))
        indexed_batch = list(enumerate(data[batch_start:batch_end], start=batch_start))
        worker_count = min(BATCH_SIZE, len(indexed_batch))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(process_record, idx, record, system_padding_suffix)
                for idx, record in indexed_batch
            ]
            for future in as_completed(futures):
                i, result_item, error_case, record_stats, request_time_seconds = future.result()
                results[i] = result_item
                total_request_time_seconds += request_time_seconds
                if error_case is not None:
                    error_cases.append(error_case)
                for key, stats in record_stats.items():
                    if key not in global_stats:
                        global_stats[key] = {'tp': 0, 'fp': 0, 'fn': 0}
                    global_stats[key]['tp'] += stats['tp']
                    global_stats[key]['fp'] += stats['fp']
                    global_stats[key]['fn'] += stats['fn']
        
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

    total_fields = len(sorted_keys)
    category_avg_f1 = sum(final_metrics[key]['f1'] for key in sorted_keys) / total_fields if total_fields > 0 else 0.0
    total_tp = sum(global_stats[key]['tp'] for key in sorted_keys)
    total_fp = sum(global_stats[key]['fp'] for key in sorted_keys)
    total_fn = sum(global_stats[key]['fn'] for key in sorted_keys)
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    sample_aggregate_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
    request_count = len(data)
    avg_request_time_seconds = total_request_time_seconds / request_count if request_count > 0 else 0.0

    final_metrics['summary'] = {
        'total_category_avg_f1': category_avg_f1,
        'total_sample_aggregate_f1': sample_aggregate_f1,
        'precision': total_precision,
        'recall': total_recall,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'field_count': total_fields,
        'request_count': request_count,
        'total_request_time_seconds': total_request_time_seconds,
        'avg_request_time_seconds': avg_request_time_seconds
    }

    print("\nSummary metrics:")
    print(f"Total Category Avg F1: {category_avg_f1:.4f}")
    print(f"Total Sample Aggregate F1: {sample_aggregate_f1:.4f} (P={total_precision:.4f}, R={total_recall:.4f}, TP={total_tp}, FP={total_fp}, FN={total_fn})")
    print(f"Average Request Time: {avg_request_time_seconds:.4f}s ({request_count} requests)")
        
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
    main()
