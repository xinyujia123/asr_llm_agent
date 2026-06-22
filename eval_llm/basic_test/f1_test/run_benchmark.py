import sys
import os
import json
import re
import time
import base64
import binascii
import subprocess
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import (
    MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1,
    DYNAMIC_CONTENT,
    adapt_system_prompt_to_kv_block,
    fetch_kv_cache_block_size
)

enable_padding  = os.getenv("ENABLE_SYSTEM_PROMPT_BLOCK_PADDING", "true").lower()
ENABLE_SYSTEM_PROMPT_BLOCK_PADDING = enable_padding in ("true", "1", "yes")

BATCH_SIZE = 10
REQUEST_TIMEOUT_S = int(os.getenv("BENCHMARK_REQUEST_TIMEOUT_S", "120"))
EXPERIMENT_ARGS = os.getenv("VLLM_EXPERIMENT_ARGS", "") 
EXP_ID = os.getenv("VLLM_EXPERIMENT_ID", "0")
REQUEST_AUDIO = os.getenv("BENCHMARK_REQUEST_AUDIO", "true").lower() in ("true", "1", "yes")
AUDIO_FORMAT = os.getenv("BENCHMARK_AUDIO_FORMAT", "wav").lower()
AUDIO_VOICE = os.getenv("BENCHMARK_AUDIO_VOICE", "alloy")
AUDIO_DIR_NAME = os.getenv("BENCHMARK_AUDIO_DIR_NAME", "audio")

MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "/root/autodl-tmp/models/models/Qwen/Qwen3.6-35B-A3B-FP8")
RESULT_MODEL_DIR = os.getenv("BENCHMARK_OUTPUT_MODEL_DIR", MODEL_NAME.split('/')[-1])

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.jsonl')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), RESULT_MODEL_DIR, f'exp_{EXP_ID}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
AUDIO_DIR = os.path.join(OUTPUT_DIR, AUDIO_DIR_NAME)
if REQUEST_AUDIO:
    os.makedirs(AUDIO_DIR, exist_ok=True)

# Model
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
CHAT_COMPLETIONS_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"
METRICS_URL = os.getenv("VLLM_METRICS_URL", f"{VLLM_BASE_URL.rstrip('/')}/metrics")

def normalize_vllm_launch_args(process_line):
    if not process_line:
        return ""
    raw_line = str(process_line).strip()
    columns = raw_line.split(None, 10)
    command_line = columns[10] if len(columns) >= 11 else raw_line
    match = re.search(r"((?:\S*/)?vllm\s+serve\b.*)$", command_line)
    if not match:
        return command_line.strip()
    extracted = match.group(1).strip()
    tokens = extracted.split(None, 1)
    if not tokens:
        return ""
    head = os.path.basename(tokens[0])
    if head != "vllm":
        return extracted
    tail = tokens[1] if len(tokens) > 1 else ""
    return f"vllm {tail}".strip()

def collect_vllm_launch_args(port=8000):
    lsof_command = ["lsof", "-t", f"-iTCP:{int(port)}", "-sTCP:LISTEN"]
    try:
        lsof_result = subprocess.run(
            lsof_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
    except Exception:
        return ""
    pid_list = []
    for line in (lsof_result.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.isdigit():
            pid_list.append(int(stripped))
    if not pid_list:
        return ""
    selected_line = ""
    for pid in pid_list:
        grep_command = f"ps -auxww | grep -w '{pid}' | grep -v grep"
        try:
            ps_result = subprocess.run(
                ["bash", "-lc", grep_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3
            )
        except Exception:
            continue
        lines = [line.strip() for line in (ps_result.stdout or "").splitlines() if line.strip()]
        if not lines:
            continue
        candidate_line = lines[0]
        if not selected_line:
            selected_line = candidate_line
        lowered = candidate_line.lower()
        if ("vllm" in lowered) or ("api_server" in lowered):
            selected_line = candidate_line
            break
    return normalize_vllm_launch_args(selected_line)

def resolve_vllm_port(base_url):
    try:
        parsed = urllib.parse.urlparse(base_url)
    except Exception:
        return 8000
    return int(parsed.port) if parsed.port else 8000

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

def extract_message_text(message):
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text") or part.get("transcript") or ""))
            else:
                text_parts.append(str(part))
        text = "".join(text_parts)
    elif content is None:
        text = ""
    else:
        text = str(content)

    audio = message.get("audio") if isinstance(message, dict) else None
    if not text and isinstance(audio, dict) and audio.get("transcript"):
        text = str(audio["transcript"])
    return text

def collect_audio_candidates(value):
    candidates = []
    if isinstance(value, str):
        candidates.append(value)
        return candidates
    if isinstance(value, list):
        for item in value:
            candidates.extend(collect_audio_candidates(item))
        return candidates
    if not isinstance(value, dict):
        return candidates

    for key in ("data", "audio", "base64", "b64_json"):
        nested_value = value.get(key)
        if isinstance(nested_value, str):
            candidates.append(nested_value)
        elif isinstance(nested_value, (dict, list)):
            candidates.extend(collect_audio_candidates(nested_value))

    for key in ("content", "output"):
        nested_value = value.get(key)
        if isinstance(nested_value, list):
            candidates.extend(collect_audio_candidates(nested_value))

    return candidates

def decode_audio_candidate(candidate):
    raw = candidate.strip()
    if not raw:
        return None
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None

def save_response_audio(message, audio_path):
    audio_meta = {
        "audio_path": None,
        "audio_format": AUDIO_FORMAT,
        "audio_bytes": 0,
        "audio_error": None
    }
    if not REQUEST_AUDIO:
        return audio_meta

    candidates = []
    if isinstance(message, dict):
        candidates.extend(collect_audio_candidates(message.get("audio")))
        candidates.extend(collect_audio_candidates(message.get("content")))

    for candidate in candidates:
        audio_bytes = decode_audio_candidate(candidate)
        if not audio_bytes:
            continue
        try:
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            audio_meta.update({
                "audio_path": audio_path,
                "audio_bytes": len(audio_bytes)
            })
            return audio_meta
        except OSError as e:
            audio_meta["audio_error"] = str(e)
            return audio_meta

    audio_meta["audio_error"] = "audio_requested_but_not_returned"
    return audio_meta

def request_vllm(messages, temperature=0.0, max_tokens=1024, timeout=REQUEST_TIMEOUT_S):
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
    if REQUEST_AUDIO:
        payload["modalities"] = ["text", "audio"]
        payload["audio"] = {
            "voice": AUDIO_VOICE,
            "format": AUDIO_FORMAT
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
            message = result["choices"][0]["message"]
            return result, extract_message_text(message)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        print(f"HTTP Error when calling vLLM: {e.code} {error_body}")
    except Exception as e:
        print(f"Error when calling vLLM: {e}")
    return None, ""

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
    audio_path = os.path.join(AUDIO_DIR, f"{i:05d}.{AUDIO_FORMAT}") if REQUEST_AUDIO else None
    start_time = time.perf_counter()
    response_json, generated_text = request_vllm(messages)
    request_time_seconds = time.perf_counter() - start_time
    message = {}
    if response_json and response_json.get("choices"):
        message = response_json["choices"][0].get("message") or {}
    audio_meta = save_response_audio(message, audio_path) if audio_path else {
        "audio_path": None,
        "audio_format": AUDIO_FORMAT,
        "audio_bytes": 0,
        "audio_error": None
    }
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
            'audio_path': audio_meta['audio_path'],
            'audio_error': audio_meta['audio_error'],
            'request_time_seconds': request_time_seconds
        }
    result_item = {
        'input': record['input_text'],
        'expected': expected_json,
        'predicted': predicted_json,
        'generated_text': generated_text,
        'audio_path': audio_meta['audio_path'],
        'audio_format': audio_meta['audio_format'],
        'audio_bytes': audio_meta['audio_bytes'],
        'audio_error': audio_meta['audio_error'],
        'request_time_seconds': request_time_seconds,
        'metrics': record_stats
    }
    return i, result_item, error_case, record_stats, request_time_seconds

def main():
    print(f"Using remote vLLM service: {CHAT_COMPLETIONS_URL}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Experiment args: {EXPERIMENT_ARGS if EXPERIMENT_ARGS else '(default)'}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Request timeout(s): {REQUEST_TIMEOUT_S}")
    print(f"Enable system prompt block padding: {ENABLE_SYSTEM_PROMPT_BLOCK_PADDING}")
    print(f"Request audio output: {REQUEST_AUDIO}")
    if REQUEST_AUDIO:
        print(f"Audio output dir: {AUDIO_DIR}")
        print(f"Audio request format: {AUDIO_FORMAT}, voice: {AUDIO_VOICE}")
    
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
    vllm_port = resolve_vllm_port(VLLM_BASE_URL)
    vllm_launch_args = collect_vllm_launch_args(vllm_port)
    block_size = fetch_kv_cache_block_size(METRICS_URL)
    final_metrics['summary'].update({
        'vllm_launch_args': vllm_launch_args,
        'model_name': MODEL_NAME,
        'experiment_args': EXPERIMENT_ARGS,
        'base_url': VLLM_BASE_URL,
        'chat_completions_url': CHAT_COMPLETIONS_URL,
        'metrics_url': METRICS_URL,
        'batch_size': BATCH_SIZE,
        'request_timeout_s': REQUEST_TIMEOUT_S,
        'enable_system_prompt_block_padding': ENABLE_SYSTEM_PROMPT_BLOCK_PADDING,
        'request_audio': REQUEST_AUDIO,
        'audio_dir': AUDIO_DIR if REQUEST_AUDIO else None,
        'audio_format': AUDIO_FORMAT,
        'audio_voice': AUDIO_VOICE,
        'block_size': block_size
    })

    print("\nSummary metrics:")
    print(f"Total Category Avg F1: {category_avg_f1:.4f}")
    print(f"Total Sample Aggregate F1: {sample_aggregate_f1:.4f} (P={total_precision:.4f}, R={total_recall:.4f}, TP={total_tp}, FP={total_fp}, FN={total_fn})")
    print(f"Average Request Time: {avg_request_time_seconds:.4f}s ({request_count} requests)")
    print(f"KV Cache Block Size: {block_size}")
        
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
