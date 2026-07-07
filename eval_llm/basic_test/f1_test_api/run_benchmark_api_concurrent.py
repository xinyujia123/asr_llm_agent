import sys
import os
import json
import re
import asyncio
import time

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:
    AsyncOpenAI = None

try:
    from dotenv import load_dotenv, find_dotenv
except ModuleNotFoundError:
    def find_dotenv(*args, **kwargs):
        return ""

    def load_dotenv(*args, **kwargs):
        return False

# Load env vars
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# --- 配置项 ---
BAICHUAN_DEFAULT_BASE_URL = "https://api.baichuan-ai.com/v1/"


def first_env(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def parse_bool_env(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "y", "on")


def parse_int_env(name, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def parse_float_env(name, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def infer_provider():
    explicit_provider = os.getenv("LLM_PROVIDER")
    if explicit_provider:
        return explicit_provider.strip().lower()

    model_hint = first_env("BAICHUAN_MODEL_NAME", "LLM_MODEL_NAME") or ""
    base_url_hint = first_env("BAICHUAN_BASE_URL", "LLM_BASE_URL") or ""
    if (
        os.getenv("BAICHUAN_API_KEY")
        or "baichuan" in model_hint.lower()
        or "baichuan-ai.com" in base_url_hint.lower()
    ):
        return "baichuan"
    return "qwen"


def normalize_openai_base_url(base_url):
    if not base_url:
        return base_url
    normalized = base_url.strip()
    suffix = "/chat/completions"
    if normalized.rstrip("/").endswith(suffix):
        normalized = normalized.rstrip("/")[: -len(suffix)]
    return normalized.rstrip("/") + "/"


def safe_filename(value):
    name = str(value or "").strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name.strip("-._") or "model"


LLM_PROVIDER = infer_provider()
if LLM_PROVIDER == "baichuan":
    LLM_API_KEY = first_env("BAICHUAN_API_KEY", "LLM_API_KEY")
    LLM_BASE_URL = normalize_openai_base_url(
        os.getenv("BAICHUAN_BASE_URL") or BAICHUAN_DEFAULT_BASE_URL
    )
    LLM_MODEL_NAME = first_env("BAICHUAN_MODEL_NAME", "LLM_MODEL_NAME") or "Baichuan-M3"
else:
    LLM_API_KEY = first_env("LLM_API_KEY", "QWEN_API_KEY")
    LLM_BASE_URL = normalize_openai_base_url(os.getenv("LLM_BASE_URL"))
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-flash")

LLM_TEMPERATURE = parse_float_env("LLM_TEMPERATURE", 0.1)
LLM_TOP_P = parse_float_env("LLM_TOP_P")
LLM_REQUEST_TIMEOUT_S = parse_float_env("LLM_REQUEST_TIMEOUT_S", 120.0)
BENCHMARK_CONCURRENCY = max(1, min(parse_int_env("BENCHMARK_CONCURRENCY", 20), 20))
BENCHMARK_PREFLIGHT = parse_bool_env("BENCHMARK_PREFLIGHT", True)
BENCHMARK_LOG_REQUESTS = parse_bool_env("BENCHMARK_LOG_REQUESTS", True)
BENCHMARK_ABORT_ON_TIMEOUT = parse_bool_env("BENCHMARK_ABORT_ON_TIMEOUT", False)
DEFAULT_MAX_TOKENS = 8192 if LLM_PROVIDER == "baichuan" else None
LLM_MAX_TOKENS = parse_int_env("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)

BAICHUAN_THINKING_BUDGET_TOKENS = parse_int_env("BAICHUAN_THINKING_BUDGET_TOKENS")
BAICHUAN_TOP_K = parse_int_env("BAICHUAN_TOP_K")
BAICHUAN_EVIDENCE_SCOPE = os.getenv("BAICHUAN_EVIDENCE_SCOPE")
BAICHUAN_OUTPUT_STYLE = os.getenv("BAICHUAN_OUTPUT_STYLE")
BAICHUAN_DISABLE_FOLLOWUP = os.getenv("BAICHUAN_DISABLE_FOLLOWUP")

# Add prompt path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))
try:
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1
except ImportError:
    # Fallback if running from a different directory
    sys.path.append('/workspace/audio_llm_agent/code/eval_llm/basic_test')
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1

# Paths
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATASET_PATH = os.getenv("BENCHMARK_DATASET_PATH", os.path.join(BASE_DIR, 'dataset.jsonl'))
OUTPUT_ROOT = os.getenv("BENCHMARK_OUTPUT_ROOT", SCRIPT_DIR)
BENCHMARK_EXP_DIR = os.getenv("BENCHMARK_EXP_DIR", "exp_1")
OUTPUT_DIR = os.getenv("BENCHMARK_OUTPUT_DIR", os.path.join(OUTPUT_ROOT, BENCHMARK_EXP_DIR))
OUTPUT_PREFIX = os.getenv("BENCHMARK_OUTPUT_PREFIX") or safe_filename(LLM_MODEL_NAME)
WRITE_LEGACY_OUTPUTS = parse_bool_env("BENCHMARK_WRITE_LEGACY_OUTPUTS", False)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def baichuan_should_flatten_system():
    explicit = os.getenv("BAICHUAN_FLATTEN_SYSTEM")
    if explicit is not None:
        return parse_bool_env("BAICHUAN_FLATTEN_SYSTEM")
    return LLM_PROVIDER == "baichuan" and LLM_MODEL_NAME.lower().endswith("-plus")


def message_content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(json.dumps(part, ensure_ascii=False))
            else:
                text_parts.append(str(part))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def flatten_system_messages(messages):
    system_parts = [
        message_content_to_text(message.get("content"))
        for message in messages
        if message.get("role") == "system"
    ]
    system_prompt = "\n\n".join(part for part in system_parts if part)
    if not system_prompt:
        return messages

    flattened_messages = []
    system_merged = False
    for message in messages:
        if message.get("role") == "system":
            continue
        new_message = dict(message)
        if not system_merged and new_message.get("role") == "user":
            user_text = message_content_to_text(new_message.get("content"))
            new_message["content"] = f"系统指令：\n{system_prompt}\n\n用户输入：\n{user_text}"
            system_merged = True
        flattened_messages.append(new_message)

    if not system_merged:
        flattened_messages.insert(0, {"role": "user", "content": f"系统指令：\n{system_prompt}"})
    return flattened_messages


def normalize_messages_for_provider(messages):
    if LLM_PROVIDER == "baichuan" and baichuan_should_flatten_system():
        return flatten_system_messages(messages)
    return messages


def build_extra_body():
    extra_body = {}
    if LLM_PROVIDER != "baichuan":
        return extra_body

    metadata = {}
    if BAICHUAN_EVIDENCE_SCOPE:
        metadata["evidence_scope"] = BAICHUAN_EVIDENCE_SCOPE
    if BAICHUAN_OUTPUT_STYLE:
        metadata["output_style"] = BAICHUAN_OUTPUT_STYLE
    if BAICHUAN_DISABLE_FOLLOWUP is not None:
        metadata["disable_follow-up_question_extension"] = parse_bool_env("BAICHUAN_DISABLE_FOLLOWUP")
    if metadata:
        extra_body["metadata"] = metadata

    if BAICHUAN_THINKING_BUDGET_TOKENS:
        extra_body["thinking"] = {"budget_tokens": BAICHUAN_THINKING_BUDGET_TOKENS}
    if BAICHUAN_TOP_K is not None:
        extra_body["top_k"] = BAICHUAN_TOP_K
    return extra_body


def build_chat_request(messages):
    request_kwargs = {
        "model": LLM_MODEL_NAME,
        "messages": normalize_messages_for_provider(messages),
        "temperature": LLM_TEMPERATURE,
        "stream": False,
    }
    if LLM_TOP_P is not None:
        request_kwargs["top_p"] = LLM_TOP_P
    if LLM_MAX_TOKENS is not None:
        request_kwargs["max_tokens"] = LLM_MAX_TOKENS

    extra_body = build_extra_body()
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    return request_kwargs


def extract_response_text(response):
    if not response or not response.choices:
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        return message_content_to_text(content)
    return str(content)


class FatalAPIError(RuntimeError):
    pass


def extract_api_error_info(exc):
    body = getattr(exc, "body", None)
    status_code = getattr(exc, "status_code", None)
    error_obj = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error_obj, dict):
        error_obj = body if isinstance(body, dict) else {}

    code = error_obj.get("code") or getattr(exc, "code", None)
    error_type = error_obj.get("type")
    message = error_obj.get("message") or str(exc)
    return {
        "status_code": status_code,
        "code": code,
        "type": error_type,
        "message": message,
    }


def is_fatal_api_error(exc):
    info = extract_api_error_info(exc)
    text = json.dumps(info, ensure_ascii=False).lower()
    fatal_codes = (
        "invalid_api_key",
        "insufficient_quota",
        "billing_not_active",
        "account_deactivated",
        "permission_denied",
    )
    return (
        info["status_code"] in (401, 403)
        or any(code in text for code in fatal_codes)
    )


def format_api_error(exc):
    info = extract_api_error_info(exc)
    parts = []
    if info["status_code"]:
        parts.append(f"status={info['status_code']}")
    if info["code"]:
        parts.append(f"code={info['code']}")
    if info["type"]:
        parts.append(f"type={info['type']}")
    parts.append(f"message={info['message']}")
    return ", ".join(parts)


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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

async def process_record(client, record, semaphore, record_id=None):
    async with semaphore:
        messages = construct_messages(record)
        start_time = time.perf_counter()
        label = f"record {record_id}" if record_id is not None else "record"
        if BENCHMARK_LOG_REQUESTS:
            print(f"Starting request for {label}...", flush=True)
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**build_chat_request(messages)),
                timeout=LLM_REQUEST_TIMEOUT_S
            )
            generated_text = extract_response_text(response)
            request_duration_seconds = time.perf_counter() - start_time
            if BENCHMARK_LOG_REQUESTS:
                print(
                    f"Finished request for {label} in {request_duration_seconds:.2f}s "
                    f"({len(generated_text)} chars).",
                    flush=True
                )
            return generated_text, request_duration_seconds, None
        except asyncio.TimeoutError as e:
            request_duration_seconds = time.perf_counter() - start_time
            error_message = (
                f"Request timeout after {request_duration_seconds:.2f}s. "
                f"Increase LLM_REQUEST_TIMEOUT_S or reduce LLM_MAX_TOKENS."
            )
            if BENCHMARK_ABORT_ON_TIMEOUT:
                raise FatalAPIError(error_message) from e
            print(f"Timeout processing {label}: {error_message}", flush=True)
            return "", request_duration_seconds, error_message
        except Exception as e:
            request_duration_seconds = time.perf_counter() - start_time
            if is_fatal_api_error(e):
                raise FatalAPIError(format_api_error(e)) from e
            error_message = str(e)
            print(f"Error processing {label}: {error_message}", flush=True)
            return "", request_duration_seconds, error_message

async def main():
    if AsyncOpenAI is None:
        raise RuntimeError("Missing dependency 'openai'. Install it with: pip install openai")
    if not LLM_API_KEY:
        if LLM_PROVIDER == "baichuan":
            raise RuntimeError("Missing API key. Set BAICHUAN_API_KEY for Baichuan.")
        raise RuntimeError("Missing API key. Set LLM_API_KEY or QWEN_API_KEY.")

    print(f"Initializing API client: provider={LLM_PROVIDER}, model={LLM_MODEL_NAME}")
    print(f"Base URL: {LLM_BASE_URL}")
    print(f"Execution mode: concurrent (max {BENCHMARK_CONCURRENCY})")
    print(f"Preflight: {BENCHMARK_PREFLIGHT}")
    print(f"Max tokens: {LLM_MAX_TOKENS}")
    print(f"Request timeout: {LLM_REQUEST_TIMEOUT_S}s")
    print(f"Abort on timeout: {BENCHMARK_ABORT_ON_TIMEOUT}")
    print(f"Output dir: {OUTPUT_DIR}")
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_REQUEST_TIMEOUT_S)
    
    print(f"Loading data from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    print(f"Loaded {len(data)} records.")
    
    # Concurrent mode is for throughput. Per-record durations include any provider-side queueing.
    semaphore = asyncio.Semaphore(BENCHMARK_CONCURRENCY)
    
    print("Generating responses...")
    record_outputs = []
    records_to_process = data
    if BENCHMARK_PREFLIGHT and data:
        print("Running preflight request on the first record...")
        record_outputs.append(await process_record(client, data[0], semaphore, record_id=0))
        records_to_process = data[1:]

    start_index = len(record_outputs)
    tasks = [
        process_record(client, record, semaphore, record_id=start_index + idx)
        for idx, record in enumerate(records_to_process)
    ]
    if tasks:
        record_outputs.extend(await asyncio.gather(*tasks))
    
    results = []
    error_cases = []
    global_stats = {} # {field: {'tp': 0, 'fp': 0, 'fn': 0}}
    total_request_duration_seconds = 0.0
    
    for i, (generated_text, request_duration_seconds, request_error) in enumerate(record_outputs):
        total_request_duration_seconds += request_duration_seconds
        predicted_json = parse_json(generated_text)
        expected_json = data[i]['expected_output']
        
        record_stats = compare_record(expected_json, predicted_json)
        
        # Check for errors
        has_error = False
        error_details = {}
        if request_error:
            has_error = True
            error_details["_request_error"] = {
                "expected": None,
                "predicted": request_error,
                "type": "RequestError"
            }
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
                'generated_text': generated_text,
                'request_duration_seconds': request_duration_seconds,
                'request_error': request_error
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
            'request_duration_seconds': request_duration_seconds,
            'request_error': request_error,
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

    total_fields = len(sorted_keys)
    category_avg_f1 = (
        sum(final_metrics[key]['f1'] for key in sorted_keys) / total_fields
        if total_fields > 0 else 0.0
    )
    total_tp = sum(global_stats[key]['tp'] for key in sorted_keys)
    total_fp = sum(global_stats[key]['fp'] for key in sorted_keys)
    total_fn = sum(global_stats[key]['fn'] for key in sorted_keys)
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    sample_aggregate_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0 else 0.0
    )
    request_count = len(record_outputs)
    avg_request_duration_seconds = (
        total_request_duration_seconds / request_count if request_count > 0 else 0.0
    )

    final_metrics['summary'] = {
        'total_category_avg_f1': category_avg_f1,
        'total_sample_aggregate_f1': sample_aggregate_f1,
        'avg_request_duration_seconds': avg_request_duration_seconds,
        'precision': total_precision,
        'recall': total_recall,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'field_count': total_fields,
        'request_count': request_count,
        'total_request_duration_seconds': total_request_duration_seconds,
        'provider': LLM_PROVIDER,
        'model_name': LLM_MODEL_NAME,
        'base_url': LLM_BASE_URL,
        'output_prefix': OUTPUT_PREFIX,
        'output_dir': OUTPUT_DIR,
        'execution_mode': 'concurrent',
        'concurrency': BENCHMARK_CONCURRENCY
    }

    print("\nSummary metrics:")
    print(f"Total Category Avg F1: {category_avg_f1:.4f}")
    print(f"Total Sample Aggregate F1: {sample_aggregate_f1:.4f} (P={total_precision:.4f}, R={total_recall:.4f}, TP={total_tp}, FP={total_fp}, FN={total_fn})")
    print(f"Average Request Duration: {avg_request_duration_seconds:.4f}s ({request_count} requests)")
        
    # Save results
    results_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_results.json')
    metrics_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_metrics.json')
    error_cases_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_error_cases.json')

    write_json(results_path, results)
    write_json(metrics_path, final_metrics)
    write_json(error_cases_path, error_cases)

    if WRITE_LEGACY_OUTPUTS:
        write_json(os.path.join(OUTPUT_DIR, 'results.json'), results)
        write_json(os.path.join(OUTPUT_DIR, 'metrics.json'), final_metrics)
        write_json(os.path.join(OUTPUT_DIR, 'error_cases.json'), error_cases)
        
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Results file: {results_path}")
    print(f"Metrics file: {metrics_path}")
    print(f"Error cases file: {error_cases_path}")
    print(f"Found {len(error_cases)} error cases.")

if __name__ == "__main__":
    asyncio.run(main())
