import os
import json
import asyncio
import time
from prompts import (
    MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1,
    DYNAMIC_CONTENT,
    prepare_system_prompt_padding_suffix
)
from hardware_benchmark_runtime import (
    average_rounds,
    build_messages_for_count,
    parse_env_bool,
    run_burst_round_async,
    run_steady_round_async,
    save_json
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.jsonl')
OUTPUT_DIR = os.path.dirname(__file__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
CHAT_COMPLETIONS_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"
METRICS_URL = os.getenv("VLLM_METRICS_URL", f"{VLLM_BASE_URL.rstrip('/')}/metrics")
BURST_CONCURRENCY_LEVELS = [1, 10, 20, 40, 60, 80, 100]
ROUND_REPEATS = 3
REQUEST_TIMEOUT_S = int(os.getenv("BENCHMARK_REQUEST_TIMEOUT_S", "120"))
ENABLE_STREAM = False
ROUND_COOLDOWN_S = 3
OUTPUT_ROUND_DETAILS = False
ENABLE_SYSTEM_PROMPT_BLOCK_PADDING = True
ENABLE_DCGMI_METRICS = parse_env_bool("BENCHMARK_ENABLE_DCGMI_METRICS", False)
DCGMI_FIELD_IDS = os.getenv("BENCHMARK_DCGMI_FIELD_IDS", "1002,1003,1004,1005,1009,1010")
DCGMI_GPU_SELECTOR = os.getenv("BENCHMARK_DCGMI_GPU_SELECTOR", "0")
DCGMI_SAMPLE_INTERVAL_S = float(os.getenv("BENCHMARK_DCGMI_SAMPLE_INTERVAL_S", "0.2"))

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_padding_suffix():
    padding_suffix, _ = prepare_system_prompt_padding_suffix(
        system_prompt=MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1,
        enable_padding=ENABLE_SYSTEM_PROMPT_BLOCK_PADDING,
        metrics_url=METRICS_URL,
        model_name=MODEL_NAME,
        vllm_base_url=VLLM_BASE_URL
    )
    return padding_suffix


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

async def run_warmup(messages):
    return await run_steady_round_async(
        messages_for_round=[messages],
        target_rps=1,
        monitor_metrics_url=METRICS_URL,
        model_name=MODEL_NAME,
        chat_completions_url=CHAT_COMPLETIONS_URL,
        timeout_s=REQUEST_TIMEOUT_S,
        enable_stream=ENABLE_STREAM
    )


def main():
    print(f"Using remote vLLM service: {CHAT_COMPLETIONS_URL}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Enable stream mode: {ENABLE_STREAM}")
    print(f"Enable system prompt block padding: {ENABLE_SYSTEM_PROMPT_BLOCK_PADDING}")
    print(f"Enable dcgmi metrics: {ENABLE_DCGMI_METRICS}")
    print("Set BENCHMARK_ENABLE_DCGMI_METRICS=1 to enable dcgmi metrics")
    print(f"DCGMI field ids: {DCGMI_FIELD_IDS}")
    print(f"DCGMI gpu selector: {DCGMI_GPU_SELECTOR}")
    print(f"DCGMI sample interval(s): {DCGMI_SAMPLE_INTERVAL_S}")
    print(f"Loading data from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    system_padding_suffix = prepare_padding_suffix()
    base_messages = [construct_messages(record, system_padding_suffix) for record in data]
    if not base_messages:
        raise RuntimeError("No data loaded from dataset.jsonl")

    print("Warming up...")
    asyncio.run(run_warmup(base_messages[0]))

    all_results = []
    for concurrency in BURST_CONCURRENCY_LEVELS:
        messages_for_round = build_messages_for_count(base_messages, concurrency)
        print(f"\n=== Burst concurrency test: {concurrency} ===")
        rounds = []
        for round_idx in range(1, ROUND_REPEATS + 1):
            print(f"Round {round_idx}/{ROUND_REPEATS}")
            round_result = asyncio.run(
                run_burst_round_async(
                    messages_for_round=messages_for_round,
                    concurrency=concurrency,
                    monitor_metrics_url=METRICS_URL,
                    model_name=MODEL_NAME,
                    chat_completions_url=CHAT_COMPLETIONS_URL,
                    timeout_s=REQUEST_TIMEOUT_S,
                    enable_stream=ENABLE_STREAM
                )
            )
            rounds.append(round_result)
            print(
                f"success_rate={round_result['success_rate']:.3f}, "
                f"lat_p50={round_result['latency_p50_s']:.3f}s, "
                f"lat_p95={round_result['latency_p95_s']:.3f}s"
            )
            if round_idx < ROUND_REPEATS:
                time.sleep(ROUND_COOLDOWN_S)
        all_results.append({
            "test_type": "burst_concurrency",
            "concurrency": concurrency,
            "round_repeats": ROUND_REPEATS,
            "requests_per_round": concurrency,
            "average": average_rounds(rounds)
        })
        if OUTPUT_ROUND_DETAILS:
            all_results[-1]["rounds"] = rounds

    output_basename = "hardware_benchmark_burst_concurrency_stream.json" if ENABLE_STREAM else "hardware_benchmark_burst_concurrency.json"
    output_file = os.path.join(OUTPUT_DIR, output_basename)
    output_payload = {
        "model_name": MODEL_NAME,
        "base_url": VLLM_BASE_URL,
        "chat_completions_url": CHAT_COMPLETIONS_URL,
        "metrics_url": METRICS_URL,
        "enable_stream": ENABLE_STREAM,
        "enable_system_prompt_block_padding": ENABLE_SYSTEM_PROMPT_BLOCK_PADDING,
        "enable_dcgmi_metrics": ENABLE_DCGMI_METRICS,
        "output_round_details": OUTPUT_ROUND_DETAILS,
        "round_repeats": ROUND_REPEATS,
        "burst_concurrency_levels": BURST_CONCURRENCY_LEVELS,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "results": all_results
    }
    if ENABLE_DCGMI_METRICS:
        output_payload.update({
            "dcgmi_field_ids": DCGMI_FIELD_IDS,
            "dcgmi_gpu_selector": DCGMI_GPU_SELECTOR,
            "dcgmi_sample_interval_s": DCGMI_SAMPLE_INTERVAL_S
        })
    save_json(output_file, output_payload)
    print(f"\nBurst benchmark completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
