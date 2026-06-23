import sys
import os
import json
import time
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

# Load env vars
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# --- 配置项 ---
LLM_API_KEY = os.getenv("QWEN_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_NAME = "qwen-flash" 

# Pricing for Qwen-Turbo (qwen-flash is often an alias or similar tier)
# Based on Alibaba Cloud pricing (approximate):
# Input: 0.002 RMB / 1000 tokens
# Output: 0.006 RMB / 1000 tokens
PRICE_INPUT_PER_1K = 0.00015
PRICE_OUTPUT_PER_1K = 0.0015

# Add prompt path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1
except ImportError:
    sys.path.append('/workspace/audio_llm_agent/code/eval_llm/basic_test')
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1

# Paths
DATASETS = [
    '/workspace/audio_llm_agent/code/eval_llm/basic_test/dataset_5s.jsonl',
    '/workspace/audio_llm_agent/code/eval_llm/basic_test/dataset_20s.jsonl',
    '/workspace/audio_llm_agent/code/eval_llm/basic_test/dataset_60s.jsonl'
]
OUTPUT_DIR = '/workspace/audio_llm_agent/code/eval_llm/basic_test/hardware_test_api'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def construct_messages(record):
    sys_time = record.get('current_sys_time', '')
    day_before = record.get('the_day_before_yesterday', '')
    
    system_content = MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1.replace(
        "{CURRENT_SYS_TIME}", sys_time
    ).replace(
        "{THE_DAY_BEFORE_YESTERDAY}", day_before
    )
    
    user_input = record['input_text']
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"分析文本：{user_input}"}
    ]
    return messages

async def process_request(client, messages):
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            temperature=0.1,
        )
        latency = time.time() - start_time
        
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        return {
            'success': True,
            'latency': latency,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'success': False,
            'latency': time.time() - start_time,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }

async def delayed_request(client, messages, delay):
    if delay > 0:
        await asyncio.sleep(delay)
    return await process_request(client, messages)

async def run_batch_test(client, data, batch_size):
    print(f"\n--- Testing concurrency: {batch_size} requests (distributed in 1s) ---")
    
    # Select batch
    batch_data = data[:batch_size]
    tasks = []
    
    # Distribute requests over 1 second
    interval = 1.0 / batch_size if batch_size > 0 else 0
    
    for i, record in enumerate(batch_data):
        messages = construct_messages(record)
        delay = i * interval
        tasks.append(delayed_request(client, messages, delay))
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Calculate stats
    successful_requests = [r for r in results if r['success']]
    success_count = len(successful_requests)
    
    total_input_tokens = sum(r['input_tokens'] for r in successful_requests)
    total_output_tokens = sum(r['output_tokens'] for r in successful_requests)
    total_tokens = total_input_tokens + total_output_tokens
    
    avg_latency = sum(r['latency'] for r in successful_requests) / success_count if success_count > 0 else 0
    throughput = total_tokens / total_time if total_time > 0 else 0
    
    # Calculate cost
    cost_input = (total_input_tokens / 1000) * PRICE_INPUT_PER_1K
    cost_output = (total_output_tokens / 1000) * PRICE_OUTPUT_PER_1K
    total_cost = cost_input + cost_output
    avg_cost_per_sentence = total_cost / success_count if success_count > 0 else 0
    
    print(f"Total time: {total_time:.4f}s")
    print(f"Average latency: {avg_latency:.4f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Estimated Cost: {total_cost:.6f} RMB (Input: {cost_input:.6f}, Output: {cost_output:.6f})")
    print(f"Average Cost per Sentence: {avg_cost_per_sentence:.6f} RMB")
    
    return {
        'concurrency': batch_size,
        'total_time_s': total_time,
        'avg_latency_s': avg_latency,
        'throughput_tokens_per_s': throughput,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'cost_rmb': total_cost,
        'avg_cost_per_sentence': avg_cost_per_sentence,
        'cost_details': {
            'input_cost': cost_input,
            'output_cost': cost_output
        }
    }

async def main():
    print(f"Initializing API client with model {LLM_MODEL_NAME}...")
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    
    all_results = {}

    for dataset_path in DATASETS:
        dataset_name = os.path.basename(dataset_path)
        print(f"\n=== Testing Dataset: {dataset_name} ===")
        print(f"Loading data from {dataset_path}...")
        
        try:
            data = load_data(dataset_path)
        except FileNotFoundError:
            print(f"File not found: {dataset_path}, skipping...")
            continue

        # Ensure we have enough data
        if not data:
            print(f"No data found in {dataset_path}, skipping...")
            continue
            
        # Increase sample size by looping 5 times
        data = data * 5
        
        # Ensure we have enough data for max concurrency
        while len(data) < 10:
            data.extend(data)
        
        concurrency_levels = [1, 3, 5, 10]
        dataset_results = []
        
        # Warmup
        print("Warming up...")
        if data:
            await process_request(client, construct_messages(data[0]))
        
        for level in concurrency_levels:
            res = await run_batch_test(client, data, level)
            dataset_results.append(res)
            # Sleep to avoid rate limits
            time.sleep(1)
        
        # Save individual result for this dataset
        result_filename = f'api_benchmark_cost_{dataset_name.replace(".jsonl", "")}.json'
        output_file = os.path.join(OUTPUT_DIR, result_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)
            
        print(f"Results for {dataset_name} saved to {output_file}")
        
    print(f"\nBenchmark completed. All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
