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
PRICE_INPUT_PER_1K = 0.002
PRICE_OUTPUT_PER_1K = 0.006

# Add prompt path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1
except ImportError:
    sys.path.append('/workspace/audio_llm_agent/code/eval_llm/basic_test')
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1

# Paths
DATASET_PATH = '/workspace/audio_llm_agent/code/eval_llm/basic_test/dataset.jsonl'
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

async def run_batch_test(client, data, batch_size):
    print(f"\n--- Testing concurrency: {batch_size} requests ---")
    
    # Select batch
    batch_data = data[:batch_size]
    tasks = []
    
    for record in batch_data:
        messages = construct_messages(record)
        tasks.append(process_request(client, messages))
    
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
    
    print(f"Total time: {total_time:.4f}s")
    print(f"Average latency: {avg_latency:.4f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Estimated Cost: {total_cost:.6f} RMB (Input: {cost_input:.6f}, Output: {cost_output:.6f})")
    
    return {
        'concurrency': batch_size,
        'total_time_s': total_time,
        'avg_latency_s': avg_latency,
        'throughput_tokens_per_s': throughput,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'cost_rmb': total_cost,
        'cost_details': {
            'input_cost': cost_input,
            'output_cost': cost_output
        }
    }

async def main():
    print(f"Initializing API client with model {LLM_MODEL_NAME}...")
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    
    print(f"Loading data from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    # Ensure we have enough data
    while len(data) < 10:
        data.extend(data)
    
    concurrency_levels = [1, 3, 5, 10]
    results = []
    
    # Warmup
    print("Warming up...")
    await process_request(client, construct_messages(data[0]))
    
    for level in concurrency_levels:
        res = await run_batch_test(client, data, level)
        results.append(res)
        # Sleep to avoid rate limits
        time.sleep(1)
        
    # Save results
    output_file = os.path.join(OUTPUT_DIR, 'api_benchmark_cost.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nBenchmark completed. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
