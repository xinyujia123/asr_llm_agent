import sys
import os
import json
import time
import asyncio
import threading
import subprocess
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Add prompt path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1
except ImportError:
    sys.path.append('/workspace/audio_llm_agent/code/eval_llm/basic_test')
    from prompts import MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1

# Paths
DATASET_PATH = '/workspace/audio_llm_agent/code/eval_llm/basic_test/dataset.jsonl'
OUTPUT_DIR = '/workspace/audio_llm_agent/code/eval_llm/basic_test/hardware_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

class NvidiaSmiMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.gpu_stats = []
        self.thread = threading.Thread(target=self._monitor)

    def start(self):
        self.stop_event.clear()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                # Run nvidia-smi to get memory usage
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    # Output format: "used, total" (e.g., "1024, 8192")
                    # Handle multiple GPUs if present, though we assume 1 for now
                    for line in output.split('\n'):
                        used, total = map(int, line.split(','))
                        self.gpu_stats.append({
                            'timestamp': time.time(),
                            'memory_used_mb': used,
                            'memory_total_mb': total
                        })
                        # Break after first GPU for simplicity in this script
                        break
            except Exception as e:
                print(f"Error running nvidia-smi: {e}")
            
            time.sleep(self.interval)

    def get_max_memory(self):
        if not self.gpu_stats:
            return 0, 0
        max_used = max(s['memory_used_mb'] for s in self.gpu_stats)
        # Assume total is constant
        total = self.gpu_stats[0]['memory_total_mb']
        return max_used, total

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def construct_prompt(record, tokenizer):
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
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return text

def run_batch_test(llm, prompts, batch_size, tokenizer):
    print(f"\n--- Testing concurrency: {batch_size} requests/sec (simulated batch) ---")
    
    # We will pick 'batch_size' prompts
    test_prompts = prompts[:batch_size]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    
    monitor = NvidiaSmiMonitor()
    monitor.start()
    
    start_time = time.time()
    
    # vllm handles batching internally, so passing a list is effectively concurrent processing
    outputs = llm.generate(test_prompts, sampling_params)
    
    end_time = time.time()
    monitor.stop()
    
    total_time = end_time - start_time
    avg_latency = total_time / len(test_prompts)
    max_used, total_mem = monitor.get_max_memory()
    
    # Calculate tokens per second
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / total_time
    
    print(f"Total time: {total_time:.4f}s")
    print(f"Average latency per request: {avg_latency:.4f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Max GPU Memory Used: {max_used} MB / {total_mem} MB")
    
    return {
        'concurrency': batch_size,
        'total_time_s': total_time,
        'avg_latency_s': avg_latency,
        'throughput_tokens_per_s': throughput,
        'max_gpu_memory_used_mb': max_used,
        'gpu_memory_total_mb': total_mem
    }

def main():
    print(f"Loading model {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        # Initialize vLLM engine with a slightly lower memory utilization to allow room for overhead
        # gpu_memory_utilization=0.9 is default, we can keep it or adjust if needed.
        llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.7,trust_remote_code=True, tensor_parallel_size=1, max_model_len=8192)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading data from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    # Ensure we have enough data for the largest batch
    while len(data) < 10:
        data.extend(data)
    
    prompts = [construct_prompt(record, tokenizer) for record in data]
    
    concurrency_levels = [1, 3, 5, 10, 30]
    results = []
    
    # Warmup
    print("Warming up...")
    llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=10))
    
    for level in concurrency_levels:
        res = run_batch_test(llm, prompts, level, tokenizer)
        results.append(res)
        
    # Save results
    output_file = os.path.join(OUTPUT_DIR, 'hardware_benchmark_nvidia_smi.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nBenchmark completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
