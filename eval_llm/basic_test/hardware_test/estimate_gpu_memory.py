import sys
import os
import time
import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Paths
OUTPUT_DIR = '/workspace/audio_llm_agent/code/eval_llm/basic_test/hardware_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

def test_memory_utilization(utilization):
    print(f"\n--- Testing gpu_memory_utilization: {utilization:.2f} ---")
    
    try:
        # Initialize vLLM engine with specific memory utilization
        llm = LLM(
            model=MODEL_NAME, 
            trust_remote_code=True, 
            tensor_parallel_size=1, 
            max_model_len=8192,
            gpu_memory_utilization=utilization
        )
        
        # Simple generation to ensure it works
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = llm.generate(prompts, sampling_params)
        
        print(f"Success! Output: {outputs[0].outputs[0].text}")
        
        # Get actual memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / (1024 * 1024)
            mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            print(f"Memory Allocated: {mem_used:.2f} MB")
            print(f"Memory Reserved: {mem_reserved:.2f} MB")
            
        return True, mem_reserved
        
    except Exception as e:
        print(f"Failed with utilization {utilization:.2f}: {e}")
        return False, 0
    finally:
        # Clean up
        if 'llm' in locals():
            del llm
        gc.collect()
        torch.cuda.empty_cache()
        # Wait a bit for memory to be released
        time.sleep(2)

def main():
    print(f"Estimating minimum GPU memory for {MODEL_NAME}...")
    
    # Start from a reasonable low value and increase, or start high and decrease
    # Let's try a few specific points to find the threshold
    test_values = [0.9, 0.7, 0.5, 0.4, 0.3, 0.2]
    
    results = []
    
    for util in test_values:
        success, mem_reserved = test_memory_utilization(util)
        results.append({
            'utilization': util,
            'success': success,
            'memory_reserved_mb': mem_reserved
        })
        
        if not success:
            print(f"Stopping as utilization {util} failed.")
            break
            
    print("\n--- Summary ---")
    for res in results:
        status = "Pass" if res['success'] else "Fail"
        print(f"Utilization {res['utilization']}: {status} (Reserved: {res['memory_reserved_mb']:.2f} MB)")
        
    # Find minimum successful utilization
    successful_utils = [r for r in results if r['success']]
    if successful_utils:
        min_util = min(r['utilization'] for r in successful_utils)
        min_mem = min(r['memory_reserved_mb'] for r in successful_utils)
        print(f"\nMinimum successful utilization tested: {min_util}")
        print(f"Approximate minimum memory required: {min_mem:.2f} MB")
    else:
        print("\nCould not determine minimum memory.")

if __name__ == "__main__":
    main()
