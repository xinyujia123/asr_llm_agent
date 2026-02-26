import json
import os
import sys
import re
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Add the current directory to path so we can import local modules
sys.path.append("/workspace/audio_llm_agent/eval_llm")

try:
    from .schema import CERTAIN_KEYS, UNCERTAIN_KEYS
    from .llm_client import LLMClient
except ImportError:
    # Fallback if run directly not as module (though .config in llm_client might still fail)
    try:
        from schema import CERTAIN_KEYS, UNCERTAIN_KEYS
        from llm_client import LLMClient
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def compare_semantic(val1: Any, val2: Any, key: str, client: LLMClient) -> bool:
    # Handle None/null
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False
    
    # Convert to string
    val1_str = str(val1).strip()
    val2_str = str(val2).strip()
    
    # If identical strings, no need to ask LLM
    if val1_str == val2_str:
        return True

    prompt = f"""
Compare the following two medical values for the field "{key}".

Value 1: {val1_str}
Value 2: {val2_str}

Are they semantically equivalent? Reply with ONLY "YES" or "NO".
"""
    
    try:
        # Use system prompt to guide behavior
        response, _ = client.chat(
            system_prompt="You are a medical data comparison assistant. Reply with YES or NO only.", 
            user_input=prompt
        )
        if response:
            response_clean = response.strip().upper()
            if "YES" in response_clean:
                return True
            else:
                return False
        return False
    except Exception as e:
        print(f"Error calling LLM for semantic comparison: {e}")
        return False

def main():
    model_1 = "glm"
    model_2 = "glm"
    exp = "exp2"
    think = True
    think_ex = "think" if think else "no_think"
    infer = True
    infer_ex = "infer" if infer else "no_infer"
    file_dir = os.path.join("/workspace/audio_llm_agent/eval_llm/benchmark", exp)
    file1_path = f"{file_dir}/{model_1}_{think_ex}_{infer_ex}_30.jsonl"
    file2_path = f"{file_dir}/{model_2}_{think_ex}_{infer_ex}_30.jsonl"

    custom = True
    if custom:
        model_1 = "kimi"
        model_2 = "kimi"
        exp = "gold_exp_2"
        file1_path = f"/workspace/audio_llm_agent/eval_llm/benchmark/gold_exp/{model_1}_think_infer_20.jsonl"
        file2_path = f"/workspace/audio_llm_agent/eval_llm/benchmark/gold_exp/{model_2}_think_infer_20_1.jsonl"


    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print("One or both input files do not exist.")
        return

    print(f"Loading {file1_path}...")
    data1 = load_jsonl(file1_path)
    print(f"Loading {file2_path}...")
    data2 = load_jsonl(file2_path)
    
    limit = min(len(data1), len(data2))
    print(f"Comparing {limit} records...")
    
    # Initialize LLM Client
    try:
        # Using qwen as judge, assuming it is available
        llm_client = LLMClient("qwen") 
        print("LLM Client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize LLM client: {e}")
        print("Please check your API configuration.")
        return

    results = {
        "total_records": limit,
        "perfect_matches": 0,
        "mismatches": []
    }
    
    for i in tqdm(range(limit)):
        record1 = data1[i]
        record2 = data2[i]
        
        json1 = record1.get("cleaned_json", {})
        json2 = record2.get("cleaned_json", {})
        
        # If cleaned_json is string (sometimes happens if parsing failed?), try to parse it
        if isinstance(json1, str):
            try:
                json1 = json.loads(json1)
            except:
                pass
        if isinstance(json2, str):
            try:
                json2 = json.loads(json2)
            except:
                pass
        
        if not isinstance(json1, dict): json1 = {}
        if not isinstance(json2, dict): json2 = {}

        mismatches_in_record = []
        
        # Check CERTAIN_KEYS
        for key in CERTAIN_KEYS:
            if key == "admissionTime":
                continue

            val1 = json1.get(key)
            val2 = json2.get(key)
            
            # Normalize for string comparison
            v1 = str(val1).strip() if val1 is not None else None
            v2 = str(val2).strip() if val2 is not None else None
            
            if v1 != v2:
                mismatches_in_record.append({
                    "key": key,
                    "type": "certain",
                    "val1": val1,
                    "val2": val2
                })
        
        # Check UNCERTAIN_KEYS
        for key in UNCERTAIN_KEYS:
            val1 = json1.get(key)
            val2 = json2.get(key)
            
            is_match = compare_semantic(val1, val2, key, llm_client)
            
            if not is_match:
                mismatches_in_record.append({
                    "key": key,
                    "type": "uncertain",
                    "val1": val1,
                    "val2": val2
                })
        
        if not mismatches_in_record:
            results["perfect_matches"] += 1
        else:
            results["mismatches"].append({
                "index": i+1,
                "diffs": mismatches_in_record,
                "reasoning_1": record1.get("raw_output"),
                "reasoning_2": record2.get("raw_output"),
            })

    # Output results
    print("\n=== Comparison Results ===")
    print(f"Total Records: {results['total_records']}")
    print(f"Perfect Matches: {results['perfect_matches']}")
    print(f"Mismatch Records: {len(results['mismatches'])}")
    print(f"Match Rate: {results['perfect_matches']/results['total_records']*100:.2f}%")
    
    if results["mismatches"]:
        print("\n--- Detailed Mismatches ---")
        output_dir = os.path.join("/workspace/audio_llm_agent/eval_llm/eval_output/", exp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        file_name = file1_path.split("/")[-1].replace('.jsonl','')+'_'+file2_path.split("/")[-1].replace('.jsonl','')+'.txt'
        output_file = os.path.join(output_dir, file_name)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Comparison Results ===\n")
            f.write(f"Total Records: {results['total_records']}\n")
            f.write(f"Perfect Matches: {results['perfect_matches']}\n")
            f.write(f"Mismatch Records: {len(results['mismatches'])}\n\n")
            
            for m in results["mismatches"]:
                msg = f"Record {m['index']}:\n"
                print(msg.strip())
                f.write(msg)
                for d in m['diffs']:
                    msg = f"  Key: {d['key']} ({d['type']})\n    {model_1}: {d['val1']}\n    {model_2}: {d['val2']}\n"
                    print(msg.strip())
                    f.write(msg)
                
                # Write reasoning output
                f.write("\n  [Reasoning Output]\n")
                f.write(f"  {model_1} (File 1) Reasoning:\n{m.get('reasoning_1', 'N/A')}\n")
                f.write(f"  {model_2} (File 2) Reasoning:\n{m.get('reasoning_2', 'N/A')}\n")
                
                f.write("-" * 20 + "\n")
                print("-" * 20)
        
        print(f"\nFull report written to {output_file}")

if __name__ == "__main__":
    main()
