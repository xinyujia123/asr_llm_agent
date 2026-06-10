import os
import json
import argparse
import sys

def get_common_max(sets_dict):
    maxs = []
    for exp, s in sets_dict.items():
        if s:
            maxs.append(max(s))
    if not maxs:
        return -1
    return min(maxs)

def format_val(val, is_pct=False):
    if val == "null" or val is None:
        return "null"
    if is_pct:
        return f"{val:.2f}%"
    return f"{val:.4f}" if isinstance(val, float) and val < 10 else f"{val:.2f}"

def generate_report(model_folder):
    base_dir = os.path.join("/root/asr_llm_agent-main/eval_llm/basic_test/hardware_test", model_folder)
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        sys.exit(1)

    # Find max exp_n
    exp_dirs = []
    for d in os.listdir(base_dir):
        if d.startswith("exp_") and d[4:].isdigit():
            exp_dirs.append(d)
            
    if not exp_dirs:
        print("No exp_ folders found.")
        sys.exit(1)
        
    exp_dirs.sort(key=lambda x: int(x[4:]))
    max_exp = int(exp_dirs[-1][4:])
    # Include all from 0 to max_exp
    all_exps = [f"exp_{i}" for i in range(max_exp + 1)]

    env_records = []
    steady_data = {} # {rps: {exp: metrics}}
    burst_data = {}  # {concurrency: {exp: metrics}}

    # Track which concurrencies exist in which experiments
    steady_rps_sets = {}
    burst_conc_sets = {}
    
    all_steady_rps = set()
    all_burst_conc = set()

    for exp in all_exps:
        steady_path = os.path.join(base_dir, exp, "hardware_benchmark_steady_rps.json")
        burst_path = os.path.join(base_dir, exp, "hardware_benchmark_burst_concurrency.json")
        
        has_steady = os.path.exists(steady_path)
        has_burst = os.path.exists(burst_path)
        
        env_record = {"exp": exp, "vllm_launch_args": "null", "padding": "null"}
        
        current_rps_set = set()
        if has_steady:
            with open(steady_path, "r", encoding="utf-8") as f:
                try:
                    s_data = json.load(f)
                    env_record["vllm_launch_args"] = s_data.get("vllm_launch_args", "null")
                    env_record["padding"] = str(s_data.get("enable_system_prompt_block_padding", "null")).lower()
                    
                    for res in s_data.get("results", []):
                        rps = res.get("target_rps")
                        if rps is not None:
                            current_rps_set.add(rps)
                            all_steady_rps.add(rps)
                            avg = res.get("average", {})
                            output_tokens_per_s = avg.get("total_output_tokens", 0) / avg.get("total_time_s", 1) if avg.get("total_time_s", 0) > 0 else 0
                            
                            metrics = {
                                "actual_success_req_per_s": avg.get("actual_success_req_per_s", "null"),
                                "kv_cache_usage_p95_pct": avg.get("kv_cache_usage_p95_pct", "null"),
                                "latency_p50_s": avg.get("latency_p50_s", "null"),
                                "latency_p95_s": avg.get("latency_p95_s", "null"),
                                "max_gpu_memory_used_mb": avg.get("max_gpu_memory_used_mb", "null"),
                                "throughput_tokens_per_s": avg.get("throughput_tokens_per_s", "null"),
                                "output_tokens_per_s": output_tokens_per_s
                            }
                            if rps not in steady_data:
                                steady_data[rps] = {}
                            steady_data[rps][exp] = metrics
                except Exception as e:
                    print(f"Warning: Failed to parse {steady_path}: {e}")
        steady_rps_sets[exp] = current_rps_set

        current_conc_set = set()
        if has_burst:
            with open(burst_path, "r", encoding="utf-8") as f:
                try:
                    b_data = json.load(f)
                    # Use burst to fill env if steady was missing
                    if env_record["vllm_launch_args"] == "null":
                        env_record["vllm_launch_args"] = b_data.get("vllm_launch_args", "null")
                        env_record["padding"] = str(b_data.get("enable_system_prompt_block_padding", "null")).lower()
                        
                    for res in b_data.get("results", []):
                        conc = res.get("concurrency")
                        if conc is not None:
                            current_conc_set.add(conc)
                            all_burst_conc.add(conc)
                            avg = res.get("average", {})
                            output_tokens_per_s = avg.get("total_output_tokens", 0) / avg.get("total_time_s", 1) if avg.get("total_time_s", 0) > 0 else 0
                            
                            metrics = {
                                "actual_success_req_per_s": avg.get("actual_success_req_per_s", "null"),
                                "kv_cache_usage_p95_pct": avg.get("kv_cache_usage_p95_pct", "null"),
                                "latency_p50_s": avg.get("latency_p50_s", "null"),
                                "latency_p95_s": avg.get("latency_p95_s", "null"),
                                "max_gpu_memory_used_mb": avg.get("max_gpu_memory_used_mb", "null"),
                                "throughput_tokens_per_s": avg.get("throughput_tokens_per_s", "null"),
                                "output_tokens_per_s": output_tokens_per_s
                            }
                            if conc not in burst_data:
                                burst_data[conc] = {}
                            burst_data[conc][exp] = metrics
                except Exception as e:
                    print(f"Warning: Failed to parse {burst_path}: {e}")
        burst_conc_sets[exp] = current_conc_set
            
        env_records.append(env_record)

    steady_common_max = get_common_max(steady_rps_sets)
    burst_common_max = get_common_max(burst_conc_sets)
    
    valid_steady_rps = sorted([rps for rps in all_steady_rps if rps <= steady_common_max])
    valid_burst_conc = sorted([conc for conc in all_burst_conc if conc <= burst_common_max])

    md_lines = []
    md_lines.append(f"# 实验记录报告\n")

    md_lines.append("## 实验环境记录\n")
    md_lines.append("| 实验 | vLLM 运行命令 | ENABLE_SYSTEM_PROMPT_BLOCK_PADDING |")
    md_lines.append("| --- | --- | --- |")
    for rec in env_records:
        args = rec['vllm_launch_args']
        args_str = f"`{args}`" if args != "null" else "null"
        md_lines.append(f"| {rec['exp']} | {args_str} | {rec['padding']} |")
    md_lines.append("\n")

    md_lines.append("## 实验结果记录\n")

    # Steady
    if valid_steady_rps:
        md_lines.append("### 稳态并发测试 (Steady RPS)\n")
        for rps in valid_steady_rps:
            md_lines.append(f"#### 目标 RPS: {rps}\n")
            md_lines.append("| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |")
            md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for exp in all_exps:
                if exp in steady_data.get(rps, {}):
                    m = steady_data[rps][exp]
                    md_lines.append(f"| {exp} | {format_val(m['actual_success_req_per_s'])} | {format_val(m['kv_cache_usage_p95_pct'], True)} | {format_val(m['latency_p50_s'])} | {format_val(m['latency_p95_s'])} | {format_val(m['max_gpu_memory_used_mb'])} | {format_val(m['throughput_tokens_per_s'])} | {format_val(m['output_tokens_per_s'])} |")
                else:
                    md_lines.append(f"| {exp} | null | null | null | null | null | null | null |")
            md_lines.append("\n")

    # Burst
    if valid_burst_conc:
        md_lines.append("### 突发并发测试 (Burst Concurrency)\n")
        for conc in valid_burst_conc:
            md_lines.append(f"#### 目标并发数 (Concurrency): {conc}\n")
            md_lines.append("| 实验 (exp) | actual_success_req_per_s | kv_cache_usage_p95_pct | latency_p50_s | latency_p95_s | max_gpu_memory_used_mb | throughput_tokens_per_s | output_tokens_per_s |")
            md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for exp in all_exps:
                if exp in burst_data.get(conc, {}):
                    m = burst_data[conc][exp]
                    md_lines.append(f"| {exp} | {format_val(m['actual_success_req_per_s'])} | {format_val(m['kv_cache_usage_p95_pct'], True)} | {format_val(m['latency_p50_s'])} | {format_val(m['latency_p95_s'])} | {format_val(m['max_gpu_memory_used_mb'])} | {format_val(m['throughput_tokens_per_s'])} | {format_val(m['output_tokens_per_s'])} |")
                else:
                    md_lines.append(f"| {exp} | null | null | null | null | null | null | null |")
            md_lines.append("\n")

    out_path = os.path.join(base_dir, "exp_record.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Report successfully written to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True, help="Model folder name, e.g. Qwen3.5-35B-A3B-AWQ-4bit")
    args = parser.parse_args()
    generate_report(args.model_folder)
