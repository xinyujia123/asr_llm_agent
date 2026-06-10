import json
import os
import glob

base_dir = "/root/asr_llm_agent-main/eval_llm/basic_test/f1_test"
metrics_files = glob.glob(os.path.join(base_dir, "*", "exp_0", "metrics.json"))

results = []
for f in metrics_files:
    model_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
    try:
        with open(f, "r") as file:
            data = json.load(file)
            summary = data.get("summary", {})
            f1 = summary.get("total_sample_aggregate_f1", "N/A")
            if f1 == "N/A":
                f1 = summary.get("total_category_avg_f1", "N/A")
            avg_time = summary.get("avg_request_time_seconds", "N/A")
            
            # Format numbers if they are float
            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            if isinstance(avg_time, float):
                avg_time = f"{avg_time:.4f}"
                
            results.append((model_name, f1, avg_time))
    except Exception as e:
        results.append((model_name, f"Error: {e}", "Error"))

# Sort results by model name
results.sort(key=lambda x: x[0])

print("| Model | F1 Score (Aggregate) | Avg Request Time (s) |")
print("|---|---|---|")
for r in results:
    print(f"| {r[0]} | {r[1]} | {r[2]} |")
