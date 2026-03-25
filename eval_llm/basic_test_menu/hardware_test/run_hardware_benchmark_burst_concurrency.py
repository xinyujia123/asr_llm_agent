import sys
import os
import json
import time
import math
import threading
import subprocess
import urllib.request
import urllib.error
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(REPO_ROOT)
from sak.prompts import MENU_EXTRACTOR_PROMPT_LITE_V1

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.jsonl')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "qwen3.5-35B-A3B-FP8")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-FP8")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
CHAT_COMPLETIONS_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"
METRICS_URL = os.getenv("VLLM_METRICS_URL", f"{VLLM_BASE_URL.rstrip('/')}/metrics")
KV_CACHE_METRIC_CANDIDATES = [
    "vllm:gpu_cache_usage_perc",
    "vllm_gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm_kv_cache_usage_perc",
    "vllm:gpu_cache_usage",
    "vllm_gpu_cache_usage",
    "vllm:kv_cache_usage",
    "vllm_kv_cache_usage"
]

BURST_CONCURRENCY_LEVELS = [1, 10, 20, 40, 60, 80]
ROUND_REPEATS = int(os.getenv("BENCHMARK_REPEATS", "5"))


class NvidiaSmiMonitor:
    def __init__(self, interval=0.1, metrics_url=METRICS_URL):
        self.interval = interval
        self.metrics_url = metrics_url
        self.stop_event = threading.Event()
        self.gpu_stats = []
        self.thread = threading.Thread(target=self._monitor)

    def _fetch_kv_cache_usage_pct(self):
        try:
            with urllib.request.urlopen(self.metrics_url, timeout=2) as response:
                metrics_text = response.read().decode("utf-8", errors="ignore")
        except Exception:
            return None, ""
        values_by_name = {}
        for line in metrics_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if " " not in stripped:
                continue
            metric_with_labels, raw_value = stripped.rsplit(" ", 1)
            metric_name = metric_with_labels.split("{", 1)[0]
            if metric_name not in KV_CACHE_METRIC_CANDIDATES:
                continue
            try:
                numeric_value = float(raw_value)
            except ValueError:
                continue
            values_by_name.setdefault(metric_name, []).append(numeric_value)
        for candidate in KV_CACHE_METRIC_CANDIDATES:
            values = values_by_name.get(candidate, [])
            if not values:
                continue
            value = mean(values)
            if value <= 1.0:
                value = value * 100.0
            return value, candidate
        return None, ""

    def start(self):
        self.stop_event.clear()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    for line in output.split('\n'):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) < 2:
                            continue
                        used = int(float(parts[0]))
                        total = int(float(parts[1]))
                        usage_pct = (used / total * 100.0) if total > 0 else 0.0
                        kv_cache_usage_pct, kv_metric_name = self._fetch_kv_cache_usage_pct()
                        self.gpu_stats.append({
                            'timestamp': time.time(),
                            'memory_used_mb': used,
                            'memory_total_mb': total,
                            'memory_usage_pct': usage_pct,
                            'kv_cache_usage_pct': kv_cache_usage_pct,
                            'kv_cache_metric_name': kv_metric_name
                        })
                        break
            except Exception:
                pass
            time.sleep(self.interval)

    def get_summary(self):
        if not self.gpu_stats:
            return {
                'max_gpu_memory_used_mb': 0,
                'gpu_memory_total_mb': 0,
                'gpu_memory_usage_avg_pct': 0.0,
                'gpu_memory_usage_p95_pct': 0.0,
                'gpu_memory_usage_max_pct': 0.0,
                'kv_cache_usage_avg_pct': 0.0,
                'kv_cache_usage_p95_pct': 0.0,
                'kv_cache_usage_max_pct': 0.0,
                'kv_cache_metric_name': ""
            }
        max_used = max(s['memory_used_mb'] for s in self.gpu_stats)
        total = self.gpu_stats[0]['memory_total_mb']
        usage_values = [s['memory_usage_pct'] for s in self.gpu_stats]
        kv_values = [s['kv_cache_usage_pct'] for s in self.gpu_stats if isinstance(s.get('kv_cache_usage_pct'), (int, float))]
        kv_metric_names = [s['kv_cache_metric_name'] for s in self.gpu_stats if s.get('kv_cache_metric_name')]
        kv_metric_name = kv_metric_names[0] if kv_metric_names else ""
        return {
            'max_gpu_memory_used_mb': max_used,
            'gpu_memory_total_mb': total,
            'gpu_memory_usage_avg_pct': mean(usage_values),
            'gpu_memory_usage_p95_pct': percentile(usage_values, 95),
            'gpu_memory_usage_max_pct': max(usage_values),
            'kv_cache_usage_avg_pct': mean(kv_values) if kv_values else 0.0,
            'kv_cache_usage_p95_pct': percentile(kv_values, 95),
            'kv_cache_usage_max_pct': max(kv_values) if kv_values else 0.0,
            'kv_cache_metric_name': kv_metric_name
        }


def percentile(values, p):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    position = (len(sorted_vals) - 1) * p / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_vals[lower])
    ratio = position - lower
    return float(sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * ratio)


def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def construct_messages(record):
    system_content = MENU_EXTRACTOR_PROMPT_LITE_V1
    user_input = record['input_text']
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]


def estimate_prompt_tokens(messages):
    all_text = " ".join(str(m.get("content", "")) for m in messages)
    return len(all_text.split())


def request_vllm(messages, temperature=0.0, max_tokens=1024, timeout=120):
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
    request = urllib.request.Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    started_at = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        ended_at = time.time()
        content = result["choices"][0]["message"]["content"]
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = str(content)
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        used_server_usage = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
        if not isinstance(prompt_tokens, int):
            prompt_tokens = estimate_prompt_tokens(messages)
        if not isinstance(completion_tokens, int):
            completion_tokens = len(text.split())
        return {
            "ok": True,
            "latency_s": ended_at - started_at,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens) + int(completion_tokens),
            "used_server_usage": used_server_usage,
            "error": None
        }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        return {
            "ok": False,
            "latency_s": time.time() - started_at,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "used_server_usage": False,
            "error": f"HTTP {e.code}: {error_body}"
        }
    except Exception as e:
        return {
            "ok": False,
            "latency_s": time.time() - started_at,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "used_server_usage": False,
            "error": str(e)
        }


def build_messages_for_count(base_messages_list, count):
    if not base_messages_list:
        return []
    return [base_messages_list[i % len(base_messages_list)] for i in range(count)]


def summarize_round(round_results, total_time_s, monitor_summary, concurrency):
    latencies = [item["latency_s"] for item in round_results]
    success_items = [item for item in round_results if item["ok"]]
    error_count = len(round_results) - len(success_items)
    success_rate = (len(success_items) / len(round_results)) if round_results else 0.0
    total_prompt_tokens = sum(item["prompt_tokens"] for item in success_items)
    total_completion_tokens = sum(item["completion_tokens"] for item in success_items)
    total_tokens = total_prompt_tokens + total_completion_tokens
    server_usage_count = sum(1 for item in success_items if item["used_server_usage"])
    usage_ratio = (server_usage_count / len(success_items)) if success_items else 0.0
    throughput_tokens_per_s = total_tokens / total_time_s if total_time_s > 0 else 0.0
    actual_success_req_per_s = len(success_items) / total_time_s if total_time_s > 0 else 0.0
    return {
        "concurrency": concurrency,
        "request_count": len(round_results),
        "total_time_s": total_time_s,
        "success_count": len(success_items),
        "error_count": error_count,
        "success_rate": success_rate,
        "actual_success_req_per_s": actual_success_req_per_s,
        "latency_avg_s": mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "total_input_tokens": total_prompt_tokens,
        "total_output_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "usage_from_server_ratio": usage_ratio,
        **monitor_summary
    }


def average_rounds(rounds):
    numeric_keys = [
        "total_time_s",
        "success_count",
        "error_count",
        "success_rate",
        "actual_success_req_per_s",
        "latency_avg_s",
        "latency_p50_s",
        "latency_p95_s",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "throughput_tokens_per_s",
        "usage_from_server_ratio",
        "max_gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_memory_usage_avg_pct",
        "gpu_memory_usage_p95_pct",
        "gpu_memory_usage_max_pct",
        "kv_cache_usage_avg_pct",
        "kv_cache_usage_p95_pct",
        "kv_cache_usage_max_pct"
    ]
    avg = {}
    for key in numeric_keys:
        values = [float(item.get(key, 0.0)) for item in rounds]
        avg[key] = mean(values) if values else 0.0
    kv_metric_names = [item.get("kv_cache_metric_name", "") for item in rounds if item.get("kv_cache_metric_name")]
    avg["kv_cache_metric_name"] = kv_metric_names[0] if kv_metric_names else ""
    return avg


def run_burst_round(messages_for_round, concurrency):
    monitor = NvidiaSmiMonitor()
    monitor.start()
    started_at = time.time()
    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for messages in messages_for_round:
            futures.append(executor.submit(request_vllm, messages))
        round_results = [future.result() for future in as_completed(futures)]
    ended_at = time.time()
    monitor.stop()
    total_time_s = ended_at - started_at
    monitor_summary = monitor.get_summary()
    return summarize_round(
        round_results=round_results,
        total_time_s=total_time_s,
        monitor_summary=monitor_summary,
        concurrency=concurrency
    )


def main():
    print(f"Using remote vLLM service: {CHAT_COMPLETIONS_URL}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Loading data from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    base_messages = [construct_messages(record) for record in data]
    if not base_messages:
        raise RuntimeError("No data loaded from dataset.jsonl")

    print("Warming up...")
    request_vllm(base_messages[0], max_tokens=10)

    all_results = []
    for concurrency in BURST_CONCURRENCY_LEVELS:
        messages_for_round = build_messages_for_count(base_messages, concurrency)
        print(f"\n=== Burst concurrency test: {concurrency} ===")
        rounds = []
        for round_idx in range(1, ROUND_REPEATS + 1):
            print(f"Round {round_idx}/{ROUND_REPEATS}")
            round_result = run_burst_round(messages_for_round, concurrency)
            rounds.append(round_result)
            print(
                f"success_rate={round_result['success_rate']:.3f}, "
                f"lat_p50={round_result['latency_p50_s']:.3f}s, "
                f"lat_p95={round_result['latency_p95_s']:.3f}s, "
                f"mem_max={round_result['max_gpu_memory_used_mb']:.1f}MB"
            )
        all_results.append({
            "test_type": "burst_concurrency",
            "concurrency": concurrency,
            "round_repeats": ROUND_REPEATS,
            "requests_per_round": concurrency,
            "average": average_rounds(rounds)
        })

    output_file = os.path.join(OUTPUT_DIR, "hardware_benchmark_burst_concurrency.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": MODEL_NAME,
            "base_url": VLLM_BASE_URL,
            "chat_completions_url": CHAT_COMPLETIONS_URL,
            "metrics_url": METRICS_URL,
            "round_repeats": ROUND_REPEATS,
            "burst_concurrency_levels": BURST_CONCURRENCY_LEVELS,
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nBurst benchmark completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
