import asyncio
import json
import orjson
import math
import multiprocessing as mp
import os
import time
import urllib.request
from statistics import mean

try:
    import httpx
except Exception:
    httpx = None

try:
    import pynvml
except Exception:
    pynvml = None


TTFT_SUM_CANDIDATES = [
    "vllm:time_to_first_token_seconds_sum",
    "vllm_time_to_first_token_seconds_sum"
]
TTFT_COUNT_CANDIDATES = [
    "vllm:time_to_first_token_seconds_count",
    "vllm_time_to_first_token_seconds_count"
]
QUEUE_SUM_CANDIDATES = [
    "vllm:request_queue_time_seconds_sum",
    "vllm_request_queue_time_seconds_sum"
]
QUEUE_COUNT_CANDIDATES = [
    "vllm:request_queue_time_seconds_count",
    "vllm_request_queue_time_seconds_count"
]
GEN_TPUT_CANDIDATES = [
    "vllm:avg_generation_throughput_tokens_per_second",
    "vllm_avg_generation_throughput_tokens_per_second"
]
GPU_CACHE_CANDIDATES = [
    "vllm:gpu_cache_usage_perc",
    "vllm_gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm_kv_cache_usage_perc"
]
CPU_CACHE_CANDIDATES = [
    "vllm:cpu_cache_usage_perc",
    "vllm_cpu_cache_usage_perc"
]


def parse_env_bool(name, default=False):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


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


def estimate_prompt_tokens(messages):
    all_text = " ".join(str(m.get("content", "")) for m in messages)
    return len(all_text.split())


def pick_metric_value(metric_map, candidates):
    for name in candidates:
        values = metric_map.get(name)
        if values:
            value = mean(values)
            if "perc" in name and value <= 1.0:
                value = value * 100.0
            return float(value), name
    return None, ""


def parse_prometheus_metrics(metrics_text):
    metrics = {}
    for line in metrics_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or " " not in stripped:
            continue
        metric_with_labels, raw_value = stripped.rsplit(" ", 1)
        metric_name = metric_with_labels.split("{", 1)[0]
        try:
            value = float(raw_value)
        except Exception:
            continue
        metrics.setdefault(metric_name, []).append(value)
    return metrics


def fetch_prometheus_metrics(metrics_url, timeout=2):
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
            metrics_text = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return {}
    return parse_prometheus_metrics(metrics_text)


def read_process_memory_mb():
    rss_mb = 0.0
    vms_mb = 0.0
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_mb = float(parts[1]) / 1024.0
                elif line.startswith("VmSize:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        vms_mb = float(parts[1]) / 1024.0
    except Exception:
        pass
    return rss_mb, vms_mb


def summarize_process_memory_samples(samples, prefix):
    rss_values = [float(s.get("rss_mb", 0.0)) for s in samples if isinstance(s.get("rss_mb"), (int, float))]
    vms_values = [float(s.get("vms_mb", 0.0)) for s in samples if isinstance(s.get("vms_mb"), (int, float))]
    return {
        f"{prefix}_rss_avg_mb": mean(rss_values) if rss_values else 0.0,
        f"{prefix}_rss_p95_mb": percentile(rss_values, 95),
        f"{prefix}_rss_max_mb": max(rss_values) if rss_values else 0.0,
        f"{prefix}_vms_avg_mb": mean(vms_values) if vms_values else 0.0,
        f"{prefix}_vms_p95_mb": percentile(vms_values, 95),
        f"{prefix}_vms_max_mb": max(vms_values) if vms_values else 0.0
    }


async def sample_main_process_memory(stop_event, samples, interval_s=0.2):
    while not stop_event.is_set():
        rss_mb, vms_mb = read_process_memory_mb()
        samples.append({
            "timestamp": time.time(),
            "rss_mb": rss_mb,
            "vms_mb": vms_mb
        })
        await asyncio.sleep(interval_s)


def read_system_cpu_stat():
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = [float(x) for x in line.split()[1:]]
                    idle = parts[3] + parts[4]
                    total = sum(parts)
                    return idle, total
    except Exception:
        pass
    return 0.0, 0.0


def _monitor_process_main(metrics_url, interval_s, stop_event, result_queue):
    samples = []
    gpu_backend = "none"
    nvml_initialized = False
    handles = []
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            nvml_initialized = True
            gpu_backend = "pynvml"
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
        except Exception:
            handles = []
            
    last_cpu_idle, last_cpu_total = read_system_cpu_stat()
    
    while not stop_event.is_set():
        ts = time.time()
        metric_map = fetch_prometheus_metrics(metrics_url, timeout=2)
        process_rss_mb, process_vms_mb = read_process_memory_mb()
        
        curr_cpu_idle, curr_cpu_total = read_system_cpu_stat()
        diff_idle = curr_cpu_idle - last_cpu_idle
        diff_total = curr_cpu_total - last_cpu_total
        system_cpu_usage_pct = 100.0 * (diff_total - diff_idle) / diff_total if diff_total > 0 else 0.0
        last_cpu_idle, last_cpu_total = curr_cpu_idle, curr_cpu_total
        
        gpu_used_mb = 0.0
        gpu_total_mb = 0.0
        gpu_compute_usage_values = []
        if nvml_initialized and handles:
            try:
                for handle in handles:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_used_mb += mem.used / (1024.0 * 1024.0)
                    gpu_total_mb += mem.total / (1024.0 * 1024.0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_compute_usage_values.append(float(util.gpu))
            except Exception:
                pass
        gpu_usage_pct = (gpu_used_mb / gpu_total_mb * 100.0) if gpu_total_mb > 0 else 0.0
        gpu_compute_usage_pct = mean(gpu_compute_usage_values) if gpu_compute_usage_values else 0.0
        ttft_sum, ttft_sum_name = pick_metric_value(metric_map, TTFT_SUM_CANDIDATES)
        ttft_count, ttft_count_name = pick_metric_value(metric_map, TTFT_COUNT_CANDIDATES)
        queue_sum, queue_sum_name = pick_metric_value(metric_map, QUEUE_SUM_CANDIDATES)
        queue_count, queue_count_name = pick_metric_value(metric_map, QUEUE_COUNT_CANDIDATES)
        gen_tput, gen_tput_name = pick_metric_value(metric_map, GEN_TPUT_CANDIDATES)
        gpu_cache, gpu_cache_name = pick_metric_value(metric_map, GPU_CACHE_CANDIDATES)
        cpu_cache, cpu_cache_name = pick_metric_value(metric_map, CPU_CACHE_CANDIDATES)
        samples.append({
            "timestamp": ts,
            "monitor_process_rss_mb": process_rss_mb,
            "monitor_process_vms_mb": process_vms_mb,
            "system_cpu_usage_pct": system_cpu_usage_pct,
            "gpu_memory_used_mb": gpu_used_mb,
            "gpu_memory_total_mb": gpu_total_mb,
            "gpu_memory_usage_pct": gpu_usage_pct,
            "gpu_compute_usage_pct": gpu_compute_usage_pct,
            "ttft_sum": ttft_sum,
            "ttft_count": ttft_count,
            "queue_sum": queue_sum,
            "queue_count": queue_count,
            "gen_tput": gen_tput,
            "gpu_cache_usage_pct": gpu_cache,
            "cpu_cache_usage_pct": cpu_cache,
            "ttft_sum_metric_name": ttft_sum_name,
            "ttft_count_metric_name": ttft_count_name,
            "queue_sum_metric_name": queue_sum_name,
            "queue_count_metric_name": queue_count_name,
            "gen_tput_metric_name": gen_tput_name,
            "gpu_cache_metric_name": gpu_cache_name,
            "cpu_cache_metric_name": cpu_cache_name
        })
        time.sleep(interval_s)
    if nvml_initialized:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    result_queue.put({
        "gpu_monitor_backend": gpu_backend,
        "samples": samples
    })


def start_monitor_process(metrics_url, interval_s=0.2):
    stop_event = mp.Event()
    result_queue = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_monitor_process_main,
        args=(metrics_url, interval_s, stop_event, result_queue),
        daemon=True
    )
    process.start()
    return process, stop_event, result_queue


def stop_monitor_process(process, stop_event, result_queue, timeout_s=8):
    stop_event.set()
    try:
        # 1. First try to get data to clear the pipe buffer so the child process can finish successfully
        # Give a little buffer time for the child process to put the last data
        payload = result_queue.get(timeout=timeout_s)
    except Exception:
        payload = {"gpu_monitor_backend": "none", "samples": []}
    
    # 2. Then join
    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        
    return payload


def summarize_monitor_payload(payload):
    samples = payload.get("samples", [])
    if not samples:
        return {
            "gpu_monitor_backend": payload.get("gpu_monitor_backend", "none"),
            "max_gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
            "gpu_memory_usage_avg_pct": 0.0,
            "gpu_memory_usage_p95_pct": 0.0,
            "gpu_memory_usage_max_pct": 0.0,
            "gpu_compute_usage_avg_pct": 0.0,
            "gpu_compute_usage_p95_pct": 0.0,
            "gpu_compute_usage_max_pct": 0.0,
            "kv_cache_usage_avg_pct": 0.0,
            "kv_cache_usage_p95_pct": 0.0,
            "kv_cache_usage_max_pct": 0.0,
            "cpu_cache_usage_avg_pct": 0.0,
            "cpu_cache_usage_p95_pct": 0.0,
            "cpu_cache_usage_max_pct": 0.0,
            "prom_avg_generation_throughput_tokens_per_second": 0.0,
            "prom_ttft_avg_s": 0.0,
            "prom_queue_avg_s": 0.0,
            "prom_ttft_sum_metric_name": "",
            "prom_ttft_count_metric_name": "",
            "prom_queue_sum_metric_name": "",
            "prom_queue_count_metric_name": "",
            "prom_generation_tput_metric_name": "",
            "kv_cache_metric_name": "",
            "cpu_cache_metric_name": "",
            "monitor_process_memory_rss_avg_mb": 0.0,
            "monitor_process_memory_rss_p95_mb": 0.0,
            "monitor_process_memory_rss_max_mb": 0.0,
            "monitor_process_memory_vms_avg_mb": 0.0,
            "monitor_process_memory_vms_p95_mb": 0.0,
            "monitor_process_memory_vms_max_mb": 0.0,
            "system_cpu_usage_avg_pct": 0.0,
            "system_cpu_usage_p95_pct": 0.0,
            "system_cpu_usage_max_pct": 0.0
        }
    gpu_used_values = [float(s.get("gpu_memory_used_mb", 0.0)) for s in samples]
    gpu_total_values = [float(s.get("gpu_memory_total_mb", 0.0)) for s in samples if float(s.get("gpu_memory_total_mb", 0.0)) > 0]
    gpu_usage_values = [float(s.get("gpu_memory_usage_pct", 0.0)) for s in samples]
    gpu_compute_values = [float(s.get("gpu_compute_usage_pct", 0.0)) for s in samples]
    kv_values = [float(s.get("gpu_cache_usage_pct")) for s in samples if isinstance(s.get("gpu_cache_usage_pct"), (int, float))]
    cpu_values = [float(s.get("cpu_cache_usage_pct")) for s in samples if isinstance(s.get("cpu_cache_usage_pct"), (int, float))]
    sys_cpu_values = [float(s.get("system_cpu_usage_pct", 0.0)) for s in samples]
    monitor_rss_values = [float(s.get("monitor_process_rss_mb")) for s in samples if isinstance(s.get("monitor_process_rss_mb"), (int, float))]
    monitor_vms_values = [float(s.get("monitor_process_vms_mb")) for s in samples if isinstance(s.get("monitor_process_vms_mb"), (int, float))]
    tput_values = [float(s.get("gen_tput")) for s in samples if isinstance(s.get("gen_tput"), (int, float))]
    ttft_sum_values = [float(s.get("ttft_sum")) for s in samples if isinstance(s.get("ttft_sum"), (int, float))]
    ttft_count_values = [float(s.get("ttft_count")) for s in samples if isinstance(s.get("ttft_count"), (int, float))]
    queue_sum_values = [float(s.get("queue_sum")) for s in samples if isinstance(s.get("queue_sum"), (int, float))]
    queue_count_values = [float(s.get("queue_count")) for s in samples if isinstance(s.get("queue_count"), (int, float))]
    ttft_avg = 0.0
    if len(ttft_sum_values) >= 2 and len(ttft_count_values) >= 2:
        delta_sum = ttft_sum_values[-1] - ttft_sum_values[0]
        delta_count = ttft_count_values[-1] - ttft_count_values[0]
        if delta_count > 0:
            ttft_avg = max(0.0, delta_sum / delta_count)
    queue_avg = 0.0
    if len(queue_sum_values) >= 2 and len(queue_count_values) >= 2:
        delta_sum = queue_sum_values[-1] - queue_sum_values[0]
        delta_count = queue_count_values[-1] - queue_count_values[0]
        if delta_count > 0:
            queue_avg = max(0.0, delta_sum / delta_count)
    ttft_sum_metric_name = next((s.get("ttft_sum_metric_name", "") for s in samples if s.get("ttft_sum_metric_name")), "")
    ttft_count_metric_name = next((s.get("ttft_count_metric_name", "") for s in samples if s.get("ttft_count_metric_name")), "")
    queue_sum_metric_name = next((s.get("queue_sum_metric_name", "") for s in samples if s.get("queue_sum_metric_name")), "")
    queue_count_metric_name = next((s.get("queue_count_metric_name", "") for s in samples if s.get("queue_count_metric_name")), "")
    gen_tput_metric_name = next((s.get("gen_tput_metric_name", "") for s in samples if s.get("gen_tput_metric_name")), "")
    kv_cache_metric_name = next((s.get("gpu_cache_metric_name", "") for s in samples if s.get("gpu_cache_metric_name")), "")
    cpu_cache_metric_name = next((s.get("cpu_cache_metric_name", "") for s in samples if s.get("cpu_cache_metric_name")), "")
    return {
        "gpu_monitor_backend": payload.get("gpu_monitor_backend", "none"),
        "max_gpu_memory_used_mb": max(gpu_used_values) if gpu_used_values else 0.0,
        "gpu_memory_total_mb": mean(gpu_total_values) if gpu_total_values else 0.0,
        "gpu_memory_usage_avg_pct": mean(gpu_usage_values) if gpu_usage_values else 0.0,
        "gpu_memory_usage_p95_pct": percentile(gpu_usage_values, 95),
        "gpu_memory_usage_max_pct": max(gpu_usage_values) if gpu_usage_values else 0.0,
        "gpu_compute_usage_avg_pct": mean(gpu_compute_values) if gpu_compute_values else 0.0,
        "gpu_compute_usage_p95_pct": percentile(gpu_compute_values, 95),
        "gpu_compute_usage_max_pct": max(gpu_compute_values) if gpu_compute_values else 0.0,
        "kv_cache_usage_avg_pct": mean(kv_values) if kv_values else 0.0,
        "kv_cache_usage_p95_pct": percentile(kv_values, 95),
        "kv_cache_usage_max_pct": max(kv_values) if kv_values else 0.0,
        "cpu_cache_usage_avg_pct": mean(cpu_values) if cpu_values else 0.0,
        "cpu_cache_usage_p95_pct": percentile(cpu_values, 95),
        "cpu_cache_usage_max_pct": max(cpu_values) if cpu_values else 0.0,
        "prom_avg_generation_throughput_tokens_per_second": mean(tput_values) if tput_values else 0.0,
        "prom_ttft_avg_s": ttft_avg,
        "prom_queue_avg_s": queue_avg,
        "prom_ttft_sum_metric_name": ttft_sum_metric_name,
        "prom_ttft_count_metric_name": ttft_count_metric_name,
        "prom_queue_sum_metric_name": queue_sum_metric_name,
        "prom_queue_count_metric_name": queue_count_metric_name,
        "prom_generation_tput_metric_name": gen_tput_metric_name,
        "kv_cache_metric_name": kv_cache_metric_name,
        "cpu_cache_metric_name": cpu_cache_metric_name,
        "monitor_process_memory_rss_avg_mb": mean(monitor_rss_values) if monitor_rss_values else 0.0,
        "monitor_process_memory_rss_p95_mb": percentile(monitor_rss_values, 95),
        "monitor_process_memory_rss_max_mb": max(monitor_rss_values) if monitor_rss_values else 0.0,
        "monitor_process_memory_vms_avg_mb": mean(monitor_vms_values) if monitor_vms_values else 0.0,
        "monitor_process_memory_vms_p95_mb": percentile(monitor_vms_values, 95),
        "monitor_process_memory_vms_max_mb": max(monitor_vms_values) if monitor_vms_values else 0.0,
        "system_cpu_usage_avg_pct": mean(sys_cpu_values) if sys_cpu_values else 0.0,
        "system_cpu_usage_p95_pct": percentile(sys_cpu_values, 95),
        "system_cpu_usage_max_pct": max(sys_cpu_values) if sys_cpu_values else 0.0
    }


class AsyncVLLMClient:
    def __init__(self, chat_completions_url, model_name, timeout_s=120, max_connections=256, max_keepalive_connections=128, enable_stream=False):
        if httpx is None:
            raise RuntimeError("httpx is required for async benchmark. Please install httpx.")
        self.chat_completions_url = chat_completions_url
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.enable_stream = enable_stream
        self.client = None

    async def __aenter__(self):
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections
        )
        timeout = httpx.Timeout(self.timeout_s)
        self.client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.client is not None:
            await self.client.aclose()

    async def request(self, messages, temperature=0.0, max_tokens=1024):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": self.enable_stream,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
        enqueued_at = time.perf_counter()
        started_at = None
        try:
            started_at = time.perf_counter()
            ttft_s = None
            tpot_s = None
            if self.enable_stream:
                text_parts = []
                first_token_at = None
                usage = {}
                async with self.client.stream("POST", self.chat_completions_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = orjson.loads(data)
                        except Exception:
                            continue
                        if isinstance(event.get("usage"), dict):
                            usage = event.get("usage", {})
                        choices = event.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        delta_content = delta.get("content")
                        token_text = ""
                        if isinstance(delta_content, str):
                            token_text = delta_content
                        elif isinstance(delta_content, list):
                            token_text = "".join(part.get("text", "") for part in delta_content if isinstance(part, dict))
                        if token_text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            text_parts.append(token_text)
                ended_at = time.perf_counter()
                text = "".join(text_parts)
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                used_server_usage = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
                if not isinstance(prompt_tokens, int):
                    prompt_tokens = estimate_prompt_tokens(messages)
                if not isinstance(completion_tokens, int):
                    completion_tokens = len(text.split())
                if first_token_at is not None:
                    ttft_s = max(0.0, first_token_at - enqueued_at)
                    generation_tail_s = max(0.0, ended_at - first_token_at)
                    if int(completion_tokens) > 0:
                        tpot_s = generation_tail_s / int(completion_tokens)
            else:
                response = await self.client.post(self.chat_completions_url, json=payload)
                response.raise_for_status()
                ended_at = time.perf_counter()
                raw_bytes = response.content
                result = await asyncio.to_thread(orjson.loads, raw_bytes)
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
            queue_wait_s = max(0.0, (started_at - enqueued_at) if started_at is not None else 0.0)
            result_payload = {
                "ok": True,
                "latency_s": max(0.0, ended_at - enqueued_at),
                "queue_wait_client_s": queue_wait_s,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
                "used_server_usage": used_server_usage,
                "error": None
            }
            if self.enable_stream:
                result_payload["ttft_s"] = float(ttft_s) if isinstance(ttft_s, (int, float)) else None
                result_payload["tpot_s"] = float(tpot_s) if isinstance(tpot_s, (int, float)) else None
            return result_payload
        except Exception as e:
            ended_at = time.perf_counter()
            queue_wait_s = max(0.0, (started_at - enqueued_at) if started_at is not None else 0.0)
            result_payload = {
                "ok": False,
                "latency_s": max(0.0, ended_at - enqueued_at),
                "queue_wait_client_s": queue_wait_s,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "used_server_usage": False,
                "error": str(e)
            }
            if self.enable_stream:
                result_payload["ttft_s"] = None
                result_payload["tpot_s"] = None
            return result_payload


def build_messages_for_count(base_messages_list, count):
    if not base_messages_list:
        return []
    return [base_messages_list[i % len(base_messages_list)] for i in range(count)]


def summarize_round(round_results, total_time_s, monitor_summary, scenario_fields, include_stream_metrics):
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
    result_payload = {
        **scenario_fields,
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
    if include_stream_metrics:
        ttft_values = [float(item.get("ttft_s")) for item in success_items if isinstance(item.get("ttft_s"), (int, float))]
        tpot_values = [float(item.get("tpot_s")) for item in success_items if isinstance(item.get("tpot_s"), (int, float))]
        if not tpot_values and total_completion_tokens > 0 and ttft_values:
            ttft_avg_s = mean(ttft_values)
            generation_time_sum_s = sum(max(0.0, item["latency_s"] - ttft_avg_s) for item in success_items)
            tpot_values = [generation_time_sum_s / total_completion_tokens]
        queue_wait_values = [item.get("queue_wait_client_s", 0.0) for item in round_results]
        result_payload["ttft_avg_s"] = mean(ttft_values) if ttft_values else 0.0
        result_payload["tpot_avg_s"] = mean(tpot_values) if tpot_values else 0.0
        result_payload["queue_wait_client_avg_s"] = mean(queue_wait_values) if queue_wait_values else 0.0
        result_payload["queue_wait_client_p95_s"] = percentile(queue_wait_values, 95)
        result_payload["queue_avg_s"] = float(monitor_summary.get("prom_queue_avg_s", 0.0))
        result_payload["prom_ttft_avg_s"] = float(monitor_summary.get("prom_ttft_avg_s", 0.0))
    else:
        for key in [
            "prom_ttft_avg_s",
            "prom_queue_avg_s",
            "prom_ttft_sum_metric_name",
            "prom_ttft_count_metric_name",
            "prom_queue_sum_metric_name",
            "prom_queue_count_metric_name"
        ]:
            result_payload.pop(key, None)
    return result_payload


def average_rounds(rounds):
    if not rounds:
        return {}
    numeric_keys = set()
    for item in rounds:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)
    avg = {}
    for key in sorted(numeric_keys):
        values = [float(item.get(key, 0.0)) for item in rounds]
        avg[key] = mean(values) if values else 0.0
    string_keys = [
        "gpu_monitor_backend",
        "prom_ttft_sum_metric_name",
        "prom_ttft_count_metric_name",
        "prom_queue_sum_metric_name",
        "prom_queue_count_metric_name",
        "prom_generation_tput_metric_name",
        "kv_cache_metric_name",
        "cpu_cache_metric_name"
    ]
    for key in string_keys:
        values = [item.get(key, "") for item in rounds if item.get(key)]
        avg[key] = values[0] if values else ""
    return avg


async def run_steady_round_async(messages_for_round, target_rps, monitor_metrics_url, model_name, chat_completions_url, timeout_s=120, enable_stream=False):
    interval_s = 1.0 / float(target_rps)
    monitor_process, monitor_stop_event, monitor_queue = start_monitor_process(monitor_metrics_url)
    main_process_memory_samples = []
    main_memory_stop_event = asyncio.Event()
    main_memory_sampler = asyncio.create_task(
        sample_main_process_memory(main_memory_stop_event, main_process_memory_samples, interval_s=0.2)
    )
    started_at = time.perf_counter()
    try:
        async with AsyncVLLMClient(
            chat_completions_url=chat_completions_url,
            model_name=model_name,
            timeout_s=timeout_s,
            max_connections=max(64, target_rps * 20),
            max_keepalive_connections=max(32, target_rps * 10),
            enable_stream=enable_stream
        ) as client:
            tasks = []
            for idx, messages in enumerate(messages_for_round):
                target_at = started_at + idx * interval_s
                async def _send_once(msg=messages, send_at=target_at):
                    delay = send_at - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return await client.request(msg)
                tasks.append(asyncio.create_task(_send_once()))
            round_results = await asyncio.gather(*tasks)
    finally:
        main_memory_stop_event.set()
        await main_memory_sampler
    ended_at = time.perf_counter()
    monitor_payload = stop_monitor_process(monitor_process, monitor_stop_event, monitor_queue)
    monitor_summary = summarize_monitor_payload(monitor_payload)
    monitor_summary.update(summarize_process_memory_samples(main_process_memory_samples, "main_process_memory"))
    total_time_s = ended_at - started_at
    return summarize_round(
        round_results=round_results,
        total_time_s=total_time_s,
        monitor_summary=monitor_summary,
        scenario_fields={"target_rps": target_rps},
        include_stream_metrics=enable_stream
    )


async def run_burst_round_async(messages_for_round, concurrency, monitor_metrics_url, model_name, chat_completions_url, timeout_s=120, enable_stream=False):
    monitor_process, monitor_stop_event, monitor_queue = start_monitor_process(monitor_metrics_url)
    main_process_memory_samples = []
    main_memory_stop_event = asyncio.Event()
    main_memory_sampler = asyncio.create_task(
        sample_main_process_memory(main_memory_stop_event, main_process_memory_samples, interval_s=0.2)
    )
    started_at = time.perf_counter()
    try:
        async with AsyncVLLMClient(
            chat_completions_url=chat_completions_url,
            model_name=model_name,
            timeout_s=timeout_s,
            max_connections=max(64, concurrency * 20),
            max_keepalive_connections=max(32, concurrency * 10),
            enable_stream=enable_stream
        ) as client:
            semaphore = asyncio.Semaphore(concurrency)
            async def _send_once(messages):
                async with semaphore:
                    return await client.request(messages)
            tasks = [asyncio.create_task(_send_once(messages)) for messages in messages_for_round]
            round_results = await asyncio.gather(*tasks)
    finally:
        main_memory_stop_event.set()
        await main_memory_sampler
    ended_at = time.perf_counter()
    monitor_payload = stop_monitor_process(monitor_process, monitor_stop_event, monitor_queue)
    monitor_summary = summarize_monitor_payload(monitor_payload)
    monitor_summary.update(summarize_process_memory_samples(main_process_memory_samples, "main_process_memory"))
    total_time_s = ended_at - started_at
    return summarize_round(
        round_results=round_results,
        total_time_s=total_time_s,
        monitor_summary=monitor_summary,
        scenario_fields={"concurrency": concurrency},
        include_stream_metrics=enable_stream
    )


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
