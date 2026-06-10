import asyncio
import json
import orjson
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
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
PROMPT_TOKENS_CANDIDATES = [
    "vllm:prompt_tokens_total",
    "vllm_prompt_tokens_total"
]
PROMPT_TOKENS_CACHED_CANDIDATES = [
    "vllm:prompt_tokens_cached_total",
    "vllm_prompt_tokens_cached_total",
    "vllm:prompt_tokens_cached",
    "vllm_prompt_tokens_cached"
]
GENERATION_TOKENS_CANDIDATES = [
    "vllm:generation_tokens_total",
    "vllm_generation_tokens_total"
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
DEFAULT_DCGMI_FIELD_IDS = [1002, 1003, 1004, 1005, 1009, 1010]
DCGMI_FIELD_KEY_MAP = {
    1002: "dcgmi_sm_activity_pct",
    1003: "dcgmi_sm_occupancy_pct",
    1004: "dcgmi_tensor_core_activity_pct",
    1005: "dcgmi_dram_active_pct",
    1009: "dcgmi_pcie_rx_bytes_per_s",
    1010: "dcgmi_pcie_tx_bytes_per_s"
}


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


def parse_env_float(name, default=0.0):
    raw_value = os.getenv(name)
    if raw_value is None:
        return float(default)
    try:
        return float(raw_value)
    except Exception:
        return float(default)


def parse_env_int_list(name, default_values):
    raw_value = os.getenv(name)
    if raw_value is None:
        return list(default_values)
    parsed = []
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            parsed.append(int(stripped))
        except Exception:
            continue
    return parsed if parsed else list(default_values)


def normalize_vllm_launch_args(process_line):
    if not process_line:
        return ""
    raw_line = str(process_line).strip()
    columns = raw_line.split(None, 10)
    command_line = columns[10] if len(columns) >= 11 else raw_line
    match = re.search(r"((?:\S*/)?vllm\s+serve\b.*)$", command_line)
    if not match:
        return command_line.strip()
    extracted = match.group(1).strip()
    tokens = extracted.split(None, 1)
    if not tokens:
        return ""
    head = os.path.basename(tokens[0])
    if head != "vllm":
        return extracted
    tail = tokens[1] if len(tokens) > 1 else ""
    return f"vllm {tail}".strip()


def collect_vllm_launch_args(port=8000):
    lsof_command = ["lsof", "-t", f"-iTCP:{int(port)}", "-sTCP:LISTEN"]
    try:
        lsof_result = subprocess.run(
            lsof_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
    except Exception:
        return ""
    pid_list = []
    for line in (lsof_result.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.isdigit():
            pid_list.append(int(stripped))
    if not pid_list:
        return ""
    selected_line = ""
    for pid in pid_list:
        grep_command = f"ps -auxww | grep -w '{pid}' | grep -v grep"
        try:
            ps_result = subprocess.run(
                ["bash", "-lc", grep_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3
            )
        except Exception:
            continue
        lines = [line.strip() for line in (ps_result.stdout or "").splitlines() if line.strip()]
        if not lines:
            continue
        candidate_line = lines[0]
        if not selected_line:
            selected_line = candidate_line
        lowered = candidate_line.lower()
        if ("vllm" in lowered) or ("api_server" in lowered):
            selected_line = candidate_line
            break
    return normalize_vllm_launch_args(selected_line)


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


def pick_metric_value(metric_map, candidates, agg_func=sum):
    for name in candidates:
        values = metric_map.get(name)
        if values:
            value = agg_func(values)
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


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _parse_dcgmi_metrics(output_text, field_ids):
    expected_count = len(field_ids)
    if expected_count <= 0:
        return {}
    candidate_rows = []
    for raw_line in output_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = re.findall(r"N/A|-?\d+(?:\.\d+)?", stripped, flags=re.IGNORECASE)
        if len(tokens) >= expected_count + 1:
            candidate_rows.append(tokens[-expected_count:])
        elif len(tokens) == expected_count:
            candidate_rows.append(tokens)
    if not candidate_rows:
        return {}
    value_tokens = candidate_rows[-1]
    payload = {}
    for idx, field_id in enumerate(field_ids):
        key = DCGMI_FIELD_KEY_MAP.get(field_id, f"dcgmi_field_{field_id}")
        numeric = _safe_float(value_tokens[idx])
        if numeric is None or abs(numeric) > 1e15:
            continue
        payload[key] = float(numeric)
        if field_id == 1005:
            payload["dcgmi_memory_utilization_pct"] = float(numeric)
    return payload


def _run_dcgmi_sample(gpu_selector, field_ids, sample_interval_s):
    interval_ms = max(100, int(sample_interval_s * 1000.0))
    command = [
        "dcgmi",
        "dmon",
        "-e",
        ",".join(str(field_id) for field_id in field_ids),
        "-d",
        str(interval_ms),
        "-c",
        "1",
        "-i",
        str(gpu_selector)
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(2.0, sample_interval_s + 1.0)
    )
    stderr_text = (completed.stderr or "").strip()
    stdout_text = completed.stdout or ""
    if completed.returncode != 0:
        return {
            "metrics": {},
            "returncode": int(completed.returncode),
            "stderr": stderr_text[:1000],
            "stdout": stdout_text[:1000]
        }
    parsed = _parse_dcgmi_metrics(stdout_text, field_ids)
    return {
        "metrics": parsed,
        "returncode": 0,
        "stderr": stderr_text[:1000],
        "stdout": stdout_text[:1000]
    }


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
    enable_dcgmi = parse_env_bool("BENCHMARK_ENABLE_DCGMI_METRICS", False)
    dcgmi_available = shutil.which("dcgmi") is not None
    dcgmi_field_ids = parse_env_int_list("BENCHMARK_DCGMI_FIELD_IDS", DEFAULT_DCGMI_FIELD_IDS)
    dcgmi_gpu_selector = os.getenv("BENCHMARK_DCGMI_GPU_SELECTOR", "0")
    dcgmi_sample_interval_s = parse_env_float("BENCHMARK_DCGMI_SAMPLE_INTERVAL_S", 0.2)
    dcgmi_enabled = bool(enable_dcgmi and dcgmi_available and dcgmi_field_ids)
    dcgmi_total_samples = 0
    dcgmi_nonempty_samples = 0
    dcgmi_failed_samples = 0
    dcgmi_empty_parse_samples = 0
    dcgmi_last_returncode = 0
    dcgmi_last_stderr = ""
    dcgmi_last_stdout = ""
    dcgmi_backend = "disabled"
    if enable_dcgmi and not dcgmi_available:
        dcgmi_backend = "dcgmi_not_found"
    elif enable_dcgmi and dcgmi_available and not dcgmi_field_ids:
        dcgmi_backend = "dcgmi_no_fields"
    elif dcgmi_enabled:
        dcgmi_backend = "dcgmi"
        gpu_backend = f"{gpu_backend}+dcgmi" if gpu_backend != "none" else "dcgmi"
            
    last_cpu_idle, last_cpu_total = read_system_cpu_stat()
    
    while not stop_event.is_set():
        loop_start = time.time()
        ts = loop_start
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
        dcgmi_metrics = {}
        if dcgmi_enabled:
            try:
                dcgmi_result = _run_dcgmi_sample(
                    gpu_selector=dcgmi_gpu_selector,
                    field_ids=dcgmi_field_ids,
                    sample_interval_s=dcgmi_sample_interval_s
                )
                dcgmi_metrics = dcgmi_result.get("metrics", {})
                dcgmi_last_returncode = int(dcgmi_result.get("returncode", 0))
                dcgmi_last_stderr = str(dcgmi_result.get("stderr", ""))
                dcgmi_last_stdout = str(dcgmi_result.get("stdout", ""))
                dcgmi_total_samples += 1
                if dcgmi_last_returncode != 0:
                    dcgmi_failed_samples += 1
                elif dcgmi_metrics:
                    dcgmi_nonempty_samples += 1
                else:
                    dcgmi_empty_parse_samples += 1
            except Exception:
                dcgmi_metrics = {}
                dcgmi_total_samples += 1
                dcgmi_failed_samples += 1
                dcgmi_last_returncode = -1
                dcgmi_last_stderr = "dcgmi sampling exception"
                dcgmi_last_stdout = ""
        ttft_sum, ttft_sum_name = pick_metric_value(metric_map, TTFT_SUM_CANDIDATES, sum)
        ttft_count, ttft_count_name = pick_metric_value(metric_map, TTFT_COUNT_CANDIDATES, sum)
        queue_sum, queue_sum_name = pick_metric_value(metric_map, QUEUE_SUM_CANDIDATES, sum)
        queue_count, queue_count_name = pick_metric_value(metric_map, QUEUE_COUNT_CANDIDATES, sum)
        gen_tput, gen_tput_name = pick_metric_value(metric_map, GEN_TPUT_CANDIDATES, sum)
        prompt_tokens_total, _ = pick_metric_value(metric_map, PROMPT_TOKENS_CANDIDATES, sum)
        prompt_tokens_cached, _ = pick_metric_value(metric_map, PROMPT_TOKENS_CACHED_CANDIDATES, sum)
        generation_tokens_total, _ = pick_metric_value(metric_map, GENERATION_TOKENS_CANDIDATES, sum)
        gpu_cache, gpu_cache_name = pick_metric_value(metric_map, GPU_CACHE_CANDIDATES, mean)
        cpu_cache, cpu_cache_name = pick_metric_value(metric_map, CPU_CACHE_CANDIDATES, mean)
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
            "prompt_tokens_total": prompt_tokens_total,
            "prompt_tokens_cached": prompt_tokens_cached,
            "generation_tokens_total": generation_tokens_total,
            "gpu_cache_usage_pct": gpu_cache,
            "cpu_cache_usage_pct": cpu_cache,
            "ttft_sum_metric_name": ttft_sum_name,
            "ttft_count_metric_name": ttft_count_name,
            "queue_sum_metric_name": queue_sum_name,
            "queue_count_metric_name": queue_count_name,
            "gen_tput_metric_name": gen_tput_name,
            "gpu_cache_metric_name": gpu_cache_name,
            "cpu_cache_metric_name": cpu_cache_name,
            **dcgmi_metrics
        })
        elapsed = time.time() - loop_start
        time.sleep(max(0.0, interval_s - elapsed))
    if nvml_initialized:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    result_queue.put({
        "gpu_monitor_backend": gpu_backend,
        "dcgmi_backend": dcgmi_backend,
        "dcgmi_enabled": dcgmi_enabled,
        "dcgmi_field_ids": ",".join(str(field_id) for field_id in dcgmi_field_ids),
        "dcgmi_gpu_selector": str(dcgmi_gpu_selector),
        "dcgmi_sample_interval_s": float(dcgmi_sample_interval_s),
        "dcgmi_total_samples": int(dcgmi_total_samples),
        "dcgmi_nonempty_samples": int(dcgmi_nonempty_samples),
        "dcgmi_failed_samples": int(dcgmi_failed_samples),
        "dcgmi_empty_parse_samples": int(dcgmi_empty_parse_samples),
        "dcgmi_last_returncode": int(dcgmi_last_returncode),
        "dcgmi_last_stderr": str(dcgmi_last_stderr),
        "dcgmi_last_stdout": str(dcgmi_last_stdout),
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


def stop_monitor_process(process, stop_event, result_queue, timeout_s=15):
    stop_event.set()
    payload = None
    start_wait = time.time()
    
    while True:
        try:
            payload = result_queue.get(timeout=1.0)
            break
        except Exception:
            if not process.is_alive():
                break
            if time.time() - start_wait > timeout_s:
                break
                
    if payload is None:
        payload = {"gpu_monitor_backend": "none", "samples": []}
    
    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        
    return payload


def summarize_monitor_payload(payload, window_start=None, window_end=None):
    samples = payload.get("samples", [])
    if window_start is not None and window_end is not None:
        samples = [s for s in samples if window_start <= s.get("timestamp", 0) <= window_end]
    dcgmi_backend = payload.get("dcgmi_backend", "disabled")
    dcgmi_enabled = bool(payload.get("dcgmi_enabled", False))
    dcgmi_field_ids = payload.get("dcgmi_field_ids", "")
    dcgmi_gpu_selector = payload.get("dcgmi_gpu_selector", "")
    dcgmi_sample_interval_s = float(payload.get("dcgmi_sample_interval_s", 0.0))
    dcgmi_total_samples = int(payload.get("dcgmi_total_samples", 0))
    dcgmi_nonempty_samples = int(payload.get("dcgmi_nonempty_samples", 0))
    dcgmi_failed_samples = int(payload.get("dcgmi_failed_samples", 0))
    dcgmi_empty_parse_samples = int(payload.get("dcgmi_empty_parse_samples", 0))
    dcgmi_last_returncode = int(payload.get("dcgmi_last_returncode", 0))
    dcgmi_last_stderr = str(payload.get("dcgmi_last_stderr", ""))
    dcgmi_last_stdout = str(payload.get("dcgmi_last_stdout", ""))
    if not samples:
        result = {
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
            "prom_gross_input_tps": 0.0,
            "prom_net_computed_input_tps": 0.0,
            "prom_decode_tps": 0.0,
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
            "system_cpu_usage_max_pct": 0.0,
            "dcgmi_backend": dcgmi_backend,
            "dcgmi_enabled": dcgmi_enabled,
            "dcgmi_field_ids": dcgmi_field_ids,
            "dcgmi_gpu_selector": dcgmi_gpu_selector,
            "dcgmi_sample_interval_s": dcgmi_sample_interval_s,
            "dcgmi_total_samples": float(dcgmi_total_samples),
            "dcgmi_nonempty_samples": float(dcgmi_nonempty_samples),
            "dcgmi_failed_samples": float(dcgmi_failed_samples),
            "dcgmi_empty_parse_samples": float(dcgmi_empty_parse_samples),
            "dcgmi_last_returncode": float(dcgmi_last_returncode),
            "dcgmi_last_stderr": dcgmi_last_stderr,
            "dcgmi_last_stdout": dcgmi_last_stdout,
            "dcgmi_sm_activity_avg_pct": 0.0,
            "dcgmi_sm_activity_p95_pct": 0.0,
            "dcgmi_sm_activity_max_pct": 0.0,
            "dcgmi_sm_occupancy_avg_pct": 0.0,
            "dcgmi_sm_occupancy_p95_pct": 0.0,
            "dcgmi_sm_occupancy_max_pct": 0.0,
            "dcgmi_tensor_core_activity_avg_pct": 0.0,
            "dcgmi_tensor_core_activity_p95_pct": 0.0,
            "dcgmi_tensor_core_activity_max_pct": 0.0,
            "dcgmi_dram_active_avg_pct": 0.0,
            "dcgmi_dram_active_p95_pct": 0.0,
            "dcgmi_dram_active_max_pct": 0.0,
            "dcgmi_memory_utilization_avg_pct": 0.0,
            "dcgmi_memory_utilization_p95_pct": 0.0,
            "dcgmi_memory_utilization_max_pct": 0.0,
            "dcgmi_pcie_rx_avg_bytes_per_s": 0.0,
            "dcgmi_pcie_rx_p95_bytes_per_s": 0.0,
            "dcgmi_pcie_rx_max_bytes_per_s": 0.0,
            "dcgmi_pcie_tx_avg_bytes_per_s": 0.0,
            "dcgmi_pcie_tx_p95_bytes_per_s": 0.0,
            "dcgmi_pcie_tx_max_bytes_per_s": 0.0
        }
        if not dcgmi_enabled:
            for key in [k for k in list(result.keys()) if k.startswith("dcgmi_")]:
                result.pop(key, None)
        return result
    gpu_used_values = [float(s.get("gpu_memory_used_mb", 0.0)) for s in samples]
    gpu_total_values = [float(s.get("gpu_memory_total_mb", 0.0)) for s in samples if float(s.get("gpu_memory_total_mb", 0.0)) > 0]
    gpu_usage_values = [float(s.get("gpu_memory_usage_pct", 0.0)) for s in samples]
    gpu_compute_values = [float(s.get("gpu_compute_usage_pct", 0.0)) for s in samples]
    kv_values = [float(s.get("gpu_cache_usage_pct")) for s in samples if isinstance(s.get("gpu_cache_usage_pct"), (int, float))]
    cpu_values = [float(s.get("cpu_cache_usage_pct")) for s in samples if isinstance(s.get("cpu_cache_usage_pct"), (int, float))]
    sys_cpu_values = [float(s.get("system_cpu_usage_pct", 0.0)) for s in samples]
    monitor_rss_values = [float(s.get("monitor_process_rss_mb")) for s in samples if isinstance(s.get("monitor_process_rss_mb"), (int, float))]
    monitor_vms_values = [float(s.get("monitor_process_vms_mb")) for s in samples if isinstance(s.get("monitor_process_vms_mb"), (int, float))]
    dcgmi_sm_activity_values = [float(s.get("dcgmi_sm_activity_pct")) for s in samples if isinstance(s.get("dcgmi_sm_activity_pct"), (int, float))]
    dcgmi_sm_occupancy_values = [float(s.get("dcgmi_sm_occupancy_pct")) for s in samples if isinstance(s.get("dcgmi_sm_occupancy_pct"), (int, float))]
    dcgmi_tensor_activity_values = [float(s.get("dcgmi_tensor_core_activity_pct")) for s in samples if isinstance(s.get("dcgmi_tensor_core_activity_pct"), (int, float))]
    dcgmi_dram_active_values = [float(s.get("dcgmi_dram_active_pct")) for s in samples if isinstance(s.get("dcgmi_dram_active_pct"), (int, float))]
    dcgmi_memory_util_values = [float(s.get("dcgmi_memory_utilization_pct")) for s in samples if isinstance(s.get("dcgmi_memory_utilization_pct"), (int, float))]
    dcgmi_pcie_rx_values = [float(s.get("dcgmi_pcie_rx_bytes_per_s")) for s in samples if isinstance(s.get("dcgmi_pcie_rx_bytes_per_s"), (int, float))]
    dcgmi_pcie_tx_values = [float(s.get("dcgmi_pcie_tx_bytes_per_s")) for s in samples if isinstance(s.get("dcgmi_pcie_tx_bytes_per_s"), (int, float))]
    tput_values = [float(s.get("gen_tput")) for s in samples if isinstance(s.get("gen_tput"), (int, float))]
    
    pt_total_values = [float(s.get("prompt_tokens_total")) for s in samples if isinstance(s.get("prompt_tokens_total"), (int, float))]
    pt_cached_values = [float(s.get("prompt_tokens_cached")) for s in samples if isinstance(s.get("prompt_tokens_cached"), (int, float))]
    gt_total_values = [float(s.get("generation_tokens_total")) for s in samples if isinstance(s.get("generation_tokens_total"), (int, float))]
    timestamp_values = [float(s.get("timestamp")) for s in samples if isinstance(s.get("timestamp"), (int, float))]
    
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
            
    prom_gross_in_tps = 0.0
    prom_net_in_tps = 0.0
    prom_decode_tps = 0.0
    
    if len(timestamp_values) >= 2:
        dt = timestamp_values[-1] - timestamp_values[0]
        if dt > 0:
            if len(pt_total_values) >= 2:
                delta_pt = pt_total_values[-1] - pt_total_values[0]
                prom_gross_in_tps = max(0.0, delta_pt / dt)
                
            if len(pt_total_values) >= 2 and len(pt_cached_values) >= 2:
                delta_pt = pt_total_values[-1] - pt_total_values[0]
                delta_cached = pt_cached_values[-1] - pt_cached_values[0]
                prom_net_in_tps = max(0.0, (delta_pt - delta_cached) / dt)
                
            if len(gt_total_values) >= 2:
                delta_gt = gt_total_values[-1] - gt_total_values[0]
                prom_decode_tps = max(0.0, delta_gt / dt)
    ttft_sum_metric_name = next((s.get("ttft_sum_metric_name", "") for s in samples if s.get("ttft_sum_metric_name")), "")
    ttft_count_metric_name = next((s.get("ttft_count_metric_name", "") for s in samples if s.get("ttft_count_metric_name")), "")
    queue_sum_metric_name = next((s.get("queue_sum_metric_name", "") for s in samples if s.get("queue_sum_metric_name")), "")
    queue_count_metric_name = next((s.get("queue_count_metric_name", "") for s in samples if s.get("queue_count_metric_name")), "")
    gen_tput_metric_name = next((s.get("gen_tput_metric_name", "") for s in samples if s.get("gen_tput_metric_name")), "")
    kv_cache_metric_name = next((s.get("gpu_cache_metric_name", "") for s in samples if s.get("gpu_cache_metric_name")), "")
    cpu_cache_metric_name = next((s.get("cpu_cache_metric_name", "") for s in samples if s.get("cpu_cache_metric_name")), "")
    result = {
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
        "prom_gross_input_tps": prom_gross_in_tps,
        "prom_net_computed_input_tps": prom_net_in_tps,
        "prom_decode_tps": prom_decode_tps,
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
        "system_cpu_usage_max_pct": max(sys_cpu_values) if sys_cpu_values else 0.0,
        "dcgmi_backend": dcgmi_backend,
        "dcgmi_enabled": dcgmi_enabled,
        "dcgmi_field_ids": dcgmi_field_ids,
        "dcgmi_gpu_selector": dcgmi_gpu_selector,
        "dcgmi_sample_interval_s": dcgmi_sample_interval_s,
        "dcgmi_total_samples": float(dcgmi_total_samples),
        "dcgmi_nonempty_samples": float(dcgmi_nonempty_samples),
        "dcgmi_failed_samples": float(dcgmi_failed_samples),
        "dcgmi_empty_parse_samples": float(dcgmi_empty_parse_samples),
        "dcgmi_last_returncode": float(dcgmi_last_returncode),
        "dcgmi_last_stderr": dcgmi_last_stderr,
        "dcgmi_last_stdout": dcgmi_last_stdout,
        "dcgmi_sm_activity_avg_pct": mean(dcgmi_sm_activity_values) if dcgmi_sm_activity_values else 0.0,
        "dcgmi_sm_activity_p95_pct": percentile(dcgmi_sm_activity_values, 95),
        "dcgmi_sm_activity_max_pct": max(dcgmi_sm_activity_values) if dcgmi_sm_activity_values else 0.0,
        "dcgmi_sm_occupancy_avg_pct": mean(dcgmi_sm_occupancy_values) if dcgmi_sm_occupancy_values else 0.0,
        "dcgmi_sm_occupancy_p95_pct": percentile(dcgmi_sm_occupancy_values, 95),
        "dcgmi_sm_occupancy_max_pct": max(dcgmi_sm_occupancy_values) if dcgmi_sm_occupancy_values else 0.0,
        "dcgmi_tensor_core_activity_avg_pct": mean(dcgmi_tensor_activity_values) if dcgmi_tensor_activity_values else 0.0,
        "dcgmi_tensor_core_activity_p95_pct": percentile(dcgmi_tensor_activity_values, 95),
        "dcgmi_tensor_core_activity_max_pct": max(dcgmi_tensor_activity_values) if dcgmi_tensor_activity_values else 0.0,
        "dcgmi_dram_active_avg_pct": mean(dcgmi_dram_active_values) if dcgmi_dram_active_values else 0.0,
        "dcgmi_dram_active_p95_pct": percentile(dcgmi_dram_active_values, 95),
        "dcgmi_dram_active_max_pct": max(dcgmi_dram_active_values) if dcgmi_dram_active_values else 0.0,
        "dcgmi_memory_utilization_avg_pct": mean(dcgmi_memory_util_values) if dcgmi_memory_util_values else 0.0,
        "dcgmi_memory_utilization_p95_pct": percentile(dcgmi_memory_util_values, 95),
        "dcgmi_memory_utilization_max_pct": max(dcgmi_memory_util_values) if dcgmi_memory_util_values else 0.0,
        "dcgmi_pcie_rx_avg_bytes_per_s": mean(dcgmi_pcie_rx_values) if dcgmi_pcie_rx_values else 0.0,
        "dcgmi_pcie_rx_p95_bytes_per_s": percentile(dcgmi_pcie_rx_values, 95),
        "dcgmi_pcie_rx_max_bytes_per_s": max(dcgmi_pcie_rx_values) if dcgmi_pcie_rx_values else 0.0,
        "dcgmi_pcie_tx_avg_bytes_per_s": mean(dcgmi_pcie_tx_values) if dcgmi_pcie_tx_values else 0.0,
        "dcgmi_pcie_tx_p95_bytes_per_s": percentile(dcgmi_pcie_tx_values, 95),
        "dcgmi_pcie_tx_max_bytes_per_s": max(dcgmi_pcie_tx_values) if dcgmi_pcie_tx_values else 0.0
    }
    if not dcgmi_enabled:
        for key in [k for k in list(result.keys()) if k.startswith("dcgmi_")]:
            result.pop(key, None)
    return result


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
            try:
                await asyncio.wait_for(self.client.aclose(), timeout=5.0)
            except Exception:
                pass

    async def request(self, messages, temperature=0.0, max_tokens=1024):
        import uuid
        import copy
        
        # Inject a nonce to prevent perfect cache hits for user input across rounds
        _messages = copy.deepcopy(messages)
        for m in _messages:
            if m.get("role") == "user":
                m["content"] = f"[{uuid.uuid4().hex[:8]}]\n" + str(m.get("content", ""))
                break
                
        payload = {
            "model": self.model_name,
            "messages": _messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": self.enable_stream,
            "stream_options": {"include_usage": True} if self.enable_stream else None,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
        
        # In non-stream mode, older vLLM might not return `prompt_tokens_details` 
        # unless `stream_options` is passed (or it's simply a version difference).
        # We can also just send `stream_options` anyway or rely on newer vLLM standard.
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
                        if isinstance(event.get("usage"), dict) and event.get("usage"):
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
                pt_details = usage.get("prompt_tokens_details")
                if not isinstance(pt_details, dict):
                    pt_details = {}
                cached_tokens = pt_details.get("cached_tokens", 0)
                used_server_usage = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
                if not isinstance(prompt_tokens, int):
                    prompt_tokens = estimate_prompt_tokens(_messages)
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
                pt_details = usage.get("prompt_tokens_details")
                if not isinstance(pt_details, dict):
                    pt_details = {}
                cached_tokens = pt_details.get("cached_tokens", 0)
                used_server_usage = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
                if not isinstance(prompt_tokens, int):
                    prompt_tokens = estimate_prompt_tokens(_messages)
                if not isinstance(completion_tokens, int):
                    completion_tokens = len(text.split())
            queue_wait_s = max(0.0, (started_at - enqueued_at) if started_at is not None else 0.0)
            result_payload = {
                "ok": True,
                "latency_s": max(0.0, ended_at - enqueued_at),
                "queue_wait_client_s": queue_wait_s,
                "prompt_tokens": int(prompt_tokens),
                "cached_tokens": int(cached_tokens) if cached_tokens else 0,
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
    total_cached_tokens = sum(item.get("cached_tokens", 0) for item in success_items)
    total_completion_tokens = sum(item["completion_tokens"] for item in success_items)
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    net_prompt_tokens = max(0, total_prompt_tokens - total_cached_tokens)
    
    # We will use the prometheus-based TPS if available, falling back to client-side accumulated counts.
    prom_gross = monitor_summary.get("prom_gross_input_tps", 0.0)
    prom_decode = monitor_summary.get("prom_decode_tps", 0.0)
    
    gross_input_tps = float(prom_gross) if prom_gross > 0 else (total_prompt_tokens / total_time_s if total_time_s > 0 else 0.0)
    decode_tps = float(prom_decode) if prom_decode > 0 else (total_completion_tokens / total_time_s if total_time_s > 0 else 0.0)
    
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
        "total_cached_tokens": total_cached_tokens,
        "total_output_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "gross_input_tps": gross_input_tps,
        "decode_tps": decode_tps,
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
        "cpu_cache_metric_name",
        "dcgmi_backend",
        "dcgmi_field_ids",
        "dcgmi_gpu_selector",
        "dcgmi_last_stderr",
        "dcgmi_last_stdout"
    ]
    for key in string_keys:
        values = [item.get(key, "") for item in rounds if item.get(key)]
        if key.startswith("dcgmi_"):
            if values:
                avg[key] = values[0]
        else:
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


async def run_chromosome_round_async(messages_for_round, target_rps, monitor_metrics_url, model_name, chat_completions_url, timeout_s=120, enable_stream=False, phase1_duration_s=40, phase2_duration_s=20, phase3_drain_out_timeout_s=60.0):
    interval_s = 1.0 / float(target_rps)
    phase1_count = int(phase1_duration_s * target_rps)
    phase2_count = int(phase2_duration_s * target_rps)
    
    def _get_msg(idx):
        return messages_for_round[idx % len(messages_for_round)]
        
    monitor_process, monitor_stop_event, monitor_queue = start_monitor_process(monitor_metrics_url)
    main_process_memory_samples = []
    main_memory_stop_event = asyncio.Event()
    main_memory_sampler = asyncio.create_task(
        sample_main_process_memory(main_memory_stop_event, main_process_memory_samples, interval_s=0.2)
    )
    started_at = time.perf_counter()
    started_at_time = time.time()
    
    try:
        async with AsyncVLLMClient(
            chat_completions_url=chat_completions_url,
            model_name=model_name,
            timeout_s=timeout_s,
            max_connections=max(64, target_rps * 20),
            max_keepalive_connections=max(32, target_rps * 10),
            enable_stream=enable_stream
        ) as client:
            
            phase2_tasks = []
            bg_tasks = set()
            phase2_done = asyncio.Event()
            all_completed_times = []
            
            async def _send_once(m, send_at, is_phase2):
                delay = send_at - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                res = await client.request(m)
                if res.get("ok"):
                    all_completed_times.append(time.perf_counter())
                return res

            async def _sender_loop():
                idx = 0
                while not phase2_done.is_set():
                    target_at = started_at + idx * interval_s
                    msg = _get_msg(idx)
                    is_phase2 = phase1_count <= idx < (phase1_count + phase2_count)
                    
                    task = asyncio.create_task(_send_once(msg, target_at, is_phase2))
                    if is_phase2:
                        phase2_tasks.append(task)
                    else:
                        bg_tasks.add(task)
                        task.add_done_callback(bg_tasks.discard)
                        
                    idx += 1
                    sleep_time = (started_at + idx * interval_s) - time.perf_counter()
                    if sleep_time > 0:
                        try:
                            await asyncio.wait_for(phase2_done.wait(), timeout=sleep_time)
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await asyncio.sleep(0)

            sender_task = asyncio.create_task(_sender_loop())
            
            while len(phase2_tasks) < phase2_count:
                if sender_task.done():
                    break
                await asyncio.sleep(0.1)
                
            phase2_dispatch_end_time = time.perf_counter()
            max_drain_out_time = float(phase3_drain_out_timeout_s)
            
            # Wait for all phase 2 tasks to complete, but with a hard timeout from dispatch end
            done, pending = await asyncio.wait(
                phase2_tasks, 
                timeout=max_drain_out_time, 
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any tasks that didn't finish within the drain out limit
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.wait(pending, timeout=5.0)
                
            # Collect results for completed tasks, and mock error results for cancelled ones
            round_results = []
            for t in phase2_tasks:
                if t in done and not t.cancelled() and t.exception() is None:
                    round_results.append(t.result())
                else:
                    round_results.append({
                        "ok": False,
                        "latency_s": max_drain_out_time,
                        "queue_wait_client_s": max_drain_out_time,
                        "prompt_tokens": 0,
                        "cached_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "used_server_usage": False,
                        "error": f"Timeout in drain out phase (>{max_drain_out_time}s)"
                    })
                    
            phase2_end_time = time.perf_counter()
            
            phase2_done.set()
            await sender_task
            
            for t in list(bg_tasks):
                if not t.done():
                    t.cancel()
            if bg_tasks:
                await asyncio.wait(bg_tasks, timeout=5.0)
                    
    finally:
        main_memory_stop_event.set()
        await main_memory_sampler
        
    monitor_payload = stop_monitor_process(monitor_process, monitor_stop_event, monitor_queue)
    
    phase2_start_time = started_at + phase1_count * interval_s
    phase2_window_end = phase2_start_time + phase2_duration_s
    
    monitor_window_start = started_at_time + phase1_count * interval_s
    monitor_window_end = monitor_window_start + phase2_duration_s
    
    monitor_summary = summarize_monitor_payload(
        monitor_payload,
        window_start=monitor_window_start,
        window_end=monitor_window_end
    )
    monitor_summary.update(summarize_process_memory_samples(main_process_memory_samples, "main_process_memory"))
    
    chromosome_duration_time = max(0.001, phase2_end_time - phase2_start_time)
    
    # window_actual_rps: 20s test window period (any successful request within phase2_start_time to phase2_start_time + phase2_duration_s)
    phase2_window_end = phase2_start_time + phase2_duration_s
    window_success_count = sum(1 for t in all_completed_times if phase2_start_time <= t <= phase2_window_end)
    window_actual_rps = window_success_count / float(phase2_duration_s) if phase2_duration_s > 0 else 0.0
    
    # batch_actual_rps: specifically marked phase 2 requests successful count / chromosome_duration_time
    batch_success_count = sum(1 for item in round_results if item.get("ok"))
    batch_actual_rps = batch_success_count / chromosome_duration_time
    
    total_time_s = chromosome_duration_time
    return summarize_round(
        round_results=round_results,
        total_time_s=total_time_s,
        monitor_summary=monitor_summary,
        scenario_fields={
            "target_rps": target_rps, 
            "test_type": "chromosome",
            "window_actual_rps": window_actual_rps,
            "batch_actual_rps": batch_actual_rps,
            "chromosome_duration_time": chromosome_duration_time
        },
        include_stream_metrics=enable_stream
    )


async def run_burst_round_async(
    messages_for_round, 
    concurrency, 
    monitor_metrics_url, 
    model_name, 
    chat_completions_url, 
    timeout_s=120, 
    enable_stream=False,
    steady_background_rps=0.0,
    steady_background_duration_s=20,
    base_messages=None
):
    monitor_process, monitor_stop_event, monitor_queue = start_monitor_process(monitor_metrics_url)
    main_process_memory_samples = []
    main_memory_stop_event = asyncio.Event()
    main_memory_sampler = asyncio.create_task(
        sample_main_process_memory(main_memory_stop_event, main_process_memory_samples, interval_s=0.2)
    )
    started_at = time.perf_counter()
    burst_start_time = None
    burst_end_time = None
    burst_start_perf = None
    burst_end_perf = None
    
    try:
        async with AsyncVLLMClient(
            chat_completions_url=chat_completions_url,
            model_name=model_name,
            timeout_s=timeout_s,
            max_connections=max(64, concurrency * 5 + int(steady_background_rps) * 6),
            max_keepalive_connections=max(32, concurrency * 2 + int(steady_background_rps) * 3),
            enable_stream=enable_stream
        ) as client:
            bg_tasks = set()
            bg_stop_event = asyncio.Event()
            bg_sender_task = None
            
            async def _bg_sender_loop():
                if not base_messages or steady_background_rps <= 0:
                    return
                interval_s = 1.0 / steady_background_rps
                idx = 0
                loop_start = time.perf_counter()
                while not bg_stop_event.is_set():
                    target_at = loop_start + idx * interval_s
                    msg = base_messages[idx % len(base_messages)]
                    
                    async def _send_once_bg(m, send_at):
                        delay = send_at - time.perf_counter()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        return await client.request(m)
                        
                    task = asyncio.create_task(_send_once_bg(msg, target_at))
                    bg_tasks.add(task)
                    task.add_done_callback(bg_tasks.discard)
                    
                    idx += 1
                    sleep_time = (loop_start + idx * interval_s) - time.perf_counter()
                    if sleep_time > 0:
                        try:
                            await asyncio.wait_for(bg_stop_event.wait(), timeout=sleep_time)
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await asyncio.sleep(0)

            if steady_background_rps > 0 and base_messages:
                bg_sender_task = asyncio.create_task(_bg_sender_loop())
                await asyncio.sleep(steady_background_duration_s)
                
            burst_start_time = time.time()
            burst_start_perf = time.perf_counter()

            semaphore = asyncio.Semaphore(concurrency)
            async def _send_once(messages):
                async with semaphore:
                    return await client.request(messages)
            tasks = [asyncio.create_task(_send_once(messages)) for messages in messages_for_round]
            round_results = await asyncio.gather(*tasks)
            
            burst_end_time = time.time()
            burst_end_perf = time.perf_counter()

            if bg_sender_task:
                bg_stop_event.set()
                await bg_sender_task
                
            for t in list(bg_tasks):
                if not t.done():
                    t.cancel()
            if bg_tasks:
                await asyncio.wait(bg_tasks, timeout=5.0)
                
    finally:
        main_memory_stop_event.set()
        await main_memory_sampler
        
    ended_at = time.perf_counter()
    monitor_payload = stop_monitor_process(monitor_process, monitor_stop_event, monitor_queue)
    
    if steady_background_rps > 0 and burst_start_time and burst_end_time:
        monitor_summary = summarize_monitor_payload(
            monitor_payload,
            window_start=burst_start_time,
            window_end=burst_end_time
        )
        total_time_s = burst_end_perf - burst_start_perf
    else:
        monitor_summary = summarize_monitor_payload(monitor_payload)
        total_time_s = ended_at - started_at
        
    monitor_summary.update(summarize_process_memory_samples(main_process_memory_samples, "main_process_memory"))
    
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
