import os
import time
import asyncio
import httpx
import subprocess
from pathlib import Path
import shutil
import statistics

# Configuration
DATASET_DIR = Path("/workspace/audio_llm_agent/dataset/asr_llm/long_audio")
CURRENT_DIR = Path(__file__).resolve().parent
REPORT_PATH = CURRENT_DIR / "concurrency_benchmark_report.txt"
TEMP_20S_DIR = DATASET_DIR / "temp_20s"

# ASR Service URL
ASR_SERVICE_URL = "http://127.0.0.1:7999/transcribe_by_path"

# Hotwords (copied from original script to simulate realistic load)
HOTWORDS = "你是一个护理语音转录助手，这是需要注意的热词：诺和锐 塞来昔布 地塞米松 昂丹司琼 螺内酯 呋塞米"

# Test Scenarios
SCENARIOS = [
    {"duration": "5s", "concurrency_levels": [1, 5, 10, 30]},
    {"duration": "20s", "concurrency_levels": [1, 3, 5, 10]},
    {"duration": "1min", "concurrency_levels": [1, 3, 5]},
    {"duration": "5min", "concurrency_levels": [1, 3, 5]},
]

async def generate_20s_files():
    """Generates 20s audio files from 1min files if they don't exist."""
    if not TEMP_20S_DIR.exists():
        TEMP_20S_DIR.mkdir(parents=True)
    
    source_dir = DATASET_DIR / "1min"
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} not found.")
        return []

    generated_files = []
    source_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.wav')])
    
    # We need at most 10 files for 20s concurrency test
    files_needed = 10
    
    for i, filename in enumerate(source_files):
        if i >= files_needed:
            break
            
        source_path = source_dir / filename
        target_path = TEMP_20S_DIR / filename
        
        if not target_path.exists():
            # Cut first 20 seconds
            cmd = [
                "ffmpeg", "-y", "-i", str(source_path),
                "-t", "20", "-c", "copy", str(target_path)
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate 20s file {filename}: {e}")
                continue
        
        generated_files.append(target_path)
        
    print(f"Prepared {len(generated_files)} files for 20s scenario.")
    return generated_files

def get_audio_files(duration_label):
    if duration_label == "5s":
        path = DATASET_DIR / "5s"
    elif duration_label == "1min":
        path = DATASET_DIR / "1min"
    elif duration_label == "5min":
        path = DATASET_DIR / "5min"
    elif duration_label == "20s":
        # Use the temp directory
        path = TEMP_20S_DIR
    else:
        return []

    if not path.exists():
        return []
        
    files = sorted([path / f for f in os.listdir(path) if f.endswith('.wav')])
    return files

async def get_gpu_memory():
    """Returns the current GPU memory usage in MiB."""
    try:
        # Run nvidia-smi to get memory usage
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Parse output, assuming single GPU or taking the first one
            output = stdout.decode().strip()
            if output:
                # If multiple GPUs, it returns multiple lines. We sum them up or take max?
                # Usually for single model deployment, we care about total or max.
                # Let's return the max usage across all visible GPUs.
                memories = [int(x) for x in output.split('\n') if x.strip()]
                return max(memories) if memories else 0
        return 0
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return 0

async def monitor_gpu(stop_event, stats):
    """Monitors GPU memory usage until stop_event is set."""
    while not stop_event.is_set():
        mem = await get_gpu_memory()
        stats.append(mem)
        await asyncio.sleep(0.5) # Check every 0.5 seconds

async def send_request(client, file_path):
    start_time = time.time()
    try:
        response = await client.post(
            ASR_SERVICE_URL,
            json={"file_path": str(file_path), "hotwords": HOTWORDS},
            timeout=1200.0  # Increased timeout for 5min audio
        )
        response.raise_for_status()
        end_time = time.time()
        return {
            "success": True,
            "latency": end_time - start_time,
            "status": response.status_code
        }
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "latency": end_time - start_time,
            "error": str(e)
        }

async def run_scenario(duration_label, concurrency):
    print(f"\n--- Running Scenario: Duration {duration_label}, Concurrency {concurrency} ---")
    
    available_files = get_audio_files(duration_label)
    if not available_files:
        print(f"No audio files found for {duration_label}")
        return None

    # Select files for the test. Reuse if necessary.
    test_files = []
    for i in range(concurrency):
        test_files.append(available_files[i % len(available_files)])
        
    results = []
    gpu_stats = []
    stop_gpu_monitor = asyncio.Event()
    
    # Start GPU monitoring
    monitor_task = asyncio.create_task(monitor_gpu(stop_gpu_monitor, gpu_stats))
    
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, f) for f in test_files]
        
        scenario_start = time.time()
        results = await asyncio.gather(*tasks)
        scenario_end = time.time()
        
    # Stop GPU monitoring
    stop_gpu_monitor.set()
    await monitor_task
    
    total_time = scenario_end - scenario_start
    
    success_count = sum(1 for r in results if r["success"])
    latencies = [r["latency"] for r in results if r["success"]]
    
    avg_latency = statistics.mean(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    p95_latency = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max_latency
    
    max_gpu = max(gpu_stats) if gpu_stats else 0
    avg_gpu = statistics.mean(gpu_stats) if gpu_stats else 0
    
    report_lines = [
        f"Scenario: {duration_label} audio, {concurrency} concurrent requests",
        f"Total Wall Time: {total_time:.4f} s",
        f"Successful Requests: {success_count}/{concurrency}",
        f"Average Latency: {avg_latency:.4f} s",
        f"Max Latency: {max_latency:.4f} s",
        f"P95 Latency: {p95_latency:.4f} s",
        f"Max GPU Memory: {max_gpu} MiB",
        f"Avg GPU Memory: {avg_gpu:.1f} MiB"
    ]
    
    if success_count < concurrency:
        errors = [r["error"] for r in results if not r["success"]]
        report_lines.append(f"Errors: {errors}")

    print("\n".join(report_lines))
    return "\n".join(report_lines)

async def main():
    print("Starting Concurrency Benchmark...")
    
    # Setup 20s files
    await generate_20s_files()
    
    full_report = []
    full_report.append(f"Benchmark Report - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    full_report.append("=" * 50)
    
    for scenario in SCENARIOS:
        duration = scenario["duration"]
        for concurrency in scenario["concurrency_levels"]:
            result_text = await run_scenario(duration, concurrency)
            if result_text:
                full_report.append(result_text)
                full_report.append("-" * 30)
            
            # Cool down between tests to let server recover/stabilize
            await asyncio.sleep(2)

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_report))
    
    print(f"\nBenchmark completed. Report saved to {REPORT_PATH}")

    # Cleanup temp files (optional, maybe keep for inspection)
    # shutil.rmtree(TEMP_20S_DIR) 

if __name__ == "__main__":
    asyncio.run(main())
