import requests
import json
import time
import concurrent.futures
from pathlib import Path

# 配置信息
API_URL = "http://localhost:8000/api/agent"
AUDIO_DIR = "test/dataset/nurse_audio/"
CONCURRENCY = 15  # 并发数

def send_request(file_path):
    """单个请求的执行逻辑"""
    file_name = Path(file_path).name
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            start_time = time.time()
            
            response = requests.post(API_URL, files=files, timeout=60)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            if response.status_code == 200:
                return True, elapsed, file_name
            else:
                return False, elapsed, f"Error {response.status_code}: {file_name}"
    except Exception as e:
        return False, 0, str(e)

def run_stress_test():
    # 准备 20 个任务的文件路径
    all_files = list(Path(AUDIO_DIR).glob("*.amr"))
    # 按文件名数字大小排序，确保顺序固定为 1.amr, 2.amr ...
    all_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf'))

    if not all_files:
        print("未找到 .amr 测试文件！")
        return
    
    # 固定每次读取的文件是 1.amr - n.amr (循环取值)
    test_tasks = [all_files[i % len(all_files)] for i in range(CONCURRENCY)]
    
    print(f"🚀 开始并发压力测试: 并发数 = {CONCURRENCY}")
    print("-" * 20)
    
    total_start_time = time.time()
    results = []

    # 使用线程池模拟并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(send_request, f): f for f in test_tasks}
        
        for future in concurrent.futures.as_completed(future_to_file):
            success, elapsed, info = future.result()
            results.append((success, elapsed))
            status = "✅ 成功" if success else "❌ 失败"
            print(f"[{status}] 耗时: {elapsed:.2f}s | 文件/信息: {info}")

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    
    # 统计结果
    success_counts = sum(1 for r in results if r[0])
    avg_latency = sum(r[1] for r in results) / len(results)
    max_latency = max(r[1] for r in results)
    throughput = CONCURRENCY / total_elapsed

    print("-" * 50)
    print(f"📊 测试总结:")
    print(f"成功率: {success_counts}/{CONCURRENCY}")
    print(f"总花费时间 (Wall-clock time): {total_elapsed:.2f} 秒")
    print(f"平均单次响应耗时 (Avg Latency): {avg_latency:.2f} 秒")
    print(f"最长单次响应耗时 (Max Latency): {max_latency:.2f} 秒")
    print(f"吞吐量 (Throughput): {throughput:.2f} 请求/秒")
    print("-" * 50)

if __name__ == "__main__":
    run_stress_test()