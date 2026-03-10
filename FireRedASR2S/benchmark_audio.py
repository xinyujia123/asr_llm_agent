import time
import soundfile as sf
import numpy as np
import os
import tempfile
import sys
import torch
from pathlib import Path

# 添加项目根目录到 sys.path，以便导入 fireredasr2s
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
except ImportError:
    print("Warning: 无法导入 fireredasr2s.fireredvad，请确保环境正确。")
    FireRedVad = None

def benchmark_audio_vad():
    # 1. 生成 10 分钟的模拟音频 (16kHz, 单声道)
    # 为了让 VAD 检测到语音，我们需要模拟一些能量变化
    # 这里我们交替生成 "静音" 和 "噪声(模拟语音)"
    duration_sec = 10 * 60
    sample_rate = 16000
    
    print(f"正在生成 10 分钟的测试音频 (混合静音和噪声)...")
    
    # 每 10 秒切换一次状态 (5s 静音, 5s 噪声)
    segment_duration = 5
    num_segments = duration_sec // segment_duration
    
    audio_parts = []
    for i in range(num_segments):
        if i % 2 == 0:
            # 静音 (极低幅度的噪声，避免完全为0被优化)
            part = np.random.uniform(-0.001, 0.001, segment_duration * sample_rate).astype(np.float32)
        else:
            # 语音 (较大幅度的噪声)
            part = np.random.uniform(-0.5, 0.5, segment_duration * sample_rate).astype(np.float32)
        audio_parts.append(part)
        
    audio_data = np.concatenate(audio_parts)
    
    test_file = "benchmark_10min_vad.wav"
    sf.write(test_file, audio_data, sample_rate)
    file_size_mb = os.path.getsize(test_file) / (1024 * 1024)
    print(f"测试文件已生成: {test_file}, 大小: {file_size_mb:.2f} MB")

    # 2. 加载 VAD 模型
    print("\n--- 加载 VAD 模型 ---")
    if FireRedVad is None:
        print("VAD 模块未导入，跳过 VAD 测试。")
        return

    MODEL_DIR = os.path.join(current_dir, "pretrained_models/FireRedVAD/VAD")
    if not os.path.exists(MODEL_DIR):
        print(f"错误: 模型目录不存在 {MODEL_DIR}")
        return

    load_start = time.time()
    config = FireRedVadConfig(
        use_gpu=False, # 强制使用 CPU，避免显存冲突
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=40,
        max_speech_frame=3000,
        min_silence_frame=30,
        merge_silence_frame=400,
        extend_speech_frame=15,
        chunk_max_frame=30000
    )
    vad = FireRedVad.from_pretrained(MODEL_DIR, config)
    load_time = time.time() - load_start
    print(f"VAD 模型加载耗时: {load_time:.4f} 秒 (GPU: {torch.cuda.is_available()})")

    # 3. 测试 VAD 推理 (detect)
    print("\n--- 测试 VAD 推理 (detect) ---")
    detect_start = time.time()
    # 注意：FireRedVad.detect 内部会读取音频文件
    result, probs = vad.detect(test_file)
    detect_time = time.time() - detect_start
    
    timestamps = result.get("timestamps", [])
    print(f"VAD 推理耗时: {detect_time:.4f} 秒")
    print(f"检测到的语音片段数: {len(timestamps)}")
    print(f"音频总时长: {result.get('dur', 0.0)} 秒")

    # 4. 测试切分与写盘 (强制模拟最坏情况：假设检测到了 60 个片段)
    print("\n--- 测试切分与写盘 (模拟 60 个片段) ---")
    
    # 构造模拟的时间戳: 每 10 秒一段，每段长 5 秒
    mock_timestamps = []
    for i in range(60):
        start = i * 10.0
        end = start + 5.0
        mock_timestamps.append([start, end])
        
    split_start = time.time()
    
    # 需要重新读取音频数据进行切分 (模拟真实流程)
    full_audio, sr = sf.read(test_file)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        count = 0
        for start, end in mock_timestamps:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # 简单的边界保护
            if end_sample > len(full_audio): end_sample = len(full_audio)
            if start_sample < 0: start_sample = 0
            
            if end_sample - start_sample < 160: continue
            
            segment_audio = full_audio[start_sample:end_sample]
            segment_path = os.path.join(temp_dir, f"seg_{count}.wav")
            sf.write(segment_path, segment_audio, sr)
            count += 1
            
    split_time = time.time() - split_start
    print(f"切分并写入 {count} 个文件耗时: {split_time:.4f} 秒")
    
    print(f"\n[总结] VAD 处理 10分钟音频总额外开销 (推理 + 模拟切分): {detect_time + split_time:.4f} 秒")

    # 清理
    os.remove(test_file)

if __name__ == "__main__":
    benchmark_audio_vad()
