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

TEST_FILE = "/workspace/audio_llm_agent/dataset/asr_llm/menu_audio_wav_30db/4.wav"

try:
    from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
except ImportError:
    print("Warning: 无法导入 fireredasr2s.fireredvad，请确保环境正确。")
    FireRedVad = None

def get_gpu_memory_usage():
    if not torch.cuda.is_available():
        return 0, 0
    # 获取当前 GPU 显存使用情况 (MB)
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return allocated, reserved

def benchmark_audio_vad_gpu():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("错误: 未检测到 GPU，无法进行 GPU 测试。")
        return

    # 1. 生成 10 分钟的模拟音频 (16kHz, 单声道)
    duration_sec = 10 * 60
    sample_rate = 16000
    
    test_file = TEST_FILE

    # 记录初始显存
    torch.cuda.empty_cache()
    init_allocated, init_reserved = get_gpu_memory_usage()
    print(f"\n[初始状态] 显存已分配: {init_allocated:.2f} MB, 已预留: {init_reserved:.2f} MB")

    # 2. 加载 VAD 模型 (GPU)
    print("\n--- 加载 VAD 模型 (GPU) ---")
    MODEL_DIR = os.path.join(current_dir, "pretrained_models/FireRedVAD/VAD")
    
    load_start = time.time()
    config = FireRedVadConfig(
        use_gpu=True, # 强制开启 GPU
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
    
    load_allocated, load_reserved = get_gpu_memory_usage()
    print(f"VAD 模型加载耗时: {load_time:.4f} 秒")
    print(f"[加载后] 显存已分配: {load_allocated:.2f} MB (增加: {load_allocated - init_allocated:.2f} MB)")
    print(f"[加载后] 显存已预留: {load_reserved:.2f} MB")

    # 3. 测试 VAD 推理 (detect)
    print("\n--- 测试 VAD 推理 (detect) ---")
    detect_start = time.time()
    
    # 记录推理过程中的显存峰值需要用 torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        result, probs = vad.detect(test_file)
    except torch.cuda.OutOfMemoryError:
        print("错误: 显存不足 (OOM)！")
        # 尝试清理
        torch.cuda.empty_cache()
        return

    detect_time = time.time() - detect_start
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    timestamps = result.get("timestamps", [])
    print(f"VAD 推理耗时: {detect_time:.4f} 秒")
    print(f"检测到的语音片段数: {len(timestamps)}")
    print(f"[推理中] 峰值显存占用: {peak_memory:.2f} MB (相对于初始增加了 {peak_memory - init_allocated:.2f} MB)")

    # 清理
    del vad
    torch.cuda.empty_cache()
    os.remove(test_file)
    
    print(f"\n[总结]")
    print(f"1. 模型加载显存占用: ~{load_allocated - init_allocated:.2f} MB")
    print(f"2. 推理过程峰值显存: ~{peak_memory:.2f} MB")
    print(f"3. 10分钟音频推理耗时: {detect_time:.4f} 秒")

if __name__ == "__main__":
    benchmark_audio_vad_gpu()
