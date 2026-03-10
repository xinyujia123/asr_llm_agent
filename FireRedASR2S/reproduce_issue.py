import soundfile as sf
import numpy as np
import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到 sys.path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
except ImportError as e:
    print(f"Warning: 无法导入 fireredasr2s.fireredvad: {e}")
    FireRedVad = None

def reproduce_error():
    if FireRedVad is None: return

    # 1. 生成极短的音频
    # 比如 100 个采样点 (在 16kHz 下只有 6ms)
    sample_rate = 16000
    num_samples = 100 
    audio_data = np.random.uniform(-0.1, 0.1, num_samples).astype(np.float32)
    
    test_file = "short_audio.wav"
    sf.write(test_file, audio_data, sample_rate)
    print(f"生成了 {num_samples} 个采样点的音频文件: {test_file}")

    # 2. 加载模型
    MODEL_DIR = os.path.join(current_dir, "pretrained_models/FireRedVAD/VAD")
    config = FireRedVadConfig(use_gpu=False)
    vad = FireRedVad.from_pretrained(MODEL_DIR, config)

    # 3. 尝试推理
    try:
        print("尝试运行 VAD detect...")
        vad.detect(test_file)
        print("运行成功 (未报错)")
    except Exception as e:
        print(f"捕获到异常: {e}")

    os.remove(test_file)

if __name__ == "__main__":
    reproduce_error()
