import subprocess
import re
import json
import os
import glob
import time

def convert_audio_to_wav_sync(source_path: str, target_dir) -> str:
    """
    同步执行的 FFmpeg 转换函数，将被放入线程池运行
    """
    output_path = os.path.join(target_dir, os.path.basename(source_path).replace('.amr', '.wav'))
    command = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        #"-af", "volume=30dB",
        output_path
    ]
    try:
        subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr.decode()}")
        raise RuntimeError("音频转码失败，文件可能已损坏")
    except FileNotFoundError:
        raise RuntimeError("服务器未安装 FFmpeg")

for f in glob.glob('menu_audio/*.amr'):
    start_time = time.time()
    convert_audio_to_wav_sync(f, 'menu_audio_wav')
    end_time = time.time()
    print(f"转换 {f} 耗时: {end_time - start_time:.2f} 秒")