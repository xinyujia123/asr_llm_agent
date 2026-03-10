import os
import subprocess



def convert_audio_double_outputs(source_path: str, trimmed_time: int = 6):
    """
    一次性生成两个文件：全量转换文件 和 前6秒截取文件
    返回 (full_path, trimmed_path)
    """
    base_name = os.path.splitext(source_path)[0]
    full_path = f"{base_name}_full.wav"
    trimmed_path = f"{base_name}_trimmed.wav"

    # 构建命令
    # FFmpeg 允许在同一个命令后面指定多个输出及其对应的参数
    command = [
        "ffmpeg", "-y",
        "-i", source_path,
        
        # --- 第一个输出：前4秒 ---
        "-t", str(trimmed_time),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-af", "volume=30dB",
        trimmed_path,
        
        # --- 第二个输出：全量文件 ---
        # 注意：这里不需要再写一次 -i，FFmpeg 会自动复用输入流
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-af", "volume=30dB",
        full_path
    ]

    try:
        subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        return full_path, trimmed_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr.decode()}")
        raise RuntimeError("音频处理失败")
    except FileNotFoundError:
        raise RuntimeError("服务器未安装 FFmpeg")