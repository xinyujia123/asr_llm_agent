from funasr import AutoModel
import librosa
import subprocess
import logging
import transformers

# 设置 transformers 的日志级别为 ERROR，彻底屏蔽生成过程中的警告
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)




ASR_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
def convert_audio_to_wav_sync(source_path: str) -> str:
    """
    同步执行的 FFmpeg 转换函数，将被放入线程池运行
    """
    output_path = source_path + "_processed.wav"
    command = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
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

# 1. 直接指定 PyTorch 模型名，但通过 is_onnx=True 切换引擎
# 它会自动去寻找该模型对应的 ONNX 仓库或在本地寻找 onnx 文件
asr_model = AutoModel(
        model=ASR_MODEL_ID,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
        trust_remote_code=True,
        remote_code="./nano_model.py",
        disable_update=True
        )

# 2. 推理逻辑与 PyTorch 完全一致
for i in range(1,3):
    raw_path = f"/home/jiaxinyu/projects/audio_llm_agent/test/test_audio/{i+1}.amr"
    lib_audio, _ = librosa.load(raw_path, sr=16000)
    wav_path = convert_audio_to_wav_sync(raw_path)
    res_ffm = asr_model.generate(
        input=[wav_path],
        batch_size=1,
        hotword="脉搏 呼吸 心率 体温 血压 体重 高压 低压 度 次 分",
        language="zh",
        itn=True,
        cache={}
    )
    res_lib = asr_model.generate(
        input=lib_audio,
        batch_size=1,
        hotword="脉搏 呼吸 心率 体温 血压 体重 高压 低压 度 次 分",
        language="zh",
        itn=True,
        cache={}
    )
    print(f"res_ffm: {res_ffm[0]['text']}")
    print(f"res_lib: {res_lib[0]['text']}")