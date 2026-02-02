from funasr import AutoModel
from sak.utils import convert_audio_to_wav_sync
import librosa

ASR_MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"


# 1. 直接指定 PyTorch 模型名，但通过 is_onnx=True 切换引擎
# 它会自动去寻找该模型对应的 ONNX 仓库或在本地寻找 onnx 文件

model = AutoModel(
        model=ASR_MODEL_ID,
        model_revision="v2.0.4", 
        vad_model=VAD_MODEL_ID,
        vad_model_revision="v2.0.4", 
        vad_kwargs={
                    "max_end_silence_time": 3000,
                    "speech_to_sil_time_thres": 800,    # 1. 延长单段最长时长 (ms)
                    "lookahead_time_end_point": 1000,       # 2. 允许更长的末尾静音 (ms)
                    "speech_noise_thres": 0.7,
        },
        device="cuda",
        disable_update=True,      # 生产环境务必锁定
        trust_remote_code= False,
        use_itn=False # 关闭逆文本标准化（比如把"一百二十"转为"120"），可以减轻LLM负担
        )

if False:
    vad_model = AutoModel(
            model=VAD_MODEL_ID,
            model_revision="v2.0.4", 
            max_end_silence_time=3000,
            speech_to_sil_time_thres=800,    # 1. 延长单段最长时长 (ms)
            lookahead_time_end_point=1000,       # 2. 允许更长的末尾静音 (ms)
            speech_noise_thres=0.7,
            device="cuda",
            disable_update=True,      # 生产环境务必锁定
            trust_remote_code= False,
            )

# 2. 推理逻辑与 PyTorch 完全一致
for i in range(0,20):
    raw_path = f"/workspace/audio_llm_agent/test/test_audio/{i+1}.amr"
    audio, _ = librosa.load(raw_path, sr=16000)
    wav_path = convert_audio_to_wav_sync(raw_path)
    res_ffm = model.generate(input=wav_path)
    res_lib = model.generate(input=audio)
    print(f"res_ffm: {res_ffm[0]['text']}")
    print(f"res_lib: {res_lib[0]['text']}")
