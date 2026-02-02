from funasr import AutoModel
import librosa

# 1. ASR: 语义增强版 Paraformer (核心)
ASR_MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# 2. VAD: FSMN (标准)
# 作用：切分长语音，过滤静音
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
# 3. PUNC: CT-Transformer (标准)
# 作用：加标点，辅助 LLM 理解语义断句
PUNC_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

asr_model = AutoModel(
        model=ASR_MODEL_ID,
        model_revision="v2.0.4", 
        vad_model=VAD_MODEL_ID,
        vad_model_revision="v2.0.4", 
        punc_model=PUNC_MODEL_ID,
        punc_model_revision="v2.0.4", 
        device="cuda",
        disable_update=True,      # 生产环境务必锁定
        trust_remote_code= False,
        use_itn=False # 关闭逆文本标准化（比如把"一百二十"转为"120"），可以减轻LLM负担
        )

raw_path = f"/workspace/audio_llm_agent/test/test_audio/1.amr"
audio, _ = librosa.load(raw_path, sr=16000)
res_lib = asr_model.generate(input=audio)
print(f"res_lib: {res_lib}")