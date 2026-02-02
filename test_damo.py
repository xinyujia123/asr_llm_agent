from funasr_onnx import Paraformer
from pathlib import Path

model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = Paraformer(model_dir, batch_size=1, quantize=True)

wav_path = ['{}/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav'.format(Path.home())]

result = model(wav_path)
print(result)