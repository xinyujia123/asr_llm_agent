import torch
import time
import sys
from pathlib import Path
from qwen_asr import Qwen3ASRModel

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)
from sak.prompts import *
from sak.utils import *

model = Qwen3ASRModel.from_pretrained(
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

raw_path = "/workspace/dataset/asr_llm/nurse_audio/18.amr"
full_path, trimmed_path = convert_audio_double_outputs(raw_path)

start_time = time.time()
results = model.transcribe(
    audio=trimmed_path,
    language="Chinese",
)
run_time = time.time() - start_time

start_time = time.time()
results_2 = model.transcribe(
    audio=full_path,
    language="Chinese",
)
run_time_2 = time.time() - start_time

print(results[0].text)
print(results_2[0].text)
print(run_time)
print(run_time_2)