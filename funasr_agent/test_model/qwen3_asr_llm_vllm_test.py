import torch
import time
import sys
import gc
import ray
from pathlib import Path
from qwen_asr import Qwen3ASRModel

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)
from sak.prompts import *
from sak.utils import *

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-0.6B",
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        max_inference_batch_size=2, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=512, # Maximum number of tokens to generate. Set a larger value for long audio input.
    )

    try:
        raw_path_list = [1,2,3,4,5,6,7,8,9]
        for index in raw_path_list:
            raw_path = f"/workspace/audio_llm_agent/dataset/asr_llm/nurse_audio/{str(index)}.amr"
            full_path, trimmed_path = convert_audio_double_outputs(raw_path)
            start_time = time.time()
            results = model.transcribe(audio=trimmed_path, language="Chinese")
            run_time = time.time() - start_time
            results_2 = model.transcribe(audio=full_path, language="Chinese")
            run_time_2 = time.time() - start_time
            print(results[0].text)
            print(run_time)
            print(results_2[0].text)
            print(run_time_2)
    finally:
        # 1. 显式删除模型对象并触发垃圾回收
        if 'model' in locals():
            # 尝试调用模型自带的关闭方法
            for method in ["close", "shutdown"]:
                if hasattr(model, method):
                    try:
                        getattr(model, method)()
                    except Exception:
                        pass
            # 删除引用并强制 GC，这会释放 GPU 显存
            del model
            gc.collect()
            torch.cuda.empty_cache()
        # 2. 处理分布式进程组
        # 必须先确保模型引擎已经关闭，再销毁进程组
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                print(f"Cleanup distributed error: {e}")
        # 3. 关闭 Ray (Qwen3 ASR 通常底层使用 vLLM/Ray)
        if ray.is_initialized():
            ray.shutdown()

        # 4. 给系统一点时间清理僵尸进程
        time.sleep(2)
        print("Cleanup finished. Exiting...")
