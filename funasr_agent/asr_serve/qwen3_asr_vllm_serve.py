import os
import gc
import ray
import time
import torch
import asyncio
import tempfile
import shutil
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from qwen_asr import Qwen3ASRModel
from pathlib import Path

# 假设你的 sak 工具包在路径中
import sys
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)
from sak.utils import convert_audio_double_outputs # 保留你的音频转换工具

# --- 配置 ---
MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"

class ASRServer:
    def __init__(self):
        self.model = None
        self.lock = asyncio.Lock() # 保证 GPU 推理的原子性

    def load(self):
        print(f"正在初始化 ASR 模型: {MODEL_PATH}...")
        self.model = Qwen3ASRModel.LLM(
            model=MODEL_PATH,
            gpu_memory_utilization=0.8,
            max_model_len=8192,
            max_inference_batch_size=8,
            max_new_tokens=512,
        )
        print("模型加载完成。")

    def transcribe(self, audio_path: str, hotwords: str):
        # 调用你测试脚本中的推理逻辑
        results = self.model.transcribe(audio=audio_path, context=hotwords, language="Chinese")
        return results[0].text if results else ""

    def transcribe_batch(self, audio_paths: List[str], hotwords: str):
        # 批量推理逻辑
        # Qwen3ASRModel.transcribe 支持 audio 为 list
        results = self.model.transcribe(audio=audio_paths, context=hotwords, language="Chinese")
        return [res.text if res else "" for res in results]

    def cleanup(self):
        print("执行深度显存清理...")
        if self.model:
            # 尝试调用内部关闭方法
            for method in ["close", "shutdown"]:
                if hasattr(self.model, method):
                    try:
                        getattr(self.model, method)()
                    except: pass
            del self.model
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if ray.is_initialized():
            ray.shutdown()
        print("清理完成。")

asr_engine = ASRServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载
    asr_engine.load()
    yield
    # 关闭时清理
    asr_engine.cleanup()

app = FastAPI(title="Qwen3-ASR Dedicated Service", lifespan=lifespan)

class TranscribeRequest(BaseModel):
    file_path: str
    hotwords: str = ""

class BatchTranscribeRequest(BaseModel):
    file_paths: List[str]
    hotwords: str = ""

@app.post("/transcribe_by_path")
async def api_transcribe_path(req: TranscribeRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="文件路径无效")
    
    try:
        start_time = time.perf_counter()
        async with asr_engine.lock:
            # 模型直接读取 Agent 准备好的文件地址
            text = await asyncio.to_thread(asr_engine.transcribe, req.file_path, req.hotwords)
        duration = time.perf_counter() - start_time
        
        return {"text": text, "inference_time": duration}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe_batch_paths")
async def api_transcribe_batch_paths(req: BatchTranscribeRequest):
    # 简单的路径检查
    for p in req.file_paths:
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"文件路径无效: {p}")
    
    try:
        start_time = time.perf_counter()
        async with asr_engine.lock:
            texts = await asyncio.to_thread(asr_engine.transcribe_batch, req.file_paths, req.hotwords)
        duration = time.perf_counter() - start_time
        
        return {"texts": texts, "inference_time": duration}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 建议固定端口，方便 Agent 调用
    uvicorn.run(app, host="0.0.0.0", port=7999)