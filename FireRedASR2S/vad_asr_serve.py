import os
import sys
import time
import torch
import httpx
import shutil
import asyncio
import logging
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# 添加项目根目录到 sys.path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

# --- 配置 ---
ASR_SERVICE_URL = "http://127.0.0.1:7999/transcribe_by_path"
ASR_BATCH_SERVICE_URL = "http://127.0.0.1:7999/transcribe_batch_paths"
MODEL_DIR = os.path.join(current_dir, "pretrained_models/FireRedVAD/VAD")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VAD_Service")

class VADServer:
    def __init__(self):
        self.vad = None
        self.http_client = None

    def load(self):
        logger.info("正在初始化 VAD 模型...")
        # 检查模型路径
        if not os.path.exists(MODEL_DIR):
            logger.error(f"模型目录不存在: {MODEL_DIR}")
            # 这里可以添加自动下载逻辑，或者假设已经存在
            pass

        config = FireRedVadConfig(
            use_gpu=torch.cuda.is_available(),
            smooth_window_size=5,
            speech_threshold=0.4,
            min_speech_frame=40,
            max_speech_frame=3000,
            min_silence_frame=30,
            merge_silence_frame=400,
            extend_speech_frame=15,
            chunk_max_frame=30000
        )
        
        try:
            self.vad = FireRedVad.from_pretrained(MODEL_DIR, config)
            logger.info("VAD 模型加载完成。")
        except Exception as e:
            logger.error(f"加载 VAD 模型失败: {e}")
            raise e

        self.http_client = httpx.AsyncClient(timeout=60.0)

    async def cleanup(self):
        logger.info("清理资源...")
        if self.http_client:
            await self.http_client.aclose()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("清理完成。")

    def detect_speech(self, audio_path: str):
        """
        执行 VAD 检测
        返回: (result, probs)
        result 包含 timestamps (单位秒)
        """
        if not self.vad:
            raise RuntimeError("VAD 模型未加载")
        
        # 确保是 wav 格式且采样率正确，这里直接调用 detect
        # FireRedVad 内部使用 soundfile 读取
        try:
            # 增加音频长度预检查，避免极短音频导致模型报错
            # 16kHz 下，1600 samples = 0.1s
            # 如果音频小于 0.1s，直接视为无语音，避免 crash
            info = sf.info(audio_path)
            if info.frames < 1600: 
                logger.warning(f"音频过短 ({info.frames} samples, <0.1s)，跳过 VAD 检测")
                return {"timestamps": [], "dur": info.duration}

            result, probs = self.vad.detect(audio_path)
            return result
        except Exception as e:
            logger.error(f"VAD 检测失败: {e}")
            raise e

vad_service = VADServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    vad_service.load()
    yield
    await vad_service.cleanup()

app = FastAPI(title="FireRed VAD Service", lifespan=lifespan)

class VADRequest(BaseModel):
    file_path: str
    hotwords: str = ""

@app.post("/vad_and_transcribe")
async def vad_and_transcribe(req: VADRequest):
    """
    1. VAD 检测
    2. 如果是空语音，返回 empty
    3. 如果有有效语音，切分并批量调用 ASR
    """
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="文件路径无效")

    try:
        # 1. VAD 检测
        start_time = time.perf_counter()
        # 在线程池中运行同步的 detect 方法
        vad_result = await asyncio.to_thread(vad_service.detect_speech, req.file_path)
        
        timestamps = vad_result.get("timestamps", [])
        duration = vad_result.get("dur", 0.0)
        
        logger.info(f"VAD 检测完成，时长: {duration}s, 片段数: {len(timestamps)}")

        # 2. 空语音处理
        if not timestamps:
            return {
                "status": "empty",
                "message": "未检测到有效语音",
                "segments": [],
                "full_text": ""
            }

        # 3. 有效语音处理 - 切分与 ASR
        # 读取原始音频用于切分
        audio_data, sample_rate = sf.read(req.file_path)
        
        segment_paths = []
        segment_timestamps = []
        
        # 创建临时目录存放切片
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, (start, end) in enumerate(timestamps):
                # 计算采样点位置
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                
                # 边界检查
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                if end_sample - start_sample < 160: # 忽略过短片段 (<10ms)
                    continue
                    
                segment_audio = audio_data[start_sample:end_sample]
                
                # 保存片段
                segment_filename = f"segment_{i}.wav"
                segment_path = os.path.join(temp_dir, segment_filename)
                sf.write(segment_path, segment_audio, sample_rate)
                
                segment_paths.append(segment_path)
                segment_timestamps.append((start, end))

            # 批量执行 ASR
            if segment_paths:
                asr_results = await call_asr_batch(segment_paths, req.hotwords)
            else:
                asr_results = []
        
        # 4. 整合结果
        final_segments = []
        combined_text_parts = []
        
        for (start, end), text in zip(segment_timestamps, asr_results):
            if text:
                final_segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
                combined_text_parts.append(text)
        
        full_text = " ".join(combined_text_parts)
        
        total_time = time.perf_counter() - start_time
        
        return {
            "status": "success",
            "segments": final_segments,
            "full_text": full_text,
            "inference_time": total_time
        }

    except Exception as e:
        logger.error(f"服务处理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def call_asr(file_path: str, hotwords: str) -> str:
    """调用 ASR 服务"""
    try:
        payload = {"file_path": file_path, "hotwords": hotwords}
        response = await vad_service.http_client.post(ASR_SERVICE_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            logger.warning(f"ASR 请求失败: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        logger.error(f"ASR 请求异常: {e}")
        return ""

async def call_asr_batch(file_paths: List[str], hotwords: str) -> List[str]:
    """调用 ASR 批量服务"""
    try:
        payload = {"file_paths": file_paths, "hotwords": hotwords}
        # 增加超时时间，因为批量处理可能较慢
        response = await vad_service.http_client.post(ASR_BATCH_SERVICE_URL, json=payload, timeout=120.0)
        if response.status_code == 200:
            return response.json().get("texts", [])
        else:
            logger.warning(f"ASR 批量请求失败: {response.status_code} - {response.text}")
            return [""] * len(file_paths)
    except Exception as e:
        logger.error(f"ASR 批量请求异常: {e}")
        return [""] * len(file_paths)

if __name__ == "__main__":
    import uvicorn
    # 运行在 8002 端口，避免与 ASR (7999) 和 Agent (8001) 冲突
    uvicorn.run(app, host="0.0.0.0", port=8002)
