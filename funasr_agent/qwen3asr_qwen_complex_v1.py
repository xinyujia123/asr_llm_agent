import os
import json
import time
import sys
import torch
import shutil
import tempfile 
import uvicorn
import asyncio
from qwen_asr import Qwen3ASRModel
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from funasr import AutoModel
from openai import AsyncOpenAI  # 使用异步客户端
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from sak.prompts import *
from sak.utils import *


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
LLM_API_KEY = os.getenv("QWEN_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_NAME = "qwen-flash"

ASR_MODEL_ID = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-1.7B"
#VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# 全局变量容器
class ModelManager:
    asr_model = None
    llm_client = None
    asr_lock = asyncio.Lock()
models = ModelManager()

def run_asr(audio_path, hotwords):
    res = models.asr_model.transcribe(
        audio=audio_path,
        context=hotwords, # Using HOTWORDS as context
        language="Chinese",
    )
    return res[0].text if res else ""

# ---------------- 模型生命周期 ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[系统启动] 正在加载 asr 模型到 GPU...")
    try:
        models.asr_model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
        )
        print("[系统启动] ASR加载成功!")
        
        # 初始化异步 LLM 客户端
        models.llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        print("[系统启动] LLM 异步客户端初始化完成！")
    except Exception as e:
        print(f"[系统启动] 致命错误: {e}")
        raise e
    
    yield
    
    print("\n[系统关闭] 释放资源...")
    # 清理显存等操作（Python通常会自动处理，但显式删除引用是个好习惯）
    del models.asr_model

async def extract_menu_info_async(text: str):
    # 1. 加强 Prompt，明确要求 JSON 里的值必须是 String
    system_prompt = MENU_EXTRACTOR_PROMPT
    
    try:
        response = await models.llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"分析文本：{text}"}
            ],
            temperature=0.1,
            # 如果模型支持 json_object 模式，务必保留
            response_format={"type": "json_object"} 
        )
        content = response.choices[0].message.content
        cleaned_data = extract_json(content, MENU_KEYS)
        cleaned_data["intend"] = "menu"
        return cleaned_data

    except json.JSONDecodeError:
        print(f"LLM 返回了非 JSON 数据: {content}")
        return {"error": "LLM output format error"}
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

async def extract_medical_info_async(text: str):

    now = datetime.now()
    formatted_time= now.strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = MEDICAL_EXTRACTOR_PROMPT.replace("{CURRENT_SYS_TIME}", formatted_time)
    
    try:
        response = await models.llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"分析文本：{text}"}
            ],
            temperature=0.1,
            # 如果模型支持 json_object 模式，务必保留
            response_format={"type": "json_object"} 
        )
        content = response.choices[0].message.content
        cleaned_data = extract_json(content, MEDICAL_KEYS)
        cleaned_data["intend"] = "medical_info"
        return cleaned_data

    except json.JSONDecodeError:
        print(f"LLM 返回了非 JSON 数据: {content}")
        return {"error": "LLM output format error"}
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

# ---------------- API 接口 ----------------
app = FastAPI(title="Medical Audio Analysis API", lifespan=lifespan)

@app.post("/api/agent")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    if models.asr_model is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    temp_files = []
    
    try:
        # 1. 保存上传文件

        suffix = os.path.splitext(file.filename)[1] or ".tmp"
        # 使用 delete=False 因为 Windows 下打开的文件不能再次打开，Linux 下比较宽松
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_raw:
            shutil.copyfileobj(file.file, tmp_raw)
            raw_path = tmp_raw.name
            temp_files.append(raw_path)
        
        convert_start_time = time.time()
        full_path, trimmed_path = convert_audio_double_outputs(raw_path)
        convert_duration = time.time() - convert_start_time
        print(f"音频转换耗时: {convert_duration:.2f}s")
        temp_files.append(full_path)
        temp_files.append(trimmed_path)
        
        async with models.asr_lock:
            # 3. [异步非阻塞] ASR 推理
            asr_start_time = time.time()
            trimmed_text = await run_in_threadpool(run_asr, trimmed_path, "注意以下词语：打开 开启")
            asr_duration = time.time() - asr_start_time
            print(f"截断音频识别结果: {trimmed_text} (耗时: {asr_duration:.2f}s)")

            # 判断前五个字是否包含 '打开' 或 '开启'
            if trimmed_text and any(keyword in trimmed_text for keyword in ['打开', '开启']):
                print("智能打开menu")
                result_data = {}
                asr_start_time = time.time()
                full_text = await run_in_threadpool(run_asr, full_path, HOTWORDS_MENU_CONTEXT)
                asr_duration = time.time() - asr_start_time
                print(f"全音频识别结果: {full_text} (耗时: {asr_duration:.2f}s)")
                intend = 'menu'
            else:
                print("智能病历表单填写")
                result_data = {}
                asr_start_time = time.time()
                full_text = await run_in_threadpool(run_asr, full_path, HOTWORDS_NURSE_CONTEXT)
                asr_duration = time.time() - asr_start_time
                print(f"全音频识别结果: {full_text} (耗时: {asr_duration:.2f}s)")
                intend = 'medical_info'

        llm_start_time = time.time()
        if full_text:
            if intend == "menu":
                result_data = await extract_menu_info_async(full_text)
            else:
                result_data = await extract_medical_info_async(full_text)
        llm_duration = time.time() - llm_start_time
        print(f"LLM耗时: {llm_duration:.2f}s")

        return {
            "status": "success",
            "raw_text": full_text,
            "data": result_data
        }

    except Exception as e:
        print(f"处理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 清理文件
        for path in temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"清理临时文件失败 {path}: {e}")

if __name__ == "__main__":
    # 建议生产环境使用 workers > 1，但要注意模型显存占用
    uvicorn.run(app, host="0.0.0.0", port=8000)