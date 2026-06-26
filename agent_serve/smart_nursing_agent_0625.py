import os
import json
import time
import sys
import asyncio
import socket
import tempfile 
import uvicorn
import shutil
import uuid
from http import HTTPStatus
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import AsyncOpenAI
from pathlib import Path
import urllib3.util.connection as urllib3_connection

try:
    import dashscope
except ImportError:
    dashscope = None

# 避免 DashScope 上传本地音频时出现 DNS/IPv6 fallback 卡顿。
urllib3_connection.allowed_gai_family = lambda: socket.AF_INET

# 环境与路径配置
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from sak.prompts import *
from sak.utils import *
from sak.hotwords import *

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


DEBUG_MODE = False  # 开启调试模式
DEBUG_SAVE_DIR = os.path.abspath("debug_uploads")
if DEBUG_MODE and not os.path.exists(DEBUG_SAVE_DIR):
    os.makedirs(DEBUG_SAVE_DIR)

# --- 配置项 ---
LLM_API_KEY = os.getenv("QWEN_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_NAME = "qwen3.6-flash"
ASR_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("LLM_API_KEY")
ASR_MODEL_NAME = os.getenv("DASHSCOPE_QWEN_ASR_MODEL", "qwen3-asr-flash")
ASR_HTTP_BASE_URL = os.getenv("DASHSCOPE_HTTP_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")

class AgentManager:
    llm_client = None

models = AgentManager()

# ---------------- 模型生命周期 ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[系统启动] 初始化 Agent 业务组件...")
    try:
        # 初始化异步 LLM 客户端
        models.llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        print("[系统启动] 业务客户端初始化完成！")
    except Exception as e:
        print(f"[系统启动] 初始化失败: {e}")
        raise e
    
    yield
    
    print("\n[系统关闭] 释放资源...")

def qwen_asr_output_to_text(output):
    if not isinstance(output, dict):
        return ""
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices:
        return str(output.get("text") or "")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    if isinstance(content, str):
        return content.strip()
    return ""

def call_qwen_asr_flash_sync(file_path: str):
    if dashscope is None:
        raise RuntimeError("dashscope SDK is not installed. Run: pip install -U dashscope")
    if not ASR_API_KEY:
        raise RuntimeError("Missing DASHSCOPE_API_KEY/QWEN_API_KEY/LLM_API_KEY for ASR.")

    dashscope.base_http_api_url = ASR_HTTP_BASE_URL
    audio_uri = f"file://{Path(file_path).resolve()}"
    response = dashscope.MultiModalConversation.call(
        api_key=ASR_API_KEY,
        model=ASR_MODEL_NAME,
        messages=[{"role": "user", "content": [{"audio": audio_uri}]}],
        result_format="message",
        asr_options={
            "language": "zh",
            "enable_itn": False,
        },
    )
    if response.status_code != HTTPStatus.OK:
        message = getattr(response, "message", None) or str(response)
        raise RuntimeError(f"Qwen-ASR request failed: {message}")
    return qwen_asr_output_to_text(getattr(response, "output", {}) or {})

# ---------------- 工具函数：调用 ASR API ----------------
async def call_remote_asr_by_path(file_path: str, hotwords: str = ""):
    """通过 DashScope qwen3-asr-flash API 识别本地音频文件。"""
    try:
        text = await asyncio.to_thread(call_qwen_asr_flash_sync, file_path)
        return {"status": "success", "full_text": text}
    except Exception as e:
        print(f"请求 ASR API 异常: {e}")
        return {"status": "error", "full_text": ""}

# --- LLM 提取逻辑 (保持不变) ---
async def extract_menu_info_async(text: str):
    try:
        response = await models.llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": MENU_EXTRACTOR_PROMPT_LITE_V1},
                {"role": "user", "content": f"分析文本：{text}"}
            ],
            temperature=0.1,
            extra_body={"enable_thinking": False}
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        raw_data = json_match.group()
        json_data = json.loads(raw_data)
        json_data["intend"] = "menu"
        return json_data
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

async def extract_medical_info_async(text: str):
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V0.replace("{CURRENT_SYS_TIME}", formatted_time)
    try:
        response = await models.llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"分析文本：{text}"}
            ],
            temperature=0.1,
            extra_body={"enable_thinking": False}
            #response_format={"type": "json_object"} 
        )
        content = response.choices[0].message.content
        print(f"LLM Response: {content}")
        cleaned_json_data = extract_json(content, MEDICAL_KEYS)
        thinking_match = re.search(r"===Thinking===(.*?)===End Thinking===", content, re.DOTALL)
        if thinking_match:
            cleaned_json_data["thinking"] = thinking_match.group(1)
        else:
            cleaned_json_data["thinking"] = None
        cleaned_json_data["intend"] = "medical_info"
        return cleaned_json_data
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

# ---------------- API 接口 ----------------
app = FastAPI(title="Medical Agent (Decoupled Version)", lifespan=lifespan)

@app.post("/api/agent")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    temp_files = []
    
    try:

        # 1. 保存上传原文件到临时目录
#        suffix = os.path.splitext(file.filename)[1] or ".tmp"
#        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_raw:
#            shutil.copyfileobj(file.file, tmp_raw)
#            raw_path = tmp_raw.name
#            temp_files.append(raw_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = f"{time.time_ns()}_{uuid.uuid4().hex}"
        debug_filename = f"{timestamp}_{unique_suffix}_{file.filename}"
        save_path = os.path.join(DEBUG_SAVE_DIR, debug_filename)
        
        # 将文件写入 debug 目录
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        

        # 为了不破坏后续逻辑，我们将 raw_path 指向这个保存好的文件
        raw_path = save_path
        
        # 2. 音频转换与切片 (在 Agent 本地执行，利用 CPU)
        convert_start_time = time.time()
        full_path, trimmed_path = convert_audio_double_outputs(raw_path)
        #temp_files.extend([full_path, trimmed_path])
        if not DEBUG_MODE:
            temp_files.extend([save_path, full_path, trimmed_path])
        print(f"音频预处理耗时: {time.time() - convert_start_time:.2f}s")
        
        # 3. 第一次识别：截断音频意图识别
        asr_start_time = time.time()
        trimmed_res = await call_remote_asr_by_path(trimmed_path, "hotwords：打开 查看 进入")
        trimmed_text = trimmed_res.get("full_text", "")
        print(f"截断识别结果: {trimmed_text} (耗时: {time.time() - asr_start_time:.2f}s)")

        # 4. 根据意图进行第二次全文识别
        if trimmed_text and any(keyword in trimmed_text for keyword in ['打开', '查看', '进入']):
            print(">> 意图检测：智能打开 menu")
            intend = 'menu'
            hotwords = HOTWORDS_MENU_CONTEXT
        else:
            print(">> 意图检测：智能病历填写")
            intend = 'medical_info'
            hotwords = HOTWORDS_NURSE_CONTEXT

        asr_start_time = time.time()
        full_res = await call_remote_asr_by_path(full_path, hotwords)
        
        if full_res.get("status") == "empty":
             print(">> VAD检测为空音频")
             return {
                 "status": "success",
                 "raw_text": "",
                 "data": {"intend": "others", "message": "未检测到有效语音"}
             }

        full_text = full_res.get("full_text", "")
        print(f"全音频识别结果: {full_text} (耗时: {time.time() - asr_start_time:.2f}s)")

        if len(full_text) <= 2:
            print(">> 识别文本过短，判定为杂音")
            return {
                "status": "success", 
                "raw_text": full_text if full_text else "",
                "data": {"intend": "others", "message": "输入过短或为杂音"}
            }
        # 5. LLM 信息提取
        llm_start_time = time.time()
        result_data = {}
        if full_text:
            if intend == "menu":
                result_data = await extract_menu_info_async(full_text)
            else:
                result_data = await extract_medical_info_async(full_text)
        print(f"LLM 提取耗时: {time.time() - llm_start_time:.2f}s")

        return {
            "status": "success",
            "raw_text": full_text,
            "data": result_data
        }

    except Exception as e:
        print(f"处理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 清理临时文件
        for path in temp_files:
            if os.path.exists(path):
                try:  os.remove(path)
                except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
