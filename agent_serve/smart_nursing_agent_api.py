import asyncio
import json
import os
import re
import shutil
import socket
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, Optional

import urllib3.util.connection as urllib3_connection
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import AsyncOpenAI
from pydantic import BaseModel

try:
    import dashscope
except ImportError:  # pragma: no cover - runtime dependency hint
    dashscope = None

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - runtime dependency hint
    find_dotenv = None
    load_dotenv = None

# Avoid 15s DNS/IPv6 fallback stalls when DashScope uploads local audio to OSS.
urllib3_connection.allowed_gai_family = lambda: socket.AF_INET


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if load_dotenv:
    dotenv_path = find_dotenv() if find_dotenv else ""
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        for env_path in (PROJECT_DIR / ".env", HERE / ".env", Path.cwd() / ".env"):
            if env_path.exists():
                load_dotenv(env_path)
                break

from sak.prompts import (  # noqa: E402
    MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V0,
    MEDICAL_KEYS,
    MENU_EXTRACTOR_PROMPT_LITE_V1,
)
from sak.utils import extract_json  # noqa: E402


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


DEBUG_MODE = env_bool("SMART_NURSING_DEBUG", False)
UPLOAD_DIR = project_path(os.getenv("SMART_NURSING_UPLOAD_DIR", "debug_uploads"))

LLM_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("LLM_API_KEY")
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv(
    "DASHSCOPE_COMPATIBLE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL_NAME = os.getenv("QWEN_MODEL_NAME") or os.getenv("LLM_MODEL_NAME", "qwen3.5-flash")

ASR_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("LLM_API_KEY")
)
ASR_MODEL_NAME = os.getenv("DASHSCOPE_ASR_MODEL", "qwen3-asr-flash")
ASR_HTTP_BASE_URL = os.getenv("DASHSCOPE_HTTP_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")

MENU_KEYWORDS = ("打开", "查看", "进入")


class AgentManager:
    llm_client: Optional[AsyncOpenAI] = None


class AnalyzePathRequest(BaseModel):
    file_path: str
    intent: Optional[str] = None


class AnalyzeTextRequest(BaseModel):
    text: str
    intent: Optional[str] = None


models = AgentManager()


def require_llm_client() -> AsyncOpenAI:
    if models.llm_client is None:
        raise RuntimeError("LLM client is not initialized. Check API key and base URL.")
    return models.llm_client


def require_asr_sdk() -> None:
    if dashscope is None:
        raise RuntimeError("dashscope SDK is not installed. Run: pip install -U dashscope")
    if not ASR_API_KEY:
        raise RuntimeError("Missing DASHSCOPE_API_KEY/QWEN_API_KEY/LLM_API_KEY for ASR.")


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "audio.wav").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "audio.wav"


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(path)


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def qwen_asr_output_to_text(output: Any) -> str:
    if not isinstance(output, dict):
        return ""
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices:
        return str(output.get("text") or "").strip()

    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def call_qwen_asr_sync(file_path: Path) -> Dict[str, Any]:
    require_asr_sdk()
    dashscope.base_http_api_url = ASR_HTTP_BASE_URL

    response = dashscope.MultiModalConversation.call(
        api_key=ASR_API_KEY,
        model=ASR_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [{"audio": f"file://{file_path.resolve()}"}],
            }
        ],
        result_format="message",
        asr_options={
            "language": "zh",
            "enable_itn": False,
        },
    )
    if response.status_code != HTTPStatus.OK:
        message = getattr(response, "message", None) or str(response)
        raise RuntimeError(f"Qwen-ASR request failed: {message}")

    output = getattr(response, "output", {}) or {}
    return {
        "text": qwen_asr_output_to_text(output),
        "sentence": json_safe(output),
        "request_id": getattr(response, "request_id", None),
        "usage": json_safe(getattr(response, "usage", {}) or {}),
        "model": ASR_MODEL_NAME,
    }


def detect_intent(text: str, intent_hint: Optional[str] = None) -> str:
    if intent_hint in {"menu", "medical_info"}:
        return intent_hint
    if any(keyword in text for keyword in MENU_KEYWORDS):
        return "menu"
    return "medical_info"


async def extract_menu_info_async(text: str) -> Optional[Dict[str, Any]]:
    try:
        llm_client = require_llm_client()
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": MENU_EXTRACTOR_PROMPT_LITE_V1},
                {"role": "user", "content": f"分析文本：{text}"},
            ],
            temperature=0.1,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content or ""
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return {"intend": "menu", "message": "未解析到菜单 JSON"}
        json_data = json.loads(json_match.group())
        json_data["intend"] = "menu"
        return json_data
    except Exception as exc:
        print(f"LLM menu extraction error: {exc}")
        return None


async def extract_medical_info_async(text: str) -> Optional[Dict[str, Any]]:
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V0.replace(
        "{CURRENT_SYS_TIME}",
        formatted_time,
    )
    try:
        llm_client = require_llm_client()
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"分析文本：{text}"},
            ],
            temperature=0.1,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content or ""
        print(f"LLM Response: {content}")
        cleaned_json_data = extract_json(content, MEDICAL_KEYS)
        thinking_match = re.search(r"===Thinking===(.*?)===End Thinking===", content, re.DOTALL)
        cleaned_json_data["thinking"] = thinking_match.group(1) if thinking_match else None
        cleaned_json_data["intend"] = "medical_info"
        return cleaned_json_data
    except Exception as exc:
        print(f"LLM medical extraction error: {exc}")
        return None


async def analyze_text(text: str, intent_hint: Optional[str] = None) -> Dict[str, Any]:
    started_at = time.time()
    raw_text = (text or "").strip()
    if len(raw_text) <= 2:
        return {
            "status": "success",
            "raw_text": raw_text,
            "data": {"intend": "others", "message": "输入过短或为杂音"},
            "timing": {
                "asr_seconds": 0,
                "llm_seconds": 0,
                "total_seconds": round(time.time() - started_at, 3),
            },
        }

    intent = detect_intent(raw_text, intent_hint)
    llm_started = time.time()
    if intent == "menu":
        result_data = await extract_menu_info_async(raw_text)
    else:
        result_data = await extract_medical_info_async(raw_text)
    llm_seconds = time.time() - llm_started

    return {
        "status": "success",
        "raw_text": raw_text,
        "data": result_data,
        "timing": {
            "asr_seconds": 0,
            "llm_seconds": round(llm_seconds, 3),
            "total_seconds": round(time.time() - started_at, 3),
        },
        "llm_model": LLM_MODEL_NAME,
    }


async def analyze_audio_path(file_path: Path, intent_hint: Optional[str] = None) -> Dict[str, Any]:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    started_at = time.time()
    asr_started = time.time()
    asr_result = await asyncio.to_thread(call_qwen_asr_sync, file_path)
    asr_seconds = time.time() - asr_started

    result = await analyze_text(asr_result.get("text", ""), intent_hint)
    result["timing"]["asr_seconds"] = round(asr_seconds, 3)
    result["timing"]["total_seconds"] = round(time.time() - started_at, 3)
    result["asr"] = asr_result
    result["asr_model"] = ASR_MODEL_NAME
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[系统启动] 初始化 Smart Nursing Agent API...")
    if not LLM_API_KEY:
        print("[warn] missing LLM API key")
    else:
        models.llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        print(f"[系统启动] LLM client ready: model={LLM_MODEL_NAME}")

    if dashscope is None:
        print("[warn] dashscope package is not installed")
    elif not ASR_API_KEY:
        print("[warn] missing DashScope API key")
    else:
        print(f"[系统启动] ASR client ready: model={ASR_MODEL_NAME}")

    yield
    print("\n[系统关闭] Smart Nursing Agent API stopped")


app = FastAPI(title="Smart Nursing Agent API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "project_dir": str(PROJECT_DIR),
        "upload_dir": display_path(UPLOAD_DIR),
        "debug_mode": DEBUG_MODE,
        "llm": {
            "model": LLM_MODEL_NAME,
            "base_url": LLM_BASE_URL,
            "has_api_key": bool(LLM_API_KEY),
        },
        "asr": {
            "model": ASR_MODEL_NAME,
            "http_base_url": ASR_HTTP_BASE_URL,
            "has_api_key": bool(ASR_API_KEY),
            "sdk_installed": dashscope is not None,
        },
    }


@app.post("/api/agent")
async def analyze_audio_endpoint(
    file: UploadFile = File(...),
    intent: Optional[str] = Form(default=None),
):
    suffix = Path(file.filename or "").suffix or ".wav"
    safe_name = sanitize_filename(file.filename or f"upload{suffix}")
    upload_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}_{safe_name}"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = await analyze_audio_path(upload_path, intent)
        result["file"] = {
            "name": file.filename,
            "server_path": display_path(upload_path) if DEBUG_MODE else None,
        }
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if not DEBUG_MODE:
            upload_path.unlink(missing_ok=True)


@app.post("/api/path")
async def analyze_path_endpoint(request: AnalyzePathRequest):
    try:
        result = await analyze_audio_path(Path(request.file_path).expanduser(), request.intent)
        result["file"] = {"server_path": request.file_path}
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/text")
async def analyze_text_endpoint(request: AnalyzeTextRequest):
    try:
        return await analyze_text(request.text, request.intent)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
