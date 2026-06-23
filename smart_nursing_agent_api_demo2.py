import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - runtime dependency hint
    load_dotenv = None

try:
    import dashscope
    from dashscope.audio.asr import Recognition
except ImportError:  # pragma: no cover - handled at runtime
    dashscope = None
    Recognition = None

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - handled at runtime
    AsyncOpenAI = None

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel


# Recommended runtime dependencies:
#   pip install -U fastapi uvicorn python-multipart openai dashscope python-dotenv
# Optional for non-16k PCM WAV uploads:
#   apt-get install ffmpeg

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE if (HERE / "sak").exists() else HERE.parent
PROJECT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

if load_dotenv:
    for env_path in (CODE_DIR / ".env", HERE / ".env", Path.cwd() / ".env"):
        if env_path.exists():
            load_dotenv(env_path)
            break

from sak.hotwords import HOTWORDS_MENU_CONTEXT, HOTWORDS_NURSE_CONTEXT
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


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


DEBUG_MODE = env_bool("SMART_NURSING_DEBUG", False)
UPLOAD_DIR = Path(os.getenv("SMART_NURSING_UPLOAD_DIR", str(HERE / "debug_uploads"))).expanduser()
DATASET_DIR = Path(
    os.getenv(
        "NURSE_AUDIO_DATASET_DIR",
        str(PROJECT_DIR / "dataset" / "asr_llm" / "nurse_audio_wav"),
    )
).expanduser()

LLM_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("LLM_API_KEY")
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv(
    "DASHSCOPE_COMPATIBLE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL_NAME = os.getenv("QWEN_MODEL_NAME") or os.getenv("LLM_MODEL_NAME", "qwen3.7-plus")
LLM_MODEL_OPTIONS = ("qwen3.7-plus", "qwen3.6-flash")
LLM_MODEL_ALIASES = {
    "qwen3.7-plus": "qwen3.7-plus",
    "3.7plus": "qwen3.7-plus",
    "qwen3.7plus": "qwen3.7-plus",
    "plus": "qwen3.7-plus",
    "qwen3.6-flash": "qwen3.6-flash",
    "3.6flash": "qwen3.6-flash",
    "qwen3.6flash": "qwen3.6-flash",
    "qwen3.7-flash": "qwen3.6-flash",
    "3.7flash": "qwen3.6-flash",
    "qwen3.7flash": "qwen3.6-flash",
    "flash": "qwen3.6-flash",
}

ASR_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("LLM_API_KEY")
)
ASR_MODEL_NAME = os.getenv("DASHSCOPE_ASR_MODEL", "fun-asr-realtime")
ASR_WEBSOCKET_URL = os.getenv(
    "DASHSCOPE_WEBSOCKET_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
)
ASR_VOCABULARY_ID = os.getenv("DASHSCOPE_ASR_VOCABULARY_ID") or None
FORCE_AUDIO_PREPROCESS = env_bool("SMART_NURSING_FORCE_PREPROCESS", False)
AUDIO_FILTER = os.getenv("SMART_NURSING_AUDIO_FILTER", "volume=30dB").strip()
MAX_BATCH_SIZE = env_int("SMART_NURSING_MAX_BATCH_SIZE", 20)

FORM_DEFINITIONS = {
    "admission_basic_form": {
        "name": "入院基础信息表",
        "keys": [
            "assessmentTime",
            "education",
            "ethnicity",
            "admissionTime",
            "admissionMethod",
            "diagnosis",
            "allergyHistory",
        ],
    },
    "vital_signs_form": {
        "name": "生命体征测量表",
        "keys": [
            "assessmentTime",
            "temperature",
            "pulseRate",
            "heartRate",
            "respiratoryRate",
            "bloodPressure",
            "weight",
            "height",
            "bloodSugar",
        ],
    },
    "nursing_status_form": {
        "name": "护理状态评估表",
        "keys": [
            "assessmentTime",
            "consciousness",
            "skin",
            "catheter",
            "limbActivity",
            "diet",
            "sleep",
            "urination",
            "defecation",
            "stoolFrequency",
        ],
    },
}

FORM_ROUTER_PROMPT = """
你是护理语音表单路由器。请根据 ASR 文本判断应填写哪些护理表单，可以多选。

可选表单：
1. admission_basic_form：入院基础信息表。包含文化程度、民族、入院时间、入院方式、诊断、过敏史等。
2. vital_signs_form：生命体征测量表。包含体温、脉率、心率、呼吸、血压、体重、身高、血糖等。
3. nursing_status_form：护理状态评估表。包含神志、皮肤、导管、肢体活动、饮食、睡眠、排尿、排便、排便次数等。

规则：
- 只有文本明确记录了某张表单的实际数据、状态、结果或否定结果时，才选择该表单。
- 可以多选，例如同时提到“诊断肺炎、体温37度、尿管”时选择三张表。
- 不要因为护士的询问句选择表单，例如“昨晚睡得怎么样”“有没有过敏史”“大便怎么样”只是询问，没有患者回答时不算表单内容。
- 不要因为提醒、建议、计划或将来动作选择表单，例如“记得吃早饭”“药别忘了吃”“一会儿去排尿”不代表饮食/用药/排尿状态。
- 菜单打开、页面跳转、聊天闲谈、无关内容返回空数组。
- 仅输出 JSON，不要解释。

输出格式：
{"forms":["admission_basic_form","vital_signs_form","nursing_status_form"]}

示例：
输入：“苗族，大专文化，诊断肺炎，没有过敏史。”
输出：{"forms":["admission_basic_form"]}

输入：“早上八点测体温三十六度五，血压一百二八十。”
输出：{"forms":["vital_signs_form"]}

输入：“患者身上有尿管，皮肤有手术切口，晚上入睡困难。”
输出：{"forms":["nursing_status_form"]}

输入：“诊断肺炎，体温37度，身上有尿袋。”
输出：{"forms":["admission_basic_form","vital_signs_form","nursing_status_form"]}

输入：“昨晚睡得怎么样？来先测个血压，血压138/82，36度7，心率75，呼吸18次。一会记得吃早饭。”
输出：{"forms":["vital_signs_form"]}
"""

ADMISSION_BASIC_FORM_PROMPT = """
你是专业严谨的护理入院基础信息录入助手。请从 ASR 文本中如实提取入院基础信息。

### 【当前系统时间】{CURRENT_SYS_TIME}
若未提及具体评估时间，assessmentTime 默认输出当前系统时间。

### 【字段枚举与映射规范】
* assessmentTime: 评估时间，格式 "YYYY-MM-DD HH:MM:SS"。
* education: 文化程度，仅限 "文盲"|"小学"|"初中"|"高中或中专"|"大专以上"|"未入学"。
* ethnicity: 民族，具体民族字符串。
* admissionTime: 入院时间，格式 "YYYY-MM-DD HH:MM:SS"，需结合“昨天/上午/下午”等推算。
* admissionMethod: 入院方式，仅限 "步行"|"轮椅"|"平车"|"推床"|"背入"|"其他"；120、救护车、担架映射为"其他"。
* diagnosis: 入院诊断，提取具体病名，必须保留不确定前缀，如“考虑肺炎”。
* allergyHistory: 过敏史，具体过敏源；明确无过敏史填 "无"。

兜底规则：
- 未提及则输出 null，assessmentTime 除外。
- 局部否定不代表全局无，例如“没得糖尿病”不代表 diagnosis 为“无”。
- 所有 JSON 值均为字符串或 null。
- 仅输出纯 JSON。

### 【示例 Few-Shot】
*假设当前系统时间为：2026-02-01 10:00:00*

示例 1：时间推算、入院方式映射、过敏史无
输入：“苗族，大专文化，病人是昨天下午三点救护车送进来的，考虑肺炎，没有过敏史。”
输出：
{
  "assessmentTime": "2026-02-01 10:00:00",
  "education": "大专以上",
  "ethnicity": "苗族",
  "admissionTime": "2026-01-31 15:00:00",
  "admissionMethod": "其他",
  "diagnosis": "考虑肺炎",
  "allergyHistory": "无"
}

示例 2：ASR 口语映射、局部否定不提取诊断
输入：“患者没读过书，汉族，今天早上八点平车入院。初步诊断冠心病。没得糖尿病，对青霉素过敏。”
输出：
{
  "assessmentTime": "2026-02-01 10:00:00",
  "education": "文盲",
  "ethnicity": "汉族",
  "admissionTime": "2026-02-01 08:00:00",
  "admissionMethod": "平车",
  "diagnosis": "冠心病",
  "allergyHistory": "青霉素"
}
"""

VITAL_SIGNS_FORM_PROMPT = """
你是专业严谨的护理生命体征录入助手。请从 ASR 文本中如实提取生命体征和测量数据。

### 【当前系统时间】{CURRENT_SYS_TIME}
若未提及具体测量时间，assessmentTime 默认输出当前系统时间。

### 【字段枚举与映射规范】
* assessmentTime: 测量/评估时间，格式 "YYYY-MM-DD HH:MM:SS"。
* temperature: 体温，℃，保留一位小数。
* pulseRate: 脉率，次/分。
* heartRate: 心率，次/分，严格区分心率和脉率。
* respiratoryRate: 呼吸频率，次/分；ASR 中“夫妻/服气”等需结合语境纠正为呼吸。
* bloodPressure: 血压，格式 "120/80" 或 "120/"、"/80"；收缩压在前，舒张压在后。
* weight: 体重，KG，保留一位小数。
* height: 身高，cm。
* bloodSugar: 血糖，mmol/L，保留一位小数。

兜底规则：
- 未提及则输出 null，assessmentTime 除外。
- 模糊数值如“低烧”“九十多”不猜测。
- 明显违背生理极限的离谱值视为 ASR 错误。
- 同一数值取最后一次；区间值取平均值。
- 所有 JSON 值均为字符串或 null。
- 仅输出纯 JSON。

### 【示例 Few-Shot】
*假设当前系统时间为：2026-02-01 10:00:00*

示例 1：ASR 同音纠错、血压连读拆解
输入：“今天早上八点测的，体温三六度五，心律八十，麦博六十二，夫妻二十，学压一百一七十，血糖六点八。”
输出：
{
  "assessmentTime": "2026-02-01 08:00:00",
  "temperature": "36.5",
  "heartRate": "80",
  "pulseRate": "62",
  "respiratoryRate": "20",
  "bloodPressure": "110/70",
  "bloodSugar": "6.8"
}

示例 2：单值血压、最后一次数值优先、模糊值不猜
输入：“现在量的血压高压一百二，后面复测高压一百三十五，脉搏七十二，体温有点低烧，体重六十公斤。”
输出：
{
  "assessmentTime": "2026-02-01 10:00:00",
  "pulseRate": "72",
  "bloodPressure": "135/",
  "weight": "60.0"
}
"""

NURSING_STATUS_FORM_PROMPT = """
你是专业严谨的护理状态评估录入助手。请从 ASR 文本中如实提取意识、皮肤、导管、功能状态和排泄情况。

### 【当前系统时间】{CURRENT_SYS_TIME}
若未提及具体评估时间，assessmentTime 默认输出当前系统时间。

### 【字段枚举与映射规范】
* assessmentTime: 评估时间，格式 "YYYY-MM-DD HH:MM:SS"。
* consciousness: 神志，仅限 "清醒"|"嗜睡"|"模糊"|"昏睡"|"昏迷"|"浅昏迷"|"中昏迷"|"深昏迷"|"药物镇静状"|"麻醉未醒"；严禁通过“能交流”等行为推断。
* skin: 皮肤黏膜，数组，可选 "完整"|"手术切口"|"造口"|"钉道"|"溃疡"|"破损"|"脓疱"|"皮疹"|"出血点"|"其他"；有异常严禁选“完整”。
* catheter: 导管，具体名称或 "无"；打点滴->静脉通路，尿袋->导尿管；吸氧面罩不算导管。
* limbActivity: 肢体活动，"正常"|"异常"；局部异常即异常。
* diet: 饮食，"正常"|"异常"。
* sleep: 睡眠，"正常"|"异常"。
* urination: 排尿，"正常"|"异常"。
* defecation: 排便，"正常"|"异常"。
* stoolFrequency: 当天排便次数。

兜底规则：
- 未提及则输出 null，assessmentTime 除外。
- 局部否定不代表全局无，如“没插胃管”不等于无导管。
- 导尿管/尿袋会导致 urination 可判定为异常；造口会导致 defecation 可判定为异常。
- 皮肤出现任何异常时，skin 严禁同时输出“完整”。
- 所有 JSON 值均为字符串、数组或 null。
- 仅输出纯 JSON。

### 【示例 Few-Shot】
*假设当前系统时间为：2026-02-01 10:00:00*

示例 1：导管映射与级联异常、局部皮肤正常不等于完整
输入：“神志清醒，手部皮肤正常，接了尿袋，同时在打点滴，今天排便两次。”
输出：
{
  "assessmentTime": "2026-02-01 10:00:00",
  "consciousness": "清醒",
  "catheter": "导尿管、静脉通路",
  "urination": "异常",
  "stoolFrequency": "2"
}

示例 2：局部否定不代表全局无、皮肤异常互斥完整
输入：“目前没有插胃管，身上有手术切口和压疮，四肢活动正常，大便正常，晚上入睡困难。”
输出：
{
  "assessmentTime": "2026-02-01 10:00:00",
  "skin": ["手术切口", "其他"],
  "limbActivity": "正常",
  "defecation": "正常",
  "sleep": "异常"
}
"""

FORM_PROMPTS = {
    "admission_basic_form": ADMISSION_BASIC_FORM_PROMPT,
    "vital_signs_form": VITAL_SIGNS_FORM_PROMPT,
    "nursing_status_form": NURSING_STATUS_FORM_PROMPT,
}


class AgentManager:
    llm_client: Optional[Any] = None


models = AgentManager()


class AnalyzePathRequest(BaseModel):
    file_path: str
    intent: Optional[str] = None
    model: Optional[str] = None


class DatasetRunRequest(BaseModel):
    files: Optional[List[str]] = None
    limit: int = 1
    intent: Optional[str] = None
    model: Optional[str] = None


def require_asr_sdk() -> None:
    if dashscope is None or Recognition is None:
        raise RuntimeError("dashscope SDK is not installed. Run: pip install -U dashscope")
    if not ASR_API_KEY:
        raise RuntimeError("Missing DASHSCOPE_API_KEY/QWEN_API_KEY/LLM_API_KEY for ASR.")


def require_llm_client() -> Any:
    if models.llm_client is None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai SDK is not installed. Run: pip install -U openai")
        raise RuntimeError("LLM client is not initialized. Check API key and base URL.")
    return models.llm_client


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "audio.wav").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "audio.wav"


def numeric_sort_key(path: Path) -> Tuple[int, Any]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def normalize_llm_model(model_name: Optional[str] = None) -> str:
    key = (model_name or LLM_MODEL_NAME).strip().lower()
    normalized = LLM_MODEL_ALIASES.get(key)
    if normalized:
        return normalized
    allowed = ", ".join(LLM_MODEL_OPTIONS)
    raise ValueError(f"Unsupported model '{model_name}'. Allowed models: {allowed}")


def probe_wav(path: Path) -> Dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            metadata = {
                "format": "wav",
                "channels": wav_file.getnchannels(),
                "sample_rate": rate,
                "sample_width": wav_file.getsampwidth(),
                "duration_seconds": round(frames / rate, 3) if rate else None,
            }
            metadata["is_16k_mono_pcm16"] = (
                metadata["channels"] == 1
                and metadata["sample_rate"] == 16000
                and metadata["sample_width"] == 2
            )
            return metadata
    except Exception as exc:
        return {
            "format": "unknown",
            "is_16k_mono_pcm16": False,
            "error": str(exc),
        }


def convert_to_16k_wav(source_path: Path) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "Audio must be 16kHz mono PCM WAV, or install ffmpeg for automatic conversion."
        )

    fd, output_name = tempfile.mkstemp(prefix="smart_nursing_", suffix=".wav")
    os.close(fd)
    output_path = Path(output_name)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]
    if AUDIO_FILTER:
        command.extend(["-af", AUDIO_FILTER])
    command.append(str(output_path))

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        detail = exc.stderr.decode("utf-8", errors="ignore")[-800:]
        raise RuntimeError(f"ffmpeg audio conversion failed: {detail}") from exc
    return output_path


def prepare_audio_for_asr(file_path: Path) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    metadata = probe_wav(file_path)
    if metadata.get("is_16k_mono_pcm16") and not FORCE_AUDIO_PREPROCESS:
        return file_path, None, metadata
    converted = convert_to_16k_wav(file_path)
    converted_metadata = probe_wav(converted)
    return converted, converted, {"source": metadata, "prepared": converted_metadata}


def sentence_to_text(sentence: Any) -> str:
    if isinstance(sentence, dict):
        if "text" in sentence:
            return str(sentence.get("text") or "")
        if "sentences" in sentence and isinstance(sentence["sentences"], list):
            return "".join(str(item.get("text") or "") for item in sentence["sentences"])
    if isinstance(sentence, list):
        return "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in sentence
        )
    return str(sentence or "")


def call_fun_asr_sync(file_path: Path) -> Dict[str, Any]:
    require_asr_sdk()
    dashscope.api_key = ASR_API_KEY
    dashscope.base_websocket_api_url = ASR_WEBSOCKET_URL

    recognition = Recognition(
        model=ASR_MODEL_NAME,
        format="wav",
        sample_rate=16000,
        callback=None,
    )
    if ASR_VOCABULARY_ID:
        result = recognition.call(str(file_path), phrase_id=ASR_VOCABULARY_ID)
    else:
        result = recognition.call(str(file_path))

    if result.status_code != HTTPStatus.OK:
        message = getattr(result, "message", None) or str(result)
        raise RuntimeError(f"Fun-ASR request failed: {message}")

    sentence = result.get_sentence()
    return {
        "text": sentence_to_text(sentence).strip(),
        "sentence": json_safe(sentence),
        "request_id": recognition.get_last_request_id(),
        "first_package_delay_ms": recognition.get_first_package_delay(),
        "last_package_delay_ms": recognition.get_last_package_delay(),
    }


def extract_first_json_object(content: str) -> Dict[str, Any]:
    json_match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not json_match:
        return {}
    return json.loads(json_match.group())


def normalize_form_ids(raw_forms: Any) -> List[str]:
    if isinstance(raw_forms, str):
        candidates = re.split(r"[,，、\s]+", raw_forms)
    elif isinstance(raw_forms, list):
        candidates = raw_forms
    else:
        candidates = []

    normalized = []
    for item in candidates:
        form_id = str(item or "").strip()
        if form_id in FORM_DEFINITIONS and form_id not in normalized:
            normalized.append(form_id)
    return normalized


def prompt_with_current_time(prompt: str) -> str:
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return prompt.replace("{CURRENT_SYS_TIME}", formatted_time)


async def route_forms_async(text: str, model_name: str) -> List[str]:
    llm_client = require_llm_client()
    response = await llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": FORM_ROUTER_PROMPT},
            {"role": "user", "content": f"ASR文本：{text}"},
        ],
        temperature=0.1,
        extra_body={"enable_thinking": False},
    )
    content = response.choices[0].message.content
    data = extract_first_json_object(content)
    return normalize_form_ids(data.get("forms"))


async def extract_form_info_async(text: str, form_id: str, model_name: str) -> Dict[str, Any]:
    llm_client = require_llm_client()
    form = FORM_DEFINITIONS[form_id]
    system_prompt = prompt_with_current_time(FORM_PROMPTS[form_id])
    response = await llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"ASR文本：{text}"},
        ],
        temperature=0.1,
        extra_body={"enable_thinking": False},
    )
    content = response.choices[0].message.content
    return {
        "id": form_id,
        "name": form["name"],
        "data": extract_json(content, form["keys"]),
    }


async def extract_forms_async(
    text: str,
    model_name: str,
    form_hint: Optional[str] = None,
) -> Dict[str, Any]:
    if form_hint in FORM_DEFINITIONS:
        form_ids = [form_hint]
    else:
        form_ids = await route_forms_async(text, model_name)

    if not form_ids:
        return {
            "intend": "form_route",
            "forms": [],
            "form_ids": [],
            "form_names": [],
            "message": "未匹配到需要填写的表单",
        }

    forms = await asyncio.gather(
        *(extract_form_info_async(text, form_id, model_name) for form_id in form_ids)
    )
    return {
        "intend": "form_route",
        "forms": forms,
        "form_ids": form_ids,
        "form_names": [FORM_DEFINITIONS[form_id]["name"] for form_id in form_ids],
    }


async def analyze_audio_path(
    file_path: Path,
    intent_hint: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    llm_model = normalize_llm_model(model_name)
    started_at = time.time()
    prepared_path: Optional[Path] = None
    generated_path: Optional[Path] = None
    try:
        prepared_path, generated_path, audio_metadata = await asyncio.to_thread(
            prepare_audio_for_asr,
            file_path,
        )

        asr_started = time.time()
        asr_result = await asyncio.to_thread(call_fun_asr_sync, prepared_path)
        asr_seconds = time.time() - asr_started
        raw_text = asr_result["text"]

        if len(raw_text) <= 2:
            return {
                "status": "success",
                "raw_text": raw_text,
                "data": {"intend": "others", "message": "输入过短或为杂音"},
                "timing": {
                    "asr_seconds": round(asr_seconds, 3),
                    "llm_seconds": 0,
                    "total_seconds": round(time.time() - started_at, 3),
                },
                "asr": asr_result,
                "audio": audio_metadata,
                "llm_model": llm_model,
            }

        llm_started = time.time()
        result_data = await extract_forms_async(raw_text, llm_model, intent_hint)
        llm_seconds = time.time() - llm_started

        return {
            "status": "success",
            "raw_text": raw_text,
            "data": result_data,
            "timing": {
                "asr_seconds": round(asr_seconds, 3),
                "llm_seconds": round(llm_seconds, 3),
                "total_seconds": round(time.time() - started_at, 3),
            },
            "asr": asr_result,
            "audio": audio_metadata,
            "llm_model": llm_model,
        }
    finally:
        if generated_path and generated_path.exists():
            generated_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[system] smart nursing agent api starting")
    if AsyncOpenAI is None:
        print("[warn] openai package is not installed")
    elif not LLM_API_KEY:
        print("[warn] missing LLM API key")
    else:
        models.llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        print(f"[system] LLM client ready: model={LLM_MODEL_NAME}")

    if dashscope is None:
        print("[warn] dashscope package is not installed")
    elif not ASR_API_KEY:
        print("[warn] missing DashScope API key")
    else:
        print(f"[system] ASR client ready: model={ASR_MODEL_NAME}")

    yield
    print("\n[system] smart nursing agent api stopped")


app = FastAPI(title="Smart Nursing Agent API Demo", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    return HTML_PAGE


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "code_dir": str(CODE_DIR),
        "dataset_dir": str(DATASET_DIR),
        "dataset_exists": DATASET_DIR.exists(),
        "upload_dir": str(UPLOAD_DIR),
        "llm": {
            "model": normalize_llm_model(),
            "default_model": normalize_llm_model(),
            "model_options": list(LLM_MODEL_OPTIONS),
            "base_url": LLM_BASE_URL,
            "has_api_key": bool(LLM_API_KEY),
            "sdk_installed": AsyncOpenAI is not None,
        },
        "asr": {
            "model": ASR_MODEL_NAME,
            "websocket_url": ASR_WEBSOCKET_URL,
            "has_api_key": bool(ASR_API_KEY),
            "sdk_installed": dashscope is not None,
            "vocabulary_id_configured": bool(ASR_VOCABULARY_ID),
        },
        "ffmpeg": bool(shutil.which("ffmpeg")),
    }


@app.post("/api/agent")
async def analyze_audio_endpoint(
    file: UploadFile = File(...),
    intent: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None),
):
    suffix = Path(file.filename or "").suffix or ".wav"
    safe_name = sanitize_filename(file.filename or f"upload{suffix}")
    upload_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}_{safe_name}"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = await analyze_audio_path(upload_path, intent_hint=intent, model_name=model)
        result["file"] = {"name": file.filename, "server_path": str(upload_path) if DEBUG_MODE else None}
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if not DEBUG_MODE:
            upload_path.unlink(missing_ok=True)


@app.post("/api/path")
async def analyze_path_endpoint(request: AnalyzePathRequest):
    try:
        result = await analyze_audio_path(
            Path(request.file_path).expanduser(),
            request.intent,
            request.model,
        )
        result["file"] = {"server_path": request.file_path}
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/dataset")
async def list_dataset():
    if not DATASET_DIR.exists():
        return {"dataset_dir": str(DATASET_DIR), "count": 0, "files": []}
    files = sorted(DATASET_DIR.glob("*.wav"), key=numeric_sort_key)
    return {
        "dataset_dir": str(DATASET_DIR),
        "count": len(files),
        "files": [
            {
                "name": item.name,
                "path": str(item),
                "audio_url": f"/api/dataset/audio/{item.name}",
            }
            for item in files
        ],
    }


@app.get("/api/dataset/audio/{filename}")
async def dataset_audio(filename: str):
    dataset_root = DATASET_DIR.resolve()
    candidate = (dataset_root / Path(filename).name).resolve()
    if candidate.parent != dataset_root or candidate.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="Audio file not found")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(candidate, media_type="audio/wav")


def resolve_dataset_files(request: DatasetRunRequest) -> List[Path]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    dataset_root = DATASET_DIR.resolve()
    if request.files:
        selected = []
        for name in request.files:
            candidate = (dataset_root / Path(name).name).resolve()
            if candidate.parent != dataset_root or candidate.suffix.lower() != ".wav":
                continue
            if candidate.exists():
                selected.append(candidate)
        return sorted(selected, key=numeric_sort_key)

    limit = max(1, min(request.limit, MAX_BATCH_SIZE))
    return sorted(DATASET_DIR.glob("*.wav"), key=numeric_sort_key)[:limit]


@app.post("/api/dataset/run")
async def run_dataset(request: DatasetRunRequest):
    try:
        llm_model = normalize_llm_model(request.model)
        files = resolve_dataset_files(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    results = []
    for audio_path in files:
        row_started = time.time()
        try:
            result = await analyze_audio_path(audio_path, request.intent, llm_model)
            results.append(
                {
                    "file": audio_path.name,
                    "audio_url": f"/api/dataset/audio/{audio_path.name}",
                    "status": result.get("status", "success"),
                    "raw_text": result.get("raw_text", ""),
                    "intend": (result.get("data") or {}).get("intend"),
                    "form_names": (result.get("data") or {}).get("form_names", []),
                    "model": result.get("llm_model"),
                    "data": result.get("data"),
                    "timing": result.get("timing"),
                    "seconds": round(time.time() - row_started, 3),
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "file": audio_path.name,
                    "audio_url": f"/api/dataset/audio/{audio_path.name}",
                    "status": "error",
                    "raw_text": "",
                    "intend": None,
                    "form_names": [],
                    "model": llm_model,
                    "data": None,
                    "timing": None,
                    "seconds": round(time.time() - row_started, 3),
                    "error": str(exc),
                }
            )

    return {
        "dataset_dir": str(DATASET_DIR),
        "count": len(results),
        "results": results,
    }


HTML_PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Smart Nursing Agent Demo</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #617080;
      --line: #d8dee6;
      --accent: #126c74;
      --accent-strong: #0b4e55;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 28px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .header-tools {
      display: flex;
      align-items: end;
      gap: 14px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .header-tools label {
      min-width: 150px;
    }
    main {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 22px 28px 34px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 18px;
      align-items: start;
      margin-bottom: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    h2 {
      margin: 0;
      font-size: 15px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, auto));
      gap: 10px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, select {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      padding: 6px 9px;
    }
    select[multiple] {
      min-height: 116px;
      width: 100%;
    }
    button {
      min-height: 36px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-size: 14px;
      font-weight: 650;
      padding: 7px 12px;
      cursor: pointer;
    }
    button.secondary {
      background: #fff;
      color: var(--accent-strong);
    }
    button:disabled {
      opacity: .58;
      cursor: wait;
    }
    .upload-row {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 10px;
      align-items: end;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1080px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
      font-size: 13px;
    }
    th {
      position: sticky;
      top: 0;
      background: #eef2f5;
      color: #344253;
      font-weight: 700;
      z-index: 1;
    }
    tr:last-child td { border-bottom: 0; }
    tr.error td { color: var(--danger); }
    .audio-cell {
      min-width: 190px;
    }
    .audio-cell audio {
      width: 190px;
      max-width: 100%;
      height: 34px;
    }
    .file-cell {
      font-weight: 700;
      white-space: nowrap;
    }
    .raw-text {
      min-width: 260px;
      max-width: 420px;
      white-space: normal;
      line-height: 1.45;
    }
    .form-summary {
      min-width: 150px;
      line-height: 1.45;
    }
    .form-picker {
      min-width: 240px;
      margin-bottom: 8px;
    }
    .form-picker span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    pre {
      margin: 0;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.42;
      color: #263442;
    }
    .empty {
      padding: 28px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 900px) {
      header {
        align-items: flex-start;
        flex-direction: column;
        padding: 16px;
      }
      .header-tools {
        width: 100%;
        justify-content: flex-start;
      }
      main { padding: 16px; }
      .toolbar { grid-template-columns: 1fr; }
      .controls, .upload-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Smart Nursing Agent Demo</h1>
    <div class="header-tools">
      <label>
        Model
        <select id="modelSelect">
          <option value="qwen3.7-plus">qwen3.7-plus</option>
          <option value="qwen3.6-flash">qwen3.6-flash</option>
        </select>
      </label>
      <div class="status" id="health">Checking runtime</div>
    </div>
  </header>
  <main>
    <div class="toolbar">
      <section>
        <div class="section-title">
          <h2>Dataset</h2>
          <button class="secondary" id="refreshBtn" type="button">Refresh</button>
        </div>
        <div class="controls">
          <label>
            Files
            <select id="fileSelect" multiple></select>
          </label>
          <label>
            Limit
            <input id="limitInput" type="number" min="1" max="20" value="1" />
          </label>
          <button id="runFirstBtn" type="button">Run First</button>
          <button class="secondary" id="runSelectedBtn" type="button">Run Selected</button>
        </div>
      </section>
      <section>
        <div class="section-title">
          <h2>Upload</h2>
          <span class="status" id="uploadStatus">Idle</span>
        </div>
        <div class="upload-row">
          <label>
            Audio
            <input id="audioInput" type="file" accept="audio/*,.wav" />
          </label>
          <button id="uploadBtn" type="button">Analyze</button>
        </div>
      </section>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th>Audio</th>
            <th>Status</th>
            <th>涉及表单</th>
            <th>Seconds</th>
            <th>ASR Text</th>
            <th>Extracted Data</th>
            <th>Error</th>
          </tr>
        </thead>
        <tbody id="resultBody">
          <tr><td class="empty" colspan="8">No results</td></tr>
        </tbody>
      </table>
    </div>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const resultBody = $("resultBody");
    let uploadObjectUrl = null;
    let displayedRows = [];

    function esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, options);
      const text = await response.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; }
      catch { data = { detail: text }; }
      if (!response.ok) {
        throw new Error(data.detail || response.statusText);
      }
      return data;
    }

    function setBusy(isBusy) {
      for (const id of ["refreshBtn", "runFirstBtn", "runSelectedBtn", "uploadBtn", "modelSelect"]) {
        $(id).disabled = isBusy;
      }
    }

    function selectedModel() {
      return $("modelSelect").value || "qwen3.7-plus";
    }

    function renderAudio(row) {
      const source = row.audio_url || row.audio_object_url;
      if (!source) return "";
      return `<audio controls preload="none" src="${esc(source)}"></audio>`;
    }

    function rowForms(row) {
      return Array.isArray(row.data?.forms) ? row.data.forms : [];
    }

    function formSummary(row) {
      const names = rowForms(row).map((form) => form.name).filter(Boolean);
      if (names.length) return names.join("、");
      if (Array.isArray(row.form_names) && row.form_names.length) return row.form_names.join("、");
      return row.data?.message || "";
    }

    function formDataText(row, formIndex = 0) {
      const forms = rowForms(row);
      if (forms.length) {
        return JSON.stringify(forms[formIndex]?.data || {}, null, 2);
      }
      return row.data ? JSON.stringify(row.data, null, 2) : "";
    }

    function renderExtractedData(row, rowIndex) {
      const forms = rowForms(row);
      if (!forms.length) {
        return `<pre>${esc(formDataText(row))}</pre>`;
      }
      const options = forms.map((form, index) => (
        `<option value="${index}">${esc(form.name || form.id || `表单${index + 1}`)}</option>`
      )).join("");
      return `
        <label class="form-picker">
          <span>涉及表单</span>
          <select class="formSelect" data-row-index="${rowIndex}">${options}</select>
        </label>
        <pre id="formData-${rowIndex}">${esc(formDataText(row, 0))}</pre>`;
    }

    function renderRows(rows) {
      displayedRows = rows;
      if (!rows.length) {
        resultBody.innerHTML = '<tr><td class="empty" colspan="8">No results</td></tr>';
        return;
      }
      resultBody.innerHTML = rows.map((row, index) => {
        const seconds = row.seconds ?? row.timing?.total_seconds ?? "";
        return `
          <tr class="${row.status === "error" ? "error" : ""}">
            <td class="file-cell">${esc(row.file || row.name || "upload")}</td>
            <td class="audio-cell">${renderAudio(row)}</td>
            <td>${esc(row.status)}</td>
            <td class="form-summary">${esc(formSummary(row))}</td>
            <td>${esc(seconds)}</td>
            <td class="raw-text">${esc(row.raw_text || "")}</td>
            <td>${renderExtractedData(row, index)}</td>
            <td>${esc(row.error || "")}</td>
          </tr>`;
      }).join("");
    }

    resultBody.addEventListener("change", (event) => {
      if (!event.target.classList.contains("formSelect")) return;
      const rowIndex = Number(event.target.dataset.rowIndex);
      const formIndex = Number(event.target.value);
      const pre = $(`formData-${rowIndex}`);
      if (!pre) return;
      pre.textContent = formDataText(displayedRows[rowIndex] || {}, formIndex);
    });

    async function loadHealth() {
      try {
        const data = await requestJson("/health");
        if (data.llm.model_options?.length) {
          $("modelSelect").innerHTML = data.llm.model_options.map((model) => (
            `<option value="${esc(model)}">${esc(model)}</option>`
          )).join("");
          $("modelSelect").value = data.llm.default_model || data.llm.model || "qwen3.7-plus";
        }
        $("health").textContent = `ASR ${data.asr.model} · dataset ${data.dataset_exists ? "ready" : "missing"}`;
      } catch (error) {
        $("health").textContent = error.message;
      }
    }

    async function loadDataset() {
      const data = await requestJson("/api/dataset");
      $("fileSelect").innerHTML = data.files.slice(0, 200).map((item) => (
        `<option value="${esc(item.name)}">${esc(item.name)}</option>`
      )).join("");
      if ($("fileSelect").options.length) {
        $("fileSelect").options[0].selected = true;
      }
      $("refreshBtn").textContent = `Refresh (${data.count})`;
    }

    async function runDataset(selectedOnly) {
      setBusy(true);
      try {
        const selected = Array.from($("fileSelect").selectedOptions).map((item) => item.value);
        const body = selected.length
          ? { files: selected, model: selectedModel() }
          : { limit: Number($("limitInput").value || 1), model: selectedModel() };
        const data = await requestJson("/api/dataset/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        renderRows(data.results || []);
      } catch (error) {
        renderRows([{ file: "dataset", status: "error", error: error.message }]);
      } finally {
        setBusy(false);
      }
    }

    async function analyzeUpload() {
      const file = $("audioInput").files[0];
      if (!file) return;
      setBusy(true);
      $("uploadStatus").textContent = "Running";
      try {
        if (uploadObjectUrl) URL.revokeObjectURL(uploadObjectUrl);
        uploadObjectUrl = URL.createObjectURL(file);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("model", selectedModel());
        const data = await requestJson("/api/agent", { method: "POST", body: formData });
        renderRows([{
          file: file.name,
          audio_object_url: uploadObjectUrl,
          status: data.status,
          raw_text: data.raw_text,
          form_names: data.data?.form_names || [],
          data: data.data,
          timing: data.timing,
          error: null,
        }]);
        $("uploadStatus").textContent = "Done";
      } catch (error) {
        renderRows([{ file: file.name, status: "error", error: error.message }]);
        $("uploadStatus").textContent = "Error";
      } finally {
        setBusy(false);
      }
    }

    $("refreshBtn").addEventListener("click", loadDataset);
    $("runFirstBtn").addEventListener("click", () => runDataset(false));
    $("runSelectedBtn").addEventListener("click", () => runDataset(true));
    $("uploadBtn").addEventListener("click", analyzeUpload);

    loadHealth();
    loadDataset().catch((error) => {
      $("fileSelect").innerHTML = "";
      renderRows([{ file: "dataset", status: "error", error: error.message }]);
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    port = env_int("PORT", 8001)
    uvicorn.run(app, host="0.0.0.0", port=port)
