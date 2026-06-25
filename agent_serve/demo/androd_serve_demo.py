import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
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

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing runtime dependencies. Install with: "
        "pip install -U fastapi uvicorn python-multipart openai dashscope python-dotenv"
    ) from exc

try:
    import urllib3.util.connection as urllib3_connection

    urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
except Exception:  # pragma: no cover - safe optimization only
    pass


# Android server demo for MobileMedical RecordingPop.uploadVoiceFile().
#
# Current Android app expects:
#   POST /agent/asr2txt/{patientId}
#   multipart field: file
#   response:
#   {
#     "code": 1,
#     "msg": "...",
#     "data": {
#       "signData": "{\"intend\":\"medical_info\", ...}"
#     }
#   }
#
# Run:
#   pip install -U fastapi uvicorn python-multipart openai dashscope python-dotenv
#   export DASHSCOPE_API_KEY=...
#   python androd_serve_demo.py
#
# Useful local test without ASR:
#   curl -F 'file=@test.amr' \
#        -F 'mock_text=体温36度5，血压120/80，脉搏78' \
#        http://127.0.0.1:17090/agent/asr2txt/123

HERE = Path(__file__).resolve().parent


def find_project_dir(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "sak").is_dir():
            return candidate
    return start


PROJECT_DIR = find_project_dir(HERE)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if load_dotenv:
    for env_path in (PROJECT_DIR / ".env", HERE / ".env", Path.cwd() / ".env"):
        if env_path.exists():
            load_dotenv(env_path)
            break

try:
    from sak.utils import extract_json as sak_extract_json
except Exception:  # pragma: no cover - keep this demo runnable standalone
    sak_extract_json = None


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


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


DEBUG_MODE = env_bool("ANDROID_NURSING_DEBUG", True)
UPLOAD_DIR = project_path(os.getenv("ANDROID_NURSING_UPLOAD_DIR", "debug_uploads/android"))
AUDIO_FILTER = os.getenv("ANDROID_NURSING_AUDIO_FILTER", "volume=30dB").strip()
FORCE_AUDIO_PREPROCESS = env_bool("ANDROID_NURSING_FORCE_PREPROCESS", False)

LLM_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("LLM_API_KEY")
)
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    os.getenv("DASHSCOPE_COMPATIBLE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
LLM_MODEL_NAME = os.getenv("QWEN_MODEL_NAME") or os.getenv("LLM_MODEL_NAME", "qwen3.6-flash")

ASR_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("LLM_API_KEY")
)
QWEN_ASR_FLASH_MODEL_NAME = os.getenv("DASHSCOPE_QWEN_ASR_MODEL", "qwen3-asr-flash")
FUN_ASR_REALTIME_MODEL_NAME = os.getenv("DASHSCOPE_FUN_ASR_MODEL", "fun-asr-realtime")
ASR_MODEL_NAME = os.getenv("DASHSCOPE_ASR_MODEL", QWEN_ASR_FLASH_MODEL_NAME)
ASR_HTTP_BASE_URL = os.getenv("DASHSCOPE_HTTP_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
ASR_WEBSOCKET_URL = os.getenv(
    "DASHSCOPE_WEBSOCKET_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
)
ASR_VOCABULARY_ID = os.getenv("DASHSCOPE_ASR_VOCABULARY_ID") or None

PORT = env_int("ANDROID_NURSING_PORT", 17090)

MENU_TARGETS = [
    "标本",
    "输液",
    "皮试",
    "配液",
    "口服",
    "治疗",
    "体征采集",
    "护理记录",
    "护理文书",
    "患者巡视",
    "健康宣教",
    "不良事件",
    "推送通知",
    "首页",
    "患者",
    "消息",
    "通讯录",
    "我的",
    "计时提醒",
    "常用语管理",
    "关于我们",
    "患者详情",
]

ANDROID_SIGN_DATA_KEYS = [
    "education",
    "ethnicity",
    "admissionTime",
    "admissionMethod",
    "diagnosis",
    "allergyHistory",
    "temperature",
    "pulseRate",
    "heartRate",
    "respiratoryRate",
    "bloodPressure",
    "weight",
    "height",
    "bloodSugar",
    "stoolFrequency",
    "intend",
    "target",
]

EXTRA_FORM_KEYS = [
    "formType",
    "recordTime",
    "assessmentTime",
    "consciousness",
    "skin",
    "catheter",
    "limbActivity",
    "diet",
    "sleep",
    "urination",
    "defecation",
    "roundStates",
    "conditionRecord",
    "eventType",
    "eventLevel",
    "eventSeverity",
    "eventContent",
]

ANDROID_EXTRACTOR_PROMPT = """
你是移动护理 App 的语音表单助手。请从护士口述文本中提取结构化数据，输出纯 JSON。

当前 App 已支持的字段：
- intend: "menu" 或 "medical_info"。
- target: 当 intend="menu" 时填写目标菜单，必须是菜单候选之一。
- education: 文化程度，仅限 "文盲"|"小学"|"初中"|"高中或中专"|"大专以上"|"未入学"。
- ethnicity: 民族。
- admissionTime: 入院时间，格式 "YYYY-MM-DD HH:MM:SS"。
- admissionMethod: 入院方式，仅限 "步行"|"轮椅"|"平车"|"推床"|"背入"|"其他"。
- diagnosis: 入院诊断。
- allergyHistory: 过敏史，明确无过敏史填 "无"。
- temperature: 体温，摄氏度，保留一位小数。
- pulseRate: 脉率，次/分。
- heartRate: 心率，次/分。
- respiratoryRate: 呼吸频率，次/分。
- bloodPressure: 血压，格式 "120/80"，收缩压在前，舒张压在后。
- weight: 体重，kg，保留一位小数。
- height: 身高，cm。
- bloodSugar: 血糖，mmol/L，保留一位小数。
- stoolFrequency: 当天大便次数。

预留扩展字段：
- formType: 建议表单类型，可选 "vital_signs_form"|"admission_basic_form"|"nursing_record_form"|"patient_round_form"|"adverse_event_form"。
- recordTime 或 assessmentTime: 记录/评估时间，格式 "YYYY-MM-DD HH:MM:SS"。
- consciousness, skin, catheter, limbActivity, diet, sleep, urination, defecation。
- roundStates: 患者巡视状态数组，例如 ["在位","输液","吸氧"]。
- conditionRecord: 护理记录/巡视病情记录。
- eventType, eventLevel, eventSeverity, eventContent。

规则：
1. 如果文本是“打开/进入/去/跳转到 + 菜单”，输出 {"intend":"menu","target":"..."}。
2. 如果文本包含表单数据，输出 {"intend":"medical_info", ...字段...}。
3. 没有提到的字段输出 null 或省略，不要编造。
4. 仅输出 JSON，不要解释，不要 markdown。

菜单候选：
__MENU_TARGETS__

当前系统时间：__CURRENT_TIME__
""".strip()


class AgentManager:
    llm_client: Optional[Any] = None


models = AgentManager()


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "audio.amr").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "audio.amr"


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if sak_extract_json:
        value = sak_extract_json(text)
        if isinstance(value, dict):
            return value
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


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
        raise RuntimeError("Install ffmpeg to convert Android AMR/M4A uploads for Fun-ASR.")

    fd, output_name = tempfile.mkstemp(prefix="android_nursing_", suffix=".wav")
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


def prepare_audio_for_fun_asr(file_path: Path) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    metadata = probe_wav(file_path)
    if metadata.get("is_16k_mono_pcm16") and not FORCE_AUDIO_PREPROCESS:
        return file_path, None, metadata
    converted_path = convert_to_16k_wav(file_path)
    return converted_path, converted_path, {"source": metadata, "prepared": probe_wav(converted_path)}


def require_asr_sdk(asr_model: str) -> None:
    if dashscope is None:
        raise RuntimeError("dashscope SDK is not installed. Run: pip install -U dashscope")
    if not ASR_API_KEY:
        raise RuntimeError("Missing DASHSCOPE_API_KEY/QWEN_API_KEY/LLM_API_KEY for ASR.")
    if asr_model == FUN_ASR_REALTIME_MODEL_NAME and Recognition is None:
        raise RuntimeError("dashscope Fun-ASR SDK is not installed. Run: pip install -U dashscope")


def qwen_asr_output_to_text(output: Any) -> str:
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


def call_qwen_asr_flash_sync(file_path: Path) -> Dict[str, Any]:
    require_asr_sdk(QWEN_ASR_FLASH_MODEL_NAME)
    dashscope.base_http_api_url = ASR_HTTP_BASE_URL
    response = dashscope.MultiModalConversation.call(
        api_key=ASR_API_KEY,
        model=QWEN_ASR_FLASH_MODEL_NAME,
        messages=[{"role": "user", "content": [{"audio": f"file://{file_path.resolve()}"}]}],
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
        "request_id": getattr(response, "request_id", None),
        "model": QWEN_ASR_FLASH_MODEL_NAME,
        "raw": json_safe(output),
    }


def call_fun_asr_sync(file_path: Path) -> Dict[str, Any]:
    require_asr_sdk(FUN_ASR_REALTIME_MODEL_NAME)
    dashscope.api_key = ASR_API_KEY
    dashscope.base_websocket_api_url = ASR_WEBSOCKET_URL

    recognition = Recognition(
        model=FUN_ASR_REALTIME_MODEL_NAME,
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
        "request_id": recognition.get_last_request_id(),
        "model": FUN_ASR_REALTIME_MODEL_NAME,
        "raw": json_safe(sentence),
    }


async def transcribe_audio(file_path: Path, asr_model: Optional[str] = None) -> Dict[str, Any]:
    model_name = asr_model or ASR_MODEL_NAME
    if model_name == FUN_ASR_REALTIME_MODEL_NAME:
        prepared_path, generated_path, metadata = prepare_audio_for_fun_asr(file_path)
        try:
            result = await asyncio.to_thread(call_fun_asr_sync, prepared_path)
            result["audio_metadata"] = metadata
            return result
        finally:
            if generated_path:
                generated_path.unlink(missing_ok=True)

    result = await asyncio.to_thread(call_qwen_asr_flash_sync, file_path)
    result["audio_metadata"] = probe_wav(file_path)
    return result


def classify_intent_by_rule(text: str, intent_hint: Optional[str] = None) -> Dict[str, Any]:
    if intent_hint in {"menu", "medical_info"}:
        return {"intend": intent_hint, "target": None}

    normalized = text.strip()
    for target in MENU_TARGETS:
        if target in normalized and re.search(r"(打开|进入|跳转|切到|去|查看|到).{0,8}" + re.escape(target), normalized):
            return {"intend": "menu", "target": target}

    return {"intend": "medical_info", "target": None}


def extract_by_rule(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}

    temp_match = re.search(r"(?:体温|温度)[^\d零一二三四五六七八九十点度]{0,6}(\d{2}(?:\.\d)?|\d{2}度\d)", text)
    if temp_match:
        data["temperature"] = temp_match.group(1).replace("度", ".")

    bp_match = re.search(r"(?:血压|学压|高压低压)?[^\d]{0,6}(\d{2,3})\s*(?:/|比|、|和|的低压)\s*(\d{2,3})", text)
    if bp_match:
        data["bloodPressure"] = f"{bp_match.group(1)}/{bp_match.group(2)}"

    pulse_match = re.search(r"(?:脉搏|脉率)[^\d]{0,6}(\d{2,3})", text)
    if pulse_match:
        data["pulseRate"] = pulse_match.group(1)

    heart_match = re.search(r"(?:心率|心律)[^\d]{0,6}(\d{2,3})", text)
    if heart_match:
        data["heartRate"] = heart_match.group(1)

    resp_match = re.search(r"(?:呼吸|呼吸频率)[^\d]{0,6}(\d{1,2})", text)
    if resp_match:
        data["respiratoryRate"] = resp_match.group(1)

    weight_match = re.search(r"(?:体重)[^\d]{0,6}(\d{1,3}(?:\.\d)?)", text)
    if weight_match:
        data["weight"] = f"{float(weight_match.group(1)):.1f}"

    sugar_match = re.search(r"(?:血糖)[^\d]{0,6}(\d{1,2}(?:\.\d)?)", text)
    if sugar_match:
        data["bloodSugar"] = f"{float(sugar_match.group(1)):.1f}"

    stool_match = re.search(r"(?:大便|排便)[^\d]{0,6}(\d{1,2})\s*次", text)
    if stool_match:
        data["stoolFrequency"] = stool_match.group(1)

    if "无过敏" in text or "没有过敏" in text:
        data["allergyHistory"] = "无"

    return data


async def extract_form_data_with_llm(
    text: str,
    intent_hint: Optional[str] = None,
    patients: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rule_intent = classify_intent_by_rule(text, intent_hint)
    if rule_intent["intend"] == "menu":
        return {"intend": "menu", "target": rule_intent["target"]}

    if models.llm_client is None:
        data = extract_by_rule(text)
        data["intend"] = "medical_info"
        data.setdefault("formType", "vital_signs_form")
        return data

    patient_context = ""
    if patients:
        patient_context = "\n当前护士可选患者列表：" + json.dumps(patients, ensure_ascii=False)

    system_prompt = (
        ANDROID_EXTRACTOR_PROMPT
        .replace("__MENU_TARGETS__", "、".join(MENU_TARGETS))
        .replace("__CURRENT_TIME__", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    response = await models.llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text + patient_context},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    parsed = extract_json(content)
    if not parsed:
        raise RuntimeError(f"LLM did not return valid JSON: {content[:400]}")

    parsed.setdefault("intend", rule_intent["intend"])
    if parsed.get("intend") == "menu":
        parsed.setdefault("target", rule_intent["target"])
    return parsed


def normalize_sign_data(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in ANDROID_SIGN_DATA_KEYS + EXTRA_FORM_KEYS:
        if key in data:
            normalized[key] = data.get(key)

    normalized.setdefault("intend", "medical_info")
    normalized.setdefault("target", "")

    if normalized.get("intend") == "menu" and not normalized.get("target"):
        normalized["target"] = ""

    for key in ANDROID_SIGN_DATA_KEYS:
        normalized.setdefault(key, "" if key in {"intend", "target"} else None)
    return normalized


async def analyze_android_audio(
    file_path: Path,
    patient_id: int,
    intent_hint: Optional[str] = None,
    mock_text: Optional[str] = None,
    patients_json: Optional[str] = None,
    asr_model: Optional[str] = None,
) -> Dict[str, Any]:
    patients: Optional[List[Dict[str, Any]]] = None
    if patients_json:
        try:
            loaded = json.loads(patients_json)
            if isinstance(loaded, list):
                patients = [item for item in loaded if isinstance(item, dict)]
        except json.JSONDecodeError:
            raise ValueError("patients must be a JSON array")

    if mock_text:
        asr_result = {"text": mock_text, "model": "mock", "request_id": None}
    else:
        asr_result = await transcribe_audio(file_path, asr_model)

    asr_text = str(asr_result.get("text") or "").strip()
    if not asr_text:
        raise RuntimeError("ASR returned empty text")

    extracted = await extract_form_data_with_llm(asr_text, intent_hint, patients)
    sign_data = normalize_sign_data(extracted)
    sign_data["patientId"] = patient_id

    return {
        "patientId": patient_id,
        "asrText": asr_text,
        "signData": sign_data,
        "asr": asr_result,
    }


def android_success_response(result: Dict[str, Any]) -> Dict[str, Any]:
    sign_data = result["signData"]
    return {
        "code": 1,
        "msg": "识别成功",
        "data": {
            "patientId": result["patientId"],
            "asrText": result["asrText"],
            "signData": json.dumps(sign_data, ensure_ascii=False),
            "signDataObject": sign_data,
        },
    }


def android_error_response(message: str, code: int = 0) -> Dict[str, Any]:
    return {
        "code": code,
        "msg": message,
        "data": {
            "signData": json.dumps(
                {
                    "intend": "",
                    "target": "",
                },
                ensure_ascii=False,
            )
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[system] android nursing serve demo starting")
    print(f"[system] upload_dir={UPLOAD_DIR}")
    print(f"[system] asr_model={ASR_MODEL_NAME}, asr_key={'yes' if ASR_API_KEY else 'no'}")
    print(f"[system] llm_model={LLM_MODEL_NAME}, llm_key={'yes' if LLM_API_KEY else 'no'}")

    if AsyncOpenAI is not None and LLM_API_KEY:
        models.llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    elif AsyncOpenAI is None:
        print("[warn] openai SDK missing; will use rule-based extraction only")
    else:
        print("[warn] LLM API key missing; will use rule-based extraction only")

    yield
    print("\n[system] android nursing serve demo stopped")


app = FastAPI(title="Android Nursing Voice Serve Demo", lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "project_dir": str(PROJECT_DIR),
        "upload_dir": str(UPLOAD_DIR),
        "debug": DEBUG_MODE,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "asr": {
            "model": ASR_MODEL_NAME,
            "qwen_model": QWEN_ASR_FLASH_MODEL_NAME,
            "fun_model": FUN_ASR_REALTIME_MODEL_NAME,
            "sdk_installed": dashscope is not None,
            "fun_asr_installed": Recognition is not None,
            "has_api_key": bool(ASR_API_KEY),
        },
        "llm": {
            "model": LLM_MODEL_NAME,
            "base_url": LLM_BASE_URL,
            "sdk_installed": AsyncOpenAI is not None,
            "has_api_key": bool(LLM_API_KEY),
            "enabled": models.llm_client is not None,
        },
        "android": {
            "compatible_endpoint": "/agent/asr2txt/{patient_id}",
            "file_param": "file",
        },
    }


@app.post("/agent/asr2txt/{patient_id}")
async def android_asr_to_text_endpoint(
    patient_id: int,
    file: UploadFile = File(...),
    intent: Optional[str] = Form(default=None),
    mock_text: Optional[str] = Form(default=None),
    patients: Optional[str] = Form(default=None),
    asr_model: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    safe_name = sanitize_filename(file.filename or "android_audio.amr")
    upload_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}_{safe_name}"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = await analyze_android_audio(
            file_path=upload_path,
            patient_id=patient_id,
            intent_hint=intent,
            mock_text=mock_text,
            patients_json=patients,
            asr_model=asr_model,
        )
        if DEBUG_MODE:
            result["file"] = {
                "name": file.filename,
                "server_path": str(upload_path),
            }
        return android_success_response(result)
    except ValueError as exc:
        return android_error_response(str(exc), code=0)
    except Exception as exc:
        return android_error_response(f"识别失败: {exc}", code=0)
    finally:
        if not DEBUG_MODE:
            upload_path.unlink(missing_ok=True)


@app.post("/api/agent")
async def debug_agent_endpoint(
    file: UploadFile = File(...),
    patient_id: int = Form(default=0),
    intent: Optional[str] = Form(default=None),
    mock_text: Optional[str] = Form(default=None),
    patients: Optional[str] = Form(default=None),
    asr_model: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    safe_name = sanitize_filename(file.filename or "debug_audio.amr")
    upload_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}_{safe_name}"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = await analyze_android_audio(
            file_path=upload_path,
            patient_id=patient_id,
            intent_hint=intent,
            mock_text=mock_text,
            patients_json=patients,
            asr_model=asr_model,
        )
        result["androidResponse"] = android_success_response(result)
        if DEBUG_MODE:
            result["file"] = {
                "name": file.filename,
                "server_path": str(upload_path),
            }
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if not DEBUG_MODE:
            upload_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
