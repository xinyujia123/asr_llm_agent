import asyncio
import os
import json
import re
import time
import sys
import socket
import tempfile 
import uvicorn
import shutil
import uuid
from datetime import datetime
from http import HTTPStatus
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from openai import AsyncOpenAI
from pathlib import Path
import urllib3.util.connection as urllib3_connection

try:
    import dashscope
except ImportError:  # pragma: no cover - runtime dependency hint
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
        if dashscope is None:
            print("[系统启动] 警告：dashscope SDK 未安装，ASR 将不可用")
        elif not ASR_API_KEY:
            print("[系统启动] 警告：未配置 ASR API Key，ASR 将不可用")
        else:
            print(f"[系统启动] ASR 客户端就绪: model={ASR_MODEL_NAME}")
        print("[系统启动] 业务客户端初始化完成！")
    except Exception as e:
        print(f"[系统启动] 初始化失败: {e}")
        raise e
    
    yield
    
    print("\n[系统关闭] 释放资源...")

def require_asr_sdk():
    if dashscope is None:
        raise RuntimeError("dashscope SDK is not installed. Run: pip install -U dashscope")
    if not ASR_API_KEY:
        raise RuntimeError("Missing DASHSCOPE_API_KEY/QWEN_API_KEY/LLM_API_KEY for ASR.")


def qwen_asr_output_to_text(output):
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


def call_qwen_asr_sync(file_path: str):
    require_asr_sdk()
    dashscope.base_http_api_url = ASR_HTTP_BASE_URL

    response = dashscope.MultiModalConversation.call(
        api_key=ASR_API_KEY,
        model=ASR_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [{"audio": f"file://{Path(file_path).resolve()}"}],
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
    text = qwen_asr_output_to_text(output)
    return {
        "status": "success" if text else "empty",
        "full_text": text,
        "asr": {
            "text": text,
            "sentence": json.loads(json.dumps(output, ensure_ascii=False, default=str)),
            "request_id": getattr(response, "request_id", None),
            "usage": json.loads(json.dumps(getattr(response, "usage", {}) or {}, ensure_ascii=False, default=str)),
            "model": ASR_MODEL_NAME,
        },
    }


# ---------------- 工具函数：调用远程 ASR ----------------
async def call_remote_asr_by_path(file_path: str, hotwords: str = ""):
    """通过 DashScope ASR API 识别本地音频文件。hotwords 参数保留以兼容原调用点。"""
    try:
        return await asyncio.to_thread(call_qwen_asr_sync, file_path)
    except Exception as e:
        print(f"请求 ASR API 异常: {e}")
        return {"status": "error", "full_text": ""}

# --- 表单路由与 LLM 提取逻辑 ---
FORM_ROUTES = {
    "admission_assessment_form": {
        "name": "入院评估单",
        "aliases": ["入院评估单", "入院护理评估单", "入院护理评估", "入院评估"],
    },
    "nursing_record_form": {
        "name": "护理记录单",
        "aliases": ["护理记录单"],
    },
    "temperature_sheet_form": {
        "name": "体温单",
        "aliases": ["体温单"],
    },
}

FORM_ROUTER_PROMPT = """
你是护理文书语音指令的表单路由器。请根据 ASR 文本判断用户明确想转写哪些护理文书表单。

可选表单只有以下三个：
1. admission_assessment_form：入院评估单，也可能被说成入院护理评估单、入院护理评估、入院评估。
2. nursing_record_form：护理记录单。
3. temperature_sheet_form：体温单。

路由规则：
1. 只做表单路由，不提取字段。
2. 用户必须明确说出表单名称或表单名称的合理 ASR 近音，才选择该表单。
3. 允许纠正 ASR 对表单名的轻微识别错误、同音近音、漏字或多字，例如“体温单/体温表/体温但/体温胆”可判断为体温单，“护理纪录单/护理记入单”可判断为护理记录单。
4. 可以多选；只有一句话里明确提到多张表单名称时，才全部返回。
5. 如果只是说护理文书、记录生命体征、填写护理内容，但没有指向三张具体表单之一，返回空数组。
6. 不要因为文本中出现体温、血压、脉搏、呼吸、护理记录内容等字段信息就选择表单；字段值不是表单名称。
7. “体温三十九度/体温39度/测体温/体温偏高”只是字段或生命体征内容，不代表体温单；只有明确说“体温单/体温表/体温记录单”等表单名时才选择 temperature_sheet_form。

示例：
- ASR文本：入院评估单一床，体温三十九度。
  输出：{"form_ids": ["admission_assessment_form"]}
- ASR文本：一床体温三十九度。
  输出：{"form_ids": []}
- ASR文本：一床体温单，体温三十九度。
  输出：{"form_ids": ["temperature_sheet_form"]}

只输出合法 JSON：
{"form_ids": []}
"""

FORM_FIELDS = {
    "admission_assessment_form": [
            "patientName",
            "bedNumber",
            "admissionMethod",
            "admissionMethodOther",
            "temperature",
            "pulse",
            "respiration",
            "bloodPressureSys",
            "bloodPressureDia",
            "consciousness",
            "weight",
            "height",
            "randomBloodGlucose",
            "skinMucosaIntact",
            "limbActivity",
            "hasCatheter",
            "catheterInfo",
            "dietStatus",
            "sleepStatus",
            "urinationStatus",
            "defecationStatus",
            "notifyDoctor",
            "assessmentTime",
    ],
    "nursing_record_form": [
            "patientName",
            "bedNumber",
            "recordDate",
            "recordTime",
            "temperature",
            "pulse",
            "heartRate",
            "respiration",
            "bloodPressureSystolic",
            "bloodPressureDiastolic",
            "bloodOxygen",
            "consciousness",
            "oxygenTherapy",
            "oxygenFlow",
            "woundDressing",
            "deepVeinCatheter",
            "pupilLeft",
            "pupilRight",
            "reflexLeft",
            "reflexRight",
            "catheter",
            "catheterNormal",
            "catheterAbnormal",
            "intakeItem",
            "intakeAmount",
            "outputItem",
            "outputAmount",
            "outputCharacter",
            "basicCare",
            "observation",
            "nurseSignature",
    ],
    "temperature_sheet_form": [
            "patientName",
            "bedNumber",
            "recordDate",
            "intakeAmount",
            "outputUrine",
            "outputSputum",
            "outputDrainage",
            "outputVomit",
            "outputTotal",
            "respiratoryRate",
            "bloodPressureDia",
            "bloodPressureSys",
            "bodyWeight",
            "bedStatus",
            "skinTestResult",
            "notes",
            "temperatureTimeList",
    ],
}

FORM_PROMPTS = {
    "admission_assessment_form": """
你是专业严谨的护理文书入院评估单录入助手。请从 ASR 文本中只提取入院评估单可提交字段。

硬性规则：
1. 只输出一个合法 JSON 对象，不要 Markdown，不要解释。
2. 只按“必须按以下字段输出”中的字段输出；不要额外提取表单“基础信息/一般资料”栏中未定义的字段。
3. 文本没有明确说到的字段填 null，不要推断。
4. 数值字段只保留数值字符串，不要带单位。
5. 中文数字要转换成阿拉伯数字，例如三十九=39，八十五=85，七十=70，一百二十=120。
6. admissionMethod 输出编码字符串：步行=1，轮椅=2，平车=3，推床=4，背入=5，其他=99。
7. consciousness 输出编码字符串：清醒/清楚=1，嗜睡=2，模糊=3，昏睡=4，昏迷=5，浅昏迷=6，中昏迷=7，深昏迷=8，药物镇静状=9，麻醉未醒=10。
8. skinMucosaIntact 输出整数：完整=1，皮疹=2，出血点=3，脓疱=4，破损=5，溃疡=6，压力性损伤=7，造口=8，钉道=9，其他=99。
9. limbActivity、hasCatheter、dietStatus、sleepStatus、urinationStatus、defecationStatus 输出整数：正常/无=0，异常/有=1。
10. 如果提到“有导管”，将 hasCatheter 设为 1，并把导管名称/说明填入 catheterInfo。
11. 如果同一段 ASR 文本里包含多张表单，仍然只提取属于入院评估单的字段，不要因为出现其他表单就全部填 null。
12. 入院评估单表名附近或表名后明确提到的字段都应提取，例如“入院评估单：一床，体温三十九度”应提取 bedNumber 和 temperature。
13. “血氧/血氧饱和度”不是入院评估单字段，不要输出。
14. 示例：ASR 文本“入院评估单：一床，体温三十九度，血氧八十五。护理记录单：二床，血氧饱和度七十，意识状态清醒。”，入院评估单应提取一床对应患者、temperature="39"；血氧忽略；护理记录单部分不要提取到入院评估单。

必须按以下字段输出：
{
  "patientName": null,
  "bedNumber": null,
  "admissionMethod": null,
  "admissionMethodOther": null,
  "temperature": null,
  "pulse": null,
  "respiration": null,
  "bloodPressureSys": null,
  "bloodPressureDia": null,
  "consciousness": null,
  "weight": null,
  "height": null,
  "randomBloodGlucose": null,
  "skinMucosaIntact": null,
  "limbActivity": null,
  "hasCatheter": null,
  "catheterInfo": null,
  "dietStatus": null,
  "sleepStatus": null,
  "urinationStatus": null,
  "defecationStatus": null,
  "notifyDoctor": null,
  "assessmentTime": null
}
""",
    "nursing_record_form": """
你是专业严谨的护理文书护理记录单录入助手。请从 ASR 文本中只提取护理记录单字段。

硬性规则：
1. 只输出一个合法 JSON 对象，不要 Markdown，不要解释。
2. 只按“必须按以下字段输出”中的字段输出；不要额外提取表单“基础信息/一般资料”栏中未定义的字段。
3. 文本没有明确说到的字段填 null，不要推断。
4. 数值字段只保留数值字符串，不要带单位。
5. 中文数字要转换成阿拉伯数字，例如三十九=39，八十五=85，七十=70，一百二十=120。
6. 血压如果说“一百二十/八十”或“一百二十八十”，收缩压填 bloodPressureSystolic，舒张压填 bloodPressureDiastolic；如果只说“血压”但没有数值，血压字段填 null。
7. “血氧/血氧饱和度”填 bloodOxygen，例如“血氧饱和度七十”填 "70"。
8. consciousness 是文本字段，例如“意识状态清醒”填 "清醒"。
9. 管道状态：正常则 catheterNormal=true、catheterAbnormal=false；异常则 catheterNormal=false、catheterAbnormal=true；没说则都为 null。
10. recordDate/recordTime 只在文本明确提到护理时间时填写。
11. 如果同一段 ASR 文本里包含多张表单，仍然只提取属于护理记录单的字段，不要因为出现其他表单就全部填 null。
12. 护理记录单表名附近或表名后明确提到的字段都应提取，例如“护理记录单：二床，血氧饱和度七十，意识状态清醒”应提取 bedNumber、bloodOxygen、consciousness。
13. 示例：ASR 文本“入院评估单：一床，体温三十九度，血氧八十五。护理记录单：二床，血压，血氧饱和度七十，意识状态清醒。”，护理记录单应提取二床对应患者、bloodOxygen="70"、consciousness="清醒"；“血压”没有具体数值时血压字段为 null；入院评估单部分不要提取到护理记录单。

必须按以下字段输出：
{
  "patientName": null,
  "bedNumber": null,
  "recordDate": null,
  "recordTime": null,
  "temperature": null,
  "pulse": null,
  "heartRate": null,
  "respiration": null,
  "bloodPressureSystolic": null,
  "bloodPressureDiastolic": null,
  "bloodOxygen": null,
  "consciousness": null,
  "oxygenTherapy": null,
  "oxygenFlow": null,
  "woundDressing": null,
  "deepVeinCatheter": null,
  "pupilLeft": null,
  "pupilRight": null,
  "reflexLeft": null,
  "reflexRight": null,
  "catheter": null,
  "catheterNormal": null,
  "catheterAbnormal": null,
  "intakeItem": null,
  "intakeAmount": null,
  "outputItem": null,
  "outputAmount": null,
  "outputCharacter": null,
  "basicCare": null,
  "observation": null,
  "nurseSignature": null
}
""",
    "temperature_sheet_form": """
你是专业严谨的护理文书体温单录入助手。请从 ASR 文本中只提取体温单字段。

硬性规则：
1. 只输出一个合法 JSON 对象，不要 Markdown，不要解释。
2. 只按“必须按以下字段输出”中的字段输出；不要额外提取表单“基础信息/一般资料”栏中未定义的字段。
3. 文本没有明确说到的字段填 null，不要推断。
4. 数值字段只保留数值字符串，不要带单位。
5. 中文数字要转换成阿拉伯数字，例如三十九=39，八十五=85，七十=70，一百二十=120。
6. bedStatus 只能输出：卧床、轮椅、平车、拒测、外出；没说则 null。
7. temperatureTimeList 是数组。只记录文本明确提到的 02:00、06:00、10:00、14:00、18:00、22:00 或其他明确时间点的体温/脉搏。
8. temperatureType 输出整数：腋温=1，肛温=2，口温=3，不升=4，外出=5，请假=6，特殊值=7，耳温=8。没有说明体温类型时填 null。
9. 如果只说“体温三十七度二、脉搏八十”，但没有时间点，可以生成一条 recordTime=null 的 temperatureTimeList 记录。

必须按以下字段输出：
{
  "patientName": null,
  "bedNumber": null,
  "recordDate": null,
  "intakeAmount": null,
  "outputUrine": null,
  "outputSputum": null,
  "outputDrainage": null,
  "outputVomit": null,
  "outputTotal": null,
  "respiratoryRate": null,
  "bloodPressureDia": null,
  "bloodPressureSys": null,
  "bodyWeight": null,
  "bedStatus": null,
  "skinTestResult": null,
  "notes": null,
  "temperatureTimeList": [
    {
      "recordTime": null,
      "temperature": null,
      "temperatureType": null,
      "pulse": null
    }
  ]
}
""",
}


def extract_first_json_object(content: str):
    if not content:
        return {}
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        print(f"JSON 解析失败: {content}")
        return {}


def clean_form_json(data, keys):
    if not isinstance(data, dict):
        data = {}
    cleaned = {}
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            cleaned[key] = stripped if stripped and stripped.lower() not in {"none", "null", "undefined", "未知"} else None
        else:
            cleaned[key] = value
    return cleaned


def normalize_form_ids(raw_form_ids):
    if isinstance(raw_form_ids, str):
        candidates = re.split(r"[,，、\s]+", raw_form_ids)
    elif isinstance(raw_form_ids, list):
        candidates = raw_form_ids
    else:
        candidates = []

    form_ids = []
    for item in candidates:
        form_id = str(item or "").strip()
        if form_id in FORM_ROUTES and form_id not in form_ids:
            form_ids.append(form_id)
    return form_ids


def normalize_patient_candidates(patients: str):
    if not patients:
        return []
    try:
        raw_patients = json.loads(patients)
    except Exception:
        return []
    if not isinstance(raw_patients, list):
        return []

    candidates = []
    for item in raw_patients:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("patientName") or "").strip()
        bed_number = str(item.get("bedNumber") or item.get("bedNo") or "").strip()
        patient_id = item.get("patientId") or item.get("id")
        if not name and not bed_number:
            continue
        candidates.append(
            {
                "patientId": patient_id,
                "name": name,
                "bedNumber": bed_number,
            }
        )
    return candidates


async def route_forms_async(text: str):
    try:
        response = await models.llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": FORM_ROUTER_PROMPT},
                {"role": "user", "content": f"ASR文本：{text}"},
            ],
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content
        print(f"LLM 表单路由 Response: {content}")
        return normalize_form_ids(extract_first_json_object(content).get("form_ids"))
    except Exception as e:
        print(f"LLM 表单路由 Error: {e}")
        return []



def patient_context_prompt(patient_candidates):
    if not patient_candidates:
        return """

患者候选列表：未提供。
请从 ASR 文本中如实提取 patientName 和 bedNumber；如果无法明确判断，填 null。
"""

    lines = []
    for patient in patient_candidates:
        label = f"- 姓名：{patient.get('name') or '未知'}，床号：{patient.get('bedNumber') or '未知'}"
        if patient.get("patientId") is not None:
            label += f"，patientId：{patient.get('patientId')}"
        lines.append(label)
    return """

患者候选列表如下，只能从候选列表中选择患者：
{patients}

患者判断规则：
1. 根据 ASR 文本中的姓名、床号、称呼或上下文判断对应患者。
2. 输出 JSON 必须包含 patientName 和 bedNumber。
3. patientName 必须使用候选列表中的姓名，bedNumber 必须使用候选列表中的床号。
4. 如果 ASR 文本没有明确指向某个候选患者，patientName 和 bedNumber 都填 null。
5. 不要凭空编造候选列表之外的患者姓名或床号。
6. 中文数字床号和候选床号要等价匹配：一床=1床/01床，二床=2床/02床，三床=3床/03床，以此类推。
7. 如果当前表单名附近或表名后出现“X床”，视为当前表单对应患者。
""".format(patients="\n".join(lines))


def prompt_with_current_time(prompt: str, patient_candidates=None):
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return prompt + patient_context_prompt(patient_candidates or []) + f"\n当前系统时间：{formatted_time}\n"


async def extract_form_info_async(text: str, form_id: str, patient_candidates=None):
    form = FORM_ROUTES[form_id]
    try:
        response = await models.llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_with_current_time(FORM_PROMPTS[form_id], patient_candidates)},
                {"role": "user", "content": f"ASR文本：{text}"},
            ],
            temperature=0.1,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content
        print(f"LLM {form['name']} Response: {content}")
        return {
            "id": form_id,
            "name": form["name"],
            "data": clean_form_json(extract_first_json_object(content), FORM_FIELDS[form_id]),
        }
    except Exception as e:
        print(f"LLM {form['name']} Error: {e}")
        return {
            "id": form_id,
            "name": form["name"],
            "data": clean_form_json({}, FORM_FIELDS[form_id]),
            "error": str(e),
        }


async def extract_forms_async(text: str, patients: str = ""):
    form_ids = await route_forms_async(text)
    if not form_ids:
        return {
            "intend": "others",
            "forms": [],
            "form_ids": [],
            "form_names": [],
            "message": "未明确提到入院评估单、护理记录单或体温单，未执行表单转写",
        }

    patient_candidates = normalize_patient_candidates(patients)
    forms = await asyncio.gather(
        *(extract_form_info_async(text, form_id, patient_candidates) for form_id in form_ids)
    )
    return {
        "intend": "form_route",
        "forms": forms,
        "form_ids": form_ids,
        "form_names": [FORM_ROUTES[form_id]["name"] for form_id in form_ids],
    }


ANDROID_FORM_DTOS = {
    "admission_assessment_form": {
        "dto": "InHospitalCommitReq",
    },
    "nursing_record_form": {
        "dto": "DocNursingRecordReq",
    },
    "temperature_sheet_form": {
        "dto": "DocTemperatureReq",
    },
}


def build_android_demo_response(agent_result, patient_id: int):
    data = agent_result.get("data") or {}
    android_forms = []
    for form in data.get("forms") or []:
        form_id = form.get("id")
        dto_info = ANDROID_FORM_DTOS.get(form_id, {})
        form_data = dict(form.get("data") or {})
        if patient_id:
            form_data["patientId"] = patient_id
        android_forms.append(
            {
                "id": form_id,
                "name": form.get("name"),
                "dto": dto_info.get("dto"),
                "action": "fill_form_only",
                "formData": form_data,
                "requestBody": form_data,
            }
        )

    sign_data = {
        "intend": "form_fill" if android_forms else "others",
        "demoOnly": True,
        "action": "fill_form_only",
        "raw_text": agent_result.get("raw_text", ""),
        "form_ids": data.get("form_ids", []),
        "form_names": data.get("form_names", []),
        "forms": android_forms,
        "message": data.get("message"),
    }

    return {
        "code": 1,
        "msg": "success",
        "data": {
            "demoOnly": True,
            "action": "fill_form_only",
            "patientId": patient_id,
            "rawText": agent_result.get("raw_text", ""),
            "signData": json.dumps(sign_data, ensure_ascii=False),
            "forms": android_forms,
            "formIds": data.get("form_ids", []),
            "formNames": data.get("form_names", []),
            "message": data.get("message"),
        },
    }


# ---------------- API 接口 ----------------
app = FastAPI(title="Medical Agent (Decoupled Version)", lifespan=lifespan)

@app.post("/api/agent")
async def analyze_audio_endpoint(file: UploadFile = File(...), patients: str = Form("")):
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
        os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
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
        
        # 3. 全文识别
        asr_start_time = time.time()
        form_hotwords = "护理文书 入院评估单 入院护理评估单 护理记录单 体温单"
        full_res = await call_remote_asr_by_path(full_path, form_hotwords)
        
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
        # 4. 表单路由与 LLM 信息提取
        llm_start_time = time.time()
        result_data = {}
        if full_text:
            result_data = await extract_forms_async(full_text, patients)
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


@app.post("/agent/asr2txt/{patient_id}")
async def android_asr2txt_endpoint(patient_id: int, file: UploadFile = File(...), patients: str = Form("")):
    agent_result = await analyze_audio_endpoint(file, patients)
    return build_android_demo_response(agent_result, patient_id)


@app.post("/agent/asr2txt")
async def android_asr2txt_demo_endpoint(file: UploadFile = File(...), patients: str = Form("")):
    agent_result = await analyze_audio_endpoint(file, patients)
    return build_android_demo_response(agent_result, 0)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
