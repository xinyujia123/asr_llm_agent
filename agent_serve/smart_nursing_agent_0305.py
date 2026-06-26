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
你是专业严谨的护理文书录入助手。你的任务是从用户的语音识别文本(ASR)中，只提取【入院评估单】可提交字段。

### 【当前系统时间】{CURRENT_SYS_TIME}
若 ASR 未提及具体评估时间，assessmentTime 默认输出当前系统时间。

### 【字段枚举与映射规范】
* patientName：患者姓名。若提供患者候选列表，必须从候选列表中选择；不能编造。
* bedNumber：床号。中文数字床号需转换，如“一床”可输出候选中的“1床/01床”。
* admissionMethod：入院方式编码字符串。步行="1"；轮椅="2"；平车="3"；推床="4"；背入="5"；其他="99"。120/救护车/担架等映射为"99"，并在 admissionMethodOther 填具体方式。
* admissionMethodOther：仅 admissionMethod="99" 时输出具体入院方式。
* temperature：体温，单位℃，只输出数值字符串，保留一位小数。
* pulse：脉搏/脉率，次/分，只输出整数数字字符串。
* respiration：呼吸频率，次/分，只输出整数数字字符串。ASR 中“夫妻/服气”等需结合语境纠正为呼吸。
* bloodPressureSys：收缩压/高压，mmHg，只输出整数数字字符串。
* bloodPressureDia：舒张压/低压，mmHg，只输出整数数字字符串。血压连读如“一百一七十/一百一十八七十六”需拆为合理高低压。
* consciousness：神志编码字符串。清醒/清楚="1"；嗜睡="2"；模糊="3"；昏睡="4"；昏迷="5"；浅昏迷="6"；中昏迷="7"；深昏迷="8"；药物镇静状="9"；麻醉未醒="10"。严禁通过“能交流”等行为推断为清醒。
* weight：体重，kg，只输出数值字符串，保留一位小数。
* height：身高，cm，只输出整数数字字符串。
* randomBloodGlucose：随机血糖，mmol/L，只输出数值字符串，保留一位小数。
* skinMucosaIntact：皮肤黏膜编码字符串。完整="1"；皮疹="2"；出血点="3"；脓疱="4"；破损="5"；溃疡="6"；压力性损伤/压疮="7"；造口="8"；钉道="9"；其他异常="99"。有异常时严禁输出完整。
* limbActivity：肢体活动编码字符串。正常="0"；异常="1"。局部异常即为异常。
* hasCatheter：导管编码字符串。明确无导管="0"；明确有导管="1"。局部否定不代表全局无，如“没插胃管”不能输出"0"。
* catheterInfo：有导管时输出具体导管名称/说明；打点滴映射为静脉通路，尿袋映射为导尿管；吸氧面罩不算导管。
* dietStatus：饮食编码字符串。正常="0"；异常="1"。单纯“食欲不振”等日常情况不能判定异常。
* sleepStatus：睡眠编码字符串。正常="0"；异常="1"。单纯“睡得晚”等日常情况不能判定异常。
* urinationStatus：排尿编码字符串。正常="0"；异常="1"。
* defecationStatus：排便编码字符串。正常="0"；异常="1"。
* notifyDoctor：通知医生编码字符串。明确无需/未通知="0"；明确已通知/通知医生="1"。
* assessmentTime：评估时间，格式 "YYYY-MM-DD HH:MM:SS"。若只提及时刻，默认日期为当前系统时间日期；若有昨天/今天等相对词，结合当前系统时间推算。

### 【兜底规则】
1. ASR 未提及则不输出该 key；assessmentTime 除外，必须输出。
2. ASR 提及但不属于以上字段的内容不要输出，例如血氧不是入院评估单字段。
3. 如果一段 ASR 同时包含多张表单，只提取【入院评估单】表名附近或表名后的内容；不要提取护理记录单、体温单片段。
4. 同一字段多次出现时取最后一次；区间值取平均值。
5. 明显违背生理极限的离谱数值视为 ASR 错误，不输出该 key。
6. 所有编码和数值均输出字符串，不输出数字类型。

### 【核心处理原则】
1. 允许做医疗语境 ASR 纠错，如“学压”->“血压”，“麦博”->“脉搏”。
2. 禁止自行根据医疗常识推断，只能按 ASR 明确信息和字段映射提取。
3. 时间口语需转 24 小时制，如“上午八点”->"08:00:00"。

### 【输出要求】
仅输出纯 JSON 对象，严禁包含解释性文字、Markdown 或代码块。

### 【示例 (Few-Shot)】
假设当前系统时间为：2026-02-01 10:00:00

输入：
“入院评估单，一床王女士，今天早上八点测体温三十六度四，麦博七十二，夫妻十八，血压一百一十八七十六，神志清醒，皮肤完整，没有导管。”

输出：
{
  "patientName": "王女士",
  "bedNumber": "1床",
  "temperature": "36.4",
  "pulse": "72",
  "respiration": "18",
  "bloodPressureSys": "118",
  "bloodPressureDia": "76",
  "consciousness": "1",
  "skinMucosaIntact": "1",
  "hasCatheter": "0",
  "assessmentTime": "2026-02-01 08:00:00"
}

输入：
“入院评估单二床，推床入院，皮肤有压疮，接了尿袋在打点滴。护理记录单二床血氧八十八。”

输出：
{
  "bedNumber": "2床",
  "admissionMethod": "4",
  "skinMucosaIntact": "7",
  "hasCatheter": "1",
  "catheterInfo": "导尿管、静脉通路",
  "assessmentTime": "2026-02-01 10:00:00"
}
""",
    "nursing_record_form": """
你是专业严谨的护理文书录入助手。你的任务是从用户的语音识别文本(ASR)中，只提取【护理记录单】可提交字段。

### 【当前系统时间】{CURRENT_SYS_TIME}
若 ASR 未提及具体记录日期/时间，recordDate 默认输出当前系统日期，recordTime 默认输出当前系统时间。

### 【字段枚举与映射规范】
* patientName：患者姓名。若提供患者候选列表，必须从候选列表中选择；不能编造。
* bedNumber：床号。中文数字床号需转换，如“一床”可输出候选中的“1床/01床”。
* recordDate：记录日期，格式 "YYYY-MM-DD"。若只提及昨天/今天等相对词，结合当前系统时间推算。
* recordTime：记录时间，格式 "HH:MM:SS"。口语时间需转 24 小时制，如“下午三点半”->"15:30:00"。
* temperature：体温，单位℃，只输出数值字符串，保留一位小数。
* pulse：脉搏/脉率，次/分，只输出整数数字字符串。
* heartRate：心率，次/分，只输出整数数字字符串。严格区分心率和脉率。
* respiration：呼吸频率，次/分，只输出整数数字字符串。ASR 中“夫妻/服气”等需结合语境纠正为呼吸。
* bloodPressureSystolic：收缩压/高压，只输出整数数字字符串。
* bloodPressureDiastolic：舒张压/低压，只输出整数数字字符串。血压连读如“一百一七十/一百一十八七十六”需拆为合理高低压。
* bloodOxygen：血氧/血氧饱和度，%，只输出 0-100 范围内的整数数字字符串；明显超过 100 不输出。
* consciousness：意识/神志文本，如“清醒”“嗜睡”“模糊”“昏迷”等。严禁通过行为推断。
* oxygenTherapy：氧疗方式文本，如鼻导管吸氧、面罩吸氧、高流量吸氧。
* oxygenFlow：氧流量，L/min，只输出数值字符串；如“二升每分”输出"2"。
* woundDressing：伤口/敷料情况文本，只提取明确描述。
* deepVeinCatheter：深静脉置管情况文本，只提取明确描述。
* pupilLeft：左侧瞳孔大小/描述。
* pupilRight：右侧瞳孔大小/描述。
* reflexLeft：左侧光反射/瞳孔反射描述。
* reflexRight：右侧光反射/瞳孔反射描述。
* catheter：导管名称或说明。打点滴映射为静脉通路，尿袋映射为导尿管；吸氧面罩不算导管。
* catheterNormal / catheterAbnormal：管道状态布尔值。明确管道正常时 catheterNormal=true 且 catheterAbnormal=false；明确异常时 catheterNormal=false 且 catheterAbnormal=true；未提及状态则不输出二者。
* intakeItem：入量项目，如饮水、输液、鼻饲。
* intakeAmount：入量，ml，只输出数值字符串。
* outputItem：出量项目，如尿量、引流量、呕吐物。
* outputAmount：出量，ml，只输出数值字符串。
* outputCharacter：出量性状文本，如清亮、血性、黄色。
* basicCare：基础护理措施文本，只提取明确描述。
* observation：病情观察/护理观察文本，只提取明确描述。
* nurseSignature：护士签名/记录人。

### 【兜底规则】
1. ASR 未提及则不输出该 key；recordDate、recordTime 除外，必须按当前系统时间输出。
2. ASR 提及但不属于以上字段的内容不要输出。
3. 如果一段 ASR 同时包含多张表单，只提取【护理记录单】表名附近或表名后的内容；不要提取入院评估单、体温单片段。
4. 只有“血压”没有具体数值时，不输出 bloodPressureSystolic / bloodPressureDiastolic。
5. 同一字段多次出现时取最后一次；区间值取平均值。
6. 明显违背生理极限的离谱数值视为 ASR 错误，不输出该 key。
7. 除 catheterNormal/catheterAbnormal 必须为布尔值 true/false 外，其余值均输出字符串。

### 【核心处理原则】
1. 允许做医疗语境 ASR 纠错，如“学压”->“血压”，“麦博”->“脉搏”。
2. 禁止自行根据医疗常识推断，只能按 ASR 明确信息和字段映射提取。
3. 局部否定不代表全局无，如“没有胃管”不能推出没有其他导管。

### 【输出要求】
仅输出纯 JSON 对象，严禁包含解释性文字、Markdown 或代码块。

### 【示例 (Few-Shot)】
假设当前系统时间为：2026-02-01 10:00:00

输入：
“护理记录单二床陈女士，今天早上八点，体温三十六度四，脉搏七十二，心率七十，呼吸十八，血压一百一十八七十六，血氧九十九，意识清醒，鼻导管吸氧二升。”

输出：
{
  "patientName": "陈女士",
  "bedNumber": "2床",
  "recordDate": "2026-02-01",
  "recordTime": "08:00:00",
  "temperature": "36.4",
  "pulse": "72",
  "heartRate": "70",
  "respiration": "18",
  "bloodPressureSystolic": "118",
  "bloodPressureDiastolic": "76",
  "bloodOxygen": "99",
  "consciousness": "清醒",
  "oxygenTherapy": "鼻导管吸氧",
  "oxygenFlow": "2"
}

输入：
“入院评估单一床体温三十九度。护理记录单二床，血压，血氧饱和度七十，意识状态清醒，导尿管通畅。”

输出：
{
  "bedNumber": "2床",
  "recordDate": "2026-02-01",
  "recordTime": "10:00:00",
  "bloodOxygen": "70",
  "consciousness": "清醒",
  "catheter": "导尿管",
  "catheterNormal": true,
  "catheterAbnormal": false
}
""",
    "temperature_sheet_form": """
你是专业严谨的护理文书录入助手。你的任务是从用户的语音识别文本(ASR)中，只提取【体温单】可提交字段。

### 【当前系统时间】{CURRENT_SYS_TIME}
若 ASR 未提及记录日期，recordDate 默认输出当前系统日期。

### 【字段枚举与映射规范】
* patientName：患者姓名。若提供患者候选列表，必须从候选列表中选择；不能编造。
* bedNumber：床号。中文数字床号需转换，如“一床”可输出候选中的“1床/01床”。
* recordDate：记录日期，格式 "YYYY-MM-DD"。若只提及昨天/今天等相对词，结合当前系统时间推算。
* intakeAmount：总入量，ml，只输出数值字符串。
* outputUrine：尿量，ml，只输出数值字符串。
* outputSputum：痰量，ml，只输出数值字符串。
* outputDrainage：引流量，ml，只输出数值字符串。
* outputVomit：呕吐量，ml，只输出数值字符串。
* outputTotal：总出量，ml，只在 ASR 明确说“总出量/合计/总共”时输出，不自行相加。
* respiratoryRate：呼吸频率，次/分，只输出整数数字字符串。ASR 中“夫妻/服气”等需结合语境纠正为呼吸。
* bloodPressureDia：舒张压/低压，只输出整数数字字符串。
* bloodPressureSys：收缩压/高压，只输出整数数字字符串。血压连读需拆为合理高低压。
* bodyWeight：体重，kg，只输出数值字符串，保留一位小数。
* bedStatus：床位/测量状态，仅限 "卧床" | "轮椅" | "平车" | "拒测" | "外出"。
* skinTestResult：皮试结果文本，如青霉素阴性/阳性。未明确药物或结果时不输出。
* notes：备注，只提取 ASR 明确要求写入备注的内容。
* temperatureTimeList：体温时点数组。仅记录明确提到的体温/脉搏时点；可包含 02:00、06:00、10:00、14:00、18:00、22:00 或其他明确时间。
  - recordTime：格式必须为 "HH:MM"，不要输出秒；如果只说体温/脉搏但没有时点，可输出 null。
  - temperature：体温，单位℃，保留一位小数。
  - temperatureType：体温类型编码字符串。腋温="1"；肛温="2"；口温="3"；不升="4"；外出="5"；请假="6"；特殊值="7"；耳温="8"。未说明体温类型则不输出。
  - pulse：脉搏/脉率，次/分，只输出整数数字字符串。

### 【兜底规则】
1. ASR 未提及则不输出该 key；recordDate 除外，必须输出。
2. ASR 提及但不属于以上字段的内容不要输出。
3. 如果一段 ASR 同时包含多张表单，只提取【体温单】表名附近或表名后的内容；不要提取入院评估单、护理记录单片段。
4. 没有任何体温或脉搏时点信息时，不输出 temperatureTimeList。
5. 同一字段多次出现时取最后一次；区间值取平均值。
6. 明显违背生理极限的离谱数值视为 ASR 错误，不输出该 key。
7. 所有编码和数值均输出字符串；temperatureTimeList 是数组。

### 【核心处理原则】
1. 允许做医疗语境 ASR 纠错，如“学压”->“血压”，“麦博”->“脉搏”。
2. 禁止自行根据医疗常识推断，只能按 ASR 明确信息和字段映射提取。
3. 时间口语需转 24 小时制且不带秒，如“下午两点”->"14:00"。

### 【输出要求】
仅输出纯 JSON 对象，严禁包含解释性文字、Markdown 或代码块。

### 【示例 (Few-Shot)】
假设当前系统时间为：2026-02-01 10:00:00

输入：
“体温单三床，今天上午十点腋温三十六度七，脉搏八十，呼吸十八，血压一百二十八十，体重六十公斤。”

输出：
{
  "bedNumber": "3床",
  "recordDate": "2026-02-01",
  "respiratoryRate": "18",
  "bloodPressureSys": "120",
  "bloodPressureDia": "80",
  "bodyWeight": "60.0",
  "temperatureTimeList": [
    {
      "recordTime": "10:00",
      "temperature": "36.7",
      "temperatureType": "1",
      "pulse": "80"
    }
  ]
}

输入：
“护理记录单二床血氧九十九。体温单二床，下午两点体温三十七度二，麦博八十二，尿量五百，总出量六百。”

输出：
{
  "bedNumber": "2床",
  "recordDate": "2026-02-01",
  "outputUrine": "500",
  "outputTotal": "600",
  "temperatureTimeList": [
    {
      "recordTime": "14:00",
      "temperature": "37.2",
      "pulse": "82"
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


def normalize_temperature_time_list(value):
    if not isinstance(value, list):
        return value
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized_item = dict(item)
        record_time = normalized_item.get("recordTime")
        if isinstance(record_time, str):
            stripped = record_time.strip()
            match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", stripped)
            if match:
                normalized_item["recordTime"] = f"{int(match.group(1)):02d}:{match.group(2)}"
            elif stripped.lower() in {"none", "null", "undefined", "未知"}:
                normalized_item["recordTime"] = None
            else:
                normalized_item["recordTime"] = stripped
        normalized.append(normalized_item)
    return normalized


def clean_form_json(data, keys):
    if not isinstance(data, dict):
        data = {}
    cleaned = {}
    for key in keys:
        value = data.get(key)
        if key == "temperatureTimeList":
            cleaned[key] = normalize_temperature_time_list(value)
            continue
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
    if "{CURRENT_SYS_TIME}" in prompt:
        prompt = prompt.replace("{CURRENT_SYS_TIME}", formatted_time)
        current_time_tail = ""
    else:
        current_time_tail = f"\n当前系统时间：{formatted_time}\n"
    return prompt + patient_context_prompt(patient_candidates or []) + current_time_tail


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
