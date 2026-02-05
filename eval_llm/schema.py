from typing import Optional, Dict, Any
from jsonschema import validate, ValidationError
import json

# 根据 prompts.py 定义的 22 个字段
MEDICAL_JSON_SCHEMA = {
    "education": "文化程度（选项：文盲、小学、初中、高中或中专、大专以上、未入学）",
    "ethnicity": "民族",
    "admissionTime": "入院时间 (YYYY-MM-DD HH:MM:SS)",
    "admissionMethod": "入院方式（选项：步行、轮椅、平车、推床、背入、其他）",
    "diagnosis": "入院诊断",
    "allergyHistory": "过敏史",
    "temperature": "体温 (℃)",
    "pulseRate": "脉率",
    "heartRate": "心率",
    "respiratoryRate": "呼吸频率",
    "bloodPressure": "血压（格式：高压/低压）",
    "weight": "体重 (KG)",
    "height": "身高 (cm)",
    "bloodSugar": "随机血糖 (mmol/L)",
    "consciousness": "神志（选项：清楚、嗜睡、模糊、昏睡、昏迷、浅昏迷、中昏迷、深昏迷、药物镇静状、麻醉未醒）",
    "skin": "皮肤黏膜（选项：完整、皮疹、出血点、脓疱、破损、溃疡、压力性损伤、造口、钉道、其他）",
    "limbActivity": "肢体活动（选项：正常、异常）",
    "catheter": "导管（选项：无 或 具体名称）",
    "diet": "饮食（选项：正常、异常）",
    "sleep": "睡眠（选项：正常、异常）",
    "urination": "排尿（选项：正常、异常）",
    "defecation": "排便（选项：正常、异常）"
}

EXPECTED_KEYS = list(MEDICAL_JSON_SCHEMA.keys())
UNCERTAIN_KEYS = ["diagnosis", "allergyHistory", "catheter"]
CERTAIN_KEYS = list(set(EXPECTED_KEYS) - set(UNCERTAIN_KEYS))

# 定义标准 Schema 规范
MEDICAL_JSON_SCHEMA_E1 = {
    "type": "object",
    "properties": {
        "education": {"type": ["string", "null"], "enum": ["文盲", "小学", "初中", "高中或中专", "大专以上", "未入学", None]},
        "ethnicity": {"type": ["string", "null"]},
        "admissionTime": {
            "type": ["string", "null"], 
            "pattern": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$" # 匹配 YYYY-MM-DD HH:MM:SS
        },
        "admissionMethod": {"type": ["string", "null"], "enum": ["步行", "轮椅", "平车", "推床", "背入", "其他", None]},
        "diagnosis": {"type": ["string", "null"]},
        "allergyHistory": {"type": ["string", "null"]},
        "temperature": {"type": ["string", "null"]},
        "pulseRate": {"type": ["string", "null"]},
        "heartRate": {"type": ["string", "null"]},
        "respiratoryRate": {"type": ["string", "null"]},
        "bloodPressure": {
            "type": ["string", "null"],
            "pattern": r"^\d+/\d+$" # 匹配 高压/低压 格式
        },
        "weight": {"type": ["string", "null"]},
        "height": {"type": ["string", "null"]},
        "bloodSugar": {"type": ["string", "null"]},
        "consciousness": {"type": ["string", "null"], "enum": ["清楚", "嗜睡", "模糊", "昏睡", "昏迷", "浅昏迷", "中昏迷", "深昏迷", "药物镇静状", "麻醉未醒", None]},
        "skin": {"type": ["string", "null"]},
        "limbActivity": {"type": ["string", "null"], "enum": ["正常", "异常", None]},
        "catheter": {"type": ["string", "null"]},
        "diet": {"type": ["string", "null"], "enum": ["正常", "异常", None]},
        "sleep": {"type": ["string", "null"], "enum": ["正常", "异常", None]},
        "urination": {"type": ["string", "null"], "enum": ["正常", "异常", None]},
        "defecation": {"type": ["string", "null"], "enum": ["正常", "异常", None]}
    },
    # 强制要求所有字段必须存在（即使值为 null）
    "required": [
        "education", "ethnicity", "admissionTime", "admissionMethod", "diagnosis", 
        "allergyHistory", "temperature", "pulseRate", "heartRate", "respiratoryRate", 
        "bloodPressure", "weight", "height", "bloodSugar", "consciousness", "skin", 
        "limbActivity", "catheter", "diet", "sleep", "urination", "defecation"
    ]
}

def json_schema_check(json_data: Dict[str, Any]) -> bool:
    """
    检查 JSON 数据是否符合指定的 JSON Schema 规范。

    :param json_data: 待检查的 JSON 数据，作为字典传入。
    :return: 如果 JSON 数据符合 Schema 规范，返回 True；否则返回 False。
    """
    try:
        validate(instance=json_data, schema=MEDICAL_JSON_SCHEMA_E1)
        return True
    except ValidationError:
        return False

if __name__ == "__main__":
    # 测试数据
    jsonl_path = "/workspace/audio_llm_agent/eval_llm/benchmark/output_kimi_think_90.jsonl"
    # 读取 JSONL 文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检查每一行是否符合 Schema 规范
    valid_count = 0
    for line in lines:
        json_data = json.loads(line.strip())
        try:
            validate(instance=json_data["cleaned_json"], schema=MEDICAL_JSON_SCHEMA_E1)
            valid_count += 1
        except ValidationError as e:
            print(f"ValidationError: {e}")
    
    print(f"Total lines: {len(lines)}, Valid lines: {valid_count}")
