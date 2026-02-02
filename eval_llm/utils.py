import re
import json
from typing import List, Dict, Any
from jsonschema import validate, ValidationError

def parse_nurse_scripts(file_path: str) -> List[Dict[str, Any]]:
    """
    解析 nurse_script_tn.txt 文件，提取场景 ID、标题和内容。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配模式：数字：标题\n“内容”
    # 注意：文件中的数字可能是 1：或 1. 或 1、
    pattern = r'(\d+)[:：\.\、]\s*(.*?)\n([\s\S]*?)(?=\n\d+[:：\.\、]|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    results = []
    for match in matches:
        index, title, body = match
        #去掉引号，并且合并
        #quotes = re.findall(r'[“"](.*?)[”"]', body, re.DOTALL)
        #body = "".join(quotes)
        # 去掉标点符号
        #body = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', body)
        results.append({
            "id": index,
            "title": title.strip(),
            "text": body.strip()
        })
    return results

def validate_json_format(json_str: str, schema: Dict[str, Any] = None) -> bool:
    """
    验证字符串是否为合法的 JSON，并可选地校验 schema。
    """
    try:
        data = json.loads(json_str)
        if schema:
            # 简单的 key 存在性检查，可以使用 jsonschema 库进行更复杂的校验
            for key in schema.keys():
                if key not in data:
                    return False
        return True
    except:
        return False

def validate_json_format_e1(json_str: str,json_schema: Dict[str, Any] = None) -> tuple[bool, str]:
    """
    验证医疗表单 JSON 格式。
    返回: (是否通过, 错误信息)
    """
    try:
        data = json.loads(json_str)
        validate(instance=data, schema=json_schema)
        return True, "Success"
    except json.JSONDecodeError:
        return False, "Invalid JSON format (Syntax Error)"
    except ValidationError as e:
        # e.message 会告诉你具体是哪个字段不符合规则
        return False, f"Validation Error: {e.message}"

def extract_json(content, keys):
    # 1. 增加对 match 是否存在的判断
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        print("警告：模型输出中未找到 JSON 格式数据")
        return {key: None for key in keys}

    try:
        raw_data = json.loads(match.group())
    except json.JSONDecodeError:
        print("错误：JSON 解析失败")
        return {key: None for key in keys}

    cleaned_data = {}
    
    for key in keys:            
        val = raw_data.get(key)
        # 2. 更加精细的处理：过滤掉空值或无意义的字符串
        if val is None or str(val).lower() in ["none", "null", "n/a", "unknown", "未知"]:
            cleaned_data[key] = None
        else:
            cleaned_data[key] = str(val).strip() # 去掉多余空格
    return cleaned_data

if __name__ == "__main__":
    # 测试解析
    scripts = parse_nurse_scripts('/workspace/dataset/asr_llm/nurse_script_tn.txt')
    print(f"解析到 {len(scripts)} 条数据")
    if scripts:
        print(f"示例数据：{scripts[0]}")
        print(f"示例数据：{scripts[165]}")


