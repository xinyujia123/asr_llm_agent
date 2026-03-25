import json
import os
import re
import urllib.request

MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1 = """
你是一个专业严谨的医疗数据录入助手。你的任务是从用户的**语音识别文本(ASR)**中，如实提取完整的生命体征数据和其他相关信息。

### 【字段枚举与映射规范】
* **assessmentTime** (评估时间): 格式 "YYYY-MM-DD HH:MM:SS"。若未提及时间默认用【当前系统时间】；若语音中仅提及具体时刻（如“上午三点”、“8点”），则默认日期为【当前系统时间】的日期并拼接成完整格式；若有“昨天”等相对词，需结合系统时间推算。
* **education** (文化程度): 仅限 "文盲" | "小学" | "初中" | "高中或中专" | "大专以上" | "未入学"（自动映射近义词）。
* **ethnicity** (民族): 具体民族字符串。 
* **admissionTime** (入院时间): "YYYY-MM-DD HH:MM:SS"，需结合上下文推算。
* **admissionMethod** (入院方式): 仅限 "步行" | "轮椅" | "平车" | "推床" | "背入" | "其他"（120/救护车/担架等映射为"其他"）。
* **diagnosis** (入院诊断): 提取具体病名，必须保留不确定前缀（如“考虑肺炎”）。
* **allergyHistory** (过敏史): Array形式，提取具体过敏源（如 ["青霉素", "海鲜"]）；语音明确没有过敏史填 ["无"]。
* **temperature** (体温, ℃) ：保留一位小数
* **weight** (体重, KG)：保留一位小数
* **height** (身高, cm) 
* **bloodSugar** (血糖, mmol/L): 保留一位小数。
* **pulseRate** (脉率, 次/分) 
* **heartRate** (心率, 次/分) 
* **respiratoryRate** (呼吸频率, 次/分): 严格区分心率和脉率。
* **systolicPressure** (收缩压, mmHg): 血压的高压数值（通常为较大值）。针对 ASR 缺乏标点的连读数值（如‘一百一七十’），需根据血压常识自动拆解为合理的收缩压110和舒张压70。
* **diastolicPressure** (舒张压, mmHg): 血压的低压数值（通常为较小值）。
* **consciousness** (神志): 仅限 "清醒" | "嗜睡" | "模糊" | "昏睡" | "昏迷" | "浅昏迷" | "中昏迷" | "深昏迷" | "药物镇静状" | "麻醉未醒"。严禁仅仅通过行为推断（只有“能交流” != “清醒”，还需要说明“意识清醒”）。
* **skin** (皮肤黏膜): Array<"完整" | "手术切口" | "造口" | "钉道" | "溃疡" | "破损" | "脓疱" | "皮疹" | "出血点" | "其他">。水肿/淤斑/压疮等映射为 "其他"。有异常严禁选“完整”。
* **catheter** (导管): Array形式，提取具体导管名称（如 ["静脉通路", "导尿管"]）；语音明确没有导管填 ["无"]。（补充映射规则：打点滴->静脉通路，尿袋->导尿管；仅限侵入式管状器械，吸氧面罩不算）。
* **diet** (饮食): "正常" | "异常"。单纯“食欲不振”等日常情况不能判定为异常。
* **sleep** (睡眠): "正常" | "异常"。单纯“睡得晚”等日常情况不能判定为异常。
* **urination** (排尿): "正常" | "异常"。
* **defecation** (排便): "正常" | "异常"。
* **stoolFrequency** (排便次数, 次): 当天排便次数

**兜底规则：语音未提及则不输出该key（带有默认值说明的字段除外，如 assessmentTime）。
- 局部否定不代表全局无（“没插胃管”不代表没有导管）。
- 明显违背人类生理极限的离谱数值（如体温 50，心率 300）视为 ASR 错误，不作为输出该key的依据。
- 所有json值均为字符串形式（"xxx"）或者列表(["xx","xx"])，不可以是数字或特殊字符。
- 同一数值取**最后一次**；区间值取**平均值**。保留小数时四舍五入。
- 提取的数值根据格式要求保留小数位（如“体温36度”取“36.0”），未明确要求小数位的字段默认输出整数。
- 时间推导：对语音中的口语化时间（如“上午两点”、“半夜三点”）必须自动转换为 24 小时制的 HH:MM:SS（如 02:00:00、03:00:00），并结合当前系统时间的日期补全。


### 【核心处理原则】
1. **ASR语意纠错**：结合医疗语境修正音近词（如“夫妻”->“呼吸”，“学压”->“血压”）。
2. **如实提取**：
   - 禁止自行根据医疗常识推断，除了字段映射规范中明确要求的常识推导（如血压连读拆解、日常俗语转医疗术语）外，严禁对未提及的体征状态进行主观猜测或推断。
   - 未提及或模糊数值（如“低烧”、“九十多”）不输出该key，严禁猜数。
   - 语音提及但【字段枚举与映射规范】中没有的字段不需要输出。

### 【输出要求】
- 仅输出纯 JSON 字符串，严禁包含任何其他解释性文字。严禁使用 ```json 等 Markdown 代码块标记包裹。

### 【示例 (Few-Shot)】
*假设当前系统时间为：2026-02-01 10:00:00*

输入：
“苗族，大专文化。昨天下午三点推床送入，考虑肺炎，皮肤上有手术切口。今天早上八点进行的体征测量，体重60公斤，心率60，麦博62，学压一百一九十，今天排便两次，神志清醒，手部皮肤正常，接了尿袋在打点滴，没有过敏史。”

**输出**：
{
  "assessmentTime": "2026-02-01 08:00:00",
  "education": "大专以上",
  "ethnicity": "苗族",
  "admissionTime": "2026-01-31 15:00:00",
  "admissionMethod": "推床",
  "diagnosis": "考虑肺炎",
  "allergyHistory": ["无"],
  "weight": "60.0",
  "pulseRate": "62",
  "heartRate": "60",
  "systolicPressure": "110",
  "diastolicPressure": "90",
  "consciousness": "清醒",
  "skin": ["手术切口"],
  "catheter": ["静脉通路", "导尿管"],
  "stoolFrequency": "2"
}

输入：
“血压高压一百一十，低压一百，脉搏七十二。”

**输出**：
{
  "assessmentTime": "2026-02-01 10:00:00",
  "pulseRate": "72",
  "systolicPressure": "110",
  "diastolicPressure": "100"
}

输入：
“早上八点，脉搏八十二，血压八十一百二十。”

输出：
{
  "assessmentTime": "2026-02-01 08:00:00",
  "pulseRate": "82",
  "systolicPressure": "120",
  "diastolicPressure": "80"
}

输入：
“刚刚量的体征，低压八十高压一百二，新率一百零五，但是麦博只有九十，其他没啥。”

输出：
{
"assessmentTime": "2026-02-01 10:00:00",
"systolicPressure": "120",
"diastolicPressure": "80",
"heartRate": "105",
"pulseRate": "90"
}

输入：
“这病人昨天半夜两点自己走进来的。血压一百四九十。”

输出：
{
"assessmentTime": "2026-02-01 10:00:00",
"admissionTime": "2026-01-31 02:00:00",
"admissionMethod": "步行",
"systolicPressure": "140",
"diastolicPressure": "90"
}

输入：
“今天中午十二点办的住院，推床进来的。下午一点半测了下，心率大概一百一十多，脉博一百一。”

输出：
{
"assessmentTime": "2026-02-01 13:30:00",
"admissionTime": "2026-02-01 12:00:00",
"admissionMethod": "推床",
"pulseRate": "110"
}

"""

DYNAMIC_CONTENT = """
正式语音提取病历任务开始
### 【当前系统时间】{CURRENT_SYS_TIME}（ 若未提及具体评估/测试时间，assessmentTime默认输出当前系统时间。）
- 【前天系统时间】{THE_DAY_BEFORE_YESTERDAY}（方便进行时间推算）
语音输入：
"""


def _parse_prometheus_labels(label_text):
    labels = {}
    for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', label_text):
        labels[match.group(1)] = match.group(2)
    return labels


def fetch_kv_cache_block_size(metrics_url, timeout_s=2):
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout_s) as response:
            metrics_text = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    for line in metrics_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "{" not in stripped or "}" not in stripped:
            continue
        metric_name = stripped.split("{", 1)[0].strip().lower()
        if "cache" not in metric_name and "kv" not in metric_name:
            continue
        labels_part = stripped.split("{", 1)[1].split("}", 1)[0]
        labels = _parse_prometheus_labels(labels_part)
        block_size = labels.get("block_size")
        if block_size and block_size.isdigit():
            return int(block_size)
    match = re.search(r'block_size="(\d+)"', metrics_text)
    if match:
        return int(match.group(1))
    return None


def _estimate_token_count(text):
    word_count = len(text.split())
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    return max(word_count, int(cjk_count * 0.7), len(text) // 4, 1)


def fetch_token_count_from_vllm(text, model_name, vllm_base_url, timeout_s=3):
    if not model_name or not vllm_base_url:
        return None
    payload = json.dumps({
        "model": model_name,
        "prompt": text
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{vllm_base_url.rstrip('/')}/tokenize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="ignore")
        data = json.loads(body)
    except Exception:
        return None
    if isinstance(data, dict):
        if isinstance(data.get("tokens"), list):
            return len(data["tokens"])
        if isinstance(data.get("token_ids"), list):
            return len(data["token_ids"])
        if isinstance(data.get("num_tokens"), int):
            return int(data["num_tokens"])
        if isinstance(data.get("count"), int):
            return int(data["count"])
    return None


def adapt_system_prompt_to_kv_block(
    system_prompt,
    enable_padding=False,
    metrics_url=None,
    model_name=None,
    vllm_base_url=None,
    timeout_s=2,
    padding_unit="。"
):
    meta = {
        "enabled": bool(enable_padding),
        "applied": False,
        "block_size": None,
        "original_tokens": None,
        "padded_tokens": None,
        "padding_units": 0,
        "token_count_source": "estimate"
    }
    if not enable_padding:
        return system_prompt, meta
    resolved_metrics_url = metrics_url or os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")
    resolved_base_url = vllm_base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    block_size = fetch_kv_cache_block_size(resolved_metrics_url, timeout_s=timeout_s)
    meta["block_size"] = block_size
    if not isinstance(block_size, int) or block_size <= 0:
        return system_prompt, meta
    token_count = fetch_token_count_from_vllm(system_prompt, model_name, resolved_base_url, timeout_s=timeout_s)
    if token_count is None:
        token_count = _estimate_token_count(system_prompt)
    else:
        meta["token_count_source"] = "vllm_tokenize"
    meta["original_tokens"] = int(token_count)
    remainder = token_count % block_size
    if remainder == 0:
        meta["padded_tokens"] = int(token_count)
        return system_prompt, meta
    missing_tokens = block_size - remainder
    probe_count = fetch_token_count_from_vllm(system_prompt + (padding_unit * 32), model_name, resolved_base_url, timeout_s=timeout_s)
    if probe_count is not None:
        meta["token_count_source"] = "vllm_tokenize"
        per_unit = max((probe_count - token_count) / 32.0, 0.1)
        padding_units = max(1, int(round(missing_tokens / per_unit)))
    else:
        padding_units = int(missing_tokens)
    padded_prompt = system_prompt + (padding_unit * padding_units)
    final_count = fetch_token_count_from_vllm(padded_prompt, model_name, resolved_base_url, timeout_s=timeout_s)
    if final_count is None:
        final_count = _estimate_token_count(padded_prompt)
    else:
        meta["token_count_source"] = "vllm_tokenize"
        guard = 0
        while final_count % block_size != 0 and guard < block_size * 2:
            padded_prompt += padding_unit
            final_count = fetch_token_count_from_vllm(padded_prompt, model_name, resolved_base_url, timeout_s=timeout_s)
            if final_count is None:
                break
            padding_units += 1
            guard += 1
    meta["applied"] = True
    meta["padding_units"] = int(max(padding_units, 0))
    meta["padded_tokens"] = int(final_count) if isinstance(final_count, int) else None
    return padded_prompt, meta


def prepare_system_prompt_padding_suffix(
    system_prompt,
    enable_padding=False,
    metrics_url=None,
    model_name=None,
    vllm_base_url=None,
    timeout_s=2
):
    if not enable_padding:
        return "", {
            "enabled": False,
            "applied": False,
            "padding_units": 0
        }
    padded_prompt, meta = adapt_system_prompt_to_kv_block(
        system_prompt=system_prompt,
        enable_padding=enable_padding,
        metrics_url=metrics_url,
        model_name=model_name,
        vllm_base_url=vllm_base_url,
        timeout_s=timeout_s
    )
    padding_units = int(meta.get("padding_units") or 0)
    if padding_units <= 0:
        return "", meta
    return padded_prompt[-padding_units:], meta


def fetch_token_count_from_transformers(text, local_model_path):
    if not local_model_path:
        return None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return len(token_ids)
    except Exception:
        return None


def get_prompt_token_count(
    prompt_text,
    model_name=None,
    vllm_base_url=None,
    local_model_path=None,
    backend="auto",
    timeout_s=3
):
    resolved_base_url = vllm_base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    backend = str(backend or "auto").strip().lower()
    if backend not in {"auto", "vllm", "transformers"}:
        backend = "auto"
    if backend in {"auto", "vllm"}:
        token_count = fetch_token_count_from_vllm(prompt_text, model_name, resolved_base_url, timeout_s=timeout_s)
        if isinstance(token_count, int):
            return {
                "token_count": token_count,
                "source": "vllm",
                "model_name": model_name,
                "base_url": resolved_base_url
            }
        if backend == "vllm":
            return {
                "token_count": None,
                "source": "vllm",
                "model_name": model_name,
                "base_url": resolved_base_url
            }
    if backend in {"auto", "transformers"}:
        token_count = fetch_token_count_from_transformers(prompt_text, local_model_path)
        if isinstance(token_count, int):
            return {
                "token_count": token_count,
                "source": "transformers",
                "model_name": local_model_path
            }
        if backend == "transformers":
            return {
                "token_count": None,
                "source": "transformers",
                "model_name": local_model_path
            }
    return {
        "token_count": _estimate_token_count(prompt_text),
        "source": "estimate",
        "model_name": model_name
    }


def get_vllm_model_block_size(metrics_url=None, timeout_s=2):
    resolved_metrics_url = metrics_url or os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")
    block_size = fetch_kv_cache_block_size(resolved_metrics_url, timeout_s=timeout_s)
    return {
        "block_size": block_size,
        "metrics_url": resolved_metrics_url
    }


def main():
    model_name = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    metrics_url = os.getenv("VLLM_METRICS_URL", f"{vllm_base_url.rstrip('/')}/metrics")
    local_model_path = os.getenv("LOCAL_MODEL_PATH", "")
    test_prompt = MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1
    token_result_auto = get_prompt_token_count(
        prompt_text=test_prompt,
        model_name=model_name,
        vllm_base_url=vllm_base_url,
        local_model_path=local_model_path,
        backend="auto"
    )
    token_result_vllm = get_prompt_token_count(
        prompt_text=test_prompt,
        model_name=model_name,
        vllm_base_url=vllm_base_url,
        backend="vllm"
    )
    token_result_transformers = get_prompt_token_count(
        prompt_text=test_prompt,
        local_model_path=local_model_path,
        backend="transformers"
    )
    block_size_result = get_vllm_model_block_size(
        metrics_url=metrics_url
    )
    print(json.dumps({
        "test_prompt_characters": len(test_prompt),
        "token_count_auto": token_result_auto,
        "token_count_vllm": token_result_vllm,
        "token_count_transformers": token_result_transformers,
        "vllm_block_size": block_size_result
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
