import argparse
import asyncio
import json
import os
import re
import time

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:
    AsyncOpenAI = None

try:
    from dotenv import find_dotenv, load_dotenv
except ModuleNotFoundError:
    def find_dotenv(*args, **kwargs):
        return ""

    def load_dotenv(*args, **kwargs):
        return False


load_dotenv(find_dotenv())

BAICHUAN_DEFAULT_BASE_URL = "https://api.baichuan-ai.com/v1/"
QWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def first_env(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def parse_bool_env(name, default=False):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("true", "1", "yes", "y", "on")


def parse_int_env(name, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def parse_float_env(name, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def normalize_openai_base_url(base_url):
    if not base_url:
        return base_url
    normalized = base_url.strip()
    suffix = "/chat/completions"
    if normalized.rstrip("/").endswith(suffix):
        normalized = normalized.rstrip("/")[: -len(suffix)]
    return normalized.rstrip("/") + "/"


def safe_filename(value):
    name = str(value or "").strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name.strip("-._") or "model"


def infer_provider():
    explicit_provider = os.getenv("LLM_PROVIDER")
    if explicit_provider:
        return explicit_provider.strip().lower()

    model_hint = first_env("BAICHUAN_MODEL_NAME", "LLM_MODEL_NAME") or ""
    base_url_hint = first_env("BAICHUAN_BASE_URL", "LLM_BASE_URL") or ""
    if (
        os.getenv("BAICHUAN_API_KEY")
        or "baichuan" in model_hint.lower()
        or "baichuan-ai.com" in base_url_hint.lower()
    ):
        return "baichuan"
    return "qwen"


LLM_PROVIDER = infer_provider()
if LLM_PROVIDER == "baichuan":
    LLM_API_KEY = first_env("BAICHUAN_API_KEY", "LLM_API_KEY")
    LLM_BASE_URL = normalize_openai_base_url(
        os.getenv("BAICHUAN_BASE_URL") or BAICHUAN_DEFAULT_BASE_URL
    )
    LLM_MODEL_NAME = first_env("BAICHUAN_MODEL_NAME", "LLM_MODEL_NAME") or "Baichuan-M3"
else:
    LLM_API_KEY = first_env("LLM_API_KEY", "QWEN_API_KEY")
    LLM_BASE_URL = normalize_openai_base_url(os.getenv("LLM_BASE_URL") or QWEN_DEFAULT_BASE_URL)
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-flash")

LLM_TEMPERATURE = parse_float_env("LLM_TEMPERATURE", 0.1)
LLM_TOP_P = parse_float_env("LLM_TOP_P")
LLM_MAX_TOKENS = parse_int_env("LLM_MAX_TOKENS", 4096)
LLM_REQUEST_TIMEOUT_S = parse_float_env("LLM_REQUEST_TIMEOUT_S", 300.0)
BENCHMARK_CONCURRENCY = max(1, min(parse_int_env("BENCHMARK_CONCURRENCY", 10), 20))
BENCHMARK_PREFLIGHT = parse_bool_env("BENCHMARK_PREFLIGHT", True)
BENCHMARK_LOG_REQUESTS = parse_bool_env("BENCHMARK_LOG_REQUESTS", True)
BENCHMARK_ABORT_ON_TIMEOUT = parse_bool_env("BENCHMARK_ABORT_ON_TIMEOUT", False)

BAICHUAN_THINKING_BUDGET_TOKENS = parse_int_env("BAICHUAN_THINKING_BUDGET_TOKENS")
BAICHUAN_TOP_K = parse_int_env("BAICHUAN_TOP_K")
BAICHUAN_EVIDENCE_SCOPE = os.getenv("BAICHUAN_EVIDENCE_SCOPE")
BAICHUAN_OUTPUT_STYLE = os.getenv("BAICHUAN_OUTPUT_STYLE")
BAICHUAN_DISABLE_FOLLOWUP = os.getenv("BAICHUAN_DISABLE_FOLLOWUP")
BAICHUAN_WITH_SEARCH_ENHANCE = os.getenv("BAICHUAN_WITH_SEARCH_ENHANCE", "false")


class FatalAPIError(RuntimeError):
    pass


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("id", f"case_{line_no:03d}")
            records.append(record)
    return records


def message_content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return str(content)


def flatten_system_messages(messages):
    system_parts = [
        message_content_to_text(message.get("content"))
        for message in messages
        if message.get("role") == "system"
    ]
    system_prompt = "\n\n".join(part for part in system_parts if part)
    if not system_prompt:
        return messages

    flattened_messages = []
    system_merged = False
    for message in messages:
        if message.get("role") == "system":
            continue
        new_message = dict(message)
        if not system_merged and new_message.get("role") == "user":
            user_text = message_content_to_text(new_message.get("content"))
            new_message["content"] = f"系统指令：\n{system_prompt}\n\n用户输入：\n{user_text}"
            system_merged = True
        flattened_messages.append(new_message)

    if not system_merged:
        flattened_messages.insert(0, {"role": "user", "content": f"系统指令：\n{system_prompt}"})
    return flattened_messages


def baichuan_should_flatten_system():
    explicit = os.getenv("BAICHUAN_FLATTEN_SYSTEM")
    if explicit is not None:
        return parse_bool_env("BAICHUAN_FLATTEN_SYSTEM")
    return LLM_PROVIDER == "baichuan" and LLM_MODEL_NAME.lower().endswith("-plus")


def normalize_messages_for_provider(messages):
    if LLM_PROVIDER == "baichuan" and baichuan_should_flatten_system():
        return flatten_system_messages(messages)
    return messages


def build_extra_body():
    extra_body = {}
    if LLM_PROVIDER != "baichuan":
        return extra_body

    if BAICHUAN_WITH_SEARCH_ENHANCE is not None:
        extra_body["with_search_enhance"] = parse_bool_env(
            "BAICHUAN_WITH_SEARCH_ENHANCE",
            False,
        )

    metadata = {}
    if BAICHUAN_EVIDENCE_SCOPE:
        metadata["evidence_scope"] = BAICHUAN_EVIDENCE_SCOPE
    if BAICHUAN_OUTPUT_STYLE:
        metadata["output_style"] = BAICHUAN_OUTPUT_STYLE
    if BAICHUAN_DISABLE_FOLLOWUP is not None:
        metadata["disable_follow-up_question_extension"] = parse_bool_env(
            "BAICHUAN_DISABLE_FOLLOWUP"
        )
    if metadata:
        extra_body["metadata"] = metadata

    if BAICHUAN_THINKING_BUDGET_TOKENS:
        extra_body["thinking"] = {"budget_tokens": BAICHUAN_THINKING_BUDGET_TOKENS}
    if BAICHUAN_TOP_K is not None:
        extra_body["top_k"] = BAICHUAN_TOP_K
    return extra_body


def build_chat_request(messages):
    request_kwargs = {
        "model": LLM_MODEL_NAME,
        "messages": normalize_messages_for_provider(messages),
        "temperature": LLM_TEMPERATURE,
        "stream": False,
    }
    if LLM_TOP_P is not None:
        request_kwargs["top_p"] = LLM_TOP_P
    if LLM_MAX_TOKENS is not None:
        request_kwargs["max_tokens"] = LLM_MAX_TOKENS

    extra_body = build_extra_body()
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    return request_kwargs


def construct_messages(system_prompt, record):
    user_input = record.get("input") or record.get("input_text") or ""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]


def extract_response_text(response):
    if not response or not response.choices:
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        return message_content_to_text(content)
    return str(content)


def parse_json_output(text):
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def usage_to_dict(usage):
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage

    result = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result


def add_usage(total, usage):
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value


def extract_api_error_info(exc):
    body = getattr(exc, "body", None)
    status_code = getattr(exc, "status_code", None)
    error_obj = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error_obj, dict):
        error_obj = body if isinstance(body, dict) else {}

    code = error_obj.get("code") or getattr(exc, "code", None)
    error_type = error_obj.get("type") or getattr(exc, "type", None)
    message = error_obj.get("message") or str(exc)
    return status_code, code, error_type, message


def is_fatal_api_error(exc):
    status_code, code, error_type, message = extract_api_error_info(exc)
    text = " ".join(str(part or "").lower() for part in (code, error_type, message))
    if status_code in (401, 403):
        return True
    fatal_markers = (
        "invalid_api_key",
        "incorrect api key",
        "access_denied",
        "insufficient_quota",
        "quota",
        "billing",
        "permission",
        "unauthorized",
        "forbidden",
    )
    return any(marker in text for marker in fatal_markers)


def format_api_error(exc):
    status_code, code, error_type, message = extract_api_error_info(exc)
    parts = []
    if status_code is not None:
        parts.append(f"status={status_code}")
    if code:
        parts.append(f"code={code}")
    if error_type:
        parts.append(f"type={error_type}")
    parts.append(f"message={message}")
    return ", ".join(parts)


async def process_record(client, system_prompt, record, semaphore, record_id):
    async with semaphore:
        label = record.get("id", f"record_{record_id}")
        messages = construct_messages(system_prompt, record)
        start_time = time.perf_counter()
        if BENCHMARK_LOG_REQUESTS:
            print(f"Starting request for {label}...", flush=True)

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**build_chat_request(messages)),
                timeout=LLM_REQUEST_TIMEOUT_S,
            )
            request_duration_seconds = time.perf_counter() - start_time
            output_text = extract_response_text(response)
            usage = usage_to_dict(getattr(response, "usage", None))
            if BENCHMARK_LOG_REQUESTS:
                print(
                    f"Finished request for {label} in {request_duration_seconds:.2f}s "
                    f"({len(output_text)} chars).",
                    flush=True,
                )
            return {
                "id": label,
                "input": record.get("input") or record.get("input_text") or "",
                "expected_output": record.get("expected_output"),
                "model_output_text": output_text,
                "model_output_json": parse_json_output(output_text),
                "request_duration_seconds": request_duration_seconds,
                "request_error": None,
                "usage": usage,
            }
        except asyncio.TimeoutError as e:
            request_duration_seconds = time.perf_counter() - start_time
            error_message = (
                f"Request timeout after {request_duration_seconds:.2f}s. "
                f"Increase LLM_REQUEST_TIMEOUT_S or reduce LLM_MAX_TOKENS."
            )
            if BENCHMARK_ABORT_ON_TIMEOUT:
                raise FatalAPIError(error_message) from e
            print(f"Timeout processing {label}: {error_message}", flush=True)
            return {
                "id": label,
                "input": record.get("input") or record.get("input_text") or "",
                "expected_output": record.get("expected_output"),
                "model_output_text": "",
                "model_output_json": None,
                "request_duration_seconds": request_duration_seconds,
                "request_error": error_message,
                "usage": {},
            }
        except Exception as e:
            request_duration_seconds = time.perf_counter() - start_time
            if is_fatal_api_error(e):
                raise FatalAPIError(format_api_error(e)) from e
            error_message = str(e)
            print(f"Error processing {label}: {error_message}", flush=True)
            return {
                "id": label,
                "input": record.get("input") or record.get("input_text") or "",
                "expected_output": record.get("expected_output"),
                "model_output_text": "",
                "model_output_json": None,
                "request_duration_seconds": request_duration_seconds,
                "request_error": error_message,
                "usage": {},
            }


async def run_benchmark(args):
    if AsyncOpenAI is None:
        raise RuntimeError("Missing dependency 'openai'. Install it with: pip install openai")
    if not LLM_API_KEY:
        if LLM_PROVIDER == "baichuan":
            raise RuntimeError("Missing API key. Set BAICHUAN_API_KEY for Baichuan.")
        raise RuntimeError("Missing API key. Set LLM_API_KEY or QWEN_API_KEY.")

    scenario_dir = os.path.abspath(args.scenario_dir)
    prompt_path = os.path.abspath(args.prompt or os.path.join(scenario_dir, "prompt.md"))
    dataset_path = os.path.abspath(args.dataset or os.path.join(scenario_dir, "dataset.jsonl"))
    output_dir = os.path.abspath(args.output_dir or os.path.join(scenario_dir, "results"))
    output_prefix = args.output_prefix or os.getenv("BENCHMARK_OUTPUT_PREFIX") or safe_filename(LLM_MODEL_NAME)

    os.makedirs(output_dir, exist_ok=True)

    system_prompt = read_text(prompt_path)
    records = load_jsonl(dataset_path)

    print(f"Initializing API client: provider={LLM_PROVIDER}, model={LLM_MODEL_NAME}")
    print(f"Base URL: {LLM_BASE_URL}")
    print(f"Scenario dir: {scenario_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Prompt: {prompt_path}")
    print(f"Output dir: {output_dir}")
    print(f"Concurrency: {BENCHMARK_CONCURRENCY}")
    print(f"Preflight: {BENCHMARK_PREFLIGHT}")
    print(f"Max tokens: {LLM_MAX_TOKENS}")
    print(f"Request timeout: {LLM_REQUEST_TIMEOUT_S}s")
    print(f"Baichuan search enhance: {BAICHUAN_WITH_SEARCH_ENHANCE}")
    print(f"Loaded {len(records)} records.")

    client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_REQUEST_TIMEOUT_S,
    )
    semaphore = asyncio.Semaphore(BENCHMARK_CONCURRENCY)

    outputs = []
    records_to_process = records
    if BENCHMARK_PREFLIGHT and records:
        print("Running preflight request on the first record...")
        outputs.append(await process_record(client, system_prompt, records[0], semaphore, 0))
        records_to_process = records[1:]

    start_index = len(outputs)
    tasks = [
        process_record(client, system_prompt, record, semaphore, start_index + idx)
        for idx, record in enumerate(records_to_process)
    ]
    if tasks:
        outputs.extend(await asyncio.gather(*tasks))

    total_duration = sum(item["request_duration_seconds"] for item in outputs)
    usage_total = {}
    for item in outputs:
        add_usage(usage_total, item.get("usage") or {})

    request_count = len(outputs)
    error_count = sum(1 for item in outputs if item.get("request_error"))
    parsed_json_count = sum(1 for item in outputs if item.get("model_output_json") is not None)
    result = {
        "summary": {
            "scenario_dir": scenario_dir,
            "dataset": dataset_path,
            "prompt": prompt_path,
            "provider": LLM_PROVIDER,
            "model_name": LLM_MODEL_NAME,
            "base_url": LLM_BASE_URL,
            "output_prefix": output_prefix,
            "request_count": request_count,
            "error_count": error_count,
            "parsed_json_count": parsed_json_count,
            "avg_request_duration_seconds": total_duration / request_count if request_count else 0.0,
            "total_request_duration_seconds": total_duration,
            "usage_total": usage_total,
            "concurrency": BENCHMARK_CONCURRENCY,
            "max_tokens": LLM_MAX_TOKENS,
            "baichuan_with_search_enhance": BAICHUAN_WITH_SEARCH_ENHANCE,
        },
        "records": outputs,
    }

    output_path = os.path.join(output_dir, f"{output_prefix}_results.json")
    write_json(output_path, result)
    print(f"Wrote results to {output_path}")
    print(
        f"Summary: requests={request_count}, errors={error_count}, "
        f"parsed_json={parsed_json_count}, avg_duration={result['summary']['avg_request_duration_seconds']:.2f}s"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run one nurse LLM scenario benchmark.")
    parser.add_argument("--scenario-dir", default=os.getcwd())
    parser.add_argument("--dataset")
    parser.add_argument("--prompt")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-prefix")
    return parser.parse_args()


def main():
    asyncio.run(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
