import json
import os
import re
import time
from typing import Any, Optional


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
BASIC_TEST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
F1_TEST_DIR = os.path.join(BASIC_TEST_DIR, "f1_test")

import sys

if F1_TEST_DIR not in sys.path:
    sys.path.insert(0, F1_TEST_DIR)

from prompts import (  # noqa: E402
    DYNAMIC_CONTENT,
    MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1,
)


MODEL_PATH = os.getenv(
    "NATIVE_MODEL_PATH",
    "/root/autodl-tmp/models/models/OpenBMB/MiniCPM-o-4_5",
)
DATASET_PATH = os.getenv(
    "BENCHMARK_DATASET_PATH",
    os.path.join(BASIC_TEST_DIR, "dataset.jsonl"),
)
OUTPUT_DIR = os.getenv("NATIVE_AUDIO_OUTPUT_DIR", SCRIPT_DIR)
AUDIO_DIR = os.getenv("NATIVE_AUDIO_DIR", os.path.join(OUTPUT_DIR, "audio_native"))
RESULTS_PATH = os.path.join(OUTPUT_DIR, "native_audio_results.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "native_audio_metrics.json")
ERROR_CASES_PATH = os.path.join(OUTPUT_DIR, "native_audio_error_cases.json")

START_INDEX = int(os.getenv("NATIVE_AUDIO_START_INDEX", "0"))
LIMIT = int(os.getenv("NATIVE_AUDIO_LIMIT", "10"))
MAX_NEW_TOKENS = int(os.getenv("NATIVE_AUDIO_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("NATIVE_AUDIO_TEMPERATURE", "0.1"))
DO_SAMPLE = os.getenv("NATIVE_AUDIO_DO_SAMPLE", "true").lower() in ("true", "1", "yes")
ATTN_IMPLEMENTATION = os.getenv("NATIVE_AUDIO_ATTN_IMPL", "sdpa")
TORCH_DTYPE = os.getenv("NATIVE_AUDIO_TORCH_DTYPE", "bfloat16")
INIT_VISION = os.getenv("NATIVE_AUDIO_INIT_VISION", "false").lower() in ("true", "1", "yes")
INIT_AUDIO = os.getenv("NATIVE_AUDIO_INIT_AUDIO", "false").lower() in ("true", "1", "yes")
INIT_TTS = os.getenv("NATIVE_AUDIO_INIT_TTS", "true").lower() in ("true", "1", "yes")
TTS_TOP_P = float(os.getenv("NATIVE_AUDIO_TTS_TOP_P", "0.85"))
TTS_TOP_K = int(os.getenv("NATIVE_AUDIO_TTS_TOP_K", "25"))
TTS_REPETITION_PENALTY = float(os.getenv("NATIVE_AUDIO_TTS_REPETITION_PENALTY", "1.05"))


def load_data(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def construct_messages(record: dict[str, Any]) -> list[dict[str, list[str]]]:
    sys_time = record.get("current_sys_time", "")
    day_before = record.get("the_day_before_yesterday", "")
    user_input = (
        DYNAMIC_CONTENT.replace("{CURRENT_SYS_TIME}", sys_time).replace(
            "{THE_DAY_BEFORE_YESTERDAY}", day_before
        )
        + record["input_text"]
    )
    return [
        {"role": "system", "content": [MEDICAL_EXTRACTOR_PROMPT_NOINFER_NOCOT_V1]},
        {"role": "user", "content": [user_input]},
    ]


def clean_generated_text(text: Any) -> str:
    cleaned = "" if text is None else str(text)
    cleaned = cleaned.split("<|tts_eos|>")[0]
    cleaned = re.sub(r"<\|[^|]+?\|>", "", cleaned)
    return cleaned.strip()


def parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        json_str = match.group(1) if match else text

    try:
        return json.loads(json_str)
    except Exception:
        return {}


def normalize_value(val: Any) -> Any:
    if isinstance(val, list):
        return sorted([str(v).strip() for v in val])
    s = str(val).strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def compare_record(expected: dict[str, Any], predicted: dict[str, Any]) -> dict[str, dict[str, int]]:
    field_stats = {}
    all_keys = set(expected.keys()) | set(predicted.keys())

    for key in all_keys:
        field_stats[key] = {"tp": 0, "fp": 0, "fn": 0}
        exp_val = expected.get(key)
        pred_val = predicted.get(key)

        if exp_val is not None and pred_val is not None:
            if normalize_value(exp_val) == normalize_value(pred_val):
                field_stats[key]["tp"] = 1
            else:
                field_stats[key]["fp"] = 1
                field_stats[key]["fn"] = 1
        elif exp_val is not None:
            field_stats[key]["fn"] = 1
        elif pred_val is not None:
            field_stats[key]["fp"] = 1

    return field_stats


def build_error_case(
    i: int,
    record: dict[str, Any],
    expected_json: dict[str, Any],
    predicted_json: dict[str, Any],
    generated_text: str,
    record_stats: dict[str, dict[str, int]],
    audio_path: Optional[str],
    audio_error: Optional[str],
    request_time_seconds: float,
) -> Optional[dict[str, Any]]:
    error_details = {}
    for key, stats in record_stats.items():
        if stats["fp"] > 0 or stats["fn"] > 0:
            error_details[key] = {
                "expected": expected_json.get(key),
                "predicted": predicted_json.get(key),
                "type": "FP"
                if stats["fp"] > 0 and stats["fn"] == 0
                else ("FN" if stats["fn"] > 0 and stats["fp"] == 0 else "Mismatch"),
            }

    if not error_details:
        return None

    return {
        "id": i,
        "input": record["input_text"],
        "expected": expected_json,
        "predicted": predicted_json,
        "error_details": error_details,
        "generated_text": generated_text,
        "audio_path": audio_path,
        "audio_error": audio_error,
        "request_time_seconds": request_time_seconds,
    }


def calculate_metrics(global_stats: dict[str, dict[str, int]], request_count: int, total_request_time: float):
    final_metrics = {}
    sorted_keys = sorted(global_stats.keys())

    for key in sorted_keys:
        stats = global_stats[key]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        final_metrics[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total_fields = len(sorted_keys)
    category_avg_f1 = (
        sum(final_metrics[key]["f1"] for key in sorted_keys) / total_fields if total_fields > 0 else 0.0
    )
    total_tp = sum(global_stats[key]["tp"] for key in sorted_keys)
    total_fp = sum(global_stats[key]["fp"] for key in sorted_keys)
    total_fn = sum(global_stats[key]["fn"] for key in sorted_keys)
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    sample_aggregate_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0
        else 0.0
    )

    final_metrics["summary"] = {
        "total_category_avg_f1": category_avg_f1,
        "total_sample_aggregate_f1": sample_aggregate_f1,
        "precision": total_precision,
        "recall": total_recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "field_count": total_fields,
        "request_count": request_count,
        "total_request_time_seconds": total_request_time,
        "avg_request_time_seconds": total_request_time / request_count if request_count else 0.0,
        "model_path": MODEL_PATH,
        "dataset_path": DATASET_PATH,
        "audio_dir": AUDIO_DIR,
        "start_index": START_INDEX,
        "limit": LIMIT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "do_sample": DO_SAMPLE,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "torch_dtype": TORCH_DTYPE,
        "init_vision": INIT_VISION,
        "init_audio": INIT_AUDIO,
        "init_tts": INIT_TTS,
        "tts_top_p": TTS_TOP_P,
        "tts_top_k": TTS_TOP_K,
        "tts_repetition_penalty": TTS_REPETITION_PENALTY,
    }
    return final_metrics


def apply_minicpmo_config_overrides(config):
    config.init_vision = INIT_VISION
    config.init_audio = INIT_AUDIO
    config.init_tts = INIT_TTS

    if INIT_TTS and hasattr(config, "tts_config"):
        # Some MiniCPM-o 4.5 checkpoints miss these fields, while newer
        # remote-code TTS modules read them during initialization.
        config.tts_config.top_p = TTS_TOP_P
        config.tts_config.top_k = TTS_TOP_K
        config.tts_config.repetition_penalty = TTS_REPETITION_PENALTY
    return config


def patch_transformers_tied_weights_compat(pretrained_model_cls):
    if hasattr(pretrained_model_cls, "all_tied_weights_keys"):
        return

    def get_all_tied_weights_keys(self):
        tied_keys = self.__dict__.get("all_tied_weights_keys")
        if tied_keys is None:
            tied_keys = getattr(self, "_tied_weights_keys", None) or []
        if isinstance(tied_keys, dict):
            return tied_keys
        return {key: None for key in tied_keys}

    def set_all_tied_weights_keys(self, value):
        if value is None:
            self.__dict__["all_tied_weights_keys"] = {}
        elif isinstance(value, dict):
            self.__dict__["all_tied_weights_keys"] = value
        else:
            self.__dict__["all_tied_weights_keys"] = {key: None for key in value}

    pretrained_model_cls.all_tied_weights_keys = property(
        get_all_tied_weights_keys,
        set_all_tied_weights_keys,
    )


def load_model():
    try:
        import torch
        from transformers import AutoConfig
        from transformers import AutoModel
        from transformers import PreTrainedModel
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing Python dependency. Run this script in the vllm_0.22.1 conda "
            "environment, or use ./run_benchmark_native_audio.sh from this folder."
        ) from e

    patch_transformers_tied_weights_compat(PreTrainedModel)

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(TORCH_DTYPE.lower())
    if torch_dtype is None:
        raise ValueError(f"Unsupported NATIVE_AUDIO_TORCH_DTYPE={TORCH_DTYPE}")

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = apply_minicpmo_config_overrides(config)

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=torch_dtype,
    )
    model.eval().cuda()
    if INIT_TTS:
        model.init_tts()
    return model


def process_record(model, i: int, record: dict[str, Any]):
    messages = construct_messages(record)
    audio_path = os.path.join(AUDIO_DIR, f"{i:05d}.wav")

    start_time = time.perf_counter()
    generated_text = clean_generated_text(
        model.chat(
            msgs=messages,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            use_tts_template=True,
            enable_thinking=False,
            generate_audio=True,
            output_audio_path=audio_path,
        )
    )
    request_time_seconds = time.perf_counter() - start_time

    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        saved_audio_path = audio_path
        audio_error = None
        audio_bytes = os.path.getsize(audio_path)
    else:
        saved_audio_path = None
        audio_error = "audio_file_not_created"
        audio_bytes = 0

    predicted_json = parse_json(generated_text)
    expected_json = record["expected_output"]
    record_stats = compare_record(expected_json, predicted_json)
    error_case = build_error_case(
        i,
        record,
        expected_json,
        predicted_json,
        generated_text,
        record_stats,
        saved_audio_path,
        audio_error,
        request_time_seconds,
    )

    result_item = {
        "id": i,
        "input": record["input_text"],
        "expected": expected_json,
        "predicted": predicted_json,
        "generated_text": generated_text,
        "audio_path": saved_audio_path,
        "audio_bytes": audio_bytes,
        "audio_error": audio_error,
        "request_time_seconds": request_time_seconds,
        "metrics": record_stats,
    }
    return result_item, error_case, record_stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    data = load_data(DATASET_PATH)
    if START_INDEX < 0 or START_INDEX >= len(data):
        raise ValueError(f"NATIVE_AUDIO_START_INDEX out of range: {START_INDEX}, dataset size={len(data)}")

    end_index = len(data) if LIMIT <= 0 else min(len(data), START_INDEX + LIMIT)
    indexed_records = list(enumerate(data[START_INDEX:end_index], start=START_INDEX))

    print(f"Native MiniCPM-o audio benchmark")
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Records: {START_INDEX}..{end_index - 1} ({len(indexed_records)} records)")
    print(f"Audio dir: {AUDIO_DIR}")
    print("If vLLM is still running, stop it first to free GPU memory.")

    model = load_model()

    results = []
    error_cases = []
    global_stats = {}
    total_request_time = 0.0

    for i, record in indexed_records:
        print(f"[{i}] generating text + audio...")
        result_item, error_case, record_stats = process_record(model, i, record)
        results.append(result_item)
        total_request_time += result_item["request_time_seconds"]
        if error_case is not None:
            error_cases.append(error_case)

        audio_status = result_item["audio_path"] or result_item["audio_error"]
        print(f"[{i}] done in {result_item['request_time_seconds']:.2f}s, audio={audio_status}")

        for key, stats in record_stats.items():
            if key not in global_stats:
                global_stats[key] = {"tp": 0, "fp": 0, "fn": 0}
            global_stats[key]["tp"] += stats["tp"]
            global_stats[key]["fp"] += stats["fp"]
            global_stats[key]["fn"] += stats["fn"]

    final_metrics = calculate_metrics(global_stats, len(results), total_request_time)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    with open(ERROR_CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, ensure_ascii=False, indent=2)

    summary = final_metrics["summary"]
    print("\nSummary:")
    print(f"Total Category Avg F1: {summary['total_category_avg_f1']:.4f}")
    print(f"Total Sample Aggregate F1: {summary['total_sample_aggregate_f1']:.4f}")
    print(f"Average Request Time: {summary['avg_request_time_seconds']:.4f}s")
    print(f"Results: {RESULTS_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Error cases: {ERROR_CASES_PATH}")


if __name__ == "__main__":
    main()
