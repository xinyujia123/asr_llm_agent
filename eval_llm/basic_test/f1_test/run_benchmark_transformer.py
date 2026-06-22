#!/usr/bin/env python3
"""Run the f1_test benchmark with a local Transformers model backend.

This mirrors run_benchmark.py's dataset, prompt construction, parsing, and
metric calculation, but replaces the vLLM HTTP request with
AutoModelForMultimodalLM.generate(). It is intended for Gemma4 Unified models
that currently do not serve correctly through vLLM.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/models/google/gemma-4-12B-it"
MODEL_PATH = os.getenv("TRANSFORMERS_MODEL_PATH") or os.getenv("HF_MODEL_PATH") or DEFAULT_MODEL_PATH
MODEL_ALIAS = os.getenv("VLLM_MODEL_NAME") or os.path.basename(MODEL_PATH.rstrip("/"))

# run_benchmark.py reads its configuration at import time. Set text-only
# defaults before importing it, while still allowing callers to override the
# usual output/model identifiers.
os.environ.setdefault("VLLM_MODEL_NAME", MODEL_ALIAS)
os.environ.setdefault("BENCHMARK_OUTPUT_MODEL_DIR", MODEL_ALIAS.split("/")[-1])
os.environ.setdefault("VLLM_EXPERIMENT_ID", "0")
os.environ["BENCHMARK_REQUEST_AUDIO"] = "false"
os.environ.setdefault("ENABLE_SYSTEM_PROMPT_BLOCK_PADDING", "false")
os.environ.setdefault("BENCHMARK_REQUEST_TIMEOUT_S", "240")
os.environ.setdefault(
    "VLLM_EXPERIMENT_ARGS",
    "transformers_backend;attn_implementation=eager;audio=false;padding=false",
)

import run_benchmark as rb  # noqa: E402


MAX_NEW_TOKENS = int(os.getenv("TRANSFORMERS_MAX_NEW_TOKENS", "1024"))
ATTN_IMPLEMENTATION = os.getenv("TRANSFORMERS_ATTN_IMPLEMENTATION", "eager")
DEVICE_MAP = os.getenv("TRANSFORMERS_DEVICE_MAP", "auto")
DTYPE_NAME = os.getenv("TRANSFORMERS_DTYPE", "bfloat16").lower()
PARTIAL_SAVE_EVERY = max(1, int(os.getenv("TRANSFORMERS_PARTIAL_SAVE_EVERY", "1")))


def resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    if dtype_name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported TRANSFORMERS_DTYPE: {dtype_name}")


def move_inputs_to_model(inputs: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generate_text(processor: Any, model: torch.nn.Module, messages: list[dict[str, str]]) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = move_inputs_to_model(dict(inputs), model)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


def build_error_case(
    i: int,
    record: dict[str, Any],
    expected_json: dict[str, Any],
    predicted_json: dict[str, Any],
    generated_text: str,
    record_stats: dict[str, dict[str, int]],
    request_time_seconds: float,
) -> dict[str, Any] | None:
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
        "audio_path": None,
        "audio_error": None,
        "request_time_seconds": request_time_seconds,
    }


def add_stats(global_stats: dict[str, dict[str, int]], record_stats: dict[str, dict[str, int]]) -> None:
    for key, stats in record_stats.items():
        if key not in global_stats:
            global_stats[key] = {"tp": 0, "fp": 0, "fn": 0}
        global_stats[key]["tp"] += stats["tp"]
        global_stats[key]["fp"] += stats["fp"]
        global_stats[key]["fn"] += stats["fn"]


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_partials(
    results: list[dict[str, Any] | None],
    error_cases: list[dict[str, Any]],
    completed: int,
    total: int,
    elapsed_seconds: float,
) -> None:
    save_json(os.path.join(rb.OUTPUT_DIR, "results.partial.json"), results)
    save_json(os.path.join(rb.OUTPUT_DIR, "error_cases.partial.json"), error_cases)
    save_json(
        os.path.join(rb.OUTPUT_DIR, "transformers_progress.json"),
        {"completed": completed, "total": total, "elapsed_seconds": elapsed_seconds},
    )


def calculate_metrics(
    global_stats: dict[str, dict[str, int]],
    request_count: int,
    total_request_time_seconds: float,
) -> dict[str, Any]:
    final_metrics: dict[str, Any] = {}
    sorted_keys = sorted(global_stats.keys())

    print("\nMetrics per field:")
    for key in sorted_keys:
        stats = global_stats[key]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
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
        print(f"{key}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f} (TP={tp}, FP={fp}, FN={fn})")

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
    avg_request_time_seconds = total_request_time_seconds / request_count if request_count > 0 else 0.0

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
        "total_request_time_seconds": total_request_time_seconds,
        "avg_request_time_seconds": avg_request_time_seconds,
        "model_name": rb.MODEL_NAME,
        "model_path": MODEL_PATH,
        "backend": "transformers",
        "experiment_args": rb.EXPERIMENT_ARGS,
        "batch_size": 1,
        "request_timeout_s": rb.REQUEST_TIMEOUT_S,
        "enable_system_prompt_block_padding": rb.ENABLE_SYSTEM_PROMPT_BLOCK_PADDING,
        "request_audio": False,
        "audio_dir": None,
        "audio_format": rb.AUDIO_FORMAT,
        "audio_voice": rb.AUDIO_VOICE,
        "block_size": None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "device_map": DEVICE_MAP,
        "dtype": DTYPE_NAME,
    }
    return final_metrics


def main() -> None:
    dtype = resolve_dtype(DTYPE_NAME)
    print("Using local Transformers backend")
    print(f"Using model path: {MODEL_PATH}")
    print(f"Using model name: {rb.MODEL_NAME}")
    print(f"Output dir: {rb.OUTPUT_DIR}")
    print(f"Experiment args: {rb.EXPERIMENT_ARGS if rb.EXPERIMENT_ARGS else '(default)'}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Attention implementation: {ATTN_IMPLEMENTATION}")
    print(f"Device map: {DEVICE_MAP}")
    print(f"Dtype: {DTYPE_NAME}")
    print(f"Enable system prompt block padding: {rb.ENABLE_SYSTEM_PROMPT_BLOCK_PADDING}")
    print("Request audio output: False")
    print(f"Loading data from {rb.DATASET_PATH}...")
    data = rb.load_data(rb.DATASET_PATH)
    print(f"Loaded {len(data)} records.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        device_map=DEVICE_MAP,
        attn_implementation=ATTN_IMPLEMENTATION,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    system_padding_suffix = rb.prepare_padding_suffix()
    results: list[dict[str, Any] | None] = [None] * len(data)
    error_cases: list[dict[str, Any]] = []
    global_stats: dict[str, dict[str, int]] = {}
    total_request_time_seconds = 0.0
    start_all = time.perf_counter()

    print("Generating...")
    for i, record in enumerate(data):
        print(f"[{i + 1}/{len(data)}] generating...", flush=True)
        messages = rb.construct_messages(record, system_padding_suffix)
        start = time.perf_counter()
        generated_text = ""
        try:
            generated_text = generate_text(processor, model, messages)
        except Exception as exc:
            print(f"[{i + 1}/{len(data)}] generation error: {exc}", flush=True)
        request_time_seconds = time.perf_counter() - start

        predicted_json = rb.parse_json(generated_text)
        expected_json = record["expected_output"]
        record_stats = rb.compare_record(expected_json, predicted_json)
        error_case = build_error_case(
            i,
            record,
            expected_json,
            predicted_json,
            generated_text,
            record_stats,
            request_time_seconds,
        )
        if error_case is not None:
            error_cases.append(error_case)

        results[i] = {
            "input": record["input_text"],
            "expected": expected_json,
            "predicted": predicted_json,
            "generated_text": generated_text,
            "audio_path": None,
            "audio_format": rb.AUDIO_FORMAT,
            "audio_bytes": 0,
            "audio_error": None,
            "request_time_seconds": request_time_seconds,
            "metrics": record_stats,
        }
        total_request_time_seconds += request_time_seconds
        add_stats(global_stats, record_stats)

        if (i + 1) % PARTIAL_SAVE_EVERY == 0 or i + 1 == len(data):
            save_partials(results, error_cases, i + 1, len(data), time.perf_counter() - start_all)
        print(
            f"[{i + 1}/{len(data)}] done in {request_time_seconds:.2f}s; "
            f"parsed_keys={len(predicted_json)}; errors={len(error_cases)}",
            flush=True,
        )

    final_metrics = calculate_metrics(global_stats, len(data), total_request_time_seconds)
    save_json(os.path.join(rb.OUTPUT_DIR, "results.json"), results)
    save_json(os.path.join(rb.OUTPUT_DIR, "metrics.json"), final_metrics)
    save_json(os.path.join(rb.OUTPUT_DIR, "error_cases.json"), error_cases)

    summary = final_metrics["summary"]
    print("\nSummary metrics:")
    print(f"Total Category Avg F1: {summary['total_category_avg_f1']:.4f}")
    print(
        "Total Sample Aggregate F1: "
        f"{summary['total_sample_aggregate_f1']:.4f} "
        f"(P={summary['precision']:.4f}, R={summary['recall']:.4f}, "
        f"TP={summary['tp']}, FP={summary['fp']}, FN={summary['fn']})"
    )
    print(f"Average Request Time: {summary['avg_request_time_seconds']:.4f}s ({len(data)} requests)")
    print(f"Results saved to {rb.OUTPUT_DIR}")
    print(f"Found {len(error_cases)} error cases.")


if __name__ == "__main__":
    main()
