import argparse
import json
import os
import re
import subprocess
import time
import wave
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/models/google/gemma-4-12B-it"
DEFAULT_DATASET_DIR = "/workspace/audio_llm_agent/dataset/asr_llm"

PROMPTS = {
    "official": (
        "Transcribe the following speech segment in Chinese into Chinese text. "
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* Do not translate.\n"
        "* When transcribing numbers, write the digits."
    ),
    "zh_strict": (
        "请将下面这段中文语音逐字转写为中文文本。只输出转写结果，不要解释，"
        "不要换行，不要添加没有听到的内容。"
    ),
    "menu_terms": (
        "Transcribe the following Chinese speech segment into Chinese text.\n"
        "The speech is usually a short hospital app menu command. Useful domain words include: "
        "打开, 查看, 标本, 输液, 皮试, 配液, 口服, 治疗, 体征采集, 护理记录, "
        "护理文书, 患者巡视, 健康宣教, 不良事件, 推送通知, 首页, 患者, 消息, "
        "通讯录, 我的, 计时提醒, 常用语管理, 关于我们, 患者详情, 功能, 界面.\n"
        "Use these words only when they match the audio. Do not infer intent. "
        "Only output the transcription, with no newlines."
    ),
    "menu_hot": (
        "你是医疗护理系统菜单指令的中文语音转写助手。常见词包括：打开、查看、标本、"
        "输液、皮试、配液、口服、治疗、体征采集、护理记录、护理文书、患者巡视、"
        "健康宣教、不良事件、推送通知、首页、患者、消息、通讯录、我的、计时提醒、"
        "常用语管理、关于我们、患者详情。请根据音频逐字转写，只输出中文转写结果，"
        "不要解释，不要换行。"
    ),
}

ENABLE_FILLER_FILTER = True
FILLER_PHRASES = ["嗯嗯", "呃呃", "嗯", "呃", "啊", "呀", "额", "欸", "诶", "哎"]


def get_audio_duration(file_path: Path) -> float:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception:
        if file_path.suffix.lower() == ".wav":
            try:
                with wave.open(str(file_path), "rb") as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    return frames / float(rate) if rate else 0.0
            except Exception as exc:
                print(f"Error getting wav duration for {file_path}: {exc}")
        else:
            print(f"ffprobe unavailable; cannot get duration for non-wav file: {file_path}")
        return 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def remove_filler_phrases(text: str) -> str:
    out = text or ""
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        out = out.replace(phrase, "")
    return out


def normalize_for_cer(text: str) -> str:
    if text is None:
        text = ""
    if ENABLE_FILLER_FILTER:
        text = remove_filler_phrases(text)
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text)


def calculate_cer(reference: str, hypothesis: str):
    ref_clean = normalize_for_cer(reference)
    hyp_clean = normalize_for_cer(hypothesis)

    if not ref_clean:
        cer = 0.0 if not hyp_clean else 1.0
        return cer, len(hyp_clean), 0, ref_clean, hyp_clean

    dist = levenshtein_distance(ref_clean, hyp_clean)
    cer = dist / len(ref_clean)
    return cer, dist, len(ref_clean), ref_clean, hyp_clean


def parse_script(script_path: Path) -> dict[str, str]:
    ground_truth = {}
    try:
        lines = script_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        print(f"Script file not found: {script_path}")
        return {}

    current_id = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        id_match = re.match(r"^(\d+)[：.]", line)
        if id_match:
            current_id = id_match.group(1)
        elif current_id:
            parts = re.findall(r"“([^”]+)”", line)
            if not parts:
                parts = re.findall(r'"([^"]+)"', line)
            transcript = " ".join(parts) if parts else line.strip("“”\"")
            ground_truth[current_id] = transcript
            current_id = None
    return ground_truth


def clean_generation(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("<turn|>", "").replace("<eos>", "")
    text = text.replace("\n", " ").strip()
    return text.strip("“”\"' ")


def build_messages(prompt: str, audio_path: Path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": str(audio_path)},
            ],
        }
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Gemma4 multimodal ASR on menu audio.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--audio-subdir", default="menu_audio_wav_30db")
    parser.add_argument("--script-name", default="menu_script.txt")
    parser.add_argument("--prompt-mode", choices=sorted(PROMPTS), default="official")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="0 means all files.")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--report-name", default="")
    parser.add_argument("--error-name", default="")
    parser.add_argument("--results-name", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    current_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir)
    audio_dir = dataset_dir / args.audio_subdir
    script_path = dataset_dir / args.script_name

    suffix = f"{args.audio_subdir}_{args.prompt_mode}"
    report_path = current_dir / (args.report_name or f"benchmark_report_{suffix}.txt")
    error_path = current_dir / (args.error_name or f"error_cases_{suffix}.txt")
    results_path = current_dir / (args.results_name or f"results_{suffix}.jsonl")

    print("Configuration")
    print(f"  model_path: {args.model_path}")
    print(f"  audio_dir: {audio_dir}")
    print(f"  script_path: {script_path}")
    print(f"  prompt_mode: {args.prompt_mode}")
    print(f"  report_path: {report_path}")

    ground_truth = parse_script(script_path)
    print(f"Loaded {len(ground_truth)} ground truth entries.")
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return

    audio_files = sorted(
        [p for p in audio_dir.iterdir() if p.suffix.lower() in {".amr", ".wav"}],
        key=lambda p: int(p.stem),
    )
    if args.start_id > 0:
        audio_files = [p for p in audio_files if int(p.stem) >= args.start_id]
    if args.limit > 0:
        audio_files = audio_files[: args.limit]
    print(f"Starting benchmark on {len(audio_files)} files.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompt = PROMPTS[args.prompt_mode]
    results = []
    error_cases = []
    total_dist = 0
    total_len = 0
    total_duration = 0.0
    total_inference_time = 0.0

    with results_path.open("w", encoding="utf-8") as results_file:
        for idx, audio_path in enumerate(audio_files, start=1):
            file_id = audio_path.stem
            ref_text = ground_truth.get(file_id)
            if ref_text is None:
                print(f"Warning: no ground truth for {audio_path.name}, skipping.")
                continue

            duration = get_audio_duration(audio_path)
            messages = build_messages(prompt, audio_path)

            start_time = time.time()
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(model.device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                generated = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
                hyp_text = clean_generation(generated)
            except Exception as exc:
                print(f"Inference failed for {audio_path.name}: {exc}")
                hyp_text = ""

            inference_time = time.time() - start_time
            cer, dist, length, ref_eval, hyp_eval = calculate_cer(ref_text, hyp_text)
            rtf = inference_time / duration if duration > 0 else 0.0

            total_dist += dist
            total_len += length
            total_duration += duration
            total_inference_time += inference_time

            item = {
                "id": file_id,
                "audio": str(audio_path),
                "ref": ref_text,
                "hyp": hyp_text,
                "ref_eval": ref_eval,
                "hyp_eval": hyp_eval,
                "cer": cer,
                "duration": duration,
                "inference_time": inference_time,
                "rtf": rtf,
                "prompt_mode": args.prompt_mode,
            }
            results.append(item)
            results_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            results_file.flush()

            if cer > 0:
                error_cases.append(
                    f"ID: {file_id}\n"
                    f"Ref: {ref_text}\n"
                    f"Hyp: {hyp_text}\n"
                    f"Ref(eval): {ref_eval}\n"
                    f"Hyp(eval): {hyp_eval}\n"
                    f"CER: {cer:.4f}\n"
                )

            print(
                f"[{idx}/{len(audio_files)}] id={file_id} "
                f"cer={cer:.4f} rtf={rtf:.4f} ref={ref_text} hyp={hyp_text}",
                flush=True,
            )

    overall_cer = total_dist / total_len if total_len > 0 else 0.0
    overall_rtf = total_inference_time / total_duration if total_duration > 0 else 0.0
    avg_latency = total_inference_time / len(results) if results else 0.0

    report = f"""
Benchmark Report
================
Model: {args.model_path}
Dataset: {dataset_dir}
Audio Subdir: {args.audio_subdir}
Script: {script_path}
Prompt Mode: {args.prompt_mode}
Total Files: {len(results)}
Overall CER: {overall_cer:.4f}
Overall Accuracy: {1 - overall_cer:.4f}
Total Duration: {total_duration:.2f} s
Total Inference Time: {total_inference_time:.2f} s
Overall RTF: {overall_rtf:.4f}
Average Latency: {avg_latency:.4f} s/file
Results JSONL: {results_path}
Error Cases: {error_path}
"""

    print(report)
    report_path.write_text(report, encoding="utf-8")
    error_path.write_text("\n".join(error_cases), encoding="utf-8")
    print(f"Report saved to {report_path}")
    print(f"Error cases saved to {error_path}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
