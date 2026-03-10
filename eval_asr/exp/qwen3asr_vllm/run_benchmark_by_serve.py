import os
import time
import re
import subprocess
from pathlib import Path

import httpx

# Configuration
dataset_dir = Path("/workspace/audio_llm_agent/dataset/asr_llm")
current_dir = Path(__file__).resolve().parent 
AUDIO_DIR = dataset_dir / "nurse_audio_wav_30db"
SCRIPT_PATH = dataset_dir / "nurse_script_tn.txt"
REPORT_PATH = current_dir / "benchmark_report_nurse_wav_30db_hot6.txt"
ERROR_LOG_PATH = current_dir / "error_cases_nurse_wav_30db_hot6.txt"

# Hotwords from sak/prompts.py
HOTWORDS_1 = "你是一个护理语音转录助手，这是需要注意的热词：诺和锐 塞来昔布 地塞米松 昂丹司琼 螺内酯 呋塞米"
HOTWORDS_2 = "hotwords：诺和锐 塞来昔布 地塞米松 昂丹司琼 螺内酯 呋塞米"
HOTWORDS_3 = "诺和锐 塞来昔布 地塞米松 昂丹司琼 螺内酯 呋塞米"
HOTWORDS_4 = "你是一个护理语音转录助手，这是需要注意的热词：诺和锐 塞来昔布 地塞米松 昂丹司琼 螺内酯 呋塞米。"
HOTWORDS_5 = "你是一个护理语音转录助手，这是需要注意的热词：诺和锐 塞来昔布 地塞米松 昂丹司琼 螺内酯 呋塞米。以下是语音识别内容："
HOTWORDS_6 = "你是一个护理语音转录助手，这是需要注意的热词：诺和锐 塞来昔布 地塞米松 昂丹司琼 螺内酯 呋塞米。如果没识别到内容，返回空字符串。"
HOTWORDS_MENU_1 = "你是打开菜单语意助手，这是热词：打开 查看 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情"
HOTWORDS_MENU_2 = "hotwords：打开 查看 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情"
HOTWORDS_MENU_3 = "打开 查看 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情"
HOTWORDS_MENU_4 = "你是打开菜单语意助手，这是热词：打开 查看 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情。如果没识别到内容，返回空字符串"
HOTWORDS_MENU_5 = "你是打开菜单语意助手，这是热词：'打开 查看 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情'。"
HOTWORDS_MENU_6 = "你是打开菜单语意助手，这是需要注意的热词：打开 查看 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情。"
HOTWORDS_MENU_7 = "打开 查看 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情"

HOTWORDS = HOTWORDS_6


ASR_SERVICE_URL = "http://127.0.0.1:7999/transcribe_by_path"

def get_audio_duration(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0.0

def levenshtein_distance(s1, s2):
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

ENABLE_FILLER_FILTER = True
FILLER_PHRASES = [
    "嗯嗯",
    "呃呃",
    "嗯",
    "呃",
    "啊",
    "呀",
    "额",
    "欸",
    "诶",
    "哎",
]


def remove_filler_phrases(text: str) -> str:
    if not text:
        return ""
    out = text
    for p in sorted(FILLER_PHRASES, key=len, reverse=True):
        out = out.replace(p, "")
    return out


def normalize_for_cer(text: str) -> str:
    if text is None:
        text = ""
    if ENABLE_FILLER_FILTER:
        text = remove_filler_phrases(text)
    return re.sub(r'[^\w\u4e00-\u9fff]', '', text)


def calculate_cer(reference, hypothesis):
    ref_clean = normalize_for_cer(reference)
    hyp_clean = normalize_for_cer(hypothesis)

    if not ref_clean:
        cer = 0.0 if not hyp_clean else 1.0
        return cer, len(hyp_clean), 0, ref_clean, hyp_clean

    dist = levenshtein_distance(ref_clean, hyp_clean)
    cer = dist / len(ref_clean)
    return cer, dist, len(ref_clean), ref_clean, hyp_clean

def parse_script(script_path):
    ground_truth = {}
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Script file not found: {script_path}")
        return {}
    
    current_id = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Regex to match ID line: "^\d+[：.]"
        # Examples: "1：..." or "10. ..."
        id_match = re.match(r'^(\d+)[：.]', line)
        if id_match:
            current_id = id_match.group(1)
        elif current_id:
             # Transcript line
             # Handle multi-speaker format: extract text within quotes to ignore role labels
             parts = re.findall(r'“([^”]+)”', line)
             if not parts:
                 # Try English quotes fallback
                 parts = re.findall(r'"([^"]+)"', line)
             
             if parts:
                 transcript = " ".join(parts)
             else:
                 transcript = line.strip('“”"')
                 
             ground_truth[current_id] = transcript
             current_id = None # Reset
    return ground_truth


def main():
    print(f"Using ASR service: {ASR_SERVICE_URL}")
    print("Parsing Ground Truth...")
    ground_truth = parse_script(SCRIPT_PATH)
    print(f"Loaded {len(ground_truth)} ground truth entries.")
    
    if not os.path.exists(AUDIO_DIR):
        print(f"Audio directory not found: {AUDIO_DIR}")
        return
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith('.amr') or f.endswith('.wav')], key=lambda x: int(os.path.splitext(x)[0]))
    
    results = []
    total_dist = 0
    total_len = 0
    total_duration = 0
    total_inference_time = 0

    error_cases = []
    
    print(f"Starting Benchmark on {len(audio_files)} files...")

    with httpx.Client(timeout=300.0) as client:
        for audio_file in audio_files:
            file_id = os.path.splitext(audio_file)[0]
            file_path = os.path.join(AUDIO_DIR, audio_file)

            if file_id not in ground_truth:
                print(f"Warning: No ground truth for {audio_file}, skipping evaluation.")
                continue

            ref_text = ground_truth[file_id]
            duration = get_audio_duration(file_path)

            request_start = time.time()
            try:
                response = client.post(
                    ASR_SERVICE_URL,
                    json={"file_path": file_path,
                         "hotwords": HOTWORDS
                         }
                )
                response.raise_for_status()
                payload = response.json()
                hyp_text = payload.get("text", "")
                inference_time = payload.get("inference_time", time.time() - request_start)
            except Exception as e:
                print(f"Inference failed for {audio_file}: {e}")
                hyp_text = ""
                inference_time = 0.0

            cer, dist, length, ref_eval, hyp_eval = calculate_cer(ref_text, hyp_text)

            total_dist += dist
            total_len += length
            total_duration += duration
            total_inference_time += inference_time

            results.append({
                "id": file_id,
                "ref": ref_text,
                "hyp": hyp_text,
                "ref_eval": ref_eval,
                "hyp_eval": hyp_eval,
                "cer": cer,
                "duration": duration,
                "inference_time": inference_time,
                "rtf": inference_time / duration if duration > 0 else 0
            })

            if cer > 0:
                error_cases.append(
                    f"ID: {file_id}\nRef: {ref_text}\nHyp: {hyp_text}\nRef(eval): {ref_eval}\nHyp(eval): {hyp_eval}\nCER: {cer:.4f}\n"
                )

            if int(file_id) % 10 == 0:
                print(f"Processed {file_id}/{len(audio_files)}...")

    # Final Stats
    overall_cer = total_dist / total_len if total_len > 0 else 0
    overall_rtf = total_inference_time / total_duration if total_duration > 0 else 0
    avg_latency = total_inference_time / len(results) if results else 0
    
    report = f"""
Benchmark Report
================
Total Files: {len(results)}
Overall CER: {overall_cer:.4f}
Overall Accuracy: {1-overall_cer:.4f}
Total Duration: {total_duration:.2f} s
Total Inference Time: {total_inference_time:.2f} s
Overall RTF: {overall_rtf:.4f}
Average Latency: {avg_latency:.4f} s/file
    """
    
    print(report)
    
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {REPORT_PATH}")
    except Exception as e:
        print(f"Failed to save report: {e}")
        
    try:
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(error_cases))
        print(f"Error cases saved to {ERROR_LOG_PATH}")
    except Exception as e:
        print(f"Failed to save error log: {e}")

if __name__ == "__main__":

    main()
