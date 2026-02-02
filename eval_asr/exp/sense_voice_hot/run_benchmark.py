import os
import sys
import time
import json
import re
import subprocess
import torch
import inspect
from pathlib import Path
from funasr import AutoModel
from modelscope import snapshot_download

model_dir = snapshot_download('dengcunqin/SenseVoiceSmall_hotword')
import sys
sys.path.append(model_dir)
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
from sensevoice_bin_hot import SenseVoiceSmall


# Configuration
dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"
current_dir = Path(__file__).resolve().parent 
AUDIO_DIR = dataset_dir / "menu_audio_wav2"
SCRIPT_PATH = dataset_dir / "menu_script.txt"
REPORT_PATH = current_dir / "benchmark_report_menu_wav2.txt"
ERROR_LOG_PATH = current_dir / "error_cases_menu_wav2.txt"

# Hotwords from sak/prompts.py
HOTWORDS = "脉搏 呼吸 心率 体温 血压 体重 高压 低压 腋温 床头铃 透析 药疹 塞来昔布 地塞米松 透析 骶尾部"
HOTWORDS_MENU = "打开 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情"
HOTWORDS_ALL = HOTWORDS + " " + HOTWORDS_MENU


# Model Configuration
ASR_MODEL_ID = "dengcunqin/SenseVoiceSmall_hotword"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

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
    print("Initializing Model...")
    model_dir = snapshot_download('dengcunqin/SenseVoiceSmall_hotword')
    sys.path.append(model_dir)
    try:
        asr_model = SenseVoiceSmall(model_dir, 
                        batch_size=10, 
                        device = "cuda", 
                        quantize=False,   
                        disable_update=True,
                        trust_remote_code=False
                        )

    except Exception as e:
        print(f"Failed to load model: {e}")
        return

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
    
    for audio_file in audio_files:
        file_id = os.path.splitext(audio_file)[0]
        file_path = os.path.join(AUDIO_DIR, audio_file)
        
        if file_id not in ground_truth:
            print(f"Warning: No ground truth for {audio_file}, skipping evaluation.")
            continue
            
        ref_text = ground_truth[file_id]
        
        # Duration
        duration = get_audio_duration(file_path)
        wav_or_scp = [file_path]
        # Inference
        start_time = time.time()
        try:
            res = asr_model(
                wav_or_scp,
                hotwords_str=HOTWORDS_MENU,
                hotwords_score=1
            )
            hyp_text = ' '.join([rich_transcription_postprocess(i) for i in res])
        except Exception as e:
            print(f"Inference failed for {audio_file}: {e}")
            hyp_text = ""
            
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # Metrics
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
