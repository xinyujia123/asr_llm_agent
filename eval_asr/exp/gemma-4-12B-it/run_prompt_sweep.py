import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from run_benchmark import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_PATH,
    calculate_cer,
    get_audio_duration,
    parse_script,
)


PROMPTS = {
    "official_original_language": (
        "Transcribe the following speech segment in its original language. "
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
        "and write 3 instead of three."
    ),
    "official_chinese": (
        "Transcribe the following speech segment in Chinese into Chinese text. "
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* Do not translate.\n"
        "* When transcribing numbers, write the digits."
    ),
    "official_chinese_simple": (
        "Transcribe this Chinese speech segment into Chinese text. "
        "Only output the transcription, with no newlines."
    ),
    "zh_direct": "请把下面这段语音转写成中文。只输出转写文字，不要解释，不要换行。",
    "zh_strict": (
        "请将下面这段中文语音逐字转写为中文文本。只输出转写结果，不要解释，"
        "不要换行，不要添加没有听到的内容。"
    ),
    "menu_terms_en": (
        "Transcribe the following Chinese speech segment into Chinese text.\n"
        "The speech is usually a short hospital app menu command. Useful domain words include: "
        "打开, 查看, 进入, 启动, 标本, 输液, 皮试, 配液, 口服, 治疗, 体征采集, 护理记录, "
        "护理文书, 护理文件, 患者巡视, 健康宣教, 不良事件, 推送通知, 患者, 消息, 通讯录, "
        "功能, 界面.\n"
        "Use these words only when they match the audio. Do not infer intent. "
        "Only output the transcription, with no newlines."
    ),
    "menu_terms_zh": (
        "这是一段中文医疗护理系统菜单指令音频。常见词包括：打开、查看、进入、启动、标本、"
        "输液、皮试、配液、口服、治疗、体征采集、护理记录、护理文书、护理文件、患者巡视、"
        "健康宣教、不良事件、推送通知、患者、消息、通讯录、功能、界面。"
        "请逐字转写音频，只输出听到的中文文本，不要解释，不要换行。"
    ),
    "menu_command_constrained": (
        "这是一段中文医疗护理系统菜单指令音频。请根据声音转写出用户说的话。"
        "可能出现的动作词包括：打开、查看、进入、启动、开启、执行、记录、核对、填写、录入、"
        "进行、浏览、发送、查找、上报。可能出现的对象词包括：标本、输液、皮试、配液、口服、"
        "治疗、体征采集、护理记录、护理文书、护理文件、患者巡视、健康宣教、不良事件、推送通知、"
        "患者、患者列表、消息、通讯录。"
        "不要输出解释，不要输出拼音，不要输出日文或英文；只输出最终中文转写文本。"
    ),
    "nurse_zh_strict": (
        "这是一段中文护理查房、病房巡视、抢救、交班或用药沟通场景的音频。请逐字转写护士或医护人员说的话。"
        "只输出完整转写文本，不要解释，不要换行，不要总结，不要添加没有听到的内容。"
        "数字请尽量按音频里的中文口语写法转写，例如一百三十八、八十二、三十六度七、零点七五克、十四点一十分，"
        "不要改写成138/82、36.7或0.75g。"
        "常见医学词包括：血压、体温、心率、呼吸、血氧、血氧饱和度、脉搏、瞳孔、对光反射、意识、嗜睡、昏迷、"
        "入量、出量、尿量、引流、皮肤、破损、切口、渗血、吸氧、氧流量、静脉通道、心肺复苏、抢救、查房、"
        "阿司匹林、阿托伐他汀、头孢曲松、潘托拉唑、呋塞米、地佐辛、肾上腺素、多巴胺、甘露醇、波立维、"
        "二甲双胍、地塞米松、低分子肝素、奥美拉唑、顺铂、昂丹司琼、阿卡波糖、氨氯地平。"
    ),
    "nurse_terms_en": (
        "Transcribe the following Chinese clinical nursing audio into Chinese text.\n"
        "Only output the transcription. Do not explain, summarize, translate, use pinyin, use English, or add newlines.\n"
        "Keep numbers in the spoken Chinese form when possible, for example 一百三十八、八十二, 三十六度七, "
        "零点七五克, 十四点一十分, instead of 138/82, 36.7, 0.75g.\n"
        "This is usually a ward round, nursing handoff, rescue, medication, vital-sign, intake-output, wound, drain, "
        "oxygen, blood pressure, temperature, heart-rate, respiration, SpO2, urine, consciousness, pupil, or skin "
        "assessment scenario. Useful medical terms include: 血压, 体温, 心率, 呼吸, 血氧, 血氧饱和度, 脉搏, 瞳孔, "
        "对光反射, 意识, 嗜睡, 昏迷, 入量, 出量, 尿量, 引流, 切口, 渗血, 静脉通道, 心肺复苏, 阿司匹林, "
        "阿托伐他汀, 头孢曲松, 潘托拉唑, 呋塞米, 地佐辛, 肾上腺素, 多巴胺, 甘露醇, 波立维, 二甲双胍, "
        "地塞米松, 低分子肝素, 奥美拉唑, 顺铂, 昂丹司琼, 阿卡波糖, 氨氯地平."
    ),
    "nurse_asr_guardrail_zh": (
        "你是一个自动语音识别系统，不是聊天助手。任务只有一个：把音频里真实听到的中文语音逐字转写出来。"
        "禁止回答问题，禁止续写，禁止根据医学场景猜测，禁止补全病历，禁止总结，禁止翻译，禁止输出拼音、英文或解释。"
        "如果听不清，只写最可能听到的中文词，不要发散。"
        "只输出一行中文转写文本，不要换行。"
        "数字按音频里的中文读法写，例如一百三十八、八十二、三十六度七、九十六、幺零八。"
    ),
    "nurse_asr_guardrail_en": (
        "You are an automatic speech recognition engine, not a chatbot.\n"
        "Your only task is to transcribe the audible Chinese speech exactly into Chinese text.\n"
        "Do not answer questions, continue the scenario, infer a medical record, summarize, translate, explain, "
        "use pinyin, or invent content. If uncertain, write the closest Chinese words you hear.\n"
        "Output one line only. Keep spoken numbers in Chinese form, such as 一百三十八、八十二, 三十六度七, 九十六, 幺零八."
    ),
    "nurse_no_punct_zh": (
        "请逐字转写下面这段中文护理场景语音。只输出中文转写文本，不要解释，不要换行。"
        "尽量不要添加标点符号；不要把中文数字改成阿拉伯数字；不要把医学术语改成同音词。"
        "重点听清床号、血压、体温、心率、呼吸、血氧、入量、出量、尿量、药名、管路、伤口、意识和瞳孔描述。"
    ),
}


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


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("<turn|>", "").replace("<eos>", "")
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ").strip()
    return text.strip("“”\"' ")


def flatten_parse_response(parsed: Any) -> str:
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("text", "content", "response", "answer", "final"):
            if key in parsed:
                return flatten_parse_response(parsed[key])
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(parsed, (list, tuple)):
        parts = [flatten_parse_response(item) for item in parsed]
        return " ".join(part for part in parts if part)
    return str(parsed)


def decode_generation(processor: Any, generated_ids: torch.Tensor) -> tuple[str, str, str]:
    raw = processor.decode(generated_ids, skip_special_tokens=False)
    parsed_text = ""
    if hasattr(processor, "parse_response"):
        try:
            parsed_text = flatten_parse_response(processor.parse_response(raw))
        except Exception as exc:
            parsed_text = f"__parse_response_error__ {exc}: {raw}"
    parsed_clean = clean_text(parsed_text)
    fallback_clean = clean_text(raw)
    hyp = parsed_clean or fallback_clean
    return raw, parsed_clean, hyp


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Gemma4 ASR prompts on the same audio set.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--audio-subdir", default="menu_audio_wav_30db")
    parser.add_argument("--script-name", default="menu_script.txt")
    parser.add_argument("--prompt-modes", default=",".join(PROMPTS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-prefix", default="prompt_sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    current_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir)
    audio_dir = dataset_dir / args.audio_subdir
    script_path = dataset_dir / args.script_name

    prompt_modes = [mode.strip() for mode in args.prompt_modes.split(",") if mode.strip()]
    unknown = [mode for mode in prompt_modes if mode not in PROMPTS]
    if unknown:
        raise ValueError(f"Unknown prompt modes: {unknown}. Available: {sorted(PROMPTS)}")

    ground_truth = parse_script(script_path)
    audio_files = sorted(
        [p for p in audio_dir.iterdir() if p.suffix.lower() in {".amr", ".wav"}],
        key=lambda p: int(p.stem),
    )
    if args.limit > 0:
        audio_files = audio_files[: args.limit]

    print("Prompt sweep configuration")
    print(f"  model_path: {args.model_path}")
    print(f"  audio_dir: {audio_dir}")
    print(f"  script_path: {script_path}")
    print(f"  files: {len(audio_files)}")
    print(f"  prompt_modes: {', '.join(prompt_modes)}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"processor.parse_response: {hasattr(processor, 'parse_response')}")
    print("Loading model...")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    summary = []
    all_results = {}
    for mode in prompt_modes:
        print(f"\n=== Prompt mode: {mode} ===", flush=True)
        prompt = PROMPTS[mode]
        results = []
        total_dist = 0
        total_len = 0
        total_duration = 0.0
        total_inference_time = 0.0

        for idx, audio_path in enumerate(audio_files, start=1):
            file_id = audio_path.stem
            ref_text = ground_truth.get(file_id)
            if ref_text is None:
                continue

            messages = build_messages(prompt, audio_path)
            duration = get_audio_duration(audio_path)
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
                raw, parsed_text, hyp_text = decode_generation(processor, outputs[0][input_len:])
                error = None
            except Exception as exc:
                raw = ""
                parsed_text = ""
                hyp_text = ""
                error = str(exc)

            inference_time = time.time() - start_time
            cer, dist, length, ref_eval, hyp_eval = calculate_cer(ref_text, hyp_text)
            total_dist += dist
            total_len += length
            total_duration += duration
            total_inference_time += inference_time
            result = {
                "id": file_id,
                "ref": ref_text,
                "hyp": hyp_text,
                "raw_response": raw,
                "parsed_response": parsed_text,
                "ref_eval": ref_eval,
                "hyp_eval": hyp_eval,
                "cer": cer,
                "duration": duration,
                "inference_time": inference_time,
                "rtf": inference_time / duration if duration > 0 else 0.0,
                "error": error,
            }
            results.append(result)
            print(
                f"[{idx}/{len(audio_files)}] {mode} id={file_id} "
                f"cer={cer:.4f} ref={ref_text} hyp={hyp_text}",
                flush=True,
            )

        overall_cer = total_dist / total_len if total_len > 0 else 0.0
        overall_rtf = total_inference_time / total_duration if total_duration > 0 else 0.0
        item = {
            "prompt_mode": mode,
            "total_files": len(results),
            "overall_cer": overall_cer,
            "overall_accuracy": 1 - overall_cer,
            "total_duration": total_duration,
            "total_inference_time": total_inference_time,
            "overall_rtf": overall_rtf,
            "avg_latency": total_inference_time / len(results) if results else 0.0,
            "prompt": prompt,
        }
        summary.append(item)
        all_results[mode] = results
        print(
            f"SUMMARY {mode}: CER={overall_cer:.4f}, "
            f"Accuracy={1 - overall_cer:.4f}, RTF={overall_rtf:.4f}",
            flush=True,
        )

    summary = sorted(summary, key=lambda x: x["overall_cer"])
    payload = {
        "model_path": args.model_path,
        "dataset_dir": str(dataset_dir),
        "audio_subdir": args.audio_subdir,
        "script_path": str(script_path),
        "limit": args.limit,
        "summary": summary,
        "results": all_results,
    }
    output_path = current_dir / f"{args.output_prefix}_{args.audio_subdir}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Ranking by CER ===")
    for rank, item in enumerate(summary, start=1):
        print(
            f"{rank}. {item['prompt_mode']}: CER={item['overall_cer']:.4f}, "
            f"Accuracy={item['overall_accuracy']:.4f}, RTF={item['overall_rtf']:.4f}"
        )
    print(f"Saved prompt sweep results to {output_path}")


if __name__ == "__main__":
    main()
