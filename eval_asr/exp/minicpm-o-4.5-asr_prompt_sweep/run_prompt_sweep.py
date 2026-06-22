import argparse
import json
import os
import re
import time
import types
import wave
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/models/OpenBMB/MiniCPM-o-4_5"
DEFAULT_DATASET_DIR = "/workspace/audio_llm_agent/dataset/asr_llm"

PROMPTS = {
    "minicpm_zh_official": "请仔细听这段音频片段，并将其内容逐字记录。",
    "minicpm_zh_no_explain": "请仔细听这段音频片段，并将其内容逐字记录。只输出转写文本，不要解释，不要换行。",
    "minicpm_en_official": "Please listen to the audio snippet carefully and transcribe the content.",
    "menu_terms_zh": (
        "这是一段中文医疗护理系统菜单指令音频。请逐字转写用户说的话，只输出中文转写文本，"
        "不要解释，不要换行。常见词包括：打开、查看、进入、启动、开启、执行、记录、核对、"
        "填写、录入、进行、浏览、发送、查找、上报、标本、输液、皮试、配液、口服、治疗、"
        "体征采集、护理记录、护理文书、护理文件、患者巡视、健康宣教、不良事件、推送通知、"
        "患者、患者列表、消息、通讯录、功能、界面。"
    ),
    "menu_terms_en": (
        "Transcribe the following Chinese speech segment into Chinese text.\n"
        "The speech is usually a short hospital app menu command. Useful domain words include: "
        "打开, 查看, 进入, 启动, 开启, 执行, 记录, 核对, 填写, 录入, 进行, 浏览, 发送, "
        "查找, 上报, 标本, 输液, 皮试, 配液, 口服, 治疗, 体征采集, 护理记录, 护理文书, "
        "护理文件, 患者巡视, 健康宣教, 不良事件, 推送通知, 患者, 患者列表, 消息, 通讯录, "
        "功能, 界面.\n"
        "Use these words only when they match the audio. Only output the transcription, with no newlines."
    ),
    "menu_terms_en_strict": (
        "Transcribe the following Chinese hospital app voice command into Chinese text.\n"
        "Only output the transcription, no explanation, no newline, no English, no pinyin, no dataset IDs, "
        "and do not repeat the phrase.\n"
        "The command is short and usually uses exactly one action plus one menu item. Common actions: 打开, "
        "查看, 进入, 启动, 开启, 执行, 记录, 核对, 填写, 录入, 进行, 浏览, 发送, 查找, 上报, 开始.\n"
        "Common menu/domain terms: 标本, 输液, 皮试, 配液, 口服, 治疗, 体征采集, 护理记录, 护理文书, "
        "护理文件, 患者巡视, 健康宣教, 不良事件, 推送通知, 患者, 患者列表, 消息, 通讯录.\n"
        "Important corrections: if the sound is a medical skin-test command, write 皮试, not 提示; write 输液, "
        "not 书页; write 配液, not 配页/配音; write 口服, not 口福/口红; write 通讯录 only once; "
        "distinguish 护理文件 from 护理文书 by the audio."
    ),
    "nurse_zh_strict": (
        "这是一段中文护理查房、病房巡视、抢救、交班或用药沟通场景的音频。请逐字转写护士或医护人员说的话。\n"
        "只输出完整转写文本，不要解释，不要换行，不要总结，不要添加没有听到的内容。\n"
        "数字请尽量按音频里的中文口语写法转写，例如一百三十八、八十二、三十六度七、零点七五克、十四点一十分，"
        "不要改写成138/82、36.7或0.75g。\n"
        "常见医学词包括：血压、体温、心率、呼吸、血氧、血氧饱和度、脉搏、瞳孔、对光反射、意识、嗜睡、昏迷、"
        "入量、出量、尿量、引流、皮肤、破损、切口、渗血、吸氧、氧流量、静脉通道、心肺复苏、抢救、查房、"
        "阿司匹林、阿托伐他汀、头孢曲松、潘托拉唑、呋塞米、地佐辛、肾上腺素、多巴胺、甘露醇、波立维、"
        "二甲双胍、地塞米松、低分子肝素、奥美拉唑、顺铂、昂丹司琼、阿卡波糖、氨氯地平。"
    ),
}

FILLER_PHRASES = ["嗯嗯", "呃呃", "嗯", "呃", "啊", "呀", "额", "欸", "诶", "哎"]


def get_audio_duration(file_path: Path) -> float:
    if file_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate) if rate else 0.0
        except Exception as exc:
            print(f"Error getting wav duration for {file_path}: {exc}")
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


def normalize_for_cer(text: str) -> str:
    text = "" if text is None else str(text)
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text)


def calculate_cer(reference: str, hypothesis: str):
    ref_clean = normalize_for_cer(reference)
    hyp_clean = normalize_for_cer(hypothesis)
    if not ref_clean:
        cer = 0.0 if not hyp_clean else 1.0
        return cer, len(hyp_clean), 0, ref_clean, hyp_clean
    dist = levenshtein_distance(ref_clean, hyp_clean)
    return dist / len(ref_clean), dist, len(ref_clean), ref_clean, hyp_clean


def parse_script(script_path: Path) -> dict[str, str]:
    ground_truth = {}
    current_id = None
    for line in script_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        id_match = re.match(r"^(\d+)[：.]", line)
        if id_match:
            current_id = id_match.group(1)
            continue
        if current_id:
            parts = re.findall(r"“([^”]+)”", line) or re.findall(r'"([^"]+)"', line)
            ground_truth[current_id] = " ".join(parts) if parts else line.strip("“”\"")
            current_id = None
    return ground_truth


def clean_generated_text(text: Any) -> str:
    cleaned = "" if text is None else str(text)
    cleaned = cleaned.split("<|tts_eos|>")[0]
    cleaned = re.sub(r"<\|[^|]+?\|>", "", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = cleaned.replace("\n", " ").strip()
    return cleaned.strip("“”\"' ")


def resolve_dtype(dtype_name: str):
    lowered = dtype_name.lower()
    if lowered == "auto":
        return "auto"
    if lowered in ("bf16", "bfloat16"):
        return torch.bfloat16
    if lowered in ("fp16", "float16"):
        return torch.float16
    if lowered in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


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


def load_model(model_path: str, dtype_name: str, attn_implementation: str):
    from transformers import AutoConfig, AutoModel, PreTrainedModel

    patch_transformers_dynamic_cache_compat()
    patch_transformers_tied_weights_compat(PreTrainedModel)
    dtype = resolve_dtype(dtype_name)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.init_vision = False
    config.init_audio = True
    config.init_tts = False

    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
    )
    model.eval().cuda()
    patch_minicpm_audio_encoder_attention(model)
    return model


def patch_transformers_dynamic_cache_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "seen_tokens"):
        return

    def get_seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        try:
            return self.get_seq_length()
        except Exception:
            return 0

    def set_seen_tokens(self, value):
        self._seen_tokens = value

    DynamicCache.seen_tokens = property(get_seen_tokens, set_seen_tokens)


def patch_minicpm_audio_encoder_attention(model: Any) -> None:
    """Make MiniCPM-o 4.5 audio encoder compatible with newer WhisperAttention.

    The bundled remote code expects Whisper attention to return
    (hidden_states, attn_weights, past_key_values). In this environment it
    returns only (hidden_states, attn_weights), which is enough for non-streaming
    ASR. Patch the layer forward method at runtime instead of editing cached
    model code.
    """

    apm = getattr(model, "apm", None)
    layers = getattr(apm, "layers", None)
    if not layers:
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )
        if len(attn_outputs) >= 3:
            hidden_states, attn_weights, past_key_values = attn_outputs[:3]
        else:
            hidden_states, attn_weights = attn_outputs[:2]
            past_key_values = None

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs

    for layer in layers:
        layer.forward = types.MethodType(forward, layer)


def transcribe(model: Any, prompt: str, audio_path: Path, max_new_tokens: int, do_sample: bool, temperature: float):
    import librosa

    audio_input, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    msgs = [{"role": "user", "content": [prompt, audio_input]}]
    if hasattr(model, "reset_session"):
        try:
            model.reset_session(reset_token2wav_cache=False)
        except TypeError:
            model.reset_session()

    kwargs = {
        "msgs": msgs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_tts_template": True,
        "generate_audio": False,
        "enable_thinking": False,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    return clean_generated_text(model.chat(**kwargs))


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep MiniCPM-o ASR prompts on menu audio.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--audio-subdir", default="menu_audio_wav_30db")
    parser.add_argument("--script-name", default="menu_script.txt")
    parser.add_argument("--prompt-modes", default=",".join(PROMPTS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
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

    print("MiniCPM-o ASR prompt sweep")
    print(f"  model_path: {args.model_path}")
    print(f"  audio_dir: {audio_dir}")
    print(f"  script_path: {script_path}")
    print(f"  files: {len(audio_files)}")
    print(f"  prompt_modes: {', '.join(prompt_modes)}")
    print(f"  dtype: {args.dtype}; attn: {args.attn_implementation}; do_sample: {args.do_sample}")

    model = load_model(args.model_path, args.dtype, args.attn_implementation)
    print("Model loaded.")

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

            duration = get_audio_duration(audio_path)
            start_time = time.time()
            error = None
            try:
                hyp_text = transcribe(
                    model=model,
                    prompt=prompt,
                    audio_path=audio_path,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                )
            except Exception as exc:
                hyp_text = ""
                error = str(exc)
                print(f"Inference failed for {audio_path}: {exc}", flush=True)

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
