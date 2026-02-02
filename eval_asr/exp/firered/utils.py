import subprocess
import re
import json
# ---------------- 格式转换工具 ----------------
def convert_audio_to_wav_sync(source_path: str) -> str:
    """
    同步执行的 FFmpeg 转换函数，将被放入线程池运行
    """
    output_path = source_path + "_processed.wav"
    command = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    try:
        subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr.decode()}")
        raise RuntimeError("音频转码失败，文件可能已损坏")
    except FileNotFoundError:
        raise RuntimeError("服务器未安装 FFmpeg")

# ---------------- 语义提取agent ----------------
async def extract_medical_info_async(text: str, llm_client, llm_model_name: str, system_prompt: str):
    try:
        response = await llm_client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"分析文本：{text}"}
            ],
            temperature=0.1,
            # 如果模型支持 json_object 模式，务必保留
            response_format={"type": "json_object"} 
        )
        content = response.choices[0].message.content
        cleaned_data = extract_json(content)
        return cleaned_data

    except json.JSONDecodeError:
        print(f"LLM 返回了非 JSON 数据: {content}")
        return {"error": "LLM output format error"}
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

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
