import os
import json
import shutil
import tempfile
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from funasr import AutoModel
from openai import OpenAI

# ---------------- 配置部分 ----------------
ASR_MODEL_ID = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# LLM 配置
LLM_API_KEY = "sk-c4e4c4c4d3704dbd87a14759be773f6c"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen-flash"

# 全局变量
asr_model_instance = None
llm_client = None

# ---------------- 核心工具：格式转换 ----------------
def convert_audio_to_wav(source_path: str) -> str:
    """
    使用 FFmpeg 将任意音频转换为 Paraformer 最佳格式：
    格式: WAV (PCM)
    采样率: 16000Hz
    声道: 单声道 (Mono)
    """
    # 生成输出文件名，保留在临时目录
    output_path = source_path + "_processed.wav"
    
    # 构建 FFmpeg 命令
    # -y: 覆盖同名文件
    # -i: 输入文件
    # -ar 16000: 重采样为 16k (模型要求)
    # -ac 1: 混音为单声道 (模型要求)
    # -c:a pcm_s16le: 编码为 PCM 16bit (标准无压缩格式)
    command = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    
    try:
        # 执行命令，不输出冗余日志，除非出错
        subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 转换失败: {e.stderr.decode()}")
        raise RuntimeError("音频格式转换失败，请检查上传文件是否损坏。")
    except FileNotFoundError:
        raise RuntimeError("未找到 FFmpeg 工具，请确保服务器已安装 FFmpeg。")

# ---------------- 模型加载与生命周期管理 ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model_instance, llm_client
    print("\n[系统启动] 正在加载 Paraformer 模型到 GPU...")
    try:
        asr_model_instance = AutoModel(
            model=ASR_MODEL_ID,
            device="cuda", 
            disable_update=True
        )
        print("[系统启动] ASR 模型加载成功！")
        llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        print("[系统启动] LLM 客户端初始化完成！")
    except Exception as e:
        print(f"[系统启动] 模型加载失败: {e}")
        raise e
    yield
    print("\n[系统关闭] 释放资源...")
    del asr_model_instance

app = FastAPI(title="Medical Audio Analysis API", lifespan=lifespan)

# ---------------- 功能函数 ----------------
def extract_medical_info(text):
    system_prompt = """
你是一个专业的医疗数据录入助手。你的任务是从用户的**语音识别文本**中提取生命体征数据。
**注意：输入文本可能包含严重的语音识别错误（同音字），你需要根据发音相似性和医疗上下文推断正确的术语。**

### 提取目标字段
1. **temperature** (体温)
2. **pulse** (脉搏) - *常见误识：麦宝、麦博、脉博、买包*
3. **heartRate** (心率) - *常见误识：心律、心理、行率、刑率*
4. **breath** (呼吸) - *常见误识：夫妻、腹气、复习、呼气*
5. **bloodPressure** (血压) - *常见误识：雪丫、学压*
6. **weight** (体重)

### 关键约束规则
1. **模糊纠错**：必须基于拼音/发音相似性修正术语。例如将 "夫妻" 纠正为 "呼吸"，"麦宝" 纠正为 "脉搏"，“刑率” 纠正为 “心率”
2. **数据类型**：所有提取的数字必须严格转换为**字符串格式**（例如："36.5"）。
3. **血压格式**：必须统一转换为 "**低压/高压**" 的字符串格式。
   - 示例："高压120低压80" -> "80/120"。
   - 示例："血压80 110" -> "80/110"。
4. **缺失处理**：未提及的字段值设为 `null`。
5. **输出格式**：仅输出标准的 JSON 字符串，严禁包含 Markdown 标记。
6. **口语数值修正（关键）**：处理三位数时，若用户说“一百X”且未明确说“零”，需按口语习惯补全十位。
   - 示例：“一百一” -> "110"（而非101）。
   - 示例：“一百二” -> "120"（而非102）。
   - 示例：“一百三” -> "130"（而非103）。
   - 示例：“一百四” -> "140"（而非104）。
   - 示例：“一百五” -> "150"（而非105）。
   - 只有明确说“一百零三”时才输出 "103"。

### 示例 (Few-Shot Examples)

**输入：**
"病人也是个老病号了，今天体温三十七度二，心率每分钟88次，血压量了一下是85到130。"
**输出：**
{
  "temperature": "37.2",
  "pulse": null,
  "heartRate": "88",
  "breath": null,
  "bloodPressure": "85/130",
  "weight": null
}

**输入：**
"麦宝六十，夫妻三十五，学压八十一百二。"
**输出：**
{
  "temperature": null,
  "pulse": "60",
  "heartRate": null,
  "breath": "35",
  "bloodPressure": "80/120",
  "weight": null
}

**输入：**
"麦宝七士，心率一百三。"
**输出：**
{
  "temperature": null,
  "pulse": "70",
  "heartRate": "130",
  "breath": null,
  "bloodPressure": null,
  "weight": null
}

**输入：**
"体温38度，脉博士84，血压九十一百四。"
**输出：**
{
  "temperature": "38",
  "pulse": "84",
  "heartRate": null,
  "breath": null,
  "bloodPressure": "90/140",
  "weight": null
}
"""
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请提取这段话里的数据：{text}"}
            ],
            temperature=0.1,
        )
        # 清理可能存在的 markdown 标记
        content = response.choices[0].message.content
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

# ---------------- API 接口定义 ----------------
@app.post("/api/agent")
def analyze_audio_endpoint(file: UploadFile = File(...)):
    global asr_model_instance
    if asr_model_instance is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    # 临时文件路径列表，用于最后统一清理
    temp_files_to_clean = []
    
    try:
        # 1. 保存原始上传文件 (可能是 m4a, amr, mp3 等)
        # 获取正确的文件后缀，这对于 FFmpeg 识别 amr 很重要
        suffix = os.path.splitext(file.filename)[1]
        if not suffix: suffix = ".tmp"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_raw:
            shutil.copyfileobj(file.file, tmp_raw)
            raw_file_path = tmp_raw.name
            temp_files_to_clean.append(raw_file_path)

        print(f"收到请求，原始文件: {raw_file_path}")

        # ==========================================
        # 2. [新增步骤] 调用 FFmpeg 进行转码
        # ==========================================
        print("正在进行音频标准化转码 (Target: 16k WAV)...")
        wav_file_path = convert_audio_to_wav(raw_file_path)
        temp_files_to_clean.append(wav_file_path) # 加入清理列表
        print(f"转码完成: {wav_file_path}")

        # 3. ASR 推理 (传入转码后的 wav 文件)
        res = asr_model_instance.generate(
            input=wav_file_path,  # <--- 注意这里改成了 wav_file_path
            batch_size_s=120,
            hotword= "脉搏 呼吸 心率 体温 血压 体重 高压 低压 度 次 分 二十 三十 四十 五十 六十 七十 八十 九十 一百"
        )
        
        asr_text = res[0]['text'] if res else ""
        print(f"识别文本: {asr_text}")

        # 4. LLM 提取
        result_data = {}
        if asr_text:
            result_data = extract_medical_info(asr_text)
        
        # 5. 构造返回
        response = {
            "status": "success",
            "raw_text": asr_text,
            "data": result_data
        }
        return response

    except RuntimeError as re:
        # 捕获转码错误
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        print(f"处理出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 6. 清理所有临时文件 (原始文件 + 转码后的 WAV)
        for path in temp_files_to_clean:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)