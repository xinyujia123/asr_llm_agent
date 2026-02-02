import os
import json
import shutil
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from funasr import AutoModel
from openai import OpenAI

# ---------------- 配置部分 ----------------
ASR_MODEL_ID = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# LLM 配置 (保持你原本可用的配置)
LLM_API_KEY = "sk-c4e4c4c4d3704dbd87a14759be773f6c"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen-flash" 

# 全局变量，用于存放加载后的模型
asr_model_instance = None
llm_client = None

# ---------------- 模型加载与生命周期管理 ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    这个函数会在服务启动前执行，并在服务关闭后清理。
    目的是确保模型只加载一次，常驻显卡。
    """
    global asr_model_instance, llm_client
    
    print("\n[系统启动] 正在加载 Paraformer 模型到 GPU...")
    try:
        # 加载 ASR 模型
        asr_model_instance = AutoModel(
            model=ASR_MODEL_ID,
            device="cuda",  # 确保你有 N卡 并且装好了 CUDA
            disable_update=True
        )
        print("[系统启动] ASR 模型加载成功！")

        # 初始化 LLM 客户端
        llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        print("[系统启动] LLM 客户端初始化完成！")
        
    except Exception as e:
        print(f"[系统启动] 模型加载失败: {e}")
        raise e
        
    yield  # 服务开始运行
    
    # 这里可以写服务关闭时的清理逻辑
    print("\n[系统关闭] 释放资源...")
    del asr_model_instance

# 初始化 FastAPI 应用
app = FastAPI(title="Medical Audio Analysis API", lifespan=lifespan)

# ---------------- 功能函数 ----------------

def extract_medical_info(text):
    """调用 LLM 提取信息的辅助函数"""
    system_prompt = """
    你是一个专业的医疗数据录入助手。你的任务是从用户的口述文本中提取生命体征数据，并输出为纯 JSON 格式。
    1. 提取字段：temperature, pulse, heartRate, breath, bloodPressure, weight。
    2. 缺失字段设为 null。
    3. 中文数字转阿拉伯数字。
    4. 只输出纯 JSON 字符串，不要 Markdown。
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
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

# ---------------- API 接口定义 ----------------

@app.post("/api/agent")
def analyze_audio_endpoint(file: UploadFile = File(...)):
    """
    HTTP 接口：接收音频文件 -> ASR 识别 -> LLM 提取 -> 返回 JSON
    """
    global asr_model_instance
    
    if asr_model_instance is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    # 1. 保存上传的文件到临时目录
    # FunASR 通常需要文件路径，而不是内存流
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        print(f"收到请求，正在处理文件: {tmp_file_path}")
        
        # 2. ASR 推理 (直接使用全局变量 asr_model_instance)
        res = asr_model_instance.generate(
            input=tmp_file_path,
            batch_size_s=5000,
            hotword='体温 脉搏 血压'
        )
        
        asr_text = res[0]['text'] if res else ""
        print(f"识别文本: {asr_text}")

        # 3. LLM 提取
        result_data = {}
        if asr_text:
            result_data = extract_medical_info(asr_text)
        
        # 4. 构造返回结构
        response = {
            "status": "success",
            "raw_text": asr_text,
            "data": result_data
        }
        return response

    except Exception as e:
        print(f"处理出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 5. 清理临时文件
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# ---------------- 启动入口 ----------------
if __name__ == "__main__":
    import uvicorn
    # 启动服务，监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)