import os
import re
import uvicorn
import sys
import json
import time
import shutil
import tempfile 
import soundfile 
import threading
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from openai import AsyncOpenAI  # 使用异步客户端
from pathlib import Path
from funasr_onnx import ContextualParaformer, Fsmn_vad, CT_Transformer
from modelscope.hub.snapshot_download import snapshot_download

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from sak.prompts import MEDICAL_EXTRACTOR_PROMPT
from sak.utils import convert_audio_to_wav_sync

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_NAME = "qwen-turbo"

ASR_MODEL_ID = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
ONNX_MODEL_ID = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx"
PUNC_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-onnx"


# 全局变量容器iic
class ModelManager:
    llm_client = None
    asr_model = None
    vad_model = None
    punc_model = None
    # 使用细粒度锁，构建流水线并发
    vad_lock = threading.Lock()
    asr_lock = threading.Lock()
    punc_lock = threading.Lock()

models = ModelManager()

def run_asr(wav_path, models):
    # 1. IO 操作不加锁
    audio, _ = soundfile.read(wav_path, dtype="float32")
    
    # 2.1 运行 VAD (独立锁)
    with models.vad_lock:
        vad_output = models.vad_model(audio)
    
    segments = vad_output[0] if vad_output else []

    if not segments:
        print("VAD: No speech detected.")
        return ""

    # 2.2 切片并运行 ASR (独立锁)
    full_text = ""
    hotword_str = "脉搏 呼吸 心率 体温 血压 体重 高压 低压 度 次 分"
    
    # 预处理数据（切片），不需要锁
    # 但由于 ONNX 不支持 Batch，我们还得在一个循环里调模型
    # 这里为了简单，我们把整个 ASR 循环锁住。
    # 如果想更极致，可以把锁放到循环内部，但那样频繁抢锁开销也不小，且容易打乱顺序。
    # 考虑到 ASR 是最耗时的，锁住整个循环是合理的折中。
    with models.asr_lock:
        for beg_ms, end_ms in segments:
            if beg_ms == -1 or end_ms == -1:
                continue
                
            beg_sample = int(beg_ms * 16000 / 1000)
            end_sample = int(end_ms * 16000 / 1000)
            
            segment_audio = audio[beg_sample:end_sample]
            
            if len(segment_audio) < 160:
                continue
                
            try:
                res = models.asr_model(segment_audio, hotwords=hotword_str)
                if res:
                    item = res[0] if isinstance(res, list) else res
                    full_text += item.get('preds', [""])[0]
            except Exception as e:
                print(f"ASR 推理异常 (segment: {beg_ms}-{end_ms}ms): {e}")
                continue

    # 2.3 运行标点模型 (独立锁)
    if full_text:
        with models.punc_lock:
            punc_text, _ = models.punc_model(full_text)
        return punc_text
    
    return ""

# ---------------- 模型生命周期 ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[系统启动] 正在加载 Paraformer 模型到 GPU...")
    try:
        # 下载 PyTorch 模型 (作为主目录)
        model_dir = snapshot_download(ASR_MODEL_ID)
        
        # 检查是否需要手动补充 ONNX 文件 (因为 AutoModel 无法自动导出此模型，且 ONNX 版 ID 不被 AutoModel 识别)
        if not os.path.exists(os.path.join(model_dir, "model_quant.onnx")):
            print("Detected missing ONNX files in PyTorch dir, attempting to download and copy from ONNX repo...")
            try:
                onnx_dir = snapshot_download(ONNX_MODEL_ID)
                # Copy model_quant.onnx
                src_quant = os.path.join(onnx_dir, "model_quant.onnx")
                dst_quant = os.path.join(model_dir, "model_quant.onnx")
                if os.path.exists(src_quant) and not os.path.exists(dst_quant):
                    shutil.copy2(src_quant, dst_quant)
                    print(f"Copied {src_quant} to {dst_quant}")
                
                # Copy model_eb.onnx -> model_eb_quant.onnx (hack for ContextualParaformer expectation)
                src_eb = os.path.join(onnx_dir, "model_eb.onnx")
                dst_eb_quant = os.path.join(model_dir, "model_eb_quant.onnx")
                if os.path.exists(src_eb) and not os.path.exists(dst_eb_quant):
                    shutil.copy2(src_eb, dst_eb_quant)
                    print(f"Copied {src_eb} to {dst_eb_quant}")
                    
            except Exception as e:
                print(f"Failed to auto-fix ONNX files: {e}")

        # 加载并开启 quantize=True (会自动寻找 model_quant.onnx)
        models.asr_model = ContextualParaformer(
            model_dir, 
            batch_size=1, 
            quantize=True, 
            device_id=0 # 如果需要GPU推理可以开启
        )
        # funasr 部分逻辑会访问 self.language，这里手动补一个字段避免 AttributeError
        models.asr_model.language = "zh"
        print(f"ASR 引擎类型: {type(models.asr_model)}")

        # 加载 VAD 模型 (ONNX)
        print("\n[系统启动] 正在加载 VAD 模型...")
        vad_dir = snapshot_download(VAD_MODEL_ID)
        models.vad_model = Fsmn_vad(
            vad_dir,
            quantize=True,
            device_id=0
        )
        print("[系统启动] VAD 模型加载成功！")

        # 加载 PUNC 模型 (ONNX)
        print("\n[系统启动] 正在加载 PUNC 模型...")
        punc_dir = snapshot_download(PUNC_MODEL_ID)
        models.punc_model = CT_Transformer(
            punc_dir,
            quantize=True,
            device_id=0
        )
        print("[系统启动] PUNC 模型加载成功！")
        
        # 初始化异步 LLM 客户端
        models.llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        print("[系统启动] LLM 异步客户端初始化完成！")
    except Exception as e:
        print(f"[系统启动] 致命错误: {e}")
        raise e
    
    yield
    
    print("\n[系统关闭] 释放资源...")
    # 清理显存等操作（Python通常会自动处理，但显式删除引用是个好习惯）
    del models.asr_model
    if models.vad_model:
        del models.vad_model
    if models.punc_model:
        del models.punc_model

app = FastAPI(title="Medical Audio Analysis API", lifespan=lifespan)

# ---------------- 修改后的提取函数 ----------------
async def extract_medical_info_async(text: str):
    # 1. 加强 Prompt，明确要求 JSON 里的值必须是 String
    system_prompt = MEDICAL_EXTRACTOR_PROMPT
    
    try:
        response = await models.llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"分析文本：{text}"}
            ],
            temperature=0.1,
            # 如果模型支持 json_object 模式，务必保留
            response_format={"type": "json_object"} 
        )
        content = response.choices[0].message.content
    
        match = re.search(r'\{.*\}', content, re.DOTALL)
        raw_data = json.loads(match.group())

        # ==================================================
        # 【核心修改点】: 强制类型清洗
        # 即使 LLM 返回了数字 80，这里也会强制转为 "80"
        # ==================================================
        cleaned_data = {}
        target_keys = ["temperature", "pulse", "heartRate", "breath", "bloodPressure", "weight"]
        
        for key in target_keys:
            val = raw_data.get(key)
            
            if val is None:
                cleaned_data[key] = None
            else:
                # 无论原本是 int, float 还是 string，统一转为 string
                cleaned_data[key] = str(val)
                
        return cleaned_data

    except json.JSONDecodeError:
        print(f"LLM 返回了非 JSON 数据: {content}")
        return {"error": "LLM output format error"}
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

# ---------------- API 接口 ----------------
@app.post("/api/agent")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    if models.asr_model is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    temp_files = []
    
    try:
        # 1. 保存上传文件
        suffix = os.path.splitext(file.filename)[1] or ".tmp"
        # 使用 delete=False 因为 Windows 下打开的文件不能再次打开，Linux 下比较宽松
        # 这里手动管理删除更稳妥
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_raw:
            shutil.copyfileobj(file.file, tmp_raw)
            raw_path = tmp_raw.name
            temp_files.append(raw_path)

        # 2. [异步非阻塞] 运行 FFmpeg
        # run_in_threadpool 将同步函数放入线程池，释放 Event Loop
        wav_path = await run_in_threadpool(convert_audio_to_wav_sync, raw_path)
        temp_files.append(wav_path)

        # 3. [异步非阻塞] ASR 推理
        # ASR 推理是计算密集型，必须放入线程池，否则会卡死整个 API
        # 3.1 记录 ASR 开始时间
        asr_start_time = time.time()
        asr_text = await run_in_threadpool(run_asr, wav_path, models)
        asr_duration = time.time() - asr_start_time
        print(f"识别结果: {asr_text} (asr耗时: {asr_duration:.2f}s)")
        # 4. [异步] LLM 提取
        result_data = {}
        if asr_text:
            llm_start_time = time.time()
            result_data = await extract_medical_info_async(asr_text)
            llm_duration = time.time() - llm_start_time
            print(f"LLM耗时: {llm_duration:.2f}s")
        
        return {
            "status": "success",
            "raw_text": asr_text,
            "data": result_data
        }

    except Exception as e:
        print(f"处理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 清理文件
        for path in temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"清理临时文件失败 {path}: {e}")

if __name__ == "__main__":
    # 建议生产环境使用 workers > 1，但要注意模型显存占用
    uvicorn.run(
        "paraformer_contextual_turbo_onnx:app", 
        host="0.0.0.0", 
        port=8000, 
        workers=2
        )
