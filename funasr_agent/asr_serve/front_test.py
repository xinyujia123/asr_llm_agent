import httpx
import asyncio
import time
import os
import sys

from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)
from sak.prompts import *
from sak.utils import *

HOTWORDS = "你是一个护理语音转录助手，这是需要注意的热词：诺和锐 塞来昔布 地塞米松 昂丹司琼 螺内酯 呋塞米"
#HOTWORDS = "你是一个护理语音转录助手，以下是热词：脉搏 呼吸 心率 体温 血压 体重 高压 低压 腋温 床头铃 透析 药疹 塞来昔布 地塞米松 透析 骶尾部"
HOTWORDS_MENU = "你是一个打开菜单语意义助手，这是热词：打开 查看 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 患者 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情"


async def run_benchmark(audio_path, url="http://127.0.0.1:8001/transcribe"):
    if not os.path.exists(audio_path):
        print(f"错误: 找不到测试文件 {audio_path}")
        return

    async with httpx.AsyncClient() as client:
        print(f"--- 开始测试: {os.path.basename(audio_path)} ---")

        # 1. 记录客户端起始时间
        client_start = time.perf_counter()
        
        with open(audio_path, "rb") as f:
            data = {
                "file_path": audio_path,  # 这里传的是字符串地址
                "hotwords": "打开,开启"
            }
            response = await client.post("http://127.0.0.1:8001/transcribe_by_path", json=data)
        
        # 2. 记录客户端结束时间
        client_end = time.perf_counter()
        
        if response.status_code == 200:
            result = response.json()
            
            # --- 核心指标计算 ---
            # 总耗时 (T_total)
            t_total = client_end - client_start
            # 模型推理耗时 (T_inference)
            t_inference = result["inference_time"]
            # 通信与系统开销 (T_comm = T_total - T_inference)
            # 这包含了：网络传输、文件写入、FastAPI 路由调度、JSON 解析等
            t_comm = t_total - t_inference

            print(f"识别结果: {result['text']}")
            print("-" * 30)
            print(f"总响应耗时 (Client Total): {t_total:.4f}s")
            print(f"模型推理耗时 (Server GPU):  {t_inference:.4f}s")
            print(f"通信/系统损耗 (Overhead):   {t_comm:.4f}s (占比: {(t_comm/t_total)*100:.1f}%)")
            print("-" * 30)
        else:
            print(f"请求失败: {response.status_code}, {response.text}")

if __name__ == "__main__":
    # 替换为你本地的一个测试音频路径
    TEST_AUDIO = "/workspace/audio_llm_agent/dataset/asr_llm/nurse_audio_wav_30db/3.wav" 
    asyncio.run(run_benchmark(TEST_AUDIO))