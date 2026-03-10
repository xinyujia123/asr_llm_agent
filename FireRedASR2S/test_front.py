import requests
import json
import time
import os

# VAD 服务地址
URL = "http://127.0.0.1:8002/vad_and_transcribe"

# 测试文件路径
# 使用一个存在的 wav 文件
TEST_FILE = "/workspace/audio_llm_agent/dataset/asr_llm/nurse_audio_wav_30db/25.wav"
#TEST_FILE = "/workspace/audio_llm_agent/code/agent_serve/debug_uploads/20260303_182623_face_10699980757000540818_full.wav"



def test_vad_service():
    if not os.path.exists(TEST_FILE):
        print(f"错误: 测试文件不存在: {TEST_FILE}")
        return

    print(f"正在测试文件: {TEST_FILE}")
    print(f"请求地址: {URL}")
    
    payload = {
        "file_path": TEST_FILE,
        "hotwords": "测试 菜单"
    }
    
    try:
        start_time = time.time()
        response = requests.post(URL, json=payload)
        end_time = time.time()
        
        print(f"请求耗时: {end_time - start_time:.4f}s")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("响应结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("错误信息:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("无法连接到服务，请确认服务是否已启动 (端口 8002)")
    except Exception as e:
        print(f"请求发生异常: {e}")

if __name__ == "__main__":
    test_vad_service()
