import requests
import json
import time  # 1. 引入 time 模块

# 服务端地址
API_URL = "http://localhost:8000/api/agent"
#API_URL = "http://118.121.197.67:8000/api/agent"
# 本地测试音频路径
TEST_AUDIO = "/workspace/dataset/asr_llm/nurse_audio/18.amr"

def test_api():
    print(f"正在准备上传音频: {TEST_AUDIO} ...")
    
    try:
        with open(TEST_AUDIO, "rb") as f:
            files = {"file": f}
            
            # 2. 记录开始时间
            start_time = time.time()
            
            print(">>> 开始发送请求...")
            # 发送 POST 请求
            response = requests.post(API_URL, files=files)
            
            # 3. 记录结束时间并计算差值
            end_time = time.time()
            elapsed_time = end_time - start_time
        
        # 4. 打印耗时 (保留2位小数)
        print(f"\n[耗时统计] 总耗时: {elapsed_time:.2f} 秒")
        
        if response.status_code == 200:
            print("--- 服务端返回成功 ---")
            result = response.json()
            
            print(f"原始识别文本: {result['raw_text']}")
            print("提取出的 JSON 数据:")
            print(json.dumps(result['data'], indent=4, ensure_ascii=False))
        else:
            print(f"请求失败: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_api()