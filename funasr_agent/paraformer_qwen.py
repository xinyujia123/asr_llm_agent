import os
import json
from funasr import AutoModel
from openai import OpenAI

# ---------------- 1. ASR 配置部分 (原有) ----------------
MODEL_ID = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
TEST_AUDIO = "autofile/test1.wav"

# ---------------- 2. LLM 配置部分 (新增) ----------------
# 这里以兼容 OpenAI 格式的 API 为例 (推荐使用 DeepSeek 或 阿里通义千问，便宜且中文能力强)
# 如果你没有 Key，需要去对应平台申请w
LLM_API_KEY = "sk-c4e4c4c4d3704dbd87a14759be773f6c" 
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 示例：DeepSeek 的地址，如果是 OpenAI 则不需要改


# 初始化 LLM 客户端
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

def extract_medical_info(text):
    """
    使用大模型从文本中提取医疗信息并转换为 JSON
    """
    print(f"--- 正在调用大模型提取信息... ---")
    
    # 定义系统提示词 (System Prompt) - 这是核心
    # 告诉模型它的角色，以及输出的严格格式
    system_prompt = """
    你是一个专业的医疗数据录入助手。你的任务是从用户的口述文本中提取生命体征数据，并输出为纯 JSON 格式。
    
    请遵循以下规则：
    1. 提取以下字段：
       - temperature (体温，浮点数)
       - pulse (脉搏，整数)
       - heartRate (心率，整数)
       - breath (呼吸，整数)
       - bloodPressure (血压，字符串，格式如 "120/80")
       - weight (体重，浮点数/整数)
    2. 如果文本中缺少某个字段，该字段的值设为 null。
    3. 自动将中文数字（如“三十七点二”）转换为阿拉伯数字（37.2）。
    4. 不要输出任何 Markdown 标记（如 ```json），只输出纯文本的 JSON 字符串。
    """

    try:
        response = client.chat.completions.create(
            model="qwen-flash", # 请根据你使用的服务商修改模型名称 (如 gpt-3.5-turbo, moonshot-v1-8k)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请提取这段话里的数据：{text}"}
            ],
            temperature=0.1, # 低温度可以让输出更稳定、格式更准确
        )
        
        json_str = response.choices[0].message.content
        
        # 尝试解析 JSON 确保格式正确
        data = json.loads(json_str)
        return data

    except Exception as e:
        print(f"LLM 提取失败: {e}")
        return None

def main():
    # ---------------- ASR 推理部分 ----------------
    print(f"1. 正在加载 Paraformer 模型...")
    
    model = AutoModel(
        model=MODEL_ID,
        device="cuda", 
        disable_update=True
    )
    
    print(f"2. 模型加载成功。开始识别: {TEST_AUDIO}")

    if not os.path.exists(TEST_AUDIO):
        print("错误：找不到音频文件，请检查路径。")
        return

    # ASR 识别
    res = model.generate(
        input=TEST_AUDIO,
        batch_size_s=5000, 
        hotword='体温 脉搏 血压', # 增加医疗相关的热词，提高识别率
    )
    
    # 获取识别后的文本
    asr_text = res[0]['text']
    print(f"\n[识别结果]: {asr_text}\n")

    # ---------------- LLM 提取部分 ----------------
    # 只有当识别出文本时才调用 LLM
    if asr_text:
        medical_json = extract_medical_info(asr_text)
        
        if medical_json:
            print("3. [提取结果 (JSON)]:")
            print(json.dumps(medical_json, indent=4, ensure_ascii=False))
            
            # 你可以在这里加入后续逻辑，比如保存到数据库
            # save_to_database(medical_json)
        else:
            print("无法生成有效的 JSON 数据。")

if __name__ == "__main__":
    main()