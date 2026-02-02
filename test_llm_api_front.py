from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="jia5201314...",          # 你的 API Key
    base_url="http://47.108.228.58:8000/v1"  # 注意最后要加 /v1
)

def test_vllm():
    try:
        response = client.chat.completions.create(
            model="qwen3-14b",         # 必须对应 --served-model-name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Qwen3 有什么新特性吗？"},
            ],
            stream=True               # 开启流式输出，体验更佳
        )

        print("AI 回复：", end="")
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n" + "-"*20)
        
    except Exception as e:
        print(f"连接失败: {e}")

if __name__ == "__main__":
    test_vllm()