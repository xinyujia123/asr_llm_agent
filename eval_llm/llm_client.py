from openai import OpenAI
from zai import ZhipuAiClient
import time
from .config import API_CONFIG
import re

class LLMClient:
    def __init__(self, provider: str = "qwen"):
        config = API_CONFIG.get(provider)
        if not config:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        self.model = config["model"]

    def chat(self, system_prompt: str, user_input: str, temperature: float = 0.0, thinking: bool = False) -> str:
        max_retries = 3
        wait_seconds = 5
        
        for attempt in range(max_retries + 1):
            try:
                kwargs= {
                    'model': self.model,
                    'messages': [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                }
                
                if 'baichuan' in self.model.lower():
                    kwargs['messages'] =[
                        {"role": "user", "content": f"【系统指令】\n{system_prompt}在json后面附上你的推理原因。\n\n【用户输入】\n{user_input}"},
                    ]
                if 'kimi' not in self.model:
                    kwargs["temperature"] = temperature
                if not thinking and 'kimi' in self.model:
                    kwargs['extra_body'] = {"thinking": {"type": "disabled"}}
                if thinking and 'qwen' in self.model:
                    kwargs['extra_body'] = {"enable_thinking": True}
                #deepseek 的thinking模型在于模型选择

                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content, getattr(response.choices[0].message, 'reasoning_content', None) if thinking else None
            except Exception as e:
                print(f"Error calling LLM ({self.model}) - Attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    sleep_time = wait_seconds * (2 ** attempt)
                    print(f"  Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    return "", None

class ZP_LLMClient:
    def __init__(self, provider: str = "qwen"):
        config = API_CONFIG.get(provider)
        if not config:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.client = ZhipuAiClient(api_key=config["api_key"])
        self.model = config["model"]

    def chat(self, system_prompt: str, user_input: str, temperature: float = 0.0, thinking: bool = False) -> str:
        max_retries = 6
        wait_seconds = 10

        for attempt in range(max_retries + 1):
            try:             
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    thinking={"type": "enabled" if thinking else "disabled"},
                    temperature= temperature,
                    max_tokens=65536,
                )
                return response.choices[0].message.content, response.choices[0].message.reasoning_content if thinking else None
                    #return response.choices[0].message.content
            except Exception as e:
                print(f"Error calling LLM ({self.model}) - Attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    sleep_time = wait_seconds * (3 ** attempt)
                    print(f"  Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    return "", None

class BaichuanClient:
    def __init__(self, provider: str = "qwen"):
        config = API_CONFIG.get(provider)
        if not config:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        self.model = config["model"]

    def chat(self, system_prompt: str, user_input: str, temperature: float = 0.0, thinking: bool = False) -> str:
        max_retries = 3
        wait_seconds = 5
        
        for attempt in range(max_retries + 1):
            try:
                kwargs= {
                    'model': self.model,
                    'messages': [
                        {"role": "user", "content": f"【系统指令】\n{system_prompt}**在输出json前先说出你的推理原因。**\n\n【用户输入】\n{user_input}"},
                    ],
                }
                kwargs["temperature"] = temperature
                response = self.client.chat.completions.create(**kwargs)
                result = response.choices[0].message.content
                match = re.search(r'^(.*?)(?=\{)', result, re.DOTALL)
                if match:
                    thinking_result = match.group(1).strip()
                else:
                    # 如果没找到 {，说明全篇可能都是推理，或者格式乱了
                    thinking_result = result
                return result, thinking_result if thinking else None
            except Exception as e:
                print(f"Error calling LLM ({self.model}) - Attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    sleep_time = wait_seconds * (2 ** attempt)
                    print(f"  Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    return "", None

if __name__ == "__main__":
    # 简单测试
    try:
        client = LLMClient("qwen")
        print("Client initialized")
    except Exception as e:
        print(f"Initialization failed: {e}")
