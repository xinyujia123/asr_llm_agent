from openai import OpenAI
from zai import ZhipuAiClient
from .config import API_CONFIG

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
        try:
            extra_body = {}
            if thinking and ('kimi' in self.model or 'glm' in self.model):
                extra_body = {"thinking": {"type": "disabled"}}
            elif thinking and 'qwen' in self.model:
                extra_body = {"enable_thinking": True}
            if 'kimi' in self.model:
                temperature = 1.0                           
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                extra_body=extra_body,
                temperature= 1.0 if 'kimi' in self.model else temperature,
                #response_format={"type": "json_object"} if "qwen" in self.model or "gpt" in self.model else None
            )
            return response.choices[0].message.content, response.choices[0].message.reasoning_content if thinking else None
                #return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM ({self.model}): {e}")
            return ""

class ZP_LLMClient:
    def __init__(self, provider: str = "qwen"):
        config = API_CONFIG.get(provider)
        if not config:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.client = ZhipuAiClient(api_key=config["api_key"])
        self.model = config["model"]

    def chat(self, system_prompt: str, user_input: str, temperature: float = 0.0, thinking: bool = False) -> str:
        try:             
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                thinking={"type": "enabled" if thinking else "disabled"},
                temperature= temperature,
                #response_format={"type": "json_object"} if "qwen" in self.model or "gpt" in self.model else None
            )
            return response.choices[0].message.content, response.choices[0].message.reasoning_content if thinking else None
                #return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM ({self.model}): {e}")
            return ""

if __name__ == "__main__":
    # 简单测试
    try:
        client = LLMClient("qwen")
        print("Client initialized")
    except Exception as e:
        print(f"Initialization failed: {e}")
