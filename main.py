import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

#加载.env文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    定制LLM客户端
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout_str = timeout or os.getenv("LLM_TIMEOUT")
        timeout = int(timeout_str) if timeout_str else None

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型的名称，密钥，还有服务商地址需要提供在.env文件里面")
        self.client = OpenAI(api_key = apiKey, base_url = baseUrl, timeout = timeout)
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应
        """
        print(f"正在调用大语言模型{self.model}")
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages =messages,
                temperature = temperature,
                stream = True,
            )
            print("大语言模型响应成功：")
            collected_content =[]
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end = "", flush = True)
                collected_content.append(content)
            print()
            return "".join(collected_content)
        except Exception as e:
            print(f"调用LLM API时出现了错误: {e}")
            return None

if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)