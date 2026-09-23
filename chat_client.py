import asyncio
from eval.tool_call.core.chat_client import chat_completion_async
from dotenv import load_dotenv
import os

load_dotenv()

base_url = os.environ.get("BASE_URL")
model_name = os.environ.get("MODEL")

async def main():
    response = await chat_completion_async(
        messages="Xin chào, giới thiệu bản thân?",
        base_url=base_url,
        model=model_name,
        system_prompt="Bạn là trợ lý AI hữu ích.",
        temperature=0.7,
        max_tokens=256
    )

    print("--- Phản hồi từ LLM ---")
    print(response["content"])
    
    print("\n--- Thống kê hiệu năng ---")
    print(f"TTFT (độ trễ token đầu): {response['throughput']['ttft']}s")
    print(f"Tốc độ sinh token: {response['throughput']['output_token_per_seconds']} tokens/s")
    print(f"Tổng thời gian: {response['throughput']['exe_time']}s")
    print(f"Usage: {response['usage']}")

if __name__ == "__main__":
    asyncio.run(main())
