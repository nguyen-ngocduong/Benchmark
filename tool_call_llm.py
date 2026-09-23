import asyncio
from eval.tool_call.core.chat_client import chat_completion_async
from dotenv import load_dotenv
import os

load_dotenv()

base_url = os.environ.get("BASE_URL")
model_name = os.environ.get("MODEL")

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Lấy thông tin thời tiết tại một địa điểm",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Tên thành phố, ví dụ: Hanoi"}
                },
                "required": ["location"]
            }
        }
    }
]

async def test_tool_call():

    response = await chat_completion_async(
        messages="Thời tiết hôm nay ở Hà Nội thế nào?",
        base_url=base_url,
        model=model_name,
        tools=tools_schema,
        tool_choice="auto"
    )

    if response.get("tool_calls"):
        print("Model đã chọn gọi Tool:", response["tool_calls"])
    else:
        print("Phản hồi text:", response["content"])

if __name__ == "__main__":
    asyncio.run(test_tool_call())
