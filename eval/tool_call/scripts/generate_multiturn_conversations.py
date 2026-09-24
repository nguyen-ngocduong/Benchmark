"""
Pipeline to generate 5 multi-turn tool calling conversations from
Na0s/sft-ready-toughdata-quora-question-answer-dataset.

Pipeline steps:
1. Select candidate questions and answers across 5 distinct domains.
2. Dịch (Translate): Translate English questions and comprehensive answers into Vietnamese.
3. Gen Assistant Turn & Tool Response:
   - Generate web_search tool_calls with search queries.
   - Generate realistic tool snippets containing evidence.
   - Generate reference assistant final answers.
4. Concatenate into multi-turn conversations matching samples_new.json format.
5. Append into eval/tool_call/datasets/multi_turn/samples_new.json.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.tool_call.core.chat_client import chat_completion_async

load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.5-flash-lite"

PROMPT_GENERATE_TURN = """Bạn là chuyên gia xây dựng dữ liệu benchmark cho Agent đàm thoại đa lượt tiếng Việt có sử dụng công cụ tìm kiếm web (web_search).

Chúng tôi có một cặp Câu hỏi (Prompt) và Câu trả lời (Completion) gốc bằng tiếng Anh từ Quora:
[QUESTION (EN)]: {en_question}
[ANSWER (EN)]: {en_answer}

Nhiệm vụ của bạn là:
1. DỊCH VÀ TINH CHỈNH:
   - "user_message": Dịch câu hỏi sang tiếng Việt tự nhiên, phù hợp với ngữ cảnh hội thoại của một người dùng thực tế.
   - "assistant_answer": Dịch và tổng hợp câu trả lời sang tiếng Việt lưu loát, khoa học, đầy đủ thông tin chi tiết từ bản gốc. Đảm bảo dùng dấu ngoặc kép hợp lệ trong chuỗi JSON (escape các dấu ngoặc kép bên trong bằng \\").
2. SINH TOOL CALL (web_search):
   - "queries": Danh sách 2-3 truy vấn tìm kiếm ngắn gọn, chuẩn xác bằng tiếng Việt (hoặc kèm thuật ngữ chuyên môn) mà một Agent thông minh sẽ dùng để tìm kiếm thông tin cho câu hỏi này.
3. SINH TOOL SNIPPET:
   - "results": Mảng gồm 1-2 kết quả tìm kiếm giả lập, mỗi kết quả có:
     * "title": Tiêu đề bài viết/nguồn tin cậy (tiếng Việt hoặc tiếng Anh).
     * "url": Đường link giả định (ví dụ: https://example.com/...).
     * "snippet": Đoạn trích dẫn thông tin thực tế (tóm lược các sự kiện, số liệu chính được nhắc tới trong câu trả lời) dùng làm bằng chứng để Assistant trả lời.

Trả về kết quả ĐÚNG định dạng JSON sau (không kèm markdown ngoài khối json):
{{
  "user_message": "...",
  "queries": ["truy vấn 1", "truy vấn 2"],
  "results": [
    {{
      "title": "...",
      "url": "https://example.com/...",
      "snippet": "..."
    }}
  ],
  "assistant_answer": "..."
}}
"""

WEB_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search for information from the internet.",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "The list of search queries."
                }
            },
            "required": ["queries"]
        }
    }
}

TOPIC_CONFIGS = [
    {
        "topic_id": "conv3",
        "name": "Kinh tế học & Tiền mã hóa (Finance & Cryptocurrency)",
        "indices": [435, 1100, 2036, 201]
    },
    {
        "topic_id": "conv4",
        "name": "Biến đổi khí hậu & Năng lượng tái tạo (Climate & Renewable Energy)",
        "indices": [576, 215, 1017, 4722]
    },
    {
        "topic_id": "conv5",
        "name": "Y học & Chăm sóc sức khỏe (Health & Medicine)",
        "indices": [1216, 2116, 1792, 345]
    },
    {
        "topic_id": "conv6",
        "name": "Lịch sử Thế chiến thứ II (WWII Military History)",
        "indices": [4656, 404, 4905, 308]
    },
    {
        "topic_id": "conv7",
        "name": "Trí tuệ nhân tạo & Điện toán tiên tiến (AI & Advanced Computing)",
        "indices": [682, 992, 875, 5635]
    }
]

CACHE_FILE = project_root / "eval" / "tool_call" / "datasets" / "multi_turn" / ".generated_topics_cache.json"


def clean_json_output(raw_text: str) -> Optional[Dict[str, Any]]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding outer braces
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Regex extraction fallback
    try:
        user_msg = ""
        u_m = re.search(r'"user_message"\s*:\s*"((?:\\.|[^"\\])*)"', text)
        if u_m:
            user_msg = bytes(u_m.group(1), "utf-8").decode("unicode_escape")

        asst_ans = ""
        a_m = re.search(r'"assistant_answer"\s*:\s*"((?:\\.|[^"\\])*)"', text)
        if a_m:
            asst_ans = bytes(a_m.group(1), "utf-8").decode("unicode_escape")

        queries = []
        q_m = re.search(r'"queries"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if q_m:
            queries = [q.strip(' "\'') for q in q_m.group(1).split(",") if q.strip(' "\'')]

        results = []
        r_m = re.search(r'"results"\s*:\s*(\[[\s\S]*?\])\s*,\s*"assistant_answer"', text)
        if r_m:
            try:
                results = json.loads(r_m.group(1))
            except Exception:
                pass

        if not results:
            results = [{
                "title": "Thông tin tra cứu trực tuyến",
                "url": "https://example.com/search-result",
                "snippet": asst_ans[:300] if asst_ans else "Kết quả tra cứu."
            }]

        if user_msg and asst_ans:
            return {
                "user_message": user_msg,
                "queries": queries or ["tra cứu thông tin"],
                "results": results,
                "assistant_answer": asst_ans
            }
    except Exception as e:
        print(f"Fallback extraction error: {e}")

    return None


async def process_turn(
    en_question: str,
    en_answer: str,
    conv_id: str,
    turn_idx: int
) -> Dict[str, Any]:
    prompt = PROMPT_GENERATE_TURN.format(
        en_question=en_question,
        en_answer=en_answer
    )
    messages = [
        {"role": "system", "content": "You are a professional benchmark dataset creator. Return only valid JSON."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(1, 4):
        try:
            res = await chat_completion_async(
                base_url=BASE_URL,
                model=MODEL,
                api_key=API_KEY,
                messages=messages,
                temperature=0.2,
                max_tokens=3000
            )
            raw = res.get("content") or ""
            parsed = clean_json_output(raw)
            if parsed and all(k in parsed for k in ["user_message", "queries", "results", "assistant_answer"]):
                return parsed
            print(f"    [Turn {turn_idx} Attempt {attempt}] Could not parse valid JSON.")
        except Exception as e:
            print(f"    [Turn {turn_idx} Attempt {attempt}] Error: {e}")
            await asyncio.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to generate turn {turn_idx} for {conv_id}")


async def build_conversations():
    print("📥 Loading HuggingFace dataset Na0s/sft-ready-toughdata-quora-question-answer-dataset...")
    ds = load_dataset("Na0s/sft-ready-toughdata-quora-question-answer-dataset", split="train")

    # Load cache if exists
    cached_convs = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached_convs = json.load(f)
            print(f"📦 Loaded {len(cached_convs)} cached conversation(s).")
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")

    new_conversations = []

    for c_idx, cfg in enumerate(TOPIC_CONFIGS, start=1):
        conv_id = cfg["topic_id"]
        t_name = cfg["name"]
        indices = cfg["indices"]

        if conv_id in cached_convs:
            print(f"⚡ Using cached conversation: {conv_id} - {t_name}")
            new_conversations.append(cached_convs[conv_id])
            continue

        print(f"\n==================================================")
        print(f"🚀 Generating Conversation {c_idx}/5: {conv_id} - {t_name}")
        print(f"==================================================")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can search the web for information. Use web_search when external information would help answer the user's question accurately."
            }
        ]

        for turn_idx, ds_idx in enumerate(indices, start=1):
            row = ds[ds_idx]
            en_q = row["prompt"].strip()
            en_a = row["completion"].strip()
            print(f"  • Turn {turn_idx}/{len(indices)} (Source Row #{ds_idx}): \"{en_q[:55]}...\"")

            turn_data = await process_turn(en_q, en_a, conv_id, turn_idx)

            user_msg = turn_data["user_message"]
            queries = turn_data["queries"]
            results = turn_data["results"]
            assistant_ans = turn_data["assistant_answer"]

            call_id = f"call_{conv_id}_turn{turn_idx}"

            # 1. User turn
            messages.append({
                "role": "user",
                "content": user_msg
            })

            # 2. Assistant tool call turn
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"queries": queries}, ensure_ascii=False)
                        }
                    }
                ]
            })

            # 3. Tool response turn
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(results, ensure_ascii=False)
            })

            # 4. Assistant final response
            messages.append({
                "role": "assistant",
                "content": assistant_ans
            })

        conversation_obj = {
            "messages": messages,
            "tools": [WEB_SEARCH_TOOL_DEF]
        }
        new_conversations.append(conversation_obj)
        cached_convs[conv_id] = conversation_obj

        # Save cache progressively
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cached_convs, f, ensure_ascii=False, indent=2)

    # Now write/append to eval/tool_call/datasets/multi_turn/samples_new.json
    samples_new_path = project_root / "eval" / "tool_call" / "datasets" / "multi_turn" / "samples_new.json"
    print(f"\n💾 Reading existing {samples_new_path}...")

    existing_conversations = []
    if samples_new_path.exists():
        with open(samples_new_path, "r", encoding="utf-8") as f:
            raw_content = json.load(f)
            if isinstance(raw_content, list):
                # Only keep original items not starting with call_conv3..conv7
                for it in raw_content:
                    first_call = ""
                    for m in it.get("messages", []):
                        if m.get("tool_calls"):
                            first_call = m["tool_calls"][0].get("id", "")
                            break
                    if not any(f"call_conv{i}" in first_call for i in range(3, 8)):
                        existing_conversations.append(it)
            elif isinstance(raw_content, dict):
                existing_conversations = [raw_content]

    print(f"   Found {len(existing_conversations)} original conversation(s).")
    all_conversations = list(existing_conversations) + new_conversations

    print(f"📝 Writing total {len(all_conversations)} conversations to {samples_new_path}...")
    with open(samples_new_path, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully wrote {len(all_conversations)} conversations to {samples_new_path}!")


if __name__ == "__main__":
    asyncio.run(build_conversations())
