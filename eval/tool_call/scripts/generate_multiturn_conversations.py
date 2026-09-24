"""
Pipeline to generate multi-turn tool calling conversations from
Na0s/sft-ready-toughdata-quora-question-answer-dataset.

Output schema follows data.json:
- conversation_id: e.g. "conversation_0001"
- conversation_index: 1, 2, ...
- conversation_topic: Topic title
- turn_count: Total turns in this conversation
- source_row_ids: list of HF dataset row indices
- source_indices: [0, 1, 2, ...]
- messages: List of turns (system, user, assistant tool_call, tool response, assistant answer)
- tools: Optional tool definitions (from list_tool.json or web_search default)

Saves all generated conversations into eval/tool_call/datasets/multi_turn/samples_new.json
"""

import argparse
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

# Read default configurations from environment or fallbacks
BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

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
     * "snippet": Đoạn trích dẫn thông tin thực tế (tóm lược các sự kiện, số liệu chính được nhắc tới trong câu trả lời) dùng làm bằng chứng để Assistant trả lời.

Trả về kết quả ĐÚNG định dạng JSON sau (không kèm markdown ngoài khối json):
{{
  "user_message": "...",
  "queries": ["truy vấn 1", "truy vấn 2"],
  "results": [
    {{
      "title": "...",
      "snippet": "..."
    }}
  ],
  "assistant_answer": "..."
}}
"""

# Default topic configurations for generating 5 multi-turn conversations
TOPIC_CONFIGS = [
    {
        "topic_id": "conversation_0001",
        "name": "Chính trị & Quan hệ quốc tế (US Political Discourse)",
        "indices": [52924, 28221, 45756, 21968, 30680]
    },
    {
        "topic_id": "conversation_0002",
        "name": "Kinh tế học & Tiền mã hóa (Finance & Cryptocurrency)",
        "indices": [435, 1100, 2036, 201]
    },
    {
        "topic_id": "conversation_0003",
        "name": "Biến đổi khí hậu & Năng lượng tái tạo (Climate & Renewable Energy)",
        "indices": [576, 215, 1017, 4722]
    },
    {
        "topic_id": "conversation_0004",
        "name": "Y học & Chăm sóc sức khỏe (Health & Medicine)",
        "indices": [1216, 2116, 1792, 345]
    },
    {
        "topic_id": "conversation_0005",
        "name": "Lịch sử Thế chiến thứ II (WWII Military History)",
        "indices": [4656, 404, 4905, 308]
    },
    {
        "topic_id": "conversation_0006",
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
        except Exception:
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
    turn_idx: int,
    base_url: str,
    model: str,
    api_key: str
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
                base_url=base_url,
                model=model,
                api_key=api_key,
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


async def build_conversations(
    base_url: str = BASE_URL,
    model: str = MODEL,
    api_key: str = API_KEY,
    max_concurrency: int = 1
):
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

    # Check if data.json exists in root and seed it if needed
    data_json_path = project_root / "data.json"
    if data_json_path.exists():
        try:
            with open(data_json_path, "r", encoding="utf-8") as f:
                data_json_obj = json.load(f)
                cid = data_json_obj.get("conversation_id", "conversation_0001")
                if cid not in cached_convs:
                    cached_convs[cid] = data_json_obj
                    print(f"📦 Preloaded seed conversation {cid} from data.json.")
        except Exception as e:
            print(f"⚠️ Could not read data.json: {e}")

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
        print(f"🚀 Generating Conversation {c_idx}/{len(TOPIC_CONFIGS)}: {conv_id} - {t_name}")
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

            turn_data = await process_turn(
                en_question=en_q,
                en_answer=en_a,
                conv_id=conv_id,
                turn_idx=turn_idx,
                base_url=base_url,
                model=model,
                api_key=api_key
            )

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

        # Match exact schema of data.json
        conversation_obj = {
            "conversation_id": conv_id,
            "conversation_index": c_idx,
            "conversation_topic": t_name,
            "turn_count": len(indices),
            "source_row_ids": indices,
            "source_indices": list(range(len(indices))),
            "messages": messages
        }
        new_conversations.append(conversation_obj)
        cached_convs[conv_id] = conversation_obj

        # Save cache progressively
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cached_convs, f, ensure_ascii=False, indent=2)

    # Now write/append to eval/tool_call/datasets/multi_turn/samples_new.json
    samples_new_path = project_root / "eval" / "tool_call" / "datasets" / "multi_turn" / "samples_new.json"
    print(f"\n💾 Reading existing {samples_new_path}...")

    # Load existing items to merge
    existing_conversations = []
    if samples_new_path.exists():
        try:
            with open(samples_new_path, "r", encoding="utf-8") as f:
                raw_content = json.load(f)
                if isinstance(raw_content, list):
                    for it in raw_content:
                        # Keep existing conversations if not overwriting current generated ones
                        curr_id = it.get("conversation_id")
                        first_call = ""
                        for m in it.get("messages", []):
                            if m.get("tool_calls"):
                                first_call = m["tool_calls"][0].get("id", "")
                                break
                        # Filter out older legacy format if replacing
                        if curr_id not in [cfg["topic_id"] for cfg in TOPIC_CONFIGS] and not any(f"call_{cfg['topic_id']}" in first_call for cfg in TOPIC_CONFIGS):
                            existing_conversations.append(it)
                elif isinstance(raw_content, dict):
                    existing_conversations = [raw_content]
        except Exception as e:
            print(f"⚠️ Could not parse existing samples_new.json: {e}")

    print(f"   Found {len(existing_conversations)} existing conversation(s).")
    all_conversations = new_conversations + existing_conversations

    print(f"📝 Writing total {len(all_conversations)} conversations to {samples_new_path}...")
    with open(samples_new_path, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully wrote {len(all_conversations)} conversations to {samples_new_path}!")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn conversation dataset matching data.json schema.")
    parser.add_argument("--base_url", type=str, default=BASE_URL, help="LLM Base URL")
    parser.add_argument("--model", type=str, default=MODEL, help="Model name (e.g. google/gemma-4-31B-it)")
    parser.add_argument("--api_key", type=str, default=API_KEY, help="API key")
    args = parser.parse_args()

    asyncio.run(build_conversations(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key
    ))


if __name__ == "__main__":
    main()
