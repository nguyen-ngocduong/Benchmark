"""
Batch Async Data Generator for Multi-Turn Tool Calling Dataset.

Features:
- Reads Q&A from Hugging Face dataset (Na0s/sft-ready-toughdata-quora-question-answer-dataset).
- Executes concurrent batch inference (32/64 workers) with asyncio.Semaphore.
- In a single pass, LLM:
    1. Dịch User Question sang tiếng Việt tự nhiên.
    2. Sinh các câu Search Queries chuẩn cho web_search.
    3. Giả lập Snippet kết quả tìm kiếm (evidence ground-truth).
    4. Sinh Reference Assistant Answer dựa trên kết quả tìm kiếm.
- Assembles turns into the exact schema of `samples_new.json`.
- Thread-safe and resumable (skips already processed items).
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiofiles
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Root path setup
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client import chat_completion_async

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Multi-turn Tool-calling samples with high concurrency")
    
    # LLM Config
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "http://localhost:8288/v1"), help="LLM API Base URL")
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", "EMPTY"), help="API Key")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL", "Qwen/Qwen3.5-35B-A3B"), help="Model name")
    parser.add_argument("--max_concurrency", type=int, default=32, help="Số request chạy song song (e.g. 32 hoặc 64)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Nhiệt độ sinh văn bản")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max output tokens")
    
    # Dataset Config
    parser.add_argument("--num_samples", type=int, default=20, help="Số lượng conversation cần gen")
    parser.add_argument("--turns_per_conv", type=int, default=4, help="Số turn trong mỗi conversation")
    parser.add_argument("--output", type=str, default="eval/tool_call/datasets/multi_turn/batch_generated_samples.json", help="File output kết quả")
    parser.add_argument("--cache_file", type=str, default="eval/tool_call/datasets/multi_turn/.batch_gen_cache.jsonl", help="File cache lưu tạm kết quả")
    
    return parser.parse_args()


ARGS = parse_args()

PROMPT_GENERATE_TURN = """Bạn là chuyên gia xây dựng dữ liệu benchmark cho Agent đàm thoại đa lượt tiếng Việt có sử dụng công cụ tìm kiếm web (web_search).

Chúng tôi có một cặp Câu hỏi (Prompt) và Câu trả lời (Completion) gốc bằng tiếng Anh từ Quora:
[QUESTION (EN)]: {en_question}
[ANSWER (EN)]: {en_answer}

Nhiệm vụ của bạn là xử lý và trả về ĐÚNG 1 khối JSON (không kèm markdown thừa) với các trường sau:
1. "user_message": Dịch câu hỏi sang tiếng Việt tự nhiên, phù hợp với ngữ cảnh hội thoại của một người dùng thực tế.
2. "queries": Danh sách 2-3 truy vấn tìm kiếm ngắn gọn, chuẩn xác bằng tiếng Việt (hoặc thuật ngữ chuyên môn) mà một Agent thông minh sẽ dùng để tìm kiếm thông tin cho câu hỏi này.
3. "results": Mảng gồm 1-2 kết quả tìm kiếm giả lập làm bằng chứng:
   - "title": Tiêu đề bài viết/nguồn tin cậy.
   - "url": Đường link giả định (ví dụ: https://example.com/...).
   - "snippet": Đoạn trích dẫn tóm lược sự kiện, số liệu chính từ câu trả lời để làm bằng chứng cho Assistant trả lời.
4. "assistant_answer": Câu trả lời tổng hợp sang tiếng Việt lưu loát, khoa học, đầy đủ thông tin chi tiết dựa trên bằng chứng đã tìm được.

Định dạng JSON yêu cầu:
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


def clean_json_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """Làm sạch markdown block và parse JSON an toàn."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


async def process_single_turn(
    en_q: str,
    en_a: str,
    conv_id: str,
    turn_idx: int,
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Xử lý 1 turn thông qua LLM với semaphore điều tiết số luồng đồng thời."""
    prompt = PROMPT_GENERATE_TURN.format(en_question=en_q, en_answer=en_a)
    messages = [
        {"role": "system", "content": "You are a professional benchmark dataset creator. Return only valid JSON."},
        {"role": "user", "content": prompt}
    ]

    async with semaphore:
        for attempt in range(1, 4):
            try:
                res = await chat_completion_async(
                    base_url=ARGS.base_url,
                    model=ARGS.model,
                    api_key=ARGS.api_key,
                    messages=messages,
                    temperature=ARGS.temperature,
                    max_tokens=ARGS.max_tokens
                )
                raw_content = res.get("content") or ""
                parsed = clean_json_output(raw_content)
                if parsed and all(k in parsed for k in ["user_message", "queries", "results", "assistant_answer"]):
                    return parsed
            except Exception as e:
                if attempt == 3:
                    print(f"❌ [Conv {conv_id} Turn {turn_idx}] Error: {e}")
                await asyncio.sleep(1.0 * attempt)

    return None


async def process_single_conversation(
    conv_idx: int,
    indices: List[int],
    ds,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    cache_path: Path
) -> Optional[Dict[str, Any]]:
    """Xử lý toàn bộ các turn của 1 conversation."""
    conv_id = f"batch_conv_{conv_idx:03d}"
    
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

        turn_data = await process_single_turn(en_q, en_a, conv_id, turn_idx, semaphore)
        if not turn_data:
            print(f"⚠️ Thất bại khi sinh turn {turn_idx} cho {conv_id}. Bỏ qua hội thoại này.")
            return None

        call_id = f"call_{conv_id}_turn{turn_idx}"
        
        # 1. User
        messages.append({
            "role": "user",
            "content": turn_data["user_message"]
        })
        # 2. Assistant Call
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"queries": turn_data["queries"]}, ensure_ascii=False)
                    }
                }
            ]
        })
        # 3. Tool Result
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(turn_data["results"], ensure_ascii=False)
        })
        # 4. Assistant Answer
        messages.append({
            "role": "assistant",
            "content": turn_data["assistant_answer"]
        })

    conversation_obj = {
        "conv_id": conv_id,
        "messages": messages,
        "tools": [WEB_SEARCH_TOOL_DEF]
    }

    # Ghi ngay kết quả vào cache dạng JSONL (thread-safe)
    async with file_lock:
        async with aiofiles.open(cache_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(conversation_obj, ensure_ascii=False) + "\n")

    return conversation_obj


async def main():
    print(f"============================================================")
    print(f"🚀 BATCH LLM MULTI-TURN DATA GENERATOR")
    print(f"   Model Endpoint   : {ARGS.base_url} ({ARGS.model})")
    print(f"   Max Concurrency  : {ARGS.max_concurrency} parallel requests")
    print(f"   Target Samples   : {ARGS.num_samples} conversations ({ARGS.turns_per_conv} turns each)")
    print(f"============================================================")

    # 1. Tải dataset từ HuggingFace
    print("📥 Loading HuggingFace dataset 'Na0s/sft-ready-toughdata-quora-question-answer-dataset'...")
    ds = load_dataset("Na0s/sft-ready-toughdata-quora-question-answer-dataset", split="train")
    print(f"   Dataset size: {len(ds)} rows.")

    # 2. Kiểm tra cache đã có sẵn
    cache_path = Path(ARGS.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    done_conv_ids = set()
    completed_conversations = []
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        done_conv_ids.add(item.get("conv_id"))
                        completed_conversations.append(item)
                    except Exception:
                        pass
        print(f"📦 Đã tìm thấy {len(completed_conversations)} conversation trong cache. Sẽ tiếp tục những mục còn thiếu.")

    # 3. Chuẩn bị danh sách chỉ số (indices) cho các conversation
    conv_indices_map = {}
    current_row = 100  # Bắt đầu từ offset 100 để lấy câu phong phú
    for c in range(1, ARGS.num_samples + 1):
        cid = f"batch_conv_{c:03d}"
        if cid not in done_conv_ids:
            conv_indices_map[c] = [current_row + i for i in range(ARGS.turns_per_conv)]
        current_row += ARGS.turns_per_conv

    if not conv_indices_map:
        print("🎉 Tất cả các sample yêu cầu đã có trong cache!")
    else:
        print(f"⚡ Cần sinh mới {len(conv_indices_map)} conversation(s) với {ARGS.max_concurrency} luồng đồng thời...")

        semaphore = asyncio.Semaphore(ARGS.max_concurrency)
        file_lock = asyncio.Lock()

        tasks = [
            process_single_conversation(
                conv_idx=c_idx,
                indices=idx_list,
                ds=ds,
                semaphore=semaphore,
                file_lock=file_lock,
                cache_path=cache_path
            )
            for c_idx, idx_list in conv_indices_map.items()
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Batch Generating")
        for res in results:
            if res:
                completed_conversations.append(res)

    # 4. Lưu ra định dạng chuẩn `samples_new.json`
    out_path = Path(ARGS.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Định dạng theo chuẩn: messages + tools (bỏ trường metadata conv_id khi lưu)
    final_output_format = [
        {"messages": c["messages"], "tools": c["tools"]}
        for c in completed_conversations
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output_format, f, ensure_ascii=False, indent=2)

    print(f"\n============================================================")
    print(f"✅ HOÀN THÀNH!")
    print(f"   Đã lưu tổng cộng {len(final_output_format)} conversations vào:")
    print(f"   👉 {out_path.resolve()}")
    print(f"============================================================")


if __name__ == "__main__":
    asyncio.run(main())
