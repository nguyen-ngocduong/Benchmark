"""
Batch Async Data Generator for Multi-Turn Tool Calling Dataset.

Features:
- Reads Q&A from Hugging Face dataset (Na0s/sft-ready-toughdata-quora-question-answer-dataset).
- Executes concurrent batch inference (16/32/64 workers) with asyncio.Semaphore.
- Clusters questions by topic using TF-IDF or LLM, ensuring NO duplicate questions in a conversation.
- Generates rich multi-turn conversations matching the exact schema of samples_new.json.
- Saves each batch of conversations (default: 16 conversations) to separate timestamped files:
  batches/samples__<timestamp>.json
- Resumable cache (.batch_gen_cache.jsonl) preventing redundant work.
- Merges all batches into final output file.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Root path setup
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from eval.tool_call.core.chat_client import chat_completion_async
except ImportError:
    try:
        from core.chat_client import chat_completion_async
    except ImportError:
        from client import chat_completion_async

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Multi-turn Tool-calling samples with high concurrency")
    
    # LLM Config
    default_base_url = os.getenv("BASE_URL", "https://more-starring-administration-cemetery.trycloudflare.com/v1")
    default_api_key = os.getenv("GEMINI_API_KEY", os.getenv("API_KEY", "EMPTY"))
    default_model = os.getenv("MODEL", "gemma-4-31B-it")
    # Normalize model name if it starts with google/
    if "more-starring-administration-cemetery" in default_base_url and default_model.startswith("google/"):
        default_model = default_model.replace("google/", "")

    parser.add_argument("--base_url", type=str, default=default_base_url, help="LLM API Base URL")
    parser.add_argument("--api_key", type=str, default=default_api_key, help="API Key")
    parser.add_argument("--model", type=str, default=default_model, help="Model name")
    parser.add_argument("--max_concurrency", type=int, default=16, help="Số request chạy song song (e.g. 16, 32)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Nhiệt độ sinh văn bản")
    parser.add_argument("--max_tokens", type=int, default=3000, help="Max output tokens")
    
    # Dataset Config
    parser.add_argument("--num_samples", type=int, default=32, help="Tổng số lượng conversation cần gen")
    parser.add_argument("--batch_size", type=int, default=16, help="Số conversation xử lý trong mỗi batch (mặc định: 16)")
    parser.add_argument("--turns_per_conv", type=int, default=4, help="Số turn trong mỗi conversation")
    parser.add_argument("--output_dir", type=str, default="eval/tool_call/datasets/multi_turn/batches", help="Thư mục chứa các file json của từng batch (samples__<thời_gian>.json)")
    parser.add_argument("--output", type=str, default="eval/tool_call/datasets/multi_turn/batch_generated_samples.json", help="File output tổng hợp tất cả batch")
    parser.add_argument("--cache_file", type=str, default="eval/tool_call/datasets/multi_turn/.batch_gen_cache.jsonl", help="File cache lưu tạm kết quả")
    parser.add_argument("--cluster_method", choices=["tfidf", "llm"], default="tfidf", help="Phương pháp phân cụm: tfidf (nhanh, chất lượng cao) hoặc llm")
    parser.add_argument("--mode", choices=["conversation", "per_turn"], default="conversation", help="Phương pháp sinh: conversation (1 pass mạch lạc) hoặc per_turn (từng lượt riêng)")

    return parser.parse_args()


ARGS = parse_args()

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Tìm kiếm thông tin trên web để thu thập dữ liệu và trả lời câu hỏi của người dùng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Danh sách các từ khóa hoặc câu truy vấn tìm kiếm thông tin trên web."
                    }
                },
                "required": ["queries"]
            }
        }
    }
]


PROMPT_GENERATE_CONVERSATION = """Bạn là chuyên gia xây dựng dữ liệu benchmark cho Agent đàm thoại đa lượt tiếng Việt có sử dụng công cụ tìm kiếm web (web_search).

Nhiệm vụ: Chuyển đổi chuỗi {turn_count} câu hỏi - câu trả lời từ Quora thành một phiên hội thoại đa lượt hoàn chỉnh giữa người dùng và trợ lý ảo:
- `conversation_topic`: Đặt tiêu đề chủ đề ngắn gọn, chuyên nghiệp bằng tiếng Việt (có thể kèm tiếng Anh trong ngoặc), ví dụ: "Kinh tế học & Tiền mã hóa (Finance & Cryptocurrency)".
- Các lượt hội thoại (turns) phải diễn ra tuần tự tương ứng với danh sách Q&A đầu vào, có sự kết nối và chuyển ý tự nhiên.
- Mỗi turn gồm:
  1. `user_message`: Dịch và trau chuốt câu hỏi sang tiếng Việt tự nhiên, phù hợp với ngữ cảnh đàm thoại thực tế.
  2. `queries`: Danh sách 1 - 2 từ khóa tìm kiếm trọng tâm (tiếng Việt hoặc tiếng Anh) để tra cứu thông tin cho câu hỏi này.
  3. `results`: 1 - 2 kết quả tìm kiếm giả lập làm bằng chứng:
     - `title`: Tiêu đề bài viết/nguồn tin cậy.
     - `snippet`: Đoạn trích dẫn tóm lược dữ kiện cốt lõi trích từ câu trả lời gốc.
  4. `assistant_answer`: Câu trả lời tổng hợp sang tiếng Việt lưu loát, khoa học, đầy đủ thông tin chi tiết dựa trên bằng chứng đã tìm được.

Định dạng JSON yêu cầu duy nhất (không kèm markdown ngoài JSON):
{
  "conversation_topic": "...",
  "turns": [
    {
      "user_message": "...",
      "queries": ["truy vấn 1", "truy vấn 2"],
      "results": [
        {"title": "...", "snippet": "..."}
      ],
      "assistant_answer": "..."
    }
  ]
}
"""

PROMPT_GENERATE_TURN = """Bạn là chuyên gia xây dựng dữ liệu benchmark cho Agent đàm thoại đa lượt tiếng Việt có sử dụng công cụ tìm kiếm web (web_search).

Chúng tôi có một cặp Câu hỏi (Prompt) và Câu trả lời (Completion) gốc bằng tiếng Anh từ Quora:
[QUESTION (EN)]: {en_question}
[ANSWER (EN)]: {en_answer}

Nhiệm vụ của bạn là xử lý và trả về ĐÚNG 1 khối JSON (không kèm markdown thừa) với các trường sau:
1. "user_message": Dịch câu hỏi sang tiếng Việt tự nhiên, phù hợp với ngữ cảnh hội thoại của một người dùng thực tế.
2. "queries": Danh sách 2-3 truy vấn tìm kiếm ngắn gọn, chuẩn xác bằng tiếng Việt (hoặc thuật ngữ chuyên môn) mà một Agent thông minh sẽ dùng để tìm kiếm thông tin cho câu hỏi này.
3. "results": Mảng gồm 1-2 kết quả tìm kiếm giả lập làm bằng chứng:
   - "title": Tiêu đề bài viết/nguồn tin cậy.
   - "snippet": Đoạn trích dẫn tóm lược sự kiện, số liệu chính từ câu trả lời để làm bằng chứng cho Assistant trả lời.
4. "assistant_answer": Câu trả lời tổng hợp sang tiếng Việt lưu loát, khoa học, đầy đủ thông tin chi tiết dựa trên bằng chứng đã tìm được.
5. "topic": Chủ đề ngắn gọn 2-4 từ mô tả lĩnh vực của câu hỏi (ví dụ: "Chính trị & Xã hội", "Khoa học vũ trụ", "Kinh tế - Tài chính"...).

Định dạng JSON yêu cầu:
{
  "user_message": "...",
  "queries": ["truy vấn 1", "truy vấn 2"],
  "results": [
    {
      "title": "...",
      "snippet": "..."
    }
  ],
  "assistant_answer": "...",
  "topic": "..."
}
"""


def clean_json_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """Làm sạch markdown block và parse JSON an toàn."""
    if not raw_text:
        return None
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        raw_json = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        raw_json = text[start:end+1] if (start != -1 and end != -1) else text

    try:
        return json.loads(raw_json)
    except Exception:
        return None


async def process_single_turn(
    en_q: str,
    en_a: str,
    conv_id: str,
    turn_idx: int,
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Xử lý 1 turn thông qua LLM với semaphore điều tiết số luồng đồng thời."""
    prompt = PROMPT_GENERATE_TURN.replace("{en_question}", en_q).replace("{en_answer}", en_a)
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


TOPICS_TAXONOMY = [
    "Công nghệ & Trí tuệ nhân tạo",
    "Kinh tế, Tài chính & Đầu tư",
    "Y học, Sức khỏe & Dinh dưỡng",
    "Chính trị, Pháp luật & Xã hội",
    "Khoa học, Vũ trụ & Môi trường",
    "Tâm lý học, Đời sống & Mối quan hệ",
    "Lịch sử, Văn hóa & Quân sự",
    "Giáo dục, Nghề nghiệp & Kỹ năng"
]


async def classify_question_chunk(chunk_tuples: List[tuple], semaphore: asyncio.Semaphore) -> Dict[int, str]:
    """Phân loại 1 chunk các câu hỏi vào các chủ đề sử dụng LLM."""
    q_text_list = [f"ID_{orig_idx}: {q_text}" for orig_idx, q_text in chunk_tuples]
    questions_block = "\n".join(q_text_list)
    topics_list_str = ", ".join(f'"{t}"' for t in TOPICS_TAXONOMY)
    
    prompt = f"""Bạn là chuyên gia phân loại dữ liệu chủ đề.
Dưới đây là danh sách các câu hỏi [ID: nội dung]:
{questions_block}

Hãy phân loại TỪNG câu hỏi vào đúng 1 trong các chủ đề sau:
[{topics_list_str}]

Định dạng JSON trả về duy nhất (không kèm giải thích markdown thừa):
{{
  "ID_xxx": "Tên chủ đề",
  "ID_yyy": "Tên chủ đề"
}}
"""
    messages = [
        {"role": "system", "content": "You are a professional topic classifier. Return valid JSON only."},
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
                    temperature=0.1,
                    max_tokens=2048
                )
                raw = res.get("content") or ""
                parsed = clean_json_output(raw)
                if parsed and isinstance(parsed, dict):
                    result = {}
                    for k, v in parsed.items():
                        m = re.search(r'\d+', str(k))
                        if m and isinstance(v, str):
                            idx = int(m.group())
                            chosen_topic = v.strip()
                            for t in TOPICS_TAXONOMY:
                                if t.lower() in chosen_topic.lower() or chosen_topic.lower() in t.lower():
                                    chosen_topic = t
                                    break
                            result[idx] = chosen_topic
                    return result
            except Exception as e:
                if attempt == 3:
                    print(f"⚠️ Lỗi phân loại chunk câu hỏi: {e}")
                await asyncio.sleep(1.0 * attempt)
    return {}


def build_topic_clusters_tfidf(
    ds,
    num_samples: int,
    turns_per_conv: int,
    sim_min: float = 0.15,
    sim_max: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Phân cụm các câu hỏi Quora theo chủ đề bằng TF-IDF và Cosine Similarity.
    Nếu chưa cài đặt scikit-learn, tự động fallback sang Token Jaccard Similarity.
    Deduplicate các câu hỏi trùng nhau trước để đảm bảo mỗi turn là 1 câu hỏi độc nhất.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    print("🔍 Đang lọc chất lượng và loại bỏ câu hỏi trùng lặp từ Quora dataset...")
    unique_map = {}
    for idx, item in enumerate(ds):
        prompt = item["prompt"].strip()
        answer = item["completion"].strip()
        if len(prompt) < 20 or len(answer) < 80:
            continue
        norm_p = prompt.lower()
        if norm_p not in unique_map or len(answer) > len(unique_map[norm_p]["completion"]):
            unique_map[norm_p] = {"row_id": idx, "prompt": prompt, "completion": answer}

    items = list(unique_map.values())
    print(f"   Số câu hỏi độc nhất chất lượng cao: {len(items):,}")

    visited = set()
    conv_groups = []
    candidate_order = sorted(range(len(items)), key=lambda i: len(items[i]["completion"]), reverse=True)

    if has_sklearn:
        prompts = [x["prompt"] for x in items]
        vectorizer = TfidfVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 2), min_df=2)
        X = vectorizer.fit_transform(prompts)

        for seed_idx in candidate_order:
            if seed_idx in visited or len(conv_groups) >= num_samples:
                if len(conv_groups) >= num_samples:
                    break
                continue

            seed_vec = X[seed_idx]
            sims = cosine_similarity(seed_vec, X).flatten()

            matching = []
            for cand_idx, score in enumerate(sims):
                if cand_idx != seed_idx and cand_idx not in visited and sim_min <= score <= sim_max:
                    matching.append((cand_idx, score))

            matching.sort(key=lambda x: x[1], reverse=True)

            needed = turns_per_conv - 1
            if len(matching) >= needed:
                selected_local = [seed_idx] + [c[0] for c in matching[:needed]]
                for idx in selected_local:
                    visited.add(idx)

                selected_row_ids = [items[i]["row_id"] for i in selected_local]
                c_idx = len(conv_groups) + 1
                conv_groups.append({
                    "conv_idx": c_idx,
                    "topic": "Chủ đề được phân cụm tự động",
                    "indices": selected_row_ids
                })

        print(f"✅ Đã tạo thành công {len(conv_groups)} cụm hội thoại ({turns_per_conv} lượt/hội thoại) bằng TF-IDF!")
    else:
        print("⚠️ Chưa có thư viện scikit-learn trong môi trường. Đang tự động phân cụm bằng Token Jaccard (Pure Python fallback)...")
        import re
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "who", "where", "when", "do", "does", "did", "can", "could", "would", "should", "in", "on", "at", "to", "for", "with", "by", "of", "and", "or", "i", "you", "my", "your", "it", "this", "that"}
        token_sets = []
        for it in items:
            words = set(re.findall(r"\w+", it["prompt"].lower())) - stop_words
            token_sets.append(words)

        for seed_idx in candidate_order:
            if seed_idx in visited or len(conv_groups) >= num_samples:
                if len(conv_groups) >= num_samples:
                    break
                continue

            seed_tokens = token_sets[seed_idx]
            if not seed_tokens:
                continue

            matching = []
            for cand_idx in range(len(items)):
                if cand_idx != seed_idx and cand_idx not in visited:
                    cand_tokens = token_sets[cand_idx]
                    if not cand_tokens:
                        continue
                    inter = len(seed_tokens & cand_tokens)
                    union = len(seed_tokens | cand_tokens)
                    if union > 0:
                        sim = inter / union
                        if 0.10 <= sim <= 0.85:
                            matching.append((cand_idx, sim))

            matching.sort(key=lambda x: x[1], reverse=True)
            needed = turns_per_conv - 1
            if len(matching) >= needed:
                selected_local = [seed_idx] + [c[0] for c in matching[:needed]]
                for idx in selected_local:
                    visited.add(idx)

                selected_row_ids = [items[i]["row_id"] for i in selected_local]
                c_idx = len(conv_groups) + 1
                conv_groups.append({
                    "conv_idx": c_idx,
                    "topic": "Chủ đề được phân cụm tự động",
                    "indices": selected_row_ids
                })

        print(f"✅ Đã tạo thành công {len(conv_groups)} cụm hội thoại ({turns_per_conv} lượt/hội thoại) bằng Token Jaccard Fallback!")

    return conv_groups


async def build_topic_clusters_llm(
    ds,
    num_samples: int,
    turns_per_conv: int,
    start_offset: int = 100
) -> List[Dict[str, Any]]:
    """Phân cụm các câu hỏi từ Quora dataset bằng LLM."""
    print(f"\n🧠 [PHÂN CỤM CHỦ ĐỀ BẰNG LLM]")
    print(f"   Đang quét dataset Quora và dùng LLM để phân loại vào {len(TOPICS_TAXONOMY)} chủ đề...")

    seen_prompts = set()
    filtered_tuples = []
    for idx, row in enumerate(ds):
        p = row["prompt"].strip()
        if len(p) >= 20 and p.lower() not in seen_prompts:
            seen_prompts.add(p.lower())
            filtered_tuples.append((idx, p))

    topic_buckets: Dict[str, List[int]] = {t: [] for t in TOPICS_TAXONOMY}
    conv_groups: List[Dict[str, Any]] = []

    semaphore = asyncio.Semaphore(min(ARGS.max_concurrency, 16))
    chunk_size = 25
    curr_idx = start_offset

    pbar = tqdm(total=num_samples, desc="Phân cụm conversations")

    while len(conv_groups) < num_samples and curr_idx < len(filtered_tuples):
        chunk_tuples = filtered_tuples[curr_idx : curr_idx + chunk_size]
        curr_idx += chunk_size

        if not chunk_tuples:
            break

        classification_map = await classify_question_chunk(chunk_tuples, semaphore)
        for orig_idx, topic in classification_map.items():
            if topic not in topic_buckets:
                topic_buckets[topic] = []
            topic_buckets[topic].append(orig_idx)

            if len(topic_buckets[topic]) >= turns_per_conv and len(conv_groups) < num_samples:
                selected_indices = topic_buckets[topic][:turns_per_conv]
                topic_buckets[topic] = topic_buckets[topic][turns_per_conv:]

                c_idx = len(conv_groups) + 1
                group_obj = {
                    "conv_idx": c_idx,
                    "topic": topic,
                    "indices": selected_indices
                }
                conv_groups.append(group_obj)
                pbar.update(1)

    pbar.close()
    print(f"✅ Đã phân cụm thành công {len(conv_groups)} conversations cùng chủ đề bằng LLM!")
    return conv_groups


async def build_topic_clusters(
    ds,
    num_samples: int,
    turns_per_conv: int,
    cache_cluster_path: Path,
    method: str = "tfidf"
) -> List[Dict[str, Any]]:
    """Lấy danh sách các nhóm hội thoại đã phân cụm (đọc từ cache nếu có)."""
    if cache_cluster_path.exists():
        try:
            with open(cache_cluster_path, "r", encoding="utf-8") as f:
                cached_groups = json.load(f)
                if len(cached_groups) >= num_samples:
                    print(f"📦 Đã tải {len(cached_groups)} kịch bản hội thoại đã phân cụm từ cache: {cache_cluster_path.name}")
                    return cached_groups[:num_samples]
        except Exception:
            pass

    if method == "tfidf":
        conv_groups = build_topic_clusters_tfidf(ds, num_samples, turns_per_conv)
    else:
        conv_groups = await build_topic_clusters_llm(ds, num_samples, turns_per_conv)

    cache_cluster_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_cluster_path, "w", encoding="utf-8") as f:
        json.dump(conv_groups, f, ensure_ascii=False, indent=2)

    return conv_groups


async def process_single_conversation_single_pass(
    conv_idx: int,
    indices: List[int],
    ds,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    cache_path: Path,
    topic: str = "General Q&A"
) -> Optional[Dict[str, Any]]:
    """Sinh toàn bộ các lượt của 1 conversation trong 1 LLM request để đảm bảo tính mạch lạc."""
    conv_id = f"conversation_{conv_idx:04d}"

    qa_list = [
        {
            "turn": t_idx + 1,
            "question": ds[ds_idx]["prompt"].strip(),
            "answer": ds[ds_idx]["completion"].strip()[:600]
        }
        for t_idx, ds_idx in enumerate(indices)
    ]

    prompt = PROMPT_GENERATE_CONVERSATION.replace("{turn_count}", str(len(qa_list)))
    user_content = f"Chủ đề gợi ý: {topic}\n\nDanh sách các cặp Q&A gốc:\n{json.dumps(qa_list, ensure_ascii=False, indent=2)}"

    messages_to_send = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]

    async with semaphore:
        for attempt in range(1, 4):
            try:
                res = await chat_completion_async(
                    base_url=ARGS.base_url,
                    model=ARGS.model,
                    api_key=ARGS.api_key,
                    messages=messages_to_send,
                    temperature=ARGS.temperature,
                    max_tokens=ARGS.max_tokens
                )
                raw_content = res.get("content") or ""
                parsed = clean_json_output(raw_content)

                if parsed and "turns" in parsed and len(parsed["turns"]) == len(indices):
                    conv_topic = parsed.get("conversation_topic") or topic
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can search the web for information. Use web_search when external information would help answer the user's question accurately."
                        }
                    ]

                    for t_idx, turn_data in enumerate(parsed["turns"], start=1):
                        call_id = f"call_{conv_id}_turn{t_idx}"
                        # 1. User
                        messages.append({
                            "role": "user",
                            "content": turn_data.get("user_message", "").strip()
                        })
                        # 2. Assistant tool call
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": json.dumps({"queries": turn_data.get("queries", [])}, ensure_ascii=False)
                                    }
                                }
                            ]
                        })
                        # 3. Tool response
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(turn_data.get("results", []), ensure_ascii=False)
                        })
                        # 4. Assistant answer
                        messages.append({
                            "role": "assistant",
                            "content": turn_data.get("assistant_answer", "").strip()
                        })

                    conversation_obj = {
                        "conversation_id": conv_id,
                        "conversation_index": conv_idx,
                        "conversation_topic": conv_topic,
                        "turn_count": len(indices),
                        "source_row_ids": indices,
                        "source_indices": list(range(len(indices))),
                        "tools": DEFAULT_TOOLS,
                        "messages": messages
                    }

                    # Ghi ngay kết quả vào cache dạng JSONL (thread-safe)
                    async with file_lock:
                        with open(cache_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(conversation_obj, ensure_ascii=False) + "\n")

                    return conversation_obj
            except Exception as e:
                if attempt == 3:
                    print(f"❌ [Conv {conv_id}] Lỗi sau 3 lần thử: {e}")
                await asyncio.sleep(1.5 * attempt)

    return None


async def process_single_conversation_per_turn(
    conv_idx: int,
    indices: List[int],
    ds,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    cache_path: Path,
    topic: str = "General Q&A"
) -> Optional[Dict[str, Any]]:
    """Xử lý từng turn riêng biệt của 1 conversation (chế độ per_turn)."""
    conv_id = f"conversation_{conv_idx:04d}"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can search the web for information. Use web_search when external information would help answer the user's question accurately."
        }
    ]

    conv_topic = topic

    for turn_idx, ds_idx in enumerate(indices, start=1):
        row = ds[ds_idx]
        en_q = row["prompt"].strip()
        en_a = row["completion"].strip()

        turn_data = await process_single_turn(en_q, en_a, conv_id, turn_idx, semaphore)
        if not turn_data:
            print(f"⚠️ Thất bại khi sinh turn {turn_idx} cho {conv_id}. Bỏ qua hội thoại này.")
            return None

        if turn_data.get("topic") and conv_topic in ["General Q&A", "Chủ đề được phân cụm tự động"]:
            conv_topic = turn_data["topic"]

        call_id = f"call_{conv_id}_turn{turn_idx}"

        messages.append({
            "role": "user",
            "content": turn_data["user_message"]
        })
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
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(turn_data["results"], ensure_ascii=False)
        })
        messages.append({
            "role": "assistant",
            "content": turn_data["assistant_answer"]
        })

    conversation_obj = {
        "conversation_id": conv_id,
        "conversation_index": conv_idx,
        "conversation_topic": conv_topic,
        "turn_count": len(indices),
        "source_row_ids": indices,
        "source_indices": list(range(len(indices))),
        "tools": DEFAULT_TOOLS,
        "messages": messages
    }

    async with file_lock:
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(conversation_obj, ensure_ascii=False) + "\n")

    return conversation_obj


async def process_single_conversation(
    conv_idx: int,
    indices: List[int],
    ds,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    cache_path: Path,
    topic: str = "General Q&A"
) -> Optional[Dict[str, Any]]:
    """Điều phối hàm sinh conversation theo cấu hình mode."""
    if ARGS.mode == "conversation":
        return await process_single_conversation_single_pass(
            conv_idx, indices, ds, semaphore, file_lock, cache_path, topic
        )
    else:
        return await process_single_conversation_per_turn(
            conv_idx, indices, ds, semaphore, file_lock, cache_path, topic
        )


async def main():
    print(f"============================================================")
    print(f"🚀 BATCH LLM MULTI-TURN DATA GENERATOR (TOPIC CLUSTERED)")
    print(f"   Model Endpoint   : {ARGS.base_url} ({ARGS.model})")
    print(f"   Max Concurrency  : {ARGS.max_concurrency} parallel requests")
    print(f"   Target Samples   : {ARGS.num_samples} conversations ({ARGS.turns_per_conv} turns each)")
    print(f"   Batch Size       : {ARGS.batch_size} conversations/batch")
    print(f"   Generation Mode  : {ARGS.mode}")
    print(f"   Cluster Method   : {ARGS.cluster_method}")
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
    processed_row_sets = set()

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        cid = item.get("conversation_id") or item.get("conv_id")
                        if cid:
                            done_conv_ids.add(cid)
                            completed_conversations.append(item)
                            processed_row_sets.add(tuple(item.get("source_row_ids", [])))
                    except Exception:
                        pass
        print(f"📦 Đã tìm thấy {len(completed_conversations)} conversation trong cache: {cache_path.name}")

    # 3. Phân cụm các câu hỏi Quora theo chủ đề
    cluster_cache_path = Path("eval/tool_call/datasets/multi_turn/.topic_clusters_cache.json")
    conv_groups = await build_topic_clusters(
        ds=ds,
        num_samples=ARGS.num_samples,
        turns_per_conv=ARGS.turns_per_conv,
        cache_cluster_path=cluster_cache_path,
        method=ARGS.cluster_method
    )

    pending_groups = [
        g for g in conv_groups
        if f"conversation_{g['conv_idx']:04d}" not in done_conv_ids
        and f"batch_conv_{g['conv_idx']:03d}" not in done_conv_ids
        and tuple(g.get("indices", [])) not in processed_row_sets
    ]

    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pending_groups:
        print("🎉 Tất cả các sample yêu cầu đã có trong cache!")
    else:
        total_to_generate = len(pending_groups)
        batch_size = max(1, ARGS.batch_size)
        total_batches = (total_to_generate + batch_size - 1) // batch_size

        print(f"\n⚡ [BƯỚC 2: SINH HỘI THOẠI ĐA LƯỢT CHO TỪNG BATCH]")
        print(f"   Cần sinh mới {total_to_generate} conversation(s) chia làm {total_batches} batch ({batch_size} convs/batch)...\n")

        semaphore = asyncio.Semaphore(ARGS.max_concurrency)
        file_lock = asyncio.Lock()

        for b_idx in range(total_batches):
            batch_items = pending_groups[b_idx * batch_size : (b_idx + 1) * batch_size]
            print(f"🚀 [Batch {b_idx + 1}/{total_batches}] Đang sinh {len(batch_items)} conversations...")

            tasks = [
                process_single_conversation(
                    conv_idx=g["conv_idx"],
                    indices=g["indices"],
                    ds=ds,
                    semaphore=semaphore,
                    file_lock=file_lock,
                    cache_path=cache_path,
                    topic=g["topic"]
                )
                for g in batch_items
            ]

            batch_results = await tqdm_asyncio.gather(*tasks, desc=f"Batch {b_idx + 1}/{total_batches}")
            valid_batch_results = [r for r in batch_results if r]
            completed_conversations.extend(valid_batch_results)

            # Chuẩn hóa format từng conversation trong batch vừa hoàn thành
            batch_output_format = []
            for c in sorted(valid_batch_results, key=lambda x: x.get("conversation_index", 0)):
                conv_obj = {
                    "conversation_id": c.get("conversation_id") or c.get("conv_id"),
                    "conversation_index": c.get("conversation_index", 1),
                    "conversation_topic": c.get("conversation_topic", "General Q&A"),
                    "turn_count": c.get("turn_count", len(c.get("messages", [])) // 4),
                    "source_row_ids": c.get("source_row_ids", []),
                    "source_indices": c.get("source_indices", list(range(c.get("turn_count", 4)))),
                    "tools": c.get("tools") or DEFAULT_TOOLS,
                    "messages": c["messages"]
                }
                batch_output_format.append(conv_obj)

            # Tạo file json riêng của batch theo thời gian hoàn tất: samples__<timestamp>.json
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_filename = f"samples__{timestamp_str}.json"
            batch_file_path = output_dir / batch_filename

            with open(batch_file_path, "w", encoding="utf-8") as f:
                json.dump(batch_output_format, f, ensure_ascii=False, indent=2)

            print(f"💾 [Batch {b_idx + 1}/{total_batches}] Hoàn thành! Đã lưu {len(batch_output_format)} conversations vào:")
            print(f"   👉 {batch_file_path.resolve()}\n")

    # 4. Lưu ra file tổng hợp tất cả các conversation nếu cấu hình --output
    if ARGS.output:
        out_path = Path(ARGS.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        final_output_format = []
        seen_ids = set()
        for c in sorted(completed_conversations, key=lambda x: x.get("conversation_index", 0)):
            cid = c.get("conversation_id") or c.get("conv_id")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            conv_obj = {
                "conversation_id": cid,
                "conversation_index": len(final_output_format) + 1,
                "conversation_topic": c.get("conversation_topic", "General Q&A"),
                "turn_count": c.get("turn_count", len(c.get("messages", [])) // 4),
                "source_row_ids": c.get("source_row_ids", []),
                "source_indices": c.get("source_indices", list(range(c.get("turn_count", 4)))),
                "messages": c["messages"]
            }
            final_output_format.append(conv_obj)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_output_format, f, ensure_ascii=False, indent=2)

        print(f"\n============================================================")
        print(f"✅ TỔNG KẾT HOÀN THÀNH!")
        print(f"   Tổng số conversations hoàn thành: {len(final_output_format)}")
        print(f"   Thư mục lưu từng batch: {output_dir.resolve()}")
        print(f"   File tổng hợp toàn bộ: {out_path.resolve()}")
        print(f"============================================================")


if __name__ == "__main__":
    asyncio.run(main())
