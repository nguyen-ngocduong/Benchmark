"""
Convert and enrich raw multi-turn conversation traces (e.g. mentor's samples_new.json)
into standardized Benchmark Ground Truth scenarios for multi_turn/samples.json.

Handles:
- Input with OpenAI / ChatML format: {"messages": [...], "tools": [...]} or list of such objects.
- Turn segmentation: extracts user query, tool calls, tool responses, and reference assistant responses.
- LLM Ground Truth enrichment via gemini-3.5-flash-lite:
    * tool_call: should_call, expected_name, required_arg_keywords
    * coreference: referent_phrase, expected_target
    * required_slots: dict of slot_name -> list of synonyms/aliases
    * golden_reference_claims: atomic verifiable claims from reference answer and tool evidence
- Appends or updates in-place in eval/tool_call/datasets/multi_turn/samples.json.

Usage:
    python -m eval.tool_call.scripts.enrich_mentor_multiturn
    python -m eval.tool_call.scripts.enrich_mentor_multiturn --input eval/tool_call/datasets/multi_turn/samples_new.json --sample-id sample_013
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.tool_call.core.chat_client import chat_completion_async

load_dotenv()

DEFAULT_BASE_URL = (
    os.environ.get("LLM_BASE_URL")
    or os.environ.get("GEMINI_BASE_URL")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
)
DEFAULT_API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or "EMPTY"
)
DEFAULT_MODEL = (
    os.environ.get("LLM_MODEL")
    or os.environ.get("GEMINI_MODEL")
    or "gemini-3.5-flash-lite"
)

PROMPT_ENRICH_TURN = """Bạn là chuyên gia gán nhãn Benchmark đánh giá Agent đàm thoại đa lượt có sử dụng công cụ (Multi-Turn Tool Calling Benchmark Annotator).

Nhiệm vụ của bạn là phân tích lượt hội thoại (turn) dưới đây của Agent và sinh bộ Ground Truth chuẩn mực để hệ thống tự động chấm điểm câu trả lời của các Agent khác.

==================================================
THÔNG TIN LƯỢT HỘI THOẠI (TURN {turn_index}):
- Lịch sử các lượt trước (Context):
{conversation_history}

- Câu hỏi của người dùng (User Message):
"{user_message}"

- Công cụ được gọi trong lượt này (Tool Call):
Tên công cụ: {tool_name}
Tham số tìm kiếm thực tế: {tool_args}

- Kết quả công cụ trả về (Tool Evidence Snippet):
\"\"\"
{tool_snippet}
\"\"\"

- Câu trả lời tham chiếu chuẩn của Assistant (Reference Answer):
\"\"\"
{assistant_answer}
\"\"\"
==================================================

HƯỚNG DẪN CHI TIẾT TỪNG TRƯỜNG:
1. "query_intent": Mục đích tìm kiếm/trả lời ngắn gọn của người dùng (1 câu ngắn).
2. "tool_call":
   - "should_call": true (nếu cần gọi công cụ) hoặc false (nếu chỉ cần trả lời từ ngữ cảnh hội thoại có sẵn).
   - "expected_name": "{tool_name}" nếu should_call=true, hoặc null nếu false.
   - "required_arg_keywords": Mảng từ 2-5 từ khóa/cụm từ then chốt (tiếng Anh hoặc tiếng Việt theo ngôn ngữ câu hỏi) mà một Agent tốt BẮT BUỘC phải đưa vào tham số tìm kiếm (ví dụ: ["tau ceti", "habitable", "atmosphere"]).
3. "coreference":
   - Nếu câu hỏi của người dùng ở lượt này có dùng đại từ/từ thay thế chỉ thực thể ở lượt trước (như "it", "they", "these planets", "the shuttle", "there", "đó", "họ", "người này"):
     {{
       "referent_phrase": "<từ thay thế trong câu hỏi>",
       "expected_target": ["<tên thực thể chính xác>", "<từ đồng nghĩa>"]
     }}
   - Nếu không có tham chiếu đa lượt hoặc là lượt 1: đặt là null.
4. "required_slots":
   - Dictionary gồm 2-4 thông tin cốt lõi (slots) mà Agent bắt buộc phải đề cập khi trả lời, mỗi slot kèm danh sách các từ/cụm từ đồng nghĩa chấp nhận được (synonyms/aliases).
   - Ví dụ:
     {{
       "planetary_mass": ["2 and 6 earth masses", "2-6 earth masses", "2 to 6 times earth mass"],
       "risk_factors": ["runaway greenhouse", "toxic atmosphere", "extreme pressure"]
     }}
5. "golden_reference_claims":
   - Danh sách từ 1 đến 3 mệnh đề sự thật nguyên tử (atomic claims) độc lập, chuẩn xác, trích xuất từ câu trả lời chuẩn và kết quả tool.
   - Các claims này sẽ dùng để LLM Judge / NLI chấm Factuality (SUPPORTED / UNSUPPORTED / CONTRADICTED).

==================================================
YÊU CẦU ĐỊNH DẠNG:
Trả về DUY NHẤT một JSON hợp lệ, KHÔNG bọc trong markdown block ```json, KHÔNG giải thích:

{{
  "query_intent": "<Mục đích>",
  "tool_call": {{
    "should_call": {should_call_str},
    "expected_name": {expected_name_json},
    "required_arg_keywords": ["keyword1", "keyword2"]
  }},
  "coreference": null,
  "required_slots": {{
    "slot_name_1": ["synonym_a", "synonym_b"],
    "slot_name_2": ["synonym_c", "synonym_d"]
  }},
  "golden_reference_claims": [
    "<Mệnh đề sự thật 1>",
    "<Mệnh đề sự thật 2>"
  ]
}}
"""


def clean_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """Strips markdown code fences and parses JSON robustly."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_messages_into_turns(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parses flat ChatML/OpenAI messages list into structured turns:
    Each turn contains:
    - turn_index
    - user_message
    - tool_call: {name, arguments} or None
    - tool_snippet: str
    - assistant_answer: str
    """
    turns = []
    current_turn: Optional[Dict[str, Any]] = None
    turn_idx = 1

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "user":
            if current_turn and current_turn.get("user_message"):
                turns.append(current_turn)
                turn_idx += 1

            current_turn = {
                "turn_index": turn_idx,
                "user_message": content.strip(),
                "tool_call": None,
                "tool_snippet": "",
                "assistant_answer": ""
            }

        elif role == "assistant":
            if not current_turn:
                continue

            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                fn = tool_calls[0].get("function", {})
                current_turn["tool_call"] = {
                    "id": tool_calls[0].get("id", f"call_turn{turn_idx}"),
                    "name": fn.get("name", "web_search"),
                    "arguments": fn.get("arguments", "{}")
                }
            elif content:
                current_turn["assistant_answer"] = content.strip()

        elif role == "tool":
            if current_turn:
                current_turn["tool_snippet"] = content.strip()

    if current_turn and current_turn.get("user_message"):
        turns.append(current_turn)

    return turns


async def enrich_turn_with_llm(
    turn_data: Dict[str, Any],
    history_summary: str,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.1
) -> Optional[Dict[str, Any]]:
    """Calls LLM to annotate one turn with full Ground Truth schema."""
    turn_idx = turn_data["turn_index"]
    user_msg = turn_data["user_message"]
    has_tool = turn_data.get("tool_call") is not None
    tool_name = turn_data["tool_call"]["name"] if has_tool else "None"
    tool_args = turn_data["tool_call"]["arguments"] if has_tool else "{}"
    tool_snippet = turn_data.get("tool_snippet") or "Không có kết quả công cụ (sử dụng kiến thức hội thoại)."
    assistant_answer = turn_data.get("assistant_answer") or "Không có câu trả lời mẫu."

    prompt = PROMPT_ENRICH_TURN.format(
        turn_index=turn_idx,
        conversation_history=history_summary if history_summary else "(Lượt đầu tiên, chưa có lịch sử)",
        user_message=user_msg,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_snippet=tool_snippet,
        assistant_answer=assistant_answer,
        should_call_str="true" if has_tool else "false",
        expected_name_json=json.dumps(tool_name if has_tool else None)
    )

    messages = [
        {"role": "system", "content": "You are an expert NLP benchmark data annotator for Multi-Turn Agent Evaluation. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(1, 4):
        try:
            res = await chat_completion_async(
                base_url=base_url,
                model=model,
                api_key=api_key,
                messages=messages,
                temperature=temperature,
                max_tokens=2000
            )
            raw_text = res.get("content") or res.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not raw_text:
                await asyncio.sleep(1.5 * attempt)
                continue

            parsed = clean_json_response(raw_text)
            if parsed and "golden_reference_claims" in parsed and "required_slots" in parsed:
                return parsed

            await asyncio.sleep(1.5 * attempt)
        except Exception as e:
            print(f"    [Turn {turn_idx} Attempt {attempt}] Error: {e}")
            await asyncio.sleep(2.0 * attempt)

    return None


async def convert_and_enrich_mentor_dataset(
    input_file: str = "eval/tool_call/datasets/multi_turn/samples_new.json",
    output_file: str = "eval/tool_call/datasets/multi_turn/samples.json",
    sample_id: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = 0.1,
    force: bool = False
):
    """
    Main workflow:
    1. Reads multi_turn/samples_new.json
    2. Parses into conversation turns
    3. Uses LLM to extract Ground Truth for each turn
    4. Formats into benchmark scenario and merges into samples.json
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Normalize raw_data to list of conversations
    conversations = []
    if isinstance(raw_data, dict):
        if "messages" in raw_data:
            conversations = [raw_data]
        else:
            print(f"❌ Unexpected dictionary format in {input_path}")
            sys.exit(1)
    elif isinstance(raw_data, list):
        conversations = raw_data
    else:
        print(f"❌ Unsupported JSON root structure in {input_path}")
        sys.exit(1)

    # Load existing scenarios to determine next sample_id or update
    existing_scenarios: List[Dict[str, Any]] = []
    existing_ids = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_scenarios = json.loads(content)
                    existing_ids = {s.get("sample_id") for s in existing_scenarios if s.get("sample_id")}
            print(f"📖 Loaded {len(existing_scenarios)} existing scenarios from {output_path.name}")
        except Exception as e:
            print(f"⚠️ Could not read {output_path}: {e}")

    results = list(existing_scenarios)
    print(f"📋 Found {len(conversations)} conversation(s) to process from {input_path.name}")
    print(f"🌐 LLM Endpoint: {base_url} | Model: {model}\n")

    for conv_idx, conv in enumerate(conversations, start=1):
        messages = conv.get("messages", [])
        if not messages:
            print(f"⚠️ Conversation #{conv_idx} has no messages, skipping.")
            continue

        # Determine target sample_id
        if sample_id:
            target_id = sample_id if len(conversations) == 1 else f"{sample_id}_{conv_idx}"
        else:
            # Auto-assign next consecutive ID
            max_num = 0
            for sid in existing_ids:
                m = re.match(r"^sample_(\d+)$", sid)
                if m:
                    max_num = max(max_num, int(m.group(1)))
            target_id = f"sample_{max_num + 1:03d}"

        if target_id in existing_ids and not force and not sample_id:
            print(f"⏭️ Skipping {target_id} (already exists in {output_path.name}).")
            continue

        print(f"⏳ Processing Conversation #{conv_idx} -> Target ID: {target_id}")
        raw_turns = parse_messages_into_turns(messages)
        print(f"   Parsed {len(raw_turns)} turn(s). Enriching Ground Truth with LLM...")

        scenario_turns = []
        history_lines = []
        all_snippets = []

        for turn_data in raw_turns:
            t_idx = turn_data["turn_index"]
            u_msg = turn_data["user_message"]
            history_summary = "\n".join(history_lines)

            print(f"   • Enriching Turn {t_idx}: \"{u_msg[:50]}...\"")
            annotation = await enrich_turn_with_llm(
                turn_data=turn_data,
                history_summary=history_summary,
                base_url=base_url,
                model=model,
                api_key=api_key,
                temperature=temperature
            )

            if not annotation:
                print(f"     ❌ Failed to annotate Turn {t_idx}. Fallback to basic schema.")
                annotation = {
                    "query_intent": u_msg,
                    "tool_call": {
                        "should_call": turn_data.get("tool_call") is not None,
                        "expected_name": "web_search" if turn_data.get("tool_call") else None,
                        "required_arg_keywords": [word for word in u_msg.lower().split() if len(word) > 3][:4]
                    },
                    "coreference": None,
                    "required_slots": {},
                    "golden_reference_claims": [turn_data.get("assistant_answer", "")[:100]]
                }

            has_tool = turn_data.get("tool_call") is not None
            tool_name = "web_search" if has_tool else None
            coref = annotation.get("coreference")

            turn_obj = {
                "turn": t_idx,
                "user_message": u_msg,
                "expected_behavior": {
                    "should_call_tool": has_tool,
                    "expected_tool": tool_name,
                    "expected_category": "knowledge" if has_tool else None,
                    "query_intent": annotation.get("query_intent", u_msg),
                    "context_required": t_idx > 1 and bool(coref),
                    "state_required": t_idx > 1
                },
                "ground_truth": {
                    "turn": t_idx,
                    "tool_call": {
                        "should_call": has_tool,
                        "expected_name": tool_name,
                        "required_arg_keywords": annotation.get("tool_call", {}).get("required_arg_keywords", [])
                    },
                    "coreference": coref,
                    "required_slots": annotation.get("required_slots", {}),
                    "golden_reference_claims": annotation.get("golden_reference_claims", [])
                }
            }
            scenario_turns.append(turn_obj)

            # Update running history for next turns
            history_lines.append(f"Turn {t_idx} User: {u_msg}")
            if turn_data.get("assistant_answer"):
                history_lines.append(f"Turn {t_idx} Assistant: {turn_data['assistant_answer'][:120]}...")
            if turn_data.get("tool_snippet"):
                all_snippets.append(turn_data["tool_snippet"])

        # Construct final scenario object
        source_text = "\n\n".join(all_snippets) if all_snippets else raw_turns[0]["user_message"]
        scenario = {
            "sample_id": target_id,
            "source": {
                "text": source_text,
                "label": 0
            },
            "scenario": {
                "objective": "multi_turn_context_tool_use",
                "turns": scenario_turns
            }
        }

        # Update in-place if exists, otherwise append
        existing_match_idx = next((i for i, x in enumerate(results) if x.get("sample_id") == target_id), None)
        if existing_match_idx is not None:
            results[existing_match_idx] = scenario
            print(f"   🔄 Updated existing {target_id} in-place.")
        else:
            results.append(scenario)
            existing_ids.add(target_id)
            print(f"   ➕ Appended new scenario {target_id}.")

        # Incremental save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"   💾 Saved progress: {len(results)} total scenarios in {output_path.name}\n")

    print(f"{'='*70}")
    print(f"🎉 MENTOR MULTI-TURN ENRICHMENT COMPLETE!")
    print(f"   • Total scenarios in {output_path.name}: {len(results)}")
    print(f"   • Output location: {output_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert mentor's raw multi-turn conversation traces (samples_new.json) into Benchmark Ground Truth"
    )
    parser.add_argument(
        "--input",
        default="eval/tool_call/datasets/multi_turn/samples_new.json",
        help="Path to input samples_new.json"
    )
    parser.add_argument(
        "--output",
        default="eval/tool_call/datasets/multi_turn/samples.json",
        help="Path to target multi_turn samples.json"
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Custom sample_id for the scenario (e.g. sample_013). Defaults to auto-incrementing next ID."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"LLM API Base URL (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="LLM API Key (default: from env GEMINI_API_KEY / LLM_API_KEY)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force overwrite if scenario with same ID exists"
    )

    args = parser.parse_args()

    asyncio.run(convert_and_enrich_mentor_dataset(
        input_file=args.input,
        output_file=args.output,
        sample_id=args.sample_id,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        force=args.force
    ))


if __name__ == "__main__":
    main()
