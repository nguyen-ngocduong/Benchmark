"""
Generate Multi-Turn Dataset with Ground Truth from Vietnamese News Samples using chat_client.

Reads source news samples from eval/tool_call/datasets/summarization/samples.json.
Calls LLM endpoints (Gemini, OpenAI, Ollama, Claude, etc.) via eval.tool_call.core.chat_client.
Generates multi-turn scenarios with embedded structured ground_truth for benchmark evaluation.
Saves incrementally to avoid losing progress.
"""

import asyncio
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

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
    or "gemini-2.5-flash"
)

PROMPT_TEMPLATE = """Bạn là một chuyên gia thiết kế benchmark cho trợ lý AI hội thoại nhiều lượt (multi-turn conversational agent) có khả năng sử dụng công cụ (tool-use).

Nhiệm vụ của bạn là nhận một bài báo tiếng Việt từ dataset và chuyển bài báo đó thành MỘT kịch bản kiểm thử hội thoại 3 lượt (3 turns), KÈM THEO BỘ GROUND TRUTH CHI TIẾT để hệ thống đánh giá tự động (evaluation pipeline).

==================================================
I. MỤC TIÊU BENCHMARK
==================================================

Kịch bản phải tập trung đánh giá các năng lực:
1. Context Retention: Agent nhớ và duy trì thông tin từ các lượt trước (có dùng đại từ/từ tham chiếu).
2. Follow-up Understanding: Agent hiểu câu hỏi nối tiếp mà user không cần nhắc lại toàn bộ bối cảnh.
3. Tool Switching: Agent biết khi nào CẦN gọi tool (lượt 1 hoặc khi cần dữ liệu mới) và khi nào KHÔNG ĐƯỢC gọi tool (khi thông tin đã có trong ngữ cảnh hội thoại trước).
4. Coreference Resolution: Agent hiểu chính xác đại từ/từ tham chiếu ("nguồn gốc đó", "chiến dịch đó", "hai nguyên liệu trên", "nhân vật này", "thương hiệu trên").
5. State Tracking & Slot Consistency: Duy trì các thực thể, thuộc tính và trạng thái qua từng lượt.

==================================================
II. TOOL DUY NHẤT
==================================================

Tool duy nhất mà Agent được phép dùng:
{{
  "type": "function",
  "function": {{
    "name": "web_search",
    "description": "Tìm kiếm thông tin trên web để trả lời câu hỏi của người dùng.",
    "parameters": {{
      "type": "object",
      "properties": {{
        "query": {{
          "type": "string",
          "description": "Truy vấn tìm kiếm."
        }},
        "category": {{
          "type": "string",
          "enum": ["news", "general"],
          "description": "Danh mục tìm kiếm."
        }}
      }},
      "required": ["query", "category"]
    }}
  }}
}}

Quy tắc tool:
- Turn 1: Thường yêu cầu tìm kiếm thông tin từ bài báo -> should_call = true, category = "news".
- Turn 2: Câu hỏi nối tiếp đào sâu chi tiết trong bài -> nên should_call = false (vì bài báo đã được trả về ở Turn 1), trừ khi hỏi sang một sự kiện độc lập hoàn toàn.
- Turn 3: Thường là tổng hợp, so sánh hoặc kiểm tra suy luận -> should_call = false.

==================================================
III. THIẾT KẾ GROUND TRUTH ĐÁNH GIÁ (BẮT BUỘC CHO MỖI TURN)
==================================================

Đối với MỖI TURN trong `scenario.turns`, BẮT BUỘC phải sinh trường `ground_truth` với 4 thành phần:

1. `tool_call`:
   - `should_call`: true nếu lượt này cần gọi tool, false nếu không.
   - `expected_name`: "web_search" (nếu should_call=true) hoặc null (nếu should_call=false).
   - `required_arg_keywords`: Danh sách 3-5 từ khóa quan trọng (chữ thường) bắt buộc phải có trong truy vấn "query" của Agent. Nếu should_call=false, để mảng rỗng [].

2. `coreference`:
   - Nếu `user_message` sử dụng đại từ hoặc cụm từ quy chiếu ngữ cảnh (ví dụ: "nguồn gốc đó", "chiến dịch đó", "hai nguyên liệu trên", "nhân vật này", "nơi đó"):
     {{
       "referent_phrase": "<cụm từ tham chiếu trong câu hỏi>",
       "expected_target": ["<thực thể mục tiêu>", "<các từ đồng nghĩa/biến thể chấp nhận được>"]
     }}
   - Nếu `user_message` không sử dụng cụm từ quy chiếu: để null.

3. `required_slots`:
   - Dictionary chứa các thông tin/thực thể cốt lõi mà câu trả lời của Trợ lý bắt buộc phải đề cập đến.
   - Mỗi key là tên slot (snake_case, ví dụ: "material_origin", "weaving_workshop", "duration").
   - Mỗi value là mảng các từ/cụm từ đồng nghĩa chấp nhận được (aliases) khi đối chiếu câu trả lời (ví dụ: ["brazil", "vườn ươm ở brazil"]).

4. `golden_reference_claims`:
   - Danh sách 2-4 phát biểu sự thật nguyên tử (atomic claims) trích xuất trực tiếp từ Text bài báo, đóng vai trò chân lý chuẩn để kiểm tra tính trung thực của câu trả lời.

==================================================
IV. SCHEMA OUTPUT BẮT BUỘC (JSON ONLY)
==================================================

CHỈ trả về JSON hợp lệ, KHÔNG markdown, KHÔNG lời giải thích:

{{
  "sample_id": "{sample_id}",
  "source": {{
    "text": "<sao chép nguyên văn trường Text đầu vào>",
    "label": {label}
  }},
  "scenario": {{
    "objective": "multi_turn_context_tool_use",
    "turns": [
      {{
        "turn": 1,
        "user_message": "<Câu hỏi mở đầu về bài báo>",
        "expected_behavior": {{
          "should_call_tool": true,
          "expected_tool": "web_search",
          "expected_category": "news",
          "query_intent": "<Ý định tìm kiếm>",
          "context_required": false,
          "state_required": false
        }},
        "ground_truth": {{
          "turn": 1,
          "tool_call": {{
            "should_call": true,
            "expected_name": "web_search",
            "required_arg_keywords": ["từ_khóa_1", "từ_khóa_2", "từ_khóa_3"]
          }},
          "coreference": null,
          "required_slots": {{
            "slot_name_1": ["alias_1a", "alias_1b"],
            "slot_name_2": ["alias_2a", "alias_2b"]
          }},
          "golden_reference_claims": [
            "Phát biểu chuẩn nguyên tử 1 từ bài báo.",
            "Phát biểu chuẩn nguyên tử 2 từ bài báo."
          ]
        }}
      }},
      {{
        "turn": 2,
        "user_message": "<Câu hỏi follow-up sử dụng cụm từ tham chiếu, ví dụ 'thực thể đó...'>",
        "expected_behavior": {{
          "should_call_tool": false,
          "expected_tool": null,
          "expected_category": null,
          "query_intent": null,
          "context_required": true,
          "state_required": true
        }},
        "ground_truth": {{
          "turn": 2,
          "tool_call": {{
            "should_call": false,
            "expected_name": null,
            "required_arg_keywords": []
          }},
          "coreference": {{
            "referent_phrase": "<cụm từ tham chiếu trong user_message>",
            "expected_target": ["<thực thể được tham chiếu>", "<alias>"]
          }},
          "required_slots": {{
            "slot_name_3": ["alias_3a", "alias_3b"]
          }},
          "golden_reference_claims": [
            "Phát biểu chuẩn nguyên tử cho turn 2."
          ]
        }}
      }},
      {{
        "turn": 3,
        "user_message": "<Câu hỏi tiếp nối kiểm tra tổng hợp hoặc trạng thái>",
        "expected_behavior": {{
          "should_call_tool": false,
          "expected_tool": null,
          "expected_category": null,
          "query_intent": null,
          "context_required": true,
          "state_required": true
        }},
        "ground_truth": {{
          "turn": 3,
          "tool_call": {{
            "should_call": false,
            "expected_name": null,
            "required_arg_keywords": []
          }},
          "coreference": null,
          "required_slots": {{
            "slot_name_4": ["alias_4a", "alias_4b"]
          }},
          "golden_reference_claims": [
            "Phát biểu chuẩn nguyên tử cho turn 3."
          ]
        }}
      }}
    ],
    "state_requirements": {{
      "turn_1_to_2": [
        "Agent phải nhớ ngữ cảnh từ lượt 1..."
      ],
      "turn_2_to_3": [
        "Agent phải duy trì thông tin từ các lượt trước..."
      ]
    }}
  }}
}}

==================================================
V. INPUT BÀI BÁO
==================================================

sample_id: {sample_id}
label: {label}
Text:
\"\"\"
{article_text}
\"\"\"

Hãy tạo MỘT kịch bản 3 lượt kèm ground_truth đầy đủ theo đúng schema trên.
"""


def _clean_json_output(raw_text: str) -> str:
    """Strip markdown code fence blocks if model wrapped output in them."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def _normalize_ground_truth(turn: Dict, turn_idx: int) -> None:
    """Ensure turn has fully populated ground_truth matching benchmark evaluation schema."""
    t_num = turn.get("turn", turn_idx)
    eb = turn.get("expected_behavior", {})
    gt = turn.get("ground_truth")

    if not gt or not isinstance(gt, dict):
        should_call = eb.get("should_call_tool", False)
        gt = {
            "turn": t_num,
            "tool_call": {
                "should_call": should_call,
                "expected_name": eb.get("expected_tool") if should_call else None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {},
            "golden_reference_claims": []
        }
        turn["ground_truth"] = gt
        return

    gt["turn"] = t_num

    # 1. tool_call
    if "tool_call" not in gt or not isinstance(gt["tool_call"], dict):
        should_call = eb.get("should_call_tool", False)
        gt["tool_call"] = {
            "should_call": should_call,
            "expected_name": eb.get("expected_tool") if should_call else None,
            "required_arg_keywords": []
        }
    else:
        tc = gt["tool_call"]
        if "should_call" not in tc:
            tc["should_call"] = eb.get("should_call_tool", False)
        if "expected_name" not in tc:
            tc["expected_name"] = "web_search" if tc["should_call"] else None
        if "required_arg_keywords" not in tc or not isinstance(tc["required_arg_keywords"], list):
            tc["required_arg_keywords"] = []

    # 2. coreference
    if "coreference" not in gt:
        gt["coreference"] = None
    elif isinstance(gt["coreference"], dict):
        coref = gt["coreference"]
        if not coref.get("referent_phrase") or not coref.get("expected_target"):
            gt["coreference"] = None

    # 3. required_slots
    if "required_slots" not in gt or not isinstance(gt["required_slots"], dict):
        gt["required_slots"] = {}

    # 4. golden_reference_claims
    if "golden_reference_claims" not in gt or not isinstance(gt["golden_reference_claims"], list):
        gt["golden_reference_claims"] = []


async def generate_scenario_for_article(
    sample: Dict,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    temperature: float = 0.2,
    max_retries: int = 3
) -> Optional[Dict]:
    """Generates a 3-turn multi-turn benchmark scenario with ground_truth via chat_client."""
    sample_id = sample.get("id") or sample.get("sample_id") or "sample_unknown"
    label = sample.get("label", 0)
    article_text = sample.get("text") or sample.get("Test") or sample.get("content", "")

    prompt = PROMPT_TEMPLATE.format(
        sample_id=sample_id,
        label=label,
        article_text=article_text.strip()
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{sample_id}] Generating scenario + ground_truth with {model} (attempt {attempt}/{max_retries})...")
            t0 = time.perf_counter()

            response = await chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                base_url=base_url,
                model=model,
                api_key=api_key,
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            if response.get("error"):
                print(f"    ⚠️ chat_client returned error on attempt {attempt}: {response['error']}")
                await asyncio.sleep(2)
                continue

            raw_content = response.get("content") or ""
            clean_content = _clean_json_output(raw_content)
            elapsed = time.perf_counter() - t0

            parsed = json.loads(clean_content)

            # Validate basic schema
            if "scenario" not in parsed or "turns" not in parsed["scenario"]:
                print(f"    ⚠️ Missing 'scenario' or 'turns' in output, retrying...")
                continue

            turns = parsed["scenario"].get("turns", [])
            if len(turns) < 3:
                print(f"    ⚠️ Expected at least 3 turns, got {len(turns)}, retrying...")
                continue

            # Ensure ground-truth text fidelity & normalize each turn
            if "source" not in parsed:
                parsed["source"] = {}
            parsed["source"]["text"] = article_text
            parsed["source"]["label"] = label
            parsed["sample_id"] = sample_id

            for idx, turn in enumerate(turns, 1):
                _normalize_ground_truth(turn, idx)

            print(f"    ✓ 3-turn scenario with ground_truth generated in {elapsed:.1f}s")
            return parsed

        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON decode error on attempt {attempt}: {e}")
        except Exception as e:
            print(f"    ⚠️ Exception on attempt {attempt}: {e}")

        await asyncio.sleep(2)

    return None


async def generate_dataset(
    input_file: str,
    output_file: str,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.2,
    single_test: bool = False,
    force: bool = False
):
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    existing_results = []
    existing_ids = set()
    if output_path.exists() and not force:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                existing_ids = {item.get("sample_id") for item in existing_results}
                print(f"Found {len(existing_results)} already generated scenarios.")
        except Exception:
            existing_results = []
            existing_ids = set()
    elif force:
        print("🔄 Force mode enabled: Overwriting all previous scenarios.")

    if single_test:
        samples = samples[:1]
        print(f"Running in single-test mode: 1 sample ({samples[0].get('id')})")
    else:
        print(f"Loaded {len(samples)} total samples from {input_path}")

    print(f"LLM Endpoint: {base_url} | Model: {model}")

    results = list(existing_results)
    for idx, sample in enumerate(samples, 1):
        s_id = sample.get("id") or sample.get("sample_id")
        if s_id in existing_ids:
            print(f"\n[{idx}/{len(samples)}] Skipping {s_id} (already generated).")
            continue

        print(f"\n[{idx}/{len(samples)}] Processing {s_id} ({sample.get('source')})...")
        scenario = await generate_scenario_for_article(
            sample=sample,
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature
        )
        if scenario:
            results.append(scenario)
            existing_ids.add(s_id)
            # Incremental save
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"    Saved progress: {len(results)}/{len(samples)} scenarios")
        else:
            print(f"    ❌ Failed to generate scenario for {s_id}")

    print(f"\n🎉 Finished: {len(results)}/{len(samples)} scenarios ready with ground truth!")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multi-turn benchmark dataset with ground_truth via chat_client"
    )
    parser.add_argument(
        "--input",
        default="eval/tool_call/datasets/summarization/samples.json",
        help="Path to input samples JSON"
    )
    parser.add_argument(
        "--output",
        default="eval/tool_call/datasets/multi_turn/samples.json",
        help="Path to output multi_turn samples JSON"
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
        help="LLM API Key (default: from env LLM_API_KEY or GEMINI_API_KEY)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        "--test-single",
        action="store_true",
        help="Generate only 1 scenario for verification"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force regenerate all scenarios and overwrite existing file"
    )
    args = parser.parse_args()

    asyncio.run(generate_dataset(
        input_file=args.input,
        output_file=args.output,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        single_test=args.test_single,
        force=args.force
    ))
