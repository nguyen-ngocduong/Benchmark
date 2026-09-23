"""
Enrich raw news samples from samples_new.json into standardized Ground Truth for Task 1 (Summarization).
Appends newly enriched samples into eval/tool_call/datasets/summarization/samples.json.

Includes:
- topic: Concise topic / headline
- article: Preserved source text
- article_word_count: Word count of article
- reference_summary: High-quality golden reference summary
- key_facts: 3-5 weighted atomic facts with verbatim evidence_spans
- key_entities: Important named entities (people, organizations, locations)
- key_numbers: Important quantitative data, numbers, and dates

Usage:
    python -m eval.tool_call.scripts.enrich_summarization_dataset
    python -m eval.tool_call.scripts.enrich_summarization_dataset --test-single
    python -m eval.tool_call.scripts.enrich_summarization_dataset --input path/to/samples_new.json --output path/to/samples.json
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

PROMPT_SUMMARIZATION_ENRICH = """Bạn là chuyên gia thẩm định và gán nhãn dữ liệu Benchmark đánh giá mô hình tóm tắt văn bản tiếng Việt (Vietnamese Document Summarization Benchmark).

Nhiệm vụ của bạn là đọc kỹ bài báo dưới đây và trích xuất/sinh bộ dữ liệu Ground Truth chuẩn mực theo schema quy định.

==================================================
BÀI BÁO ĐẦU VÀO:
- ID: {sample_id}
- Nội dung bài báo:
\"\"\"
{article_text}
\"\"\"
==================================================

HƯỚNG DẪN CHI TIẾT:
1. "topic": Tiêu đề hoặc chủ đề khái quát của bài viết (1 câu ngắn gọn, tự nhiên, dưới 20 từ).
2. "reference_summary": Bản tóm tắt tham chiếu chuẩn mực (Gold Summary) độ dài khoảng 60 - 120 từ:
   - Bao quát đầy đủ các ý chính, thông tin cốt lõi nhất của bài viết.
   - Giữ văn phong báo chí khách quan, chuẩn xác, mạch lạc.
   - Tuyệt đối không thêm thắt thông tin ngoài bài báo (Faithful).
3. "key_facts": Danh sách từ 3 đến 5 sự thật nguyên tử cốt lõi (atomic verifiable facts):
   - "id": "f1", "f2", "f3", ...
   - "fact": Phát biểu sự thật ngắn gọn, độc lập.
   - "weight": 2 (đối với thông tin/kết quả/sự kiện chính quan trọng nhất) hoặc 1 (đối với chi tiết phụ/nguyên nhân/bối cảnh).
   - "evidence_span": Câu hoặc đoạn ngắn TRÍCH DẪN NGUYÊN VĂN từ bài báo chứng minh trực tiếp cho fact đó.
4. "key_entities": Mảng các thực thể tên riêng quan trọng (tên người, công ty, tổ chức, địa danh, tên bộ phim/tác phẩm, thương hiệu) xuất hiện trong bài báo.
5. "key_numbers": Mảng các số liệu, ngày tháng, thời gian, số lượng hoặc tỷ lệ phần trăm cụ thể xuất hiện trong bài báo (ví dụ: "35 nghìn", "gần ba tháng", "1/8", "45 thiết kế").

==================================================
YÊU CẦU ĐỊNH DẠNG (JSON ONLY):
Trả về DUY NHẤT một chuỗi JSON hợp lệ, KHÔNG sử dụng markdown block ```json, KHÔNG có văn bản giải thích:

{{
  "id": "{sample_id}",
  "topic": "<Chủ đề bài viết>",
  "article": "<Giữ nguyên vẹn toàn văn bài báo>",
  "article_word_count": {word_count},
  "reference_summary": "<Bản tóm tắt chuẩn mực>",
  "key_facts": [
    {{
      "id": "f1",
      "fact": "<Nội dung fact 1>",
      "weight": 2,
      "evidence_span": "<Trích dẫn nguyên văn câu trong bài>"
    }},
    {{
      "id": "f2",
      "fact": "<Nội dung fact 2>",
      "weight": 1,
      "evidence_span": "<Trích dẫn nguyên văn câu trong bài>"
    }}
  ],
  "key_entities": ["<Thực thể 1>", "<Thực thể 2>"],
  "key_numbers": ["<Số liệu 1>", "<Số liệu 2>"]
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


async def enrich_single_article(
    sample: Dict[str, Any],
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.1
) -> Optional[Dict[str, Any]]:
    """Calls LLM to enrich a single article into Task 1 Ground Truth schema."""
    s_id = sample.get("id") or sample.get("sample_id", "sample_unknown")
    article_text = (sample.get("article") or sample.get("text") or "").strip()
    word_count = len(article_text.split())

    prompt = PROMPT_SUMMARIZATION_ENRICH.format(
        sample_id=s_id,
        article_text=article_text,
        word_count=word_count
    )

    messages = [
        {"role": "system", "content": "You are a professional NLP benchmark data annotator for Vietnamese text. Output valid JSON only."},
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
                max_tokens=2500
            )
            raw_text = res.get("content") or res.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not raw_text:
                error_msg = res.get("error", "Empty response")
                print(f"    [Attempt {attempt}] Empty response or error from LLM: {error_msg}")
                await asyncio.sleep(1.5 * attempt)
                continue

            parsed = clean_json_response(raw_text)

            if parsed and parsed.get("reference_summary") and parsed.get("key_facts"):
                # Ensure structure integrity
                parsed["id"] = s_id
                parsed["article"] = article_text
                parsed["article_word_count"] = word_count

                # Ensure key_facts have required fields
                facts = []
                for i, f in enumerate(parsed.get("key_facts", []), start=1):
                    fid = f.get("id") or f"f{i}"
                    fact_str = f.get("fact", "").strip()
                    weight = int(f.get("weight", 1))
                    if weight not in (1, 2):
                        weight = 1
                    evidence = f.get("evidence_span", "").strip()
                    if fact_str:
                        facts.append({
                            "id": fid,
                            "fact": fact_str,
                            "weight": weight,
                            "evidence_span": evidence
                        })
                parsed["key_facts"] = facts

                if "key_entities" not in parsed or not isinstance(parsed["key_entities"], list):
                    parsed["key_entities"] = []
                if "key_numbers" not in parsed or not isinstance(parsed["key_numbers"], list):
                    parsed["key_numbers"] = []

                return parsed

            print(f"    [Attempt {attempt}] Invalid schema returned for {s_id}, retrying...")
            await asyncio.sleep(1.5 * attempt)

        except Exception as e:
            print(f"    [Attempt {attempt}] Error enriching {s_id}: {e}")
            await asyncio.sleep(2.0 * attempt)

    return None


async def enrich_summarization_dataset(
    input_file: str,
    output_file: str,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 0.1,
    sample_id: Optional[str] = None,
    from_sample: Optional[str] = None,
    single_test: bool = False,
    force: bool = False
):
    """
    Reads newly loaded samples from input_file (samples_new.json),
    enriches with Task 1 Ground Truth, and appends to output_file (samples.json).
    Supports single sample execution (--sample-id) and resumption (--from-sample).
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        new_samples = json.load(f)

    # Load existing samples from output_file to avoid duplication
    existing_samples: List[Dict[str, Any]] = []
    existing_ids = set()

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_samples = json.loads(content)
                    existing_ids = {s.get("id") for s in existing_samples if s.get("id")}
            print(f"📖 Loaded {len(existing_samples)} existing samples from {output_path.name}.")
        except Exception as e:
            print(f"⚠️ Could not parse existing {output_path}: {e}")
            existing_samples = []
            existing_ids = set()

    # Filter candidate samples by sample_id, from_sample, or single_test
    if sample_id:
        matched = [s for s in new_samples if s.get("id") == sample_id]
        if matched:
            new_samples = matched
            print(f"🎯 Filtered for single sample: {sample_id} ({len(new_samples)} matched in {input_path.name})")
        else:
            existing_match = next((s for s in existing_samples if s.get("id") == sample_id), None)
            if existing_match:
                new_samples = [{
                    "id": sample_id,
                    "article": existing_match.get("article", ""),
                    "label": existing_match.get("label", 0)
                }]
                print(f"🎯 Filtered for existing sample in dataset: {sample_id}")
            else:
                print(f"❌ Sample {sample_id} not found in {input_path.name} or {output_path.name}")
                return
    elif from_sample:
        from_idx = None
        for i, s in enumerate(new_samples):
            if s.get("id") == from_sample:
                from_idx = i
                break
        if from_idx is not None:
            new_samples = new_samples[from_idx:]
            print(f"🎯 Filtered from sample {from_sample} onwards ({len(new_samples)} samples in {input_path.name})")
        else:
            print(f"⚠️ Sample {from_sample} not found in {input_path.name}. Running all candidate samples.")
    if single_test and not sample_id:
        new_samples = new_samples[:1]
        print(f"🔍 Running in single-test mode: 1 sample ({new_samples[0].get('id')})")
    elif not sample_id and not from_sample and not single_test:
        print(f"📋 Loaded {len(new_samples)} candidate samples from {input_path}")

    print(f"🌐 LLM Endpoint: {base_url} | Model: {model}\n")

    enriched_dataset = list(existing_samples)
    newly_added_count = 0

    for idx, sample in enumerate(new_samples, start=1):
        s_id = sample.get("id") or f"sample_{idx:03d}"
        if s_id in existing_ids and not force and not sample_id:
            print(f"[{idx}/{len(new_samples)}] ⏭️ Skipping {s_id} (already exists in {output_path.name}).")
            continue

        print(f"[{idx}/{len(new_samples)}] ⏳ Enriching {s_id}...")
        enriched = await enrich_single_article(
            sample=sample,
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature
        )

        if enriched:
            # If item exists in dataset, update in-place; else append
            existing_match_idx = next((i for i, x in enumerate(enriched_dataset) if x.get("id") == s_id), None)
            if existing_match_idx is not None:
                enriched_dataset[existing_match_idx] = enriched
            else:
                enriched_dataset.append(enriched)
                existing_ids.add(s_id)

            newly_added_count += 1

            # Incremental save to output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enriched_dataset, f, ensure_ascii=False, indent=2)

            print(f"    ✓ Successfully enriched {s_id}: '{enriched.get('topic')}' ({len(enriched.get('key_facts', []))} facts, {len(enriched.get('key_entities', []))} entities)")
            print(f"    💾 Saved progress: {len(enriched_dataset)} total samples in {output_path.name}")
        else:
            print(f"    ❌ Failed to enrich {s_id}")

    print(f"\n{'='*70}")
    print(f"🎉 TASK 1 ENRICHMENT COMPLETE!")
    print(f"   • Newly added/updated: {newly_added_count} samples")
    print(f"   • Total samples now in {output_path.name}: {len(enriched_dataset)}")
    print(f"   • Output location: {output_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich raw news samples from samples_new.json into Task 1 Summarization Ground Truth"
    )
    parser.add_argument(
        "--input",
        default="eval/tool_call/datasets/summarization/samples_new.json",
        help="Path to input samples_new.json"
    )
    parser.add_argument(
        "--output",
        default="eval/tool_call/datasets/summarization/samples.json",
        help="Path to target samples.json to append enriched data"
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
        "--sample-id",
        type=str,
        default=None,
        help="Enrich only a specific sample_id (e.g. sample_011)"
    )
    parser.add_argument(
        "--from-sample",
        type=str,
        default=None,
        help="Enrich starting from a specific sample_id onwards (e.g. sample_011)"
    )
    parser.add_argument(
        "--test-single",
        action="store_true",
        help="Enrich only 1 sample for verification"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-enrich and overwrite existing samples with same ID"
    )

    args = parser.parse_args()

    asyncio.run(enrich_summarization_dataset(
        input_file=args.input,
        output_file=args.output,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        sample_id=args.sample_id,
        from_sample=args.from_sample,
        single_test=args.test_single,
        force=args.force
    ))


if __name__ == "__main__":
    main()
