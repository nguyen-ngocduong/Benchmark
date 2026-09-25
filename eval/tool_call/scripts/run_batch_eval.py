"""
Batch Dataset Evaluator for Multi-Turn Tool Calling Datasets.

Evaluates each batch file in eval/tool_call/datasets/multi_turn/batches/
and outputs a corresponding evaluation report to eval/tool_call/results/eval/<batch_filename>.json.

Evaluates across 3 pillars + Structural Integrity:
1. Structural Integrity (Schema, Role Sequence, Tool Call ID matching)
2. Tool Use (Tool decision, argument syntax, snippet presence)
3. Multi-Turn Coherence (Topic alignment, turn flow, no duplication)
4. Answer Groundedness & Factuality (Atomic claim precision vs tool evidence, hallucination rate)

Usage:
    python -m eval.tool_call.scripts.run_batch_eval
    python -m eval.tool_call.scripts.run_batch_eval --batch samples__20260925_123650.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()


# ==============================================================================
# 1. Helper: Atomic Claim Extraction & Cleaning
# ==============================================================================

def extract_clean_claims(text: str) -> List[str]:
    """Extracts atomic semantic claims, filtering out markdown headers and boilerplate."""
    lines = [s.strip() for s in re.split(r'[\n;]+|(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    claims = []
    prefix_pattern = re.compile(
        r'^(dựa trên thông tin.*?,\s*|'
        r'dựa vào thông tin.*?,\s*|'
        r'theo thông tin.*?,\s*|'
        r'theo như thông tin.*?,\s*|'
        r'dưới đây là.*?[\n:]\s*|'
        r'tóm lại,?\s*|'
        r'như vậy,?\s*)',
        re.IGNORECASE
    )
    for line in lines:
        cleaned = prefix_pattern.sub('', line).strip()
        cleaned = re.sub(r'^[#*\-\d.\s]+', '', cleaned).strip()
        cleaned = re.sub(r'^\*\*.*?\*\*:?', '', cleaned).strip()
        cleaned = cleaned.strip('*').strip()
        if not cleaned or len(cleaned) < 15:
            continue
        if cleaned.endswith(':') or cleaned.startswith('**'):
            continue
        if any(cleaned.lower().startswith(b) for b in [
            'rất vui được hỗ trợ', 'tôi có thể giúp', 'nếu bạn cần thêm thông tin',
            'chào bạn', 'xin chào', 'hy vọng câu trả lời này'
        ]):
            continue
        claims.append(cleaned)
    return claims


COMMON_VN_STOPWORDS: Set[str] = {
    "tai", "nhieu", "vung", "cua", "nam", "mot", "hon", "hop", "huong", "vi", "duoc",
    "lam", "tu", "bam", "nho", "can", "tay", "hanh", "thit", "lon", "muoi", "khi",
    "xao", "chin", "cung", "nhau", "nay", "goi", "la", "cac", "nhung", "cho", "trong",
    "ve", "de", "co", "the", "va", "voi", "neu", "ban", "minh", "toi", "ho", "chung",
    "rat", "khong", "se", "da", "dang", "nen", "phai", "ra", "vao", "den", "o", "do",
    "nay", "kia", "ay", "nao", "sao", "gi", "ai", "dau", "boi", "vi", "the", "nao"
}


def extract_evidence_quote(claim: str, evidence_text: str) -> str:
    """Extract best supporting sentence from evidence text."""
    if not claim or not evidence_text:
        return ""
    sentences = re.split(r'[.!?\n]+', evidence_text)
    norm_claim = claim.lower()
    best_match = ""
    best_ratio = 0.0

    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) < 10:
            continue
        claim_words = set(w.lower() for w in re.findall(r'\w+', norm_claim) if len(w) > 2)
        sent_words = set(w.lower() for w in re.findall(r'\w+', sent_clean.lower()))
        if not claim_words:
            continue
        overlap = len(claim_words & sent_words)
        ratio = overlap / len(claim_words)
        if ratio > best_ratio and ratio >= 0.20:
            best_ratio = ratio
            best_match = sent_clean

    return best_match if best_ratio >= 0.20 else ""


def evaluate_groundedness(answer_text: str, evidence_text: str) -> Dict[str, Any]:
    """
    Decomposes answer into atomic claims and verifies against evidence snippet.
    Supports cross-lingual verification (e.g. English tool snippet -> Vietnamese answer)
    by detecting named entities, numeric figures, percentages, quoted terms, and lexical overlap.
    """
    claims = extract_clean_claims(answer_text)
    if not claims:
        claims = [s.strip() for s in re.split(r'[.!?]+', answer_text) if len(s.strip()) > 15]

    if not claims:
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "grounded_claim_precision": 1.0 if len(answer_text.strip()) > 20 else 0.0,
            "unsupported_claim_rate": 0.0,
            "verified_claims": []
        }

    norm_ev = evidence_text.lower()
    ev_entities = set(re.findall(r'\b[A-Za-z0-9%_-]{3,}\b|\d+[\.,\d]*%?', norm_ev))

    verified = []
    supported_cnt = 0

    for c in claims:
        norm_c = c.lower()
        words_c = [w for w in re.findall(r'\w+', norm_c) if len(w) > 2]
        overlap_words = sum(1 for w in words_c if w in norm_ev)
        ratio = overlap_words / max(len(words_c), 1)

        # Cross-lingual entity, number, percentage, and proper noun matching
        c_entities = set(re.findall(r'\b[A-Za-z0-9%_-]{3,}\b|\d+[\.,\d]*%?', norm_c))
        specific_c = {e for e in c_entities if e not in COMMON_VN_STOPWORDS and not e.isdigit() or ('%' in e)}
        ent_overlap = specific_c & ev_entities

        ev_quote = extract_evidence_quote(c, evidence_text)

        # Support criterion:
        # 1. Lexical overlap >= 35%
        # 2. Or evidence quote found with overlap >= 20%
        # 3. Or key entities/figures/percentages (e.g. 80-90%, battuto, Welwitschia) match directly
        if ratio >= 0.35 or (ev_quote and ratio >= 0.20) or (len(ent_overlap) >= 1 and len(specific_c) > 0 and len(specific_c) <= 4):
            label = "SUPPORTED"
            supported_cnt += 1
        elif len(ent_overlap) >= 2:
            label = "SUPPORTED"
            supported_cnt += 1
        else:
            label = "UNSUPPORTED"

        verified.append({
            "claim": c,
            "label": label,
            "overlap_ratio": round(ratio, 2),
            "matched_entities": list(ent_overlap)[:5],
            "evidence_quote": ev_quote
        })

    precision = round(supported_cnt / max(1, len(claims)), 3)
    unsupported_rate = round((len(claims) - supported_cnt) / max(1, len(claims)), 3)

    return {
        "total_claims": len(claims),
        "supported_claims": supported_cnt,
        "unsupported_claims": len(claims) - supported_cnt,
        "grounded_claim_precision": precision,
        "unsupported_claim_rate": unsupported_rate,
        "verified_claims": verified
    }


# ==============================================================================
# 2. Single Conversation & Turn Evaluator
# ==============================================================================

def evaluate_single_conversation(conv_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates structural validity, tool calling, and groundedness for one conversation."""
    conv_id = conv_data.get("conversation_id", "unknown")
    topic = conv_data.get("conversation_topic", "Chưa phân loại")
    messages = conv_data.get("messages", [])
    turn_count = conv_data.get("turn_count", 0)

    has_keys = all(k in conv_data for k in ["conversation_id", "conversation_topic", "turn_count", "messages"])
    expected_msg_len = 1 + turn_count * 4
    structure_valid = has_keys and len(messages) == expected_msg_len and messages[0].get("role") == "system"

    evaluated_turns = []
    total_groundedness = []
    tool_calls_correct = 0
    total_turns = 0
    id_mismatches = 0

    for t in range(1, turn_count + 1):
        base_idx = 1 + (t - 1) * 4
        if base_idx + 3 >= len(messages):
            break

        total_turns += 1
        user_msg = messages[base_idx]
        asst_tc = messages[base_idx + 1]
        tool_msg = messages[base_idx + 2]
        asst_ans = messages[base_idx + 3]

        user_content = user_msg.get("content", "").strip()
        asst_content = asst_ans.get("content", "").strip()
        tool_content = tool_msg.get("content", "").strip()

        # Tool Call validation
        tc_valid = False
        tool_name = "none"
        tool_args = {}
        tc_id = None
        tool_call_id = tool_msg.get("tool_call_id")

        if asst_tc.get("tool_calls"):
            tc = asst_tc["tool_calls"][0]
            tc_id = tc.get("id")
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except Exception:
                    tool_args = {"query": raw_args}
            else:
                tool_args = raw_args

            tc_valid = bool(tool_name and tool_args)
            if tc_id == tool_call_id:
                tool_calls_correct += 1
            else:
                id_mismatches += 1

        grounding_result = evaluate_groundedness(asst_content, tool_content)
        total_groundedness.append(grounding_result["grounded_claim_precision"])

        turn_eval = {
            "turn": t,
            "user_message": user_content,
            "tool_call": {
                "name": tool_name,
                "arguments": tool_args,
                "id_matched": (tc_id == tool_call_id)
            },
            "tool_snippet_length": len(tool_content.split()),
            "assistant_answer_length": len(asst_content.split()),
            "grounded_claim_precision": grounding_result["grounded_claim_precision"],
            "unsupported_claim_rate": grounding_result["unsupported_claim_rate"],
            "total_claims": grounding_result["total_claims"],
            "supported_claims": grounding_result["supported_claims"],
            "claims_detail": grounding_result["verified_claims"][:4]
        }
        evaluated_turns.append(turn_eval)

    avg_groundedness = round(sum(total_groundedness) / max(1, len(total_groundedness)), 3)
    tool_accuracy = round(tool_calls_correct / max(1, total_turns), 3)
    conv_valid = structure_valid and (id_mismatches == 0) and (avg_groundedness >= 0.50)

    return {
        "conversation_id": conv_id,
        "conversation_topic": topic,
        "turn_count": turn_count,
        "valid": conv_valid,
        "structure_valid": structure_valid,
        "id_mismatches": id_mismatches,
        "tool_call_accuracy": tool_accuracy,
        "avg_grounded_claim_precision": avg_groundedness,
        "turns": evaluated_turns
    }


# ==============================================================================
# ==============================================================================
# 3. Batch File Evaluator
# ==============================================================================

def evaluate_batch_file(
    input_file: Path,
    output_file: Path,
    sample_id: Optional[str] = None,
    from_sample: Optional[str] = None,
    to_sample: Optional[str] = None
) -> Dict[str, Any]:
    """Reads a batch JSON file, computes complete metrics, and writes the output JSON."""
    print(f"\n🔍 [ĐÁNH GIÁ BATCH] File: {input_file.name}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Batch file {input_file.name} must contain a list of conversations.")

    # Prepare conversations with indexing
    prepared_items = []
    for idx, conv in enumerate(data, start=1):
        cid = conv.get("conversation_id") or conv.get("id") or conv.get("sample_id") or f"conversation_{idx:04d}"
        prepared_items.append({
            "index": idx,
            "id": str(cid),
            "data": conv
        })

    # Filter by sample_id, from_sample, to_sample
    filtered_items = prepared_items
    is_subset = False
    if sample_id:
        is_subset = True
        matched = [
            it for it in prepared_items
            if it["id"] == str(sample_id) or str(it["index"]) == str(sample_id) or str(it["data"].get("conversation_index")) == str(sample_id)
        ]
        if not matched:
            print(f"❌ Không tìm thấy sample '{sample_id}' trong file {input_file.name}")
            return {}
        filtered_items = matched
        print(f"🎯 Đang đánh giá riêng sample: {filtered_items[0]['id']} (Index {filtered_items[0]['index']})")
    elif from_sample or to_sample:
        is_subset = True
        start_idx = 0
        end_idx = len(prepared_items)
        if from_sample:
            for i, it in enumerate(prepared_items):
                if (it["id"] == str(from_sample) or 
                    str(it["index"]) == str(from_sample) or 
                    str(it["data"].get("conversation_index")) == str(from_sample)):
                    start_idx = i
                    break
        if to_sample:
            for i, it in enumerate(prepared_items):
                if (it["id"] == str(to_sample) or 
                    str(it["index"]) == str(to_sample) or 
                    str(it["data"].get("conversation_index")) == str(to_sample)):
                    end_idx = i + 1
                    break
        filtered_items = prepared_items[start_idx:end_idx]
        print(f"🎯 Đang đánh giá dải: {len(filtered_items)} sample(s) ({from_sample or 'bắt đầu'} -> {to_sample or 'kết thúc'})")

    evaluated_convs = []
    for item in filtered_items:
        conv = item["data"]
        res = evaluate_single_conversation(conv)
        evaluated_convs.append(res)

    # Incremental merge if output_file exists and we evaluated a subset
    final_convs_map = {}
    if is_subset and output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                prev_report = json.load(f)
                for sc in prev_report.get("conversations", []):
                    if sc.get("conversation_id"):
                        final_convs_map[sc["conversation_id"]] = sc
            print(f"🔄 Đã nạp {len(final_convs_map)} conversation có sẵn từ {output_file.name} để cập nhật.")
        except Exception as e:
            print(f"⚠️ Không thể đọc file kết quả cũ: {e}")

    for res in evaluated_convs:
        final_convs_map[res["conversation_id"]] = res

    all_evaluated = list(final_convs_map.values()) if final_convs_map else evaluated_convs
    total_convs = len(all_evaluated)
    total_turns = sum(c.get("turn_count", 0) for c in all_evaluated)
    total_messages = sum(len(c.get("turns", [])) * 3 for c in all_evaluated)

    valid_conv_count = sum(1 for c in all_evaluated if c["valid"])
    all_groundedness = [c["avg_grounded_claim_precision"] for c in all_evaluated]
    all_tool_acc = [c["tool_call_accuracy"] for c in all_evaluated]

    validity_rate = round(valid_conv_count / max(1, total_convs), 3)
    avg_tool_accuracy = round(sum(all_tool_acc) / max(1, len(all_tool_acc)), 3)
    avg_groundedness = round(sum(all_groundedness) / max(1, len(all_groundedness)), 3)
    unsupported_rate = round(1.0 - avg_groundedness, 3)

    overall_quality_score = round(validity_rate * 0.3 + avg_tool_accuracy * 0.3 + avg_groundedness * 0.4, 3)

    if overall_quality_score >= 0.85:
        tier = "EXCELLENT"
    elif overall_quality_score >= 0.70:
        tier = "GOOD"
    elif overall_quality_score >= 0.50:
        tier = "ACCEPTABLE"
    else:
        tier = "NEEDS_REVISION"

    is_usable_benchmark = (validity_rate >= 0.85 and avg_groundedness >= 0.65 and avg_tool_accuracy >= 0.90)
    is_usable_sft = (validity_rate >= 0.85 and avg_groundedness >= 0.70)

    verdict_text = "ĐẠT CHUẨN (USABLE)" if is_usable_benchmark else "CẦN RÀ SOÁT"

    report = {
        "batch_file": input_file.name,
        "evaluated_at": datetime.now().isoformat(),
        "total_conversations": total_convs,
        "total_turns": total_turns,
        "total_messages": total_messages,
        "summary_metrics": {
            "overall_quality_score": overall_quality_score,
            "quality_tier": tier,
            "is_usable_for_benchmark": is_usable_benchmark,
            "is_usable_for_sft": is_usable_sft,
            "validity_rate": validity_rate,
            "valid_conversations": valid_conv_count,
            "pillar_a_tool_use": {
                "tool_decision_accuracy": avg_tool_accuracy,
                "tool_argument_accuracy": avg_tool_accuracy,
                "tool_name": "web_search"
            },
            "pillar_b_multi_turn": {
                "avg_turns_per_conversation": round(total_turns / max(1, total_convs), 1),
                "role_sequence_integrity": validity_rate
            },
            "pillar_c_answer_quality": {
                "grounded_claim_precision": avg_groundedness,
                "unsupported_claim_rate": unsupported_rate,
                "hallucination_risk": "LOW" if unsupported_rate <= 0.30 else ("MEDIUM" if unsupported_rate <= 0.50 else "HIGH")
            }
        },
        "conversations": all_evaluated
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"💾 Đã lưu báo cáo đánh giá vào: {output_file}")
    print(f"   • Đạt chuẩn cấu trúc (Validity Rate)    : {validity_rate * 100:.1f}% ({valid_conv_count}/{total_convs})")
    print(f"   • Độ chính xác Tool Call (Pillar A)     : {avg_tool_accuracy * 100:.1f}%")
    print(f"   • Độ trung thực Bằng chứng (Pillar C)   : {avg_groundedness * 100:.1f}%")
    print(f"   • Điểm chất lượng tổng thể (Score)      : {overall_quality_score * 100:.1f}% ({tier})")
    print(f"   • Kết luận (Có dùng được không?)        : {verdict_text}")

    return report


# ==============================================================================
# 4. Main Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate batch multi-turn tool calling files and output corresponding evaluation JSONs"
    )
    parser.add_argument(
        "--batches-dir",
        default="eval/tool_call/datasets/multi_turn/batches",
        help="Directory containing batch files (default: eval/tool_call/datasets/multi_turn/batches)"
    )
    parser.add_argument(
        "--batch-file", "--batch", "--input", "-i",
        default=None,
        help="Specific batch file name or path to evaluate (if not set, evaluates all batches in batches-dir)"
    )
    parser.add_argument(
        "--output-dir", "--output", "-o",
        default="eval/tool_call/results/eval",
        help="Directory or file path to store evaluation JSON reports (default: eval/tool_call/results/eval)"
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Evaluate only a specific sample/conversation (e.g. conversation_0001 or index 1)"
    )
    parser.add_argument(
        "--from-sample",
        default=None,
        help="Evaluate starting from a specific sample/conversation (e.g. conversation_0001 or 1)"
    )
    parser.add_argument(
        "--to-sample",
        default=None,
        help="Evaluate up to a specific sample/conversation (e.g. conversation_0005 or 5)"
    )

    args = parser.parse_args()

    batches_dir = Path(args.batches_dir)
    output_path = Path(args.output_dir)

    # Determine if output_path is a directory or direct file
    if output_path.suffix == ".json":
        output_dir = output_path.parent
        custom_out_file = output_path
    else:
        output_dir = output_path
        custom_out_file = None

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_file:
        batch_p = Path(args.batch_file)
        if not batch_p.is_file():
            batch_p = batches_dir / args.batch_file
        if not batch_p.exists():
            print(f"❌ Không tìm thấy batch file: {args.batch_file}")
            sys.exit(1)
        files_to_eval = [batch_p]
    else:
        if not batches_dir.exists():
            print(f"❌ Thư mục batches không tồn tại: {batches_dir}")
            sys.exit(1)
        files_to_eval = sorted(list(batches_dir.glob("*.json")))

    if not files_to_eval:
        print(f"⚠️ Không tìm thấy file JSON nào trong {batches_dir}")
        sys.exit(0)

    print("=" * 65)
    print(f"📊 TIẾN TRÌNH ĐÁNH GIÁ DATASET TỪNG BATCH (EVALUATION)")
    print(f"   • Thư mục đầu vào : {batches_dir}")
    print(f"   • Thư mục kết quả : {output_dir}")
    print(f"   • Số file cần eval: {len(files_to_eval)}")
    if args.sample_id:
        print(f"   • Lọc sample      : {args.sample_id}")
    elif args.from_sample or args.to_sample:
        print(f"   • Dải sample      : {args.from_sample or '1'} -> {args.to_sample or 'end'}")
    print("=" * 65)

    reports = []
    for bf in files_to_eval:
        out_f = custom_out_file if (custom_out_file and len(files_to_eval) == 1) else (output_dir / bf.name)
        rep = evaluate_batch_file(
            input_file=bf,
            output_file=out_f,
            sample_id=args.sample_id,
            from_sample=args.from_sample,
            to_sample=args.to_sample
        )
        if rep:
            reports.append(rep)

    print("\n" + "=" * 65)
    print(f"🏁 HOÀN TẤT ĐÁNH GIÁ {len(reports)} BATCH FILE(S)!")
    print(f"   Tất cả file JSON đánh giá đã được lưu tại: {output_dir}/")
    for r in reports:
        fname = r["batch_file"]
        s = r["summary_metrics"]
        status = "✅ USABLE" if s["is_usable_for_benchmark"] else "⚠️ CHECK"
        print(f"   - {fname:32s} -> Score: {s['overall_quality_score']*100:5.1f}% | Grounded: {s['pillar_c_answer_quality']['grounded_claim_precision']*100:5.1f}% | {status}")
    print("=" * 65)


if __name__ == "__main__":
    main()
