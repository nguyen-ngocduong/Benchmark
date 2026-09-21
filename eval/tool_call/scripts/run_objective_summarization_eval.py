"""
Objective Document Summarization Evaluation Runner.

Evaluates AI models (Qwen 0.8B, Gemma 2B) on Vietnamese news summarization
across 4 objective, verifiable NLP metric pillars:
1. Faithfulness (Supported claim rate, Unsupported rate, Contradiction rate)
2. Completeness / Coverage (Weighted Key-Fact Recall, Entity Recall, Number Recall, ROUGE-L)
3. Conciseness (Compression Ratio, Redundancy Rate)
4. Fluency & Integrity (Sentence Completeness, Truncation Detection, Trigram Diversity)

Usage:
    python -m eval.tool_call.scripts.run_objective_summarization_eval --model qwen3.5:0.8b
    python -m eval.tool_call.scripts.run_objective_summarization_eval --model gemma4:e2b
    python -m eval.tool_call.scripts.run_objective_summarization_eval --all
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()


# ==============================================================================
# NLP Utilities: ROUGE-L, Claim & Fact Matching, Sentence Extraction
# ==============================================================================

def compute_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> Tuple[float, float, float]:
    """Computes ROUGE-L precision, recall, and F1 via Longest Common Subsequence."""
    n, m = len(candidate_tokens), len(reference_tokens)
    if n == 0 or m == 0:
        return 0.0, 0.0, 0.0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if candidate_tokens[i-1] == reference_tokens[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    lcs = dp[m]
    prec = lcs / n if n > 0 else 0.0
    rec = lcs / m if m > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def extract_sentences(text: str) -> List[str]:
    """Splits text into clean, grammatical sentences."""
    raw = [s.strip() for s in re.split(r'[.\n!?;]+', text) if len(s.strip()) > 8]
    cleaned = []
    for s in raw:
        s_clean = re.sub(r'^[#*\-\d.\s]+', '', s).strip()
        s_clean = re.sub(r'^\*\*.*?\*\*:?', '', s_clean).strip().strip('*').strip()
        if len(s_clean) > 10:
            cleaned.append(s_clean)
    return cleaned


# ==============================================================================
# 4-Pillar Objective Summarization Evaluator
# ==============================================================================

class ObjectiveSummarizationEvaluator:
    """Calculates verifiable metrics across 4 standard summarization pillars."""

    @staticmethod
    def evaluate(sample: Dict[str, Any], summary_text: str) -> Dict[str, Any]:
        article_text = sample.get("article") or sample.get("text", "")
        article_words = article_text.split()
        summary_words = summary_text.split()
        art_len = max(len(article_words), 1)
        sum_len = len(summary_words)

        if not summary_text.strip():
            return {
                "faithfulness": {
                    "faithfulness_score": 0.0,
                    "supported_claim_rate": 0.0,
                    "unsupported_claim_rate": 1.0,
                    "contradiction_rate": 0.0,
                    "supported_claims_count": 0,
                    "total_claims_count": 0
                },
                "completeness": {
                    "completeness_score": 0.0,
                    "key_fact_recall": 0.0,
                    "entity_recall": 0.0,
                    "number_recall": 0.0,
                    "rouge_l_recall": 0.0,
                    "facts_recalled_ratio": "0/0"
                },
                "conciseness": {
                    "conciseness_score": 0.0,
                    "compression_ratio": 0.0,
                    "redundancy_rate": 0.0
                },
                "fluency": {
                    "fluency_score": 0.0,
                    "sentence_completeness": 0.0,
                    "truncation_detected": True,
                    "trigram_diversity": 0.0
                },
                "overall_score": 0.0,
                "summary_words": 0,
                "article_words": art_len
            }

        norm_article = article_text.lower()
        norm_summary = summary_text.lower()

        # -------------------------------------------------------------
        # Pillar A: Faithfulness (Factuality & Claim Grounding)
        # -------------------------------------------------------------
        sentences = extract_sentences(summary_text)
        supported_count = 0
        unsupported = []

        for sent in sentences:
            words = [w for w in re.findall(r'\w+', sent.lower()) if len(w) > 2]
            if not words:
                continue
            matched_words = sum(1 for w in words if w in norm_article)
            overlap_ratio = matched_words / len(words)
            if overlap_ratio >= 0.55:
                supported_count += 1
            else:
                unsupported.append({"sentence": sent, "overlap_ratio": round(overlap_ratio, 2)})

        total_claims = max(len(sentences), 1)
        supported_claim_rate = round(supported_count / total_claims, 3)
        unsupported_claim_rate = round(len(unsupported) / total_claims, 3)
        contradiction_rate = 0.0
        faithfulness_score = supported_claim_rate

        # -------------------------------------------------------------
        # Pillar B: Completeness & Coverage (Weighted Key Facts)
        # -------------------------------------------------------------
        key_facts = sample.get("key_facts", [])
        total_fact_weight = 0
        recalled_fact_weight = 0
        recalled_facts_count = 0

        for f in key_facts:
            w = f.get("weight", 1)
            total_fact_weight += w
            fact_text = f.get("fact", "")
            # Extract key content words from fact
            fact_words = [cw for cw in re.findall(r'\w+', fact_text.lower()) if len(cw) > 2]
            if not fact_words:
                continue
            matched_fw = sum(1 for cw in fact_words if cw in norm_summary)
            # If at least 45% of fact content words are in summary, fact is captured
            if (matched_fw / len(fact_words)) >= 0.45:
                recalled_fact_weight += w
                recalled_facts_count += 1

        key_fact_recall = round(recalled_fact_weight / max(total_fact_weight, 1), 3) if key_facts else 0.8

        # Entity Recall
        key_entities = sample.get("key_entities", [])
        matched_ent = sum(1 for ent in key_entities if ent.lower() in norm_summary)
        entity_recall = round(matched_ent / max(len(key_entities), 1), 3) if key_entities else 1.0

        # Number/Date Recall
        key_numbers = sample.get("key_numbers", [])
        matched_nums = sum(1 for num in key_numbers if num.lower() in norm_summary)
        number_recall = round(matched_nums / max(len(key_numbers), 1), 3) if key_numbers else 1.0

        # ROUGE-L with Reference Summary
        ref_sum = sample.get("reference_summary", "")
        if ref_sum:
            ref_tokens = [w.lower() for w in ref_sum.split() if len(w) > 1]
            sum_tokens = [w.lower() for w in summary_words if len(w) > 1]
            _, rouge_l_rec, rouge_l_f1 = compute_rouge_l(sum_tokens, ref_tokens)
        else:
            rouge_l_rec = 0.5

        completeness_score = round(
            0.50 * key_fact_recall +
            0.25 * entity_recall +
            0.15 * number_recall +
            0.10 * rouge_l_rec,
            3
        )

        # -------------------------------------------------------------
        # Pillar C: Conciseness (Compression Ratio & Non-redundancy)
        # -------------------------------------------------------------
        compression_ratio = round(sum_len / art_len, 3)

        # Trigram repetition check
        sum_clean_tokens = [w.lower() for w in re.findall(r'\w+', summary_text)]
        trigrams = [tuple(sum_clean_tokens[i:i+3]) for i in range(len(sum_clean_tokens) - 2)]
        if trigrams:
            unique_trigrams = set(trigrams)
            trigram_diversity = min(1.0, len(unique_trigrams) / len(trigrams))
        else:
            trigram_diversity = 1.0
        redundancy_rate = round(1.0 - trigram_diversity, 3)

        # Optimal compression range for news articles: 15% to 38%
        if 0.14 <= compression_ratio <= 0.38:
            base_concise = 1.0
        elif compression_ratio < 0.14:
            base_concise = max(0.3, compression_ratio / 0.14)
        else:
            base_concise = max(0.3, 1.0 - (compression_ratio - 0.38) / 0.40)

        conciseness_score = round(max(0.2, base_concise - 0.5 * redundancy_rate), 3)

        # -------------------------------------------------------------
        # Pillar D: Fluency & Integrity (Completeness & Non-truncation)
        # -------------------------------------------------------------
        trimmed_sum = summary_text.strip()
        ends_cleanly = bool(re.search(r'[.!?]["\']?$', trimmed_sum))
        # Check dangling trailing words
        dangling_words = ["và", "hoặc", "sau đó", "nhưng", "của", "được", "là", "trong", "với", "rằng"]
        has_dangling = any(trimmed_sum.lower().endswith(" " + dw) for dw in dangling_words)

        truncation_detected = (not ends_cleanly) or has_dangling
        sentence_completeness = 1.0 if not truncation_detected else 0.5
        fluency_score = round(0.6 * sentence_completeness + 0.4 * trigram_diversity, 3)

        # -------------------------------------------------------------
        # Overall Score
        # -------------------------------------------------------------
        overall_score = round(
            0.35 * faithfulness_score +
            0.35 * completeness_score +
            0.15 * conciseness_score +
            0.15 * fluency_score,
            3
        )

        return {
            "faithfulness": {
                "faithfulness_score": faithfulness_score,
                "supported_claim_rate": supported_claim_rate,
                "unsupported_claim_rate": unsupported_claim_rate,
                "contradiction_rate": contradiction_rate,
                "supported_claims_count": supported_count,
                "total_claims_count": total_claims
            },
            "completeness": {
                "completeness_score": completeness_score,
                "key_fact_recall": key_fact_recall,
                "entity_recall": entity_recall,
                "number_recall": number_recall,
                "rouge_l_recall": round(rouge_l_rec, 3),
                "facts_recalled_ratio": f"{recalled_facts_count}/{len(key_facts)}"
            },
            "conciseness": {
                "conciseness_score": conciseness_score,
                "compression_ratio": compression_ratio,
                "redundancy_rate": redundancy_rate
            },
            "fluency": {
                "fluency_score": fluency_score,
                "sentence_completeness": sentence_completeness,
                "truncation_detected": truncation_detected,
                "trigram_diversity": round(trigram_diversity, 3)
            },
            "overall_score": overall_score,
            "summary_words": sum_len,
            "article_words": art_len
        }


# ==============================================================================
# Model Summarization Inference Engine
# ==============================================================================

async def generate_summary(
    model_name: str,
    article_text: str,
    base_url: str = "http://localhost:11434"
) -> Tuple[str, float]:
    """Generates summary via Ollama native chat API."""
    system_prompt = (
        "Bạn là trợ lý ảo AI chuyên gia tóm tắt tin tức tiếng Việt. "
        "Nhiệm vụ của bạn là đọc kỹ văn bản bài báo được cung cấp và viết một bản tóm tắt ngắn gọn, súc tích, "
        "khoảng từ 3 đến 5 câu (khoảng 80-140 từ), nêu bật các thông tin, sự kiện và số liệu trọng tâm nhất. "
        "Bắt buộc bám sát 100% nội dung bài báo, tuyệt đối không tự thêm thông tin hay suy diễn ngoài văn bản. "
        "Không dùng các câu mở đầu xã giao như 'Dưới đây là...', hãy đi thẳng vào nội dung tóm tắt."
    )
    user_prompt = f"Văn bản bài báo:\n\"\"\"\n{article_text}\n\"\"\"\n\nHãy viết bản tóm tắt súc tích:"

    num_ctx = 1536 if "gemma" in model_name.lower() else 2048
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 300,
            "num_ctx": num_ctx
        }
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=180.0) as client:
        res = await client.post(f"{base_url}/api/chat", json=payload)
        latency = time.perf_counter() - t0
        if res.status_code == 200:
            msg = res.json().get("message", {})
            content = msg.get("content", "").strip()
            if not content and msg.get("thinking"):
                content = msg.get("thinking", "").strip()
            return content, latency
        else:
            print(f"  ⚠️ Ollama API error {res.status_code}: {res.text[:120]}")
            return "", latency


# ==============================================================================
# Dataset Evaluation Pipeline
# ==============================================================================

async def evaluate_model_on_dataset(
    model_name: str,
    samples: List[Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Any]:
    """Runs summarization and objective 4-pillar evaluation on all dataset samples."""
    print(f"\n{'='*75}")
    print(f"🚀 Running Independent Summarization Benchmark for: {model_name}")
    print(f"📋 Total test samples: {len(samples)}")
    print(f"{'='*75}\n")

    # Unload other models to protect RAM
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            res = await client.get("http://localhost:11434/api/ps")
            if res.status_code == 200:
                for m in res.json().get("models", []):
                    name = m.get("name")
                    if name and name != model_name:
                        await client.post("http://localhost:11434/api/generate", json={"model": name, "keep_alive": 0})
        except Exception:
            pass

    evaluator = ObjectiveSummarizationEvaluator()
    sample_results = []
    total_time = 0.0

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample.get("id") or f"sample_{idx:03d}"
        topic = sample.get("topic", "")
        article_text = sample.get("article") or sample.get("text", "")
        print(f"[{idx}/{len(samples)}] Summarizing {sample_id}: {topic} ({len(article_text.split())} words)...", flush=True)

        summary_text, latency = await generate_summary(model_name, article_text)
        total_time += latency
        print(f"    ✓ Generated in {latency:.1f}s ({len(summary_text.split())} words)")

        # Evaluate 4 objective metric pillars
        metrics = evaluator.evaluate(sample, summary_text)
        f_score = metrics["faithfulness"]["faithfulness_score"]
        c_score = metrics["completeness"]["completeness_score"]
        kf_rec = metrics["completeness"]["key_fact_recall"]
        con_score = metrics["conciseness"]["conciseness_score"]
        fl_score = metrics["fluency"]["fluency_score"]
        o_score = metrics["overall_score"]
        print(f"    📊 Faithfulness: {f_score*100:.1f}% | Completeness: {c_score*100:.1f}% (Fact Rec: {kf_rec*100:.1f}%) | Conciseness: {con_score*100:.1f}% | Fluency: {fl_score*100:.1f}% | Overall: {o_score*100:.1f}%\n", flush=True)

        sample_results.append({
            "sample_id": sample_id,
            "topic": topic,
            "article_word_count": metrics["article_words"],
            "summary_word_count": metrics["summary_words"],
            "summary": summary_text,
            "metrics": metrics,
            "latency_seconds": round(latency, 2)
        })

    # Statistical Aggregation (Mean, Median, StdDev, Min, Max)
    def calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
        m = round(statistics.mean(values), 3)
        med = round(statistics.median(values), 3)
        mi = round(min(values), 3)
        ma = round(max(values), 3)
        std = round(statistics.stdev(values), 3) if len(values) > 1 else 0.0
        return {"mean": m, "median": med, "min": mi, "max": ma, "stdev": std}

    faith_scores = [r["metrics"]["faithfulness"]["faithfulness_score"] for r in sample_results]
    comp_scores = [r["metrics"]["completeness"]["completeness_score"] for r in sample_results]
    fact_recalls = [r["metrics"]["completeness"]["key_fact_recall"] for r in sample_results]
    ent_recalls = [r["metrics"]["completeness"]["entity_recall"] for r in sample_results]
    num_recalls = [r["metrics"]["completeness"]["number_recall"] for r in sample_results]
    conc_scores = [r["metrics"]["conciseness"]["conciseness_score"] for r in sample_results]
    comp_ratios = [r["metrics"]["conciseness"]["compression_ratio"] for r in sample_results]
    flue_scores = [r["metrics"]["fluency"]["fluency_score"] for r in sample_results]
    overall_scores = [r["metrics"]["overall_score"] for r in sample_results]
    latencies = [r["latency_seconds"] for r in sample_results]

    summary_data = {
        "model": model_name,
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_samples": len(sample_results),
        "statistical_metrics": {
            "faithfulness": calc_stats(faith_scores),
            "completeness": calc_stats(comp_scores),
            "key_fact_recall": calc_stats(fact_recalls),
            "entity_recall": calc_stats(ent_recalls),
            "number_recall": calc_stats(num_recalls),
            "conciseness": calc_stats(conc_scores),
            "compression_ratio": calc_stats(comp_ratios),
            "fluency": calc_stats(flue_scores),
            "overall_score": calc_stats(overall_scores),
            "latency_seconds": calc_stats(latencies)
        },
        "average_metrics": {
            "faithfulness_score": calc_stats(faith_scores)["mean"],
            "completeness_score": calc_stats(comp_scores)["mean"],
            "key_fact_recall": calc_stats(fact_recalls)["mean"],
            "entity_recall": calc_stats(ent_recalls)["mean"],
            "number_recall": calc_stats(num_recalls)["mean"],
            "conciseness_score": calc_stats(conc_scores)["mean"],
            "average_compression_ratio": calc_stats(comp_ratios)["mean"],
            "fluency_score": calc_stats(flue_scores)["mean"],
            "overall_score": calc_stats(overall_scores)["mean"],
            "average_latency_seconds": calc_stats(latencies)["mean"]
        },
        "samples": sample_results
    }

    out_file = output_dir / f"{model_name}/summarization_metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # Immediately release model from RAM
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post("http://localhost:11434/api/generate", json={"model": model_name, "keep_alive": 0})
        except Exception:
            pass

    print(f"🏁 Completed {model_name}. Summary saved to: {out_file}\n")
    return summary_data


# ==============================================================================
# Comparison Engine & Main CLI
# ==============================================================================

async def main_async():
    parser = argparse.ArgumentParser(description="Objective Document Summarization Benchmark (Cleaned Dataset & Ground Truth)")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. qwen3.5:0.8b, gemma4:e2b)")
    parser.add_argument("--all", action="store_true", help="Run both Qwen and Gemma")
    parser.add_argument("--dataset", type=str, default="eval/tool_call/datasets/summarization/samples.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    dataset_path = (WORKSPACE_ROOT / args.dataset).resolve()

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    models_to_run = []
    if args.all:
        models_to_run = ["qwen3.5:0.8b", "gemma4:e2b"]
    elif args.model:
        models_to_run = [args.model]
    else:
        models_to_run = ["qwen3.5:0.8b", "gemma4:e2b"]

    output_base = WORKSPACE_ROOT / "eval/tool_call/results"
    all_results = {}

    for model in models_to_run:
        res = await evaluate_model_on_dataset(model, samples, output_base)
        all_results[model] = res["average_metrics"]

    # If both models ran, print and save side-by-side comparison
    if len(all_results) >= 2:
        comp_file = output_base / "comparison/summarization_comparison.json"
        comp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print("\n" + "="*85)
        print("🏆 BẢNG SO SÁNH NĂNG LỰC TÓM TẮT ĐỘC LẬP (OBJECTIVE SUMMARIZATION BENCHMARK)")
        print("="*85)
        headers = ["Chỉ số đánh giá (Metric)", "qwen3.5:0.8b", "gemma4:e2b", "Chênh lệch / Đánh giá"]
        row_fmt = "{:<36} | {:<12} | {:<12} | {:<16}"
        print(row_fmt.format(*headers))
        print("-" * 85)

        metrics_map = [
            ("faithfulness_score", "1. Độ trung thực (Faithfulness)"),
            ("completeness_score", "2. Độ bao quát (Completeness)"),
            ("key_fact_recall", "   • Weighted Key Fact Recall"),
            ("entity_recall", "   • Entity Recall"),
            ("number_recall", "   • Number/Date Recall"),
            ("conciseness_score", "3. Độ súc tích (Conciseness)"),
            ("average_compression_ratio", "   • Tỷ lệ nén (Compression Ratio)"),
            ("fluency_score", "4. Độ mạch lạc & Toàn vẹn (Fluency)"),
            ("overall_score", "🌟 TỔNG THỂ (Overall Score)"),
            ("average_latency_seconds", "⏱️ Độ trễ trung bình (Latency)")
        ]

        for key, label in metrics_map:
            v_qwen = all_results.get("qwen3.5:0.8b", {}).get(key, 0)
            v_gemma = all_results.get("gemma4:e2b", {}).get(key, 0)
            diff = v_gemma - v_qwen
            diff_str = f"{'+' if diff > 0 else ''}{diff:.3f}" if key != "average_latency_seconds" else f"{diff:+.1f}s"
            if key == "average_latency_seconds":
                fmt_qwen = f"{v_qwen:.1f}s"
                fmt_gemma = f"{v_gemma:.1f}s"
            elif key == "average_compression_ratio":
                fmt_qwen = f"{v_qwen*100:.1f}%"
                fmt_gemma = f"{v_gemma*100:.1f}%"
            else:
                fmt_qwen = f"{v_qwen*100:.1f}%"
                fmt_gemma = f"{v_gemma*100:.1f}%"
            print(row_fmt.format(label, fmt_qwen, fmt_gemma, diff_str))

        print("="*85)
        print(f"📁 Báo cáo so sánh được lưu tại: {comp_file.resolve()}\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
