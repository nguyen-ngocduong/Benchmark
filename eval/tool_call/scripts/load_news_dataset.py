"""
Load and Preprocess Vietnamese News Dataset Samples for Benchmark Pipelines.

Supports:
- Remote HuggingFace datasets (e.g. ICCIES-2025-DetectAI/vietnamese_news_human_ai)
- Local files (.json, .jsonl, .csv)
- Dynamic and balanced sampling (Human vs AI) with reproducible random seeds
- Text length and quality filtering (min_words, max_words)
- Incremental append mode (--append) with auto-incrementing sample IDs (sample_011, sample_012...)
- Key entity and pattern extraction (dates, numbers, organizations)

Usage:
    # 1. Load default 10 balanced samples:
    python -m eval.tool_call.scripts.load_news_dataset --n 10

    # 2. Append 10 more samples to existing dataset:
    python -m eval.tool_call.scripts.load_news_dataset --n 10 --append

    # 3. Load from a local file with word count filters:
    python -m eval.tool_call.scripts.load_news_dataset --input-file raw_data.jsonl --n 20 --min-words 150 --max-words 1500
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_HF_DATASET = "ICCIES-2025-DetectAI/vietnamese_news_human_ai"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "summarization"


def extract_key_info_patterns(text: str) -> List[str]:
    """
    Extract potential key information patterns from Vietnamese text.
    Uses non-capturing groups (?:...) to ensure re.findall returns the full matched string.
    """
    if not text:
        return []

    patterns = {
        "dates": r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
        "numbers_with_units": r'\b\d+[\s-]*(?:tỷ|triệu|nghìn|ngàn|trăm)\b|\b\d+(?:[\.,]\d+)?%',
        "organizations": r'(?:TP\.|Thành phố|Tỉnh|Bộ|Ngân hàng|Công ty|Tập đoàn|Ủy ban)\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][a-zàáâãèéêìíòóôõùúăđĩũơư]+(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][a-zàáâãèéêìíòóôõùúăđĩũơư]+)*',
    }

    found: List[str] = []
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            found.extend(matches[:4])

    seen = set()
    unique_found = []
    for item in found:
        cleaned = item.strip()
        if cleaned.lower() not in seen and len(cleaned) > 1:
            seen.add(cleaned.lower())
            unique_found.append(cleaned)

    return unique_found[:15]


def load_raw_data(
    input_file: Optional[str] = None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """Loads raw articles from a local file or HuggingFace dataset."""
    if input_file:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"📂 Loading raw samples from local file: {input_path}")
        records: List[Dict[str, Any]] = []

        if input_path.suffix == ".jsonl":
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        elif input_path.suffix == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
        elif input_path.suffix in [".csv", ".tsv"]:
            delimiter = "\t" if input_path.suffix == ".tsv" else ","
            with open(input_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                records = list(reader)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .json, .jsonl, or .csv")

        # Normalize field names to standard 'Text' and 'Label'
        normalized = []
        for r in records:
            text = r.get("Text") or r.get("text") or r.get("content") or r.get("article") or ""
            label_val = r.get("Label") if "Label" in r else r.get("label", 0)
            try:
                label_val = int(label_val)
            except (ValueError, TypeError):
                label_val = 0
            if text:
                normalized.append({"Text": text, "Label": label_val})
        return normalized

    # Otherwise load from HuggingFace
    print(f"🌐 Loading dataset from HuggingFace: {hf_dataset} (split='{split}')...")
    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    print(f"   Downloaded {len(ds)} total rows.")
    return [{"Text": row["Text"], "Label": int(row["Label"])} for row in ds if "Text" in row and row["Text"]]


def filter_articles(
    articles: List[Dict[str, Any]],
    min_words: int = 100,
    max_words: int = 2500
) -> List[Dict[str, Any]]:
    """Filters articles based on word length and text validity."""
    filtered = []
    for a in articles:
        text = a.get("Text", "").strip()
        word_count = len(text.split())
        if min_words <= word_count <= max_words:
            filtered.append(a)
    return filtered


def sample_records(
    articles: List[Dict[str, Any]],
    n_samples: int,
    balance: bool = True,
    seed: int = 42,
    existing_texts: Optional[set] = None
) -> List[Dict[str, Any]]:
    """Samples n_samples with reproducible seed, balanced labels, and deduplication against existing texts."""
    rng = random.Random(seed)
    existing_texts = existing_texts or set()

    # Deduplicate against existing dataset
    candidates = [
        a for a in articles 
        if a["Text"].strip() not in existing_texts and len(a["Text"].strip()) > 0
    ]

    if not candidates:
        print("⚠️ No new unique candidates available after deduplication.")
        return []

    if not balance:
        rng.shuffle(candidates)
        return candidates[:n_samples]

    human_pool = [a for a in candidates if a.get("Label") == 0]
    ai_pool = [a for a in candidates if a.get("Label") == 1]

    rng.shuffle(human_pool)
    rng.shuffle(ai_pool)

    n_human = n_samples // 2
    n_ai = n_samples - n_human

    # If one pool doesn't have enough samples, backfill from the other
    selected_human = human_pool[:n_human]
    selected_ai = ai_pool[:n_ai]

    if len(selected_human) < n_human:
        needed = n_human - len(selected_human)
        selected_ai.extend(ai_pool[n_ai:n_ai + needed])

    if len(selected_ai) < n_ai:
        needed = n_ai - len(selected_ai)
        selected_human.extend(human_pool[n_human:n_human + needed])

    combined = selected_human + selected_ai
    # Keep predictable ordering (human first, AI second)
    return combined


def get_max_sample_id_index(samples: List[Dict[str, Any]]) -> int:
    """Finds the maximum integer index in existing sample IDs (e.g. sample_010 -> 10)."""
    max_idx = 0
    for s in samples:
        s_id = str(s.get("id") or s.get("sample_id") or "")
        match = re.search(r'sample_(\d+)', s_id)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx


def get_next_sample_id_index(existing_samples: List[Dict[str, Any]]) -> int:
    """Finds the next index for sample IDs (e.g. sample_010 -> 11)."""
    return get_max_sample_id_index(existing_samples) + 1


def load_news_samples(
    n_samples: int = 10,
    balance: bool = True,
    seed: int = 42,
    min_words: int = 100,
    max_words: int = 2500,
    input_file: Optional[str] = None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    output_dir: Optional[str] = None,
    append: bool = False,
    force: bool = False
) -> List[Dict[str, Any]]:
    """
    Main entry point to load, filter, sample, and structure news articles for the benchmark.
    Saves newly loaded articles into samples_new.json while continuing IDs from samples.json.
    """
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "samples_new.json"
    base_file = out_dir / "samples.json"

    # 1. Read base reference dataset (samples.json) to preserve continuity and avoid duplicates
    existing_texts = set()
    base_max_idx = 0
    if base_file.exists():
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                base_samples = json.load(f)
                for item in base_samples:
                    txt = (item.get("text") or item.get("article") or "").strip()
                    if txt:
                        existing_texts.add(txt)
                base_max_idx = get_max_sample_id_index(base_samples)
            print(f"📖 Reference base dataset: {len(base_samples)} samples in {base_file} (Max ID: sample_{base_max_idx:03d}).")
        except Exception as e:
            print(f"⚠️ Could not read reference base file {base_file}: {e}")

    # 2. Check existing samples in output file (samples_new.json)
    existing_samples: List[Dict[str, Any]] = []
    new_max_idx = 0

    if out_file.exists() and not force:
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_samples = json.loads(content)
                    for item in existing_samples:
                        txt = (item.get("text") or item.get("article") or "").strip()
                        if txt:
                            existing_texts.add(txt)
                    new_max_idx = get_max_sample_id_index(existing_samples)
            if existing_samples:
                print(f"📋 Found {len(existing_samples)} existing samples in {out_file} (Max ID: sample_{new_max_idx:03d}).")
        except Exception as e:
            print(f"⚠️ Could not read existing {out_file}: {e}")
            existing_samples = []

    # If not in append mode and not force, but output exists and has samples, warn user
    if out_file.exists() and not append and not force and existing_samples:
        print(f"ℹ️ Output file {out_file} already exists ({len(existing_samples)} samples). Use --append to add more, or --force to overwrite/create fresh starting after samples.json.")
        return existing_samples

    # Load raw records
    raw_articles = load_raw_data(input_file=input_file, hf_dataset=hf_dataset)
    print(f"Total raw articles loaded: {len(raw_articles)}")

    # Filter quality
    valid_articles = filter_articles(raw_articles, min_words=min_words, max_words=max_words)
    print(f"Articles satisfying word bounds [{min_words}, {max_words}]: {len(valid_articles)}")

    # Sample - ALWAYS pass existing_texts to ensure no duplicates from samples.json or samples_new.json
    sampled = sample_records(
        articles=valid_articles,
        n_samples=n_samples,
        balance=balance,
        seed=seed,
        existing_texts=existing_texts
    )

    if not sampled:
        print("❌ No samples selected.")
        return existing_samples

    # Determine starting index for sample IDs:
    # If append: continue after the maximum ID among base_samples and existing samples_new
    # If fresh / overwrite: start right after base_max_idx (from samples.json)
    if append and existing_samples:
        start_idx = max(base_max_idx, new_max_idx) + 1
    else:
        start_idx = base_max_idx + 1

    new_processed: List[Dict[str, Any]] = []
    for i, s in enumerate(sampled):
        current_id = f"sample_{start_idx + i:03d}"
        txt = s["Text"].strip()
        word_count = len(txt.split())
        source_type = "human" if s["Label"] == 0 else "ai"

        record = {
            "id": current_id,
            "source": source_type,
            "label": s["Label"],
            "text": txt,
            "article": txt,                         # backward compatibility
            "word_count": word_count,
            "article_word_count": word_count,       # backward compatibility
            "expected_key_info": extract_key_info_patterns(txt)
        }
        new_processed.append(record)

    final_dataset = (existing_samples + new_processed) if append else new_processed

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Successfully saved {len(final_dataset)} total samples to {out_file}")
    print(f"   • Newly added: {len(new_processed)} samples (from {new_processed[0]['id']} to {new_processed[-1]['id']})")
    print(f"   • Total Human (label=0): {sum(1 for x in final_dataset if x.get('label') == 0)}")
    print(f"   • Total AI (label=1):    {sum(1 for x in final_dataset if x.get('label') == 1)}")

    return final_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and preprocess news articles for the evaluation benchmark"
    )
    parser.add_argument("--n", type=int, default=10, help="Number of samples to select (default: 10)")
    parser.add_argument("--balance", action="store_true", default=True, help="Balance 50/50 between human and AI (default: True)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--min-words", type=int, default=100, help="Minimum word count per article (default: 100)")
    parser.add_argument("--max-words", type=int, default=2500, help="Maximum word count per article (default: 2500)")
    parser.add_argument("--input-file", type=str, default=None, help="Path to local raw file (.json, .jsonl, .csv)")
    parser.add_argument("--hf-dataset", type=str, default=DEFAULT_HF_DATASET, help=f"HuggingFace dataset repo (default: {DEFAULT_HF_DATASET})")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output samples.json")
    parser.add_argument("--append", action="store_true", help="Append new samples to existing dataset without overwriting")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing dataset")

    args = parser.parse_args()

    load_news_samples(
        n_samples=args.n,
        balance=args.balance,
        seed=args.seed,
        min_words=args.min_words,
        max_words=args.max_words,
        input_file=args.input_file,
        hf_dataset=args.hf_dataset,
        output_dir=args.output_dir,
        append=args.append,
        force=args.force
    )
