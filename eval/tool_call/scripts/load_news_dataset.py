"""
Load samples from vietnamese_news_human_ai dataset
Dataset: https://huggingface.co/datasets/ICCIES-2025-DetectAI/vietnamese_news_human_ai

Usage:
    python scripts/load_news_dataset.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset


def load_news_samples(n_samples: int = 10, balance: bool = True, output_dir: str = None):
    """
    Load và sample articles từ HuggingFace dataset

    Args:
        n_samples: Total number of samples (default: 10)
        balance: If True, split equally between human (label=0) and AI (label=1)
        output_dir: Directory to save samples.json

    Returns:
        List of sample dictionaries
    """
    print(f"Loading samples from HuggingFace...")

    # Load toàn bộ dataset
    ds = load_dataset(
        "ICCIES-2025-DetectAI/vietnamese_news_human_ai",
        split="train"
    )
    print(f"Total dataset size: {len(ds)}")
    print(f"Dataset columns: {ds.column_names}")

    # Convert to list để access by index
    all_samples = list(ds)

    # Lấy 5 dòng đầu có Label=1 (AI)
    ai_samples = [row for row in all_samples if row["Label"] == 1][:5]

    # Lấy 5 dòng cuối có Label=0 (human)
    human_samples = [row for row in all_samples if row["Label"] == 0][-5:]

    print(f"Selected: {len(human_samples)} human (Label=0), {len(ai_samples)} AI (Label=1)")

    # Kết hợp: human trước, AI sau
    samples = human_samples + ai_samples

    # Add IDs and expected key info extraction
    processed_samples = []
    for i, sample in enumerate(samples):
        processed = {
            "id": f"sample_{i+1:03d}",
            "source": "human" if sample["Label"] == 0 else "ai",
            "label": sample["Label"],
            "text": sample["Text"],
            "word_count": len(sample["Text"].split()),
            "expected_key_info": extract_key_info_patterns(sample["Text"])
        }
        processed_samples.append(processed)

    # Save to JSON
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "datasets" / "summarization"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "samples.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(processed_samples)} samples to {output_file}")
    print(f"  - Human: {len([s for s in processed_samples if s['source'] == 'human'])}")
    print(f"  - AI: {len([s for s in processed_samples if s['source'] == 'ai'])}")

    return processed_samples


def extract_key_info_patterns(text: str) -> list:
    """
    Extract potential key information patterns from text.
    These are used as hints for the judge.
    """
    import re

    patterns = {
        "dates": r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}',
        "numbers": r'\d+[\.,]?\d*%?|\d+[\s-]*(tỷ|triệu|nghìn|trăm)',
        "organizations": r'(?:TP\.|Thành phố|Tỉnh|Bộ|Ngân hàng|Công ty)\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][a-zàáâãèéêìíòóôõùúăđĩũơư]+',
    }

    found = []
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            found.extend(matches[:3])

    seen = set()
    unique_found = []
    for item in found:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique_found.append(item)

    return unique_found[:10]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Vietnamese news dataset samples")
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--balance", action="store_true", default=True, help="Balance human/AI samples")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    samples = load_news_samples(
        n_samples=args.n,
        balance=args.balance,
        output_dir=args.output
    )

    # Show first sample as preview
    if samples:
        print("\n--- Sample Preview (first 500 chars) ---")
        print(samples[0]["text"][:500] + "...")
