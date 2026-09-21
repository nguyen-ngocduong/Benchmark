"""
Meta-Evaluation Module for Benchmark Judge Validation.

Computes statistical correlation between Automated Judge and Human Annotators:
- Spearman rank correlation (rho)
- Kendall rank correlation (tau)
- Pearson linear correlation (r)
- Cohen's Kappa for binary/categorical classification agreement

Usage:
    python -m eval.tool_call.scripts.run_meta_eval --human-file path/to/human_labels.json --judge-file path/to/judge_labels.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def compute_correlations(x: List[float], y: List[float]) -> dict:
    """Computes Spearman rho, Kendall tau, and Pearson r between two score vectors."""
    if len(x) != len(y) or len(x) < 2:
        return {"error": "Insufficient or mismatched data points"}

    spearman_res = stats.spearmanr(x, y)
    kendall_res = stats.kendalltau(x, y)
    pearson_res = stats.pearsonr(x, y)

    return {
        "sample_size": len(x),
        "spearman_rho": round(float(spearman_res.statistic), 4),
        "spearman_pvalue": round(float(spearman_res.pvalue), 6),
        "kendall_tau": round(float(kendall_res.statistic), 4),
        "kendall_pvalue": round(float(kendall_res.pvalue), 6),
        "pearson_r": round(float(pearson_res.statistic), 4),
        "pearson_pvalue": round(float(pearson_res.pvalue), 6),
    }


def compute_binary_agreement(human_labels: List[int], judge_labels: List[int]) -> dict:
    """Computes Cohen's Kappa and Raw Agreement for binary claim labels (Supported vs Unsupported)."""
    if len(human_labels) != len(judge_labels) or len(human_labels) == 0:
        return {"error": "Invalid label lists"}

    kappa = cohen_kappa_score(human_labels, judge_labels)
    raw_acc = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / len(human_labels)

    return {
        "cohens_kappa": round(float(kappa), 4),
        "raw_agreement": round(float(raw_acc), 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Meta-Evaluation Metrics between Judge and Human")
    parser.add_argument("--human-scores", nargs="+", type=float, default=[4.5, 3.0, 5.0, 2.0, 4.0])
    parser.add_argument("--judge-scores", nargs="+", type=float, default=[4.0, 3.0, 5.0, 1.5, 4.2])
    args = parser.parse_args()

    results = compute_correlations(args.human_scores, args.judge_scores)
    print("\n" + "="*60)
    print("🔬 META-EVALUATION: AUTOMATED JUDGE VS HUMAN AGREEMENT")
    print("="*60)
    print(f"Sample Size (N): {results['sample_size']}")
    print(f"Spearman's ρ:    {results['spearman_rho']} (p = {results['spearman_pvalue']})")
    print(f"Kendall's  τ:    {results['kendall_tau']} (p = {results['kendall_pvalue']})")
    print(f"Pearson's  r:    {results['pearson_r']} (p = {results['pearson_pvalue']})")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
