"""
Compute LIS (Language Interference Score) matrix from Phase 2 evaluation results.

Methodological design:
  - LIS matrix uses SIB-200 + Belebele for ALL 7 languages (including en).
    This ensures cross-language comparability: LIS(en→sw) and LIS(sw→en)
    are measured on the same task scale.
  - MMLU / HellaSwag / ARC / TruthfulQA are reported SEPARATELY as
    "English reasoning retention" metrics and are NOT included in the LIS matrix.
  - FLORES is a translation-quality auxiliary metric and is NOT in the LIS matrix.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.lis_calculator import compute_lis, build_lis_matrix

LANGUAGES = ["en", "yo", "so", "ha"]
MODELS = ["Qwen3.5-9B-Base"]

# Parallel multilingual benchmarks used for LIS matrix (same task for all langs)
LIS_METRIC_KEYS = ["sib200", "belebele"]

# English-specific benchmarks reported separately (NOT in LIS matrix)
EN_RETENTION_KEYS = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc1"]


def aggregate_lis_score(scores_dict: dict, lang: str) -> float:
    """Compute LIS-matrix score for a language using parallel benchmarks only.

    Uses SIB-200 and Belebele for ALL languages including English, so that
    LIS(A→B) and LIS(B→A) are on the same measurement scale.
    """
    ml = scores_dict.get("multilingual", {})
    vals = [ml.get(metric, {}).get(lang, 0.0) for metric in LIS_METRIC_KEYS]
    vals = [v for v in vals if v > 0]
    return float(np.mean(vals)) if vals else 0.0


def english_retention_score(scores_dict: dict) -> dict:
    """Return English reasoning benchmark scores (MMLU/HellaSwag/ARC/TruthfulQA).
    Reported as a separate table, NOT used in LIS matrix computation.
    """
    en = scores_dict.get("english", {})
    return {k: en.get(k, None) for k in EN_RETENTION_KEYS}


def load_eval_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/phase2_v2")
    parser.add_argument("--baseline_dir", default="results/phase2_v2")
    parser.add_argument("--output_dir", default="results/phase2_v2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_short in MODELS:
        print(f"\n=== Computing LIS matrix for {model_short} ===")

        # Load baseline scores
        baseline_path = os.path.join(args.baseline_dir, f"{model_short}_baseline.json")
        if not os.path.exists(baseline_path):
            print(f"Baseline not found: {baseline_path}. Skipping.")
            continue
        baseline_data = load_eval_json(baseline_path)
        baseline_scores = {
            lang: aggregate_lis_score(baseline_data["scores"], lang)
            for lang in LANGUAGES
        }
        print("Baseline LIS scores (SIB-200 + Belebele avg):")
        for lang, score in baseline_scores.items():
            print(f"  {lang}: {score:.4f}")

        # Load per-language fine-tuned scores
        finetuned_scores = {}
        en_retention_table = {}

        for train_lang in LANGUAGES:
            eval_path = os.path.join(
                args.results_dir,
                f"lis_{model_short}_train_{train_lang}_eval.json"
            )
            if not os.path.exists(eval_path):
                print(f"Missing eval: {eval_path}")
                continue
            eval_data = load_eval_json(eval_path)

            finetuned_scores[train_lang] = {
                lang: aggregate_lis_score(eval_data["scores"], lang)
                for lang in LANGUAGES
            }
            # Collect English retention for separate reporting
            en_retention_table[train_lang] = english_retention_score(eval_data["scores"])

        if not finetuned_scores:
            print(f"No fine-tuned results found for {model_short}.")
            continue

        # Build 7×7 LIS matrix (rows=train_lang, cols=eval_lang)
        lis_matrix = build_lis_matrix(baseline_scores, finetuned_scores, LANGUAGES)
        print("\nLIS matrix (rows=train_lang, cols=eval_lang):")
        print(lis_matrix.round(4))

        csv_path = os.path.join(args.output_dir, f"lis_matrix_{model_short}.csv")
        lis_matrix.to_csv(csv_path)
        print(f"LIS matrix saved to {csv_path}")

        # Per-benchmark LIS matrices for robustness check
        for metric in LIS_METRIC_KEYS:
            baseline_metric = {
                lang: baseline_data["scores"].get("multilingual", {}).get(metric, {}).get(lang, 0.0)
                for lang in LANGUAGES
            }
            finetuned_metric = {}
            for train_lang in finetuned_scores:
                eval_path = os.path.join(
                    args.results_dir,
                    f"lis_{model_short}_train_{train_lang}_eval.json"
                )
                eval_data = load_eval_json(eval_path)
                finetuned_metric[train_lang] = {
                    lang: eval_data["scores"].get("multilingual", {}).get(metric, {}).get(lang, 0.0)
                    for lang in LANGUAGES
                }
            lis_m = build_lis_matrix(baseline_metric, finetuned_metric, LANGUAGES)
            m_csv = os.path.join(args.output_dir, f"lis_matrix_{model_short}_{metric}.csv")
            lis_m.to_csv(m_csv)
            print(f"Per-metric LIS ({metric}) saved to {m_csv}")

        # English retention table
        if en_retention_table:
            retention_df = pd.DataFrame(en_retention_table).T
            retention_df.index.name = "train_lang"
            retention_csv = os.path.join(args.output_dir, f"en_retention_{model_short}.csv")
            retention_df.to_csv(retention_csv)
            print(f"\nEnglish retention (MMLU/HellaSwag/ARC/TruthfulQA) saved to {retention_csv}")
            print(retention_df.round(4))

    print("\nDone.")


if __name__ == "__main__":
    main()
