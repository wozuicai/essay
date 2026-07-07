"""
LIS (Language Interference Score) calculator.

LIS(train_lang → test_lang) = (score_after - score_before) / max(score_before, 0.01)
  negative = negative transfer (training on A harms language B performance)
  positive = positive transfer
"""

import numpy as np
import pandas as pd


def compute_lis(score_after: float, score_before: float) -> float:
    """
    Compute Language Interference Score for a single (train, test) pair.
    Denominator clamped to 0.01 to prevent explosion when baseline near 0.
    """
    return (score_after - score_before) / max(score_before, 0.01)


def build_lis_matrix(
    baseline_scores: dict,
    finetuned_scores: dict,
    languages: list[str],
) -> pd.DataFrame:
    """
    Build N×N LIS matrix.

    Args:
        baseline_scores: {lang: score} — zero-shot baseline scores
        finetuned_scores: {train_lang: {eval_lang: score}} — scores after SFT
        languages: ordered list of language codes

    Returns:
        DataFrame with rows=training language, columns=evaluated language
    """
    n = len(languages)
    matrix = np.zeros((n, n))

    for i, train_lang in enumerate(languages):
        if train_lang not in finetuned_scores:
            continue
        for j, eval_lang in enumerate(languages):
            before = baseline_scores.get(eval_lang, 0.0)
            after = finetuned_scores[train_lang].get(eval_lang, 0.0)
            matrix[i][j] = compute_lis(after, before)

    return pd.DataFrame(matrix, index=languages, columns=languages)


def compute_asymmetry_score(lis_matrix: pd.DataFrame, lang_a: str, lang_b: str) -> dict:
    """
    Compute asymmetry between A→B and B→A interference.
    Returns the two LIS values and their difference.
    """
    a_to_b = lis_matrix.loc[lang_a, lang_b]
    b_to_a = lis_matrix.loc[lang_b, lang_a]
    return {
        f"{lang_a}_→_{lang_b}": a_to_b,
        f"{lang_b}_→_{lang_a}": b_to_a,
        "asymmetry": abs(a_to_b - b_to_a),
    }
