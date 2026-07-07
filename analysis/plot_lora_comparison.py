"""
Plot LoRA strategy comparison bar charts for Phase 4.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS_ORDER = ["No-FT", "Full-FT", "Standard-LoRA", "Mixed-LoRA", "Isolated-LoRA"]
METHOD_MAP = {
    "no_ft": "No-FT", "full_ft": "Full-FT",
    "standard_lora": "Standard-LoRA", "mixed_lora": "Mixed-LoRA",
    "isolated_lora": "Isolated-LoRA",
}
COLORS = ["#7f7f7f", "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def plot_lora_comparison(results_df: pd.DataFrame, target_lang: str, model_name: str, save_path: str):
    methods_present = [m for m in METHODS_ORDER if m in results_df["method"].values]
    colors = [COLORS[METHODS_ORDER.index(m)] for m in methods_present]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: English capability retention
    en_scores = []
    for method in methods_present:
        sub = results_df[(results_df["method"] == method) & (results_df["train_lang"] == target_lang)]
        en_scores.append(sub["english_score"].mean() if not sub.empty else 0.0)

    bars0 = axes[0].bar(methods_present, en_scores, color=colors, edgecolor="black", linewidth=0.5)
    no_ft_en = results_df[results_df["method"] == "No-FT"]["english_score"].mean()
    axes[0].axhline(y=no_ft_en, color="red", linestyle="--", linewidth=1.5, label="Zero-shot baseline")
    axes[0].set_title("English Capability Retention", fontsize=13)
    axes[0].set_ylabel("English Benchmark Score (avg MMLU/HellaSwag/ARC)", fontsize=10)
    axes[0].set_ylim(bottom=max(0, no_ft_en - 0.15))
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=15)
    _add_value_labels(axes[0], bars0)

    # Right: Target language improvement
    tgt_scores = []
    for method in methods_present:
        sub = results_df[(results_df["method"] == method) & (results_df["train_lang"] == target_lang)]
        tgt_scores.append(sub["target_score"].mean() if not sub.empty else 0.0)

    bars1 = axes[1].bar(methods_present, tgt_scores, color=colors, edgecolor="black", linewidth=0.5)
    no_ft_tgt = results_df[results_df["method"] == "No-FT"]["target_score"].mean()
    axes[1].axhline(y=no_ft_tgt, color="red", linestyle="--", linewidth=1.5, label="Zero-shot baseline")
    axes[1].set_title(f"{target_lang} Capability Gain", fontsize=13)
    axes[1].set_ylabel(f"{target_lang} Benchmark Score (avg SIB-200/Belebele)", fontsize=10)
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=15)
    _add_value_labels(axes[1], bars1)

    plt.suptitle(f"LoRA Strategy Comparison: {target_lang} ({model_name})", fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )


def load_phase4_results(results_dir: str, baseline_dir: str) -> pd.DataFrame:
    rows = []

    # Load baselines (No-FT)
    for model_short in ["Qwen3.5-9B-Base"]:
        baseline_path = os.path.join(baseline_dir, f"{model_short}_baseline.json")
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                data = json.load(f)
            en_scores = data["scores"].get("english", {})
            ml_scores = data["scores"].get("multilingual", {})
            for lang in ["sw", "zh"]:
                en_avg = (en_scores.get("mmlu", 0) + en_scores.get("hellaswag", 0) + en_scores.get("arc_challenge", 0)) / 3
                sib = ml_scores.get("sib200", {}).get(lang, 0.0)
                bel = ml_scores.get("belebele", {}).get(lang, 0.0)
                rows.append({
                    "model": model_short, "method": "No-FT", "train_lang": lang,
                    "english_score": en_avg, "target_score": (sib + bel) / 2,
                })

    # Load Phase 4 results
    for fname in os.listdir(results_dir):
        if not fname.endswith("_eval.json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)

        # Filename: lora_{model}_{method}_{lang}_eval.json
        base = fname.replace("_eval.json", "")
        parts = base.split("_")
        if len(parts) < 4:
            continue

        model = parts[1]
        raw_method = parts[2]
        lang = parts[3]
        method = METHOD_MAP.get(raw_method, raw_method)

        en_scores = data["scores"].get("english", {})
        ml_scores = data["scores"].get("multilingual", {})
        en_avg = (en_scores.get("mmlu", 0) + en_scores.get("hellaswag", 0) + en_scores.get("arc_challenge", 0)) / 3
        sib = ml_scores.get("sib200", {}).get(lang, 0.0)
        bel = ml_scores.get("belebele", {}).get(lang, 0.0)

        rows.append({
            "model": model, "method": method, "train_lang": lang,
            "english_score": en_avg, "target_score": (sib + bel) / 2,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/phase4_lora_isolation")
    parser.add_argument("--baseline_dir", default="results/phase1_baseline")
    parser.add_argument("--output_dir", default="paper_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_phase4_results(args.results_dir, args.baseline_dir)

    if df.empty:
        print("No Phase 4 results found.")
        return

    for model in df["model"].unique():
        for lang in ["sw", "zh"]:
            model_df = df[df["model"] == model]
            out_path = os.path.join(args.output_dir, f"fig_lora_comparison_{lang}_{model}.pdf")
            plot_lora_comparison(model_df, lang, model, out_path)


if __name__ == "__main__":
    main()
