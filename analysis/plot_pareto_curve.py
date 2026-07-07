"""
Plot Pareto curves for Phase 3 data mixture experiments.
Identifies Pareto-optimal English/target-language tradeoff points.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-efficient points (maximize both dimensions)."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
            is_efficient[i] = True
    return is_efficient


def plot_pareto_curve(results_df: pd.DataFrame, target_lang: str, model_name: str, save_path: str):
    lang_results = results_df[results_df["train_lang"] == target_lang].copy()

    if lang_results.empty:
        print(f"No results for lang={target_lang}, model={model_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        lang_results["target_score"],
        lang_results["english_score"],
        c=lang_results["english_ratio"],
        cmap="viridis",
        s=120,
        zorder=5,
        edgecolors="black",
        linewidth=0.5,
    )

    for _, row in lang_results.iterrows():
        ax.annotate(
            f"{row['english_ratio']:.0%}",
            (row["target_score"], row["english_score"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    # Mark Pareto frontier
    pts = lang_results[["target_score", "english_score"]].values
    pareto_mask = is_pareto_efficient(pts)
    pareto_pts = lang_results[pareto_mask].sort_values("target_score")
    ax.plot(
        pareto_pts["target_score"],
        pareto_pts["english_score"],
        "r--",
        linewidth=1.5,
        label="Pareto frontier",
        zorder=4,
    )

    ax.set_xlabel(f"{target_lang} Benchmark Score", fontsize=12)
    ax.set_ylabel("English Benchmark Score", fontsize=12)
    ax.set_title(f"Pareto Curve: English vs {target_lang}\n({model_name})", fontsize=13)
    plt.colorbar(scatter, label="English Data Ratio")
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def load_phase3_results(results_dir: str) -> pd.DataFrame:
    """Load all Phase 3 eval JSONs into a single DataFrame."""
    rows = []
    for fname in os.listdir(results_dir):
        if not fname.endswith("_eval.json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)

        # Parse filename: mixture_{model}_{lang}_en{ratio}_eval.json
        parts = fname.replace("_eval.json", "").split("_")
        if len(parts) < 4:
            continue

        model = parts[1]
        lang = parts[2]
        ratio = float(parts[3].replace("en", ""))

        en_scores = data["scores"].get("english", {})
        ml_scores = data["scores"].get("multilingual", {})

        english_score = (
            en_scores.get("mmlu", 0) + en_scores.get("hellaswag", 0) + en_scores.get("arc_challenge", 0)
        ) / 3 if en_scores else 0.0

        sib = ml_scores.get("sib200", {}).get(lang, 0.0)
        bel = ml_scores.get("belebele", {}).get(lang, 0.0)
        target_score = (sib + bel) / 2 if (sib or bel) else 0.0

        rows.append({
            "model": model, "train_lang": lang, "english_ratio": ratio,
            "english_score": english_score, "target_score": target_score,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/phase3_data_mixture")
    parser.add_argument("--output_dir", default="paper_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_phase3_results(args.results_dir)

    if df.empty:
        print("No Phase 3 results found.")
        return

    for model in df["model"].unique():
        for lang in df["train_lang"].unique():
            model_df = df[df["model"] == model]
            out_path = os.path.join(args.output_dir, f"fig_pareto_{lang}_{model}.pdf")
            plot_pareto_curve(model_df, lang, model, out_path)

    # Print optimal ratios
    print("\n=== Suggested optimal ratios (max harmonic mean of english + target scores) ===")
    for model in df["model"].unique():
        for lang in df["train_lang"].unique():
            sub = df[(df["model"] == model) & (df["train_lang"] == lang)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["harmonic"] = 2 * sub["english_score"] * sub["target_score"] / (
                sub["english_score"] + sub["target_score"] + 1e-9
            )
            best = sub.loc[sub["harmonic"].idxmax()]
            print(f"  {model} / {lang}: best_ratio={best['english_ratio']:.1%}, "
                  f"en={best['english_score']:.3f}, tgt={best['target_score']:.3f}")


if __name__ == "__main__":
    main()
