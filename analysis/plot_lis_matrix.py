"""
Plot LIS matrix heatmaps for both models.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_lis_matrix(lis_df: pd.DataFrame, model_name: str, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        lis_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-0.3,
        vmax=0.1,
        ax=ax,
        annot_kws={"size": 11},
        linewidths=0.5,
    )

    ax.set_xlabel("Evaluated Language (受影响语言)", fontsize=13)
    ax.set_ylabel("Training Language (SFT 使用语言)", fontsize=13)
    ax.set_title(f"Language Interference Score Matrix\n({model_name})", fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/phase2_lis_matrix")
    parser.add_argument("--output_dir", default="paper_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    MODELS = ["Qwen3.5-9B-Base"]

    for model_short in MODELS:
        csv_path = os.path.join(args.results_dir, f"lis_matrix_{model_short}.csv")
        if not os.path.exists(csv_path):
            print(f"Matrix CSV not found: {csv_path}. Run compute_lis.py first.")
            continue

        lis_df = pd.read_csv(csv_path, index_col=0)
        out_path = os.path.join(args.output_dir, f"fig_lis_heatmap_{model_short}.pdf")
        plot_lis_matrix(lis_df, model_short, out_path)


if __name__ == "__main__":
    main()
