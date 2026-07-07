"""
Aggregate all experiment results into paper-ready CSV tables and trigger all plot scripts.
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd


def collect_all_results(results_dir: str) -> dict:
    """Walk results directory and collect all eval JSONs into phase-separated dicts."""
    phases = {
        "phase1": [], "phase2": [], "phase3": [], "phase4": []
    }

    for phase_dir, phase_key in [
        ("phase1_baseline", "phase1"),
        ("phase2_lis_matrix", "phase2"),
        ("phase3_data_mixture", "phase3"),
        ("phase4_lora_isolation", "phase4"),
    ]:
        full_dir = os.path.join(results_dir, phase_dir)
        if not os.path.isdir(full_dir):
            continue
        for fname in os.listdir(full_dir):
            if fname.endswith("_eval.json") or fname.endswith("_baseline.json"):
                with open(os.path.join(full_dir, fname)) as f:
                    phases[phase_key].append((fname, json.load(f)))

    return phases


def build_baseline_table(phase1_data: list) -> pd.DataFrame:
    rows = []
    for fname, data in phase1_data:
        model = fname.replace("_baseline.json", "")
        en = data["scores"].get("english", {})
        ml = data["scores"].get("multilingual", {})
        row = {"model": model}
        row.update({f"en_{k}": v for k, v in en.items()})
        for bench, lang_scores in ml.items():
            if isinstance(lang_scores, dict):
                for lang, score in lang_scores.items():
                    if isinstance(score, (int, float)):
                        row[f"{bench}_{lang}"] = score
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="paper_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    phases = collect_all_results(args.results_dir)

    # Table 1: Baseline scores
    if phases["phase1"]:
        baseline_df = build_baseline_table(phases["phase1"])
        baseline_df.to_csv(os.path.join(args.output_dir, "table1_baseline.csv"), index=False)
        print(f"Saved table1_baseline.csv ({len(baseline_df)} rows)")

    # Run plot scripts
    scripts = [
        ["python", "analysis/plot_lis_matrix.py", "--output_dir", args.output_dir],
        ["python", "analysis/plot_pareto_curve.py", "--output_dir", args.output_dir],
        ["python", "analysis/plot_lora_comparison.py", "--output_dir", args.output_dir],
    ]

    for cmd in scripts:
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"WARNING: {cmd[1]} failed:\n{result.stderr}")
        else:
            print(result.stdout)

    print(f"\nAll outputs written to {args.output_dir}/")


if __name__ == "__main__":
    main()
