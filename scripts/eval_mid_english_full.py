#!/usr/bin/env python3
"""
Run full English eval (MMLU / HellaSwag / ARC-Challenge) on existing mid_yo/so/ha models.
TruthfulQA already in eval JSONs — this adds the three missing tasks and recomputes en_avg.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_mid_english_full.py [--langs yo,so,ha]
"""

import argparse, json, os, sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.english_eval import ENGLISH_TASKS, run_english_eval

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SHORT = "Qwen3.5-9B-Base"
NEW_TASKS   = ["mmlu", "hellaswag", "arc_challenge"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", default="yo,so,ha", help="Comma-separated target languages")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args   = parse_args()
    langs  = [l.strip() for l in args.langs.split(",")]

    for lang in langs:
        model_path = os.path.join(BASE_DIR, f"results/mid/mid_{MODEL_SHORT}_{lang}")
        eval_path  = os.path.join(BASE_DIR, f"results/mid/mid_{MODEL_SHORT}_{lang}_eval.json")

        if not os.path.isdir(model_path):
            print(f"[{lang}] model dir not found: {model_path} — skipping")
            continue

        print(f"\n[{lang.upper()}] Running {NEW_TASKS} ...")
        scores = run_english_eval(model_path, tasks=NEW_TASKS, batch_size=args.batch_size)

        with open(eval_path) as f:
            eval_data = json.load(f)

        eng = eval_data.setdefault("scores", {}).setdefault("english", {})
        for k, v in scores.items():
            if k != "english_avg":
                eng[k] = v

        # Recompute english_avg across all four tasks (include pre-existing TruthfulQA)
        present = [t for t in ENGLISH_TASKS if t in eng]
        eng["english_avg"] = sum(eng[t] for t in present) / len(present) if present else 0.0

        with open(eval_path, "w") as f:
            json.dump(eval_data, f, indent=2)

        print(f"[{lang.upper()}] Updated English scores:")
        for t in ENGLISH_TASKS:
            v = eng.get(t)
            print(f"  {t:25s} = {v:.4f}" if v is not None else f"  {t:25s} = N/A")
        print(f"  {'english_avg':25s} = {eng['english_avg']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
