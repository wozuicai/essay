"""
Unified evaluation entry point for all phases.
Covers English benchmarks (MMLU, HellaSwag, ARC, TruthfulQA),
multilingual benchmarks (FLORES, SIB-200, Belebele),
MT-Bench multilingual (Phase 3 & 4 only),
and LCB language confusion metrics.
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.english_eval import run_english_eval
from src.evaluation.multilingual_eval import run_multilingual_eval, compute_lcb_metrics
from src.evaluation.lis_calculator import compute_lis

ENGLISH_TASKS = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc1"]
MULTILINGUAL_LANGS = ["en", "yo", "so", "ha"]
MT_BENCH_LANGS = ["yo", "so", "ha"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tasks", default="all",
                        help="Comma-separated list or 'all', 'english_main', 'target_lang'")
    parser.add_argument("--languages", default=",".join(MULTILINGUAL_LANGS))
    parser.add_argument("--target_lang", default=None)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--include_mt_bench", action="store_true",
                        help="Include MT-Bench multilingual (Phase 3 & 4 only)")
    parser.add_argument("--include_lcb", action="store_true", default=False,
                        help="Include LCB language confusion metrics (requires prepared prompt files)")
    parser.add_argument("--skip_flores", action="store_true", default=False,
                        help="Skip FLORES translation eval (slow, generation-based). "
                             "Use for Phase 2 efficiency.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--inject_lang_tag", action="store_true", default=False,
                        help="Inject <|tgt_lang:xx|> into benchmark prompts (Phase 5 tag routing eval)")
    parser.add_argument("--en_tasks", default=None,
        help="逗号分隔的英文评测任务（默认全跑）。例：truthfulqa_mc1")
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages.split(",")
    results = {"model_path": args.model_path, "scores": {}}

    run_en = args.tasks in ("all", "english_main") or "mmlu" in args.tasks or args.en_tasks is not None
    run_ml = args.tasks in ("all", "target_lang") or "flores" in args.tasks

    if run_en:
        print("=== Running English evaluation ===")
        en_task_list = args.en_tasks.split(",") if args.en_tasks else None
        en_results = run_english_eval(args.model_path, tasks=en_task_list, batch_size=args.batch_size, inject_lang_tag=args.inject_lang_tag)
        results["scores"]["english"] = en_results
        print(f"English results: {en_results}")

    if run_ml:
        print("=== Running multilingual evaluation ===")
        ml_results = run_multilingual_eval(
            args.model_path, languages,
            run_flores=not args.skip_flores,
            run_sib200=True,
            run_belebele=True,
            batch_size=args.batch_size,
            inject_lang_tag=args.inject_lang_tag,
        )
        results["scores"]["multilingual"] = ml_results

    if args.include_mt_bench:
        print("=== Running MT-Bench multilingual evaluation ===")
        from src.evaluation.multilingual_eval import run_mt_bench_multilingual
        mt_results = run_mt_bench_multilingual(
            args.model_path, MT_BENCH_LANGS, judge_model="claude-sonnet-4-6"
        )
        results["scores"]["mt_bench"] = mt_results

    if args.include_lcb:
        print("=== Running LCB language confusion evaluation ===")
        eval_langs = [args.target_lang] if args.target_lang else languages
        lcb_results = {}
        for lang in eval_langs:
            lcb_results[lang] = compute_lcb_metrics(args.model_path, lang)
        results["scores"]["lcb"] = lcb_results

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved to {args.output}")
    return results


if __name__ == "__main__":
    main()
