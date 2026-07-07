#!/usr/bin/env python3
"""Run only the required eval suite.

Outputs:
- scores.english: mmlu, hellaswag, arc_challenge, truthfulqa_mc1, english_avg
- scores.multilingual.belebele: en, yo, so, ha
- scores.multilingual.irokobench: AfriMMLU MCQ + AfriXNLI + AfriMGSM
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.english_eval import run_english_eval
from src.evaluation.irokobench_eval import run_irokobench_eval
from src.evaluation.multilingual_eval import _load_model_and_tokenizer, run_multilingual_eval


def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_path", help="HF model dir or PEFT adapter dir")
    group.add_argument("--moe_dir", help="MoE-LoRA output dir with moe_config.json")
    p.add_argument("--output", required=True)
    p.add_argument("--languages", default="en,yo,so,ha")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="Batch size for generation-style eval such as AfriMGSM.",
    )
    p.add_argument("--inject_lang_tag", action="store_true")
    return p.parse_args()


def _load_moe_model(moe_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.models.moe_lora import freeze_base, load_moe, setup_moe_lora

    with open(os.path.join(moe_dir, "moe_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    base_path = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(moe_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    setup_moe_lora(
        model,
        cfg["target_modules"],
        cfg["r"],
        cfg["lora_alpha"],
        cfg["n_experts"],
        cfg.get("dropout_p", 0.05),
    )
    freeze_base(model)
    load_moe(model, moe_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, cfg


def main():
    args = parse_args()
    languages = [x for x in args.languages.split(",") if x]

    if args.moe_dir:
        model_path = args.moe_dir
        model, tokenizer, moe_cfg = _load_moe_model(args.moe_dir)
        extra = {"method": "MoE-LoRA", "moe_config": moe_cfg}
    else:
        model_path = args.model_path
        model, tokenizer = _load_model_and_tokenizer(args.model_path)
        extra = {}

    results = {"model_path": model_path, **extra, "scores": {}}

    print("=== English eval ===")
    results["scores"]["english"] = run_english_eval(
        model_path=model_path,
        batch_size=args.batch_size,
        inject_lang_tag=args.inject_lang_tag,
        model=model,
        tokenizer=tokenizer,
    )

    print("=== Belebele eval ===")
    results["scores"]["multilingual"] = run_multilingual_eval(
        model_path=model_path,
        languages=languages,
        run_flores=False,
        run_sib200=False,
        run_belebele=True,
        batch_size=args.batch_size,
        inject_lang_tag=args.inject_lang_tag,
        model=model,
        tokenizer=tokenizer,
    )

    print("=== IrokoBench eval ===")
    results["scores"]["multilingual"]["irokobench"] = run_irokobench_eval(
        model,
        tokenizer,
        inject_lang_tag=args.inject_lang_tag,
        batch_size=args.batch_size,
        generation_batch_size=args.generation_batch_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved required eval results to {args.output}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
