#!/usr/bin/env python3
"""
SSO-LoRA training script.

--mode stage1  Train shared LoRA on all 4 languages (2 epochs).
--mode stage2  Freeze shared, train lang-specific LoRA with orthogonal penalty (1 epoch).
--mode merge   Merge shared+lang adapters into a standard HF model.
"""

import argparse
import json
import os
import sys

import torch
from datasets import concatenate_datasets
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import load_sft_dataset
from src.data.trl_dataset_utils import prepare_dataset_for_trl
from src.models.sso_lora import (
    add_lang_adapter,
    orthogonal_loss,
    set_trainable_for_stage2,
    setup_shared,
)
from src.training.trainer import (
    build_sft_config,
    build_trainer_kwargs,
    load_causal_lm,
    load_tokenizer,
    require_full_model_save_allowed,
    setup_training_environment,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument(
        "--mode", required=True, choices=["stage1", "stage2", "merge", "merge_eval"]
    )
    p.add_argument("--train_lang", default=None)
    p.add_argument("--stage1_dir", default=None)
    p.add_argument("--r_shared", type=int, default=16)
    p.add_argument("--r_lang", type=int, default=8)
    p.add_argument("--lora_alpha_shared", type=float, default=32.0)
    p.add_argument("--lora_alpha_lang", type=float, default=16.0)
    p.add_argument(
        "--orth_weight",
        type=float,
        default=0.1,
        help="Weight for orthogonal penalty loss",
    )
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument(
        "--final_dir",
        default=None,
        help="Override merged model output dir (e.g. /tmp/...). "
        "If not set, defaults to output_dir/sso_{lang}.",
    )
    p.add_argument(
        "--eval_output",
        default=None,
        help="Path for eval JSON (merge_eval mode). Skips if file already exists.",
    )
    return p.parse_args()


class SSOTrainer(SFTTrainer):
    """SFTTrainer with added orthogonal penalty for stage2."""

    def __init__(self, *args, lang=None, orth_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._sso_lang = lang
        self._orth_weight = orth_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )
        if self._sso_lang is None or self._orth_weight <= 0:
            return result

        outputs = result[1] if return_outputs else None
        loss = result[0] if return_outputs else result

        orth = orthogonal_loss(model, self._sso_lang)
        loss = loss + self._orth_weight * orth

        return (loss, outputs) if return_outputs else loss


def run_stage1(args, cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    stage1_dir = os.path.join(args.output_dir, "stage1_shared")

    if os.path.exists(os.path.join(stage1_dir, "adapter_config.json")) or os.path.exists(
        os.path.join(stage1_dir, "shared", "adapter_config.json")
    ):
        if local_rank == 0:
            print(f"Stage 1 adapter found at {stage1_dir}, skipping.")
        return

    if local_rank == 0:
        print(
            f"\n=== SSO-LoRA Stage 1: Shared LoRA (all layers, r={args.r_shared}, 4-lang) ===\n"
        )

    tokenizer = load_tokenizer(args.model)

    base = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)

    model = setup_shared(
        base,
        r=args.r_shared,
        lora_alpha=args.lora_alpha_shared,
        dropout_p=float(cfg.peft.get("lora_dropout", 0.05)),
    )

    all_langs = ["en", "yo", "so", "ha"]
    parts = [load_sft_dataset(args.data_dir, l) for l in all_langs]
    train_dataset = prepare_dataset_for_trl(
        concatenate_datasets(parts).shuffle(seed=42),
        name="sso_stage1_all_langs",
    )
    if local_rank == 0:
        sizes = " + ".join(f"{len(p)} {l}" for l, p in zip(all_langs, parts))
        print(f"Dataset: {sizes} = {len(train_dataset)}")

    sft_cfg = build_sft_config(cfg, stage1_dir)
    trainer = SFTTrainer(**build_trainer_kwargs(
        SFTTrainer,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_cfg,
    ))
    trainer.train()

    if local_rank == 0:
        model.save_pretrained(stage1_dir, selected_adapters=["shared"])
        tokenizer.save_pretrained(stage1_dir)
        print(f"Stage 1 shared adapter saved → {stage1_dir}")


def run_stage2(args, cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    lang = args.train_lang
    assert lang, "--train_lang required for stage2"

    stage1_dir = os.path.abspath(
        args.stage1_dir or os.path.join(args.output_dir, "stage1_shared")
    )
    assert os.path.exists(
        os.path.join(stage1_dir, "adapter_config.json")
    ) or os.path.exists(
        os.path.join(stage1_dir, "shared", "adapter_config.json")
    ), f"Stage 1 adapter not found at {stage1_dir}"

    stage2_dir = os.path.join(args.output_dir, f"stage2_{lang}")
    if os.path.exists(os.path.join(stage2_dir, lang, "adapter_config.json")):
        if local_rank == 0:
            print(f"[{lang}] Stage 2 adapter found, skipping.")
        return

    if local_rank == 0:
        print(
            f"\n=== SSO-LoRA Stage 2: [{lang}] specific LoRA "
            f"(r={args.r_lang}, orth_weight={args.orth_weight}) ===\n"
        )

    tokenizer = load_tokenizer(args.model)

    from peft import PeftModel

    base = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)
    model = PeftModel.from_pretrained(
        base, os.path.join(stage1_dir, "shared"), adapter_name="shared"
    )
    model = add_lang_adapter(
        model,
        lang,
        r=args.r_lang,
        lora_alpha=args.lora_alpha_lang,
        dropout_p=float(cfg.peft.get("lora_dropout", 0.05)),
    )
    # Activate both adapters in forward, but only lang is trained
    try:
        model.set_adapter(["shared", lang])
    except Exception:
        model.set_adapter(lang)
    set_trainable_for_stage2(model, lang)

    lang_data = prepare_dataset_for_trl(
        load_sft_dataset(args.data_dir, lang),
        name=f"sso_stage2_{lang}",
    )
    if local_rank == 0:
        print(f"[{lang}] dataset: {len(lang_data)} samples")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    cfg_s2 = OmegaConf.to_container(cfg, resolve=True)
    cfg_s2["training"]["num_epochs"] = 1
    cfg_s2 = OmegaConf.create(cfg_s2)
    sft_cfg = build_sft_config(cfg_s2, stage2_dir)

    trainer = SSOTrainer(**build_trainer_kwargs(
        SSOTrainer,
        model=model,
        processing_class=tokenizer,
        train_dataset=lang_data,
        args=sft_cfg,
        lang=lang,
        orth_weight=args.orth_weight,
    ))
    trainer.train()

    if local_rank == 0:
        # Save only the lang adapter
        model.set_adapter(lang)
        model.save_pretrained(stage2_dir, selected_adapters=[lang])
        tokenizer.save_pretrained(stage2_dir)
        print(f"[{lang}] Stage 2 adapter saved → {stage2_dir}")


def run_merge(args):
    require_full_model_save_allowed("train_sso_lora.py --mode merge")
    lang = args.train_lang
    assert lang, "--train_lang required for merge"

    stage1_dir = os.path.abspath(
        args.stage1_dir or os.path.join(args.output_dir, "stage1_shared")
    )
    stage2_dir = os.path.abspath(os.path.join(args.output_dir, f"stage2_{lang}"))
    final_dir = os.path.abspath(
        args.final_dir or os.path.join(args.output_dir, f"sso_{lang}")
    )

    if os.path.exists(os.path.join(final_dir, "config.json")):
        print(f"[{lang}] Merged model already exists, skipping.")
        return

    print(f"\n=== SSO-LoRA Merge [{lang}]: shared (all layers) + lang (all layers) ===")

    tokenizer = load_tokenizer(args.model)

    from peft import PeftModel

    base = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)

    # Step 1: merge shared adapter into base
    model = PeftModel.from_pretrained(
        base, os.path.join(stage1_dir, "shared"), adapter_name="shared"
    )
    merged1 = model.merge_and_unload()
    # Step 2: load lang-specific adapter on top and merge
    merged1_peft = PeftModel.from_pretrained(
        merged1, os.path.join(stage2_dir, lang), adapter_name=lang
    )
    merged = merged1_peft.merge_and_unload()

    os.makedirs(final_dir, exist_ok=True)
    merged.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    meta = {
        "method": "SSO-LoRA",
        "lang": lang,
        "r_shared": args.r_shared,
        "r_lang": args.r_lang,
        "lora_alpha_shared": args.lora_alpha_shared,
        "lora_alpha_lang": args.lora_alpha_lang,
        "orth_weight": args.orth_weight,
        "stage1_dir": stage1_dir,
        "stage2_dir": stage2_dir,
    }
    with open(os.path.join(final_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{lang}] Merged model → {final_dir}")


def run_merge_eval(args):
    """Merge adapters in memory, run eval on the live model, save only JSON. No disk save."""
    lang = args.train_lang
    assert lang, "--train_lang required for merge_eval"
    eval_out = args.eval_output
    assert eval_out, "--eval_output required for merge_eval"

    if os.path.exists(eval_out):
        print(f"[{lang}] Eval already exists at {eval_out}, skipping.")
        return

    stage1_dir = os.path.abspath(
        args.stage1_dir or os.path.join(args.output_dir, "stage1_shared")
    )
    stage2_dir = os.path.abspath(os.path.join(args.output_dir, f"stage2_{lang}"))

    print(f"\n=== SSO-LoRA Merge+Eval [{lang}] in memory (no disk save) ===")

    tokenizer = load_tokenizer(args.model)

    from peft import PeftModel

    base = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)
    # Step 1: merge shared adapter into base
    model = PeftModel.from_pretrained(
        base, os.path.join(stage1_dir, "shared"), adapter_name="shared"
    )
    merged1 = model.merge_and_unload()
    # Step 2: load lang-specific adapter on top and merge
    merged1_peft = PeftModel.from_pretrained(
        merged1, os.path.join(stage2_dir, lang), adapter_name=lang
    )
    merged = merged1_peft.merge_and_unload()
    merged.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    merged = merged.to(device)
    print(f"[{lang}] Merge complete (shared→base, {lang}→base), model on {device}.")

    from src.evaluation.english_eval import run_english_eval
    from src.evaluation.irokobench_eval import run_irokobench_eval
    from src.evaluation.multilingual_eval import run_multilingual_eval

    results = {"model_path": f"sso_{lang}_in_memory", "scores": {}}

    print("=== English eval ===")
    results["scores"]["english"] = run_english_eval(
        model_path=args.model, model=merged, tokenizer=tokenizer
    )

    print("=== Multilingual eval ===")
    results["scores"]["multilingual"] = run_multilingual_eval(
        model_path=args.model,
        model=merged,
        tokenizer=tokenizer,
        languages=["en", "yo", "so", "ha"],
        run_flores=False,
        run_sib200=False,
        run_belebele=True,
    )
    print("=== IrokoBench eval ===")
    results["scores"]["multilingual"]["irokobench"] = run_irokobench_eval(
        merged, tokenizer
    )

    os.makedirs(os.path.dirname(os.path.abspath(eval_out)) or ".", exist_ok=True)
    with open(eval_out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[{lang}] Eval saved → {eval_out}")

    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    setup_training_environment()
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "stage1":
        run_stage1(args, cfg)
    elif args.mode == "stage2":
        run_stage2(args, cfg)
    elif args.mode == "merge":
        run_merge(args)
    elif args.mode == "merge_eval":
        run_merge_eval(args)


if __name__ == "__main__":
    main()
