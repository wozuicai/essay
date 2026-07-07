"""
Unified training entry point supporting:
  - Standard LoRA (phase2, phase3)
  - Full fine-tuning (phase4)
  - Isolated LoRA / B2 (phase4)
"""

import argparse
import json
import os
import sys

import torch
from omegaconf import OmegaConf
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import load_sft_dataset
from src.data.data_mixer import create_mixed_dataset
from src.data.trl_dataset_utils import prepare_dataset_for_trl
from src.models.lora_standard import setup_standard_lora
from src.models.lora_isolated import setup_isolated_lora, train_isolated_lora
from src.training.trainer import (
    build_sft_config,
    build_trainer_kwargs,
    load_causal_lm,
    load_tokenizer,
    require_full_model_save_allowed,
    save_run_config,
    save_training_metadata,
    setup_training_environment,
)

SUPPORTED_METHODS = ["standard_lora", "full_ft", "mixed_lora", "isolated_lora"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--train_lang", required=True, help="Language code for training data"
    )
    parser.add_argument("--method", default="standard_lora", choices=SUPPORTED_METHODS)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--english_ratio", type=float, default=None)
    parser.add_argument("--total_samples", type=int, default=4000)
    parser.add_argument(
        "--mix_all",
        action="store_true",
        help="全量 en + 全量 target-lang concat 后 shuffle，不做比例截断",
    )
    parser.add_argument(
        "--mix_all_langs",
        action="store_true",
        help="全量 en+yo+so+ha 四语言 concat shuffle，用于 tag routing 实验",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--config", required=True, help="Path to experiment YAML config"
    )
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--wandb_project", default="crosslingual-interference")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def _init_wandb(args, cfg):
    if args.no_wandb:
        return
    try:
        import wandb
        from src.training.trainer import _get_report_to

        if _get_report_to() == "wandb":
            wandb.init(
                project=args.wandb_project,
                name=os.path.basename(args.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
    except Exception as e:
        print(f"[WARNING] wandb init failed, continuing without: {e}")


def main():
    setup_training_environment()
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    _init_wandb(args, cfg)

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = load_tokenizer(args.model)

    print(f"Loading base model from {args.model}...")
    model = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)

    # Prepare dataset (isolated_lora has two internal stages, prepared below).
    train_dataset = None
    if args.method != "isolated_lora":
        if args.mix_all_langs:
            from datasets import concatenate_datasets

            all_langs = ["en", "yo", "so", "ha"]
            parts = [load_sft_dataset(args.data_dir, l) for l in all_langs]
            train_dataset = concatenate_datasets(parts).shuffle(seed=42)
            sizes = " + ".join(f"{len(p)} {l}" for l, p in zip(all_langs, parts))
            print(f"mix_all_langs: {sizes} = {len(train_dataset)} total")
        elif args.mix_all:
            from datasets import concatenate_datasets

            en_data = load_sft_dataset(args.data_dir, "en")
            tgt_data = load_sft_dataset(args.data_dir, args.train_lang)
            train_dataset = concatenate_datasets([en_data, tgt_data]).shuffle(seed=42)
            print(
                f"mix_all: {len(en_data)} en + {len(tgt_data)} {args.train_lang} = {len(train_dataset)} total"
            )
        elif args.english_ratio is not None:
            en_data = load_sft_dataset(args.data_dir, "en")
            tgt_data = load_sft_dataset(args.data_dir, args.train_lang)
            train_dataset = create_mixed_dataset(
                en_data, tgt_data, args.english_ratio, args.total_samples
            )
        else:
            n_samples = args.train_samples or cfg.data.get("train_samples", 500)
            train_dataset = load_sft_dataset(
                args.data_dir, args.train_lang, n_samples=n_samples
            )
        train_dataset = prepare_dataset_for_trl(
            train_dataset, name=f"{args.method}_{args.train_lang}"
        )
        print(f"Training dataset size: {len(train_dataset)}")

    sft_cfg = build_sft_config(cfg, args.output_dir)

    if args.method == "full_ft":
        require_full_model_save_allowed("train.py --method full_ft")
        for param in model.parameters():
            param.requires_grad = True

        trainer = SFTTrainer(**build_trainer_kwargs(
            SFTTrainer,
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=sft_cfg,
        ))
        train_result = trainer.train()
        trainer.save_model(args.output_dir)

    elif args.method in ("standard_lora", "mixed_lora"):
        peft_cfg_node = cfg.peft if "peft" in cfg else cfg.methods.standard_lora
        model = setup_standard_lora(model, peft_cfg_node)

        trainer = SFTTrainer(**build_trainer_kwargs(
            SFTTrainer,
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=sft_cfg,
        ))
        train_result = trainer.train()
        trainer.save_model(args.output_dir)

    elif args.method == "isolated_lora":
        require_full_model_save_allowed("train.py --method isolated_lora final merge")
        isolated_cfg = cfg.methods.isolated_lora
        lang = args.train_lang
        en_data = prepare_dataset_for_trl(
            load_sft_dataset(args.data_dir, "en"), name="isolated_stage1_en"
        )
        lang_data = prepare_dataset_for_trl(
            load_sft_dataset(args.data_dir, lang), name=f"isolated_{lang}"
        )

        model = setup_isolated_lora(
            model, lang, isolated_cfg.shared_r, isolated_cfg.lang_r
        )
        train_isolated_lora(
            model, tokenizer, en_data, lang_data, lang, cfg, args.output_dir
        )

        # Merge both adapters into base model so evaluate.py can use it as a plain HF model
        print("\nMerging shared + lang adapters into base model...")
        model.base_model.set_adapter(
            ["shared", lang]
        )  # PeftModel.set_adapter rejects list in 0.19.1
        model = model.merge_and_unload()
        print("Merge complete.")
        train_result = None

    # Save final model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    if args.method == "isolated_lora":
        model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    n_samples_meta = (
        len(train_dataset)
        if train_dataset is not None
        else "isolated_lora: see stage logs"
    )
    metadata = {
        "model": args.model,
        "train_lang": args.train_lang,
        "method": args.method,
        "train_samples": n_samples_meta,
        "english_ratio": args.english_ratio,
        "config": str(args.config),
        "trainer_backend": "trl_prompt_completion",
        "completion_only_loss": True,
        "packing": bool(getattr(sft_cfg, "packing", False)),
    }
    if train_result is not None:
        metadata["train_metrics"] = getattr(train_result, "metrics", {})
    save_training_metadata(args.output_dir, metadata)
    save_run_config(args.output_dir, args)

    print(f"\nTraining complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
