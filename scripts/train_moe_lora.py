#!/usr/bin/env python3
"""
Soft MoE-LoRA (LA-MoA) training.

K=4 LoRA experts + token-level learnable router, applied to all target linear layers.
Trained jointly on all 4 languages (en+yo+so+ha), same data as tag_routing 4-lang.

Usage (via launch_moe_lora.sh):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate_2gpu.yaml \\
      scripts/train_moe_lora.py --model ... --output_dir ... --config ...
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
from src.models.moe_lora import freeze_base, save_moe, setup_moe_lora
from src.training.trainer import build_sft_config

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument(
        "--r",
        type=int,
        default=8,
        help="LoRA rank per expert (total params ≈ K × std-LoRA-r)",
    )
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print(
            f"\n=== MoE-LoRA Training: K={args.n_experts} r={args.r} "
            f"alpha={args.lora_alpha} ===\n"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.use_cache = False

    replaced = setup_moe_lora(
        model,
        TARGET_MODULES,
        r=args.r,
        lora_alpha=args.lora_alpha,
        n_experts=args.n_experts,
        dropout_p=float(cfg.peft.get("lora_dropout", 0.05)),
    )
    model = freeze_base(model)

    if local_rank == 0:
        print(f"Replaced {len(replaced)} linear layers with MoE-LoRA")

    all_langs = ["en", "yo", "so", "ha"]
    parts = [load_sft_dataset(args.data_dir, l) for l in all_langs]
    train_dataset = concatenate_datasets(parts).shuffle(seed=42)
    if local_rank == 0:
        sizes = " + ".join(f"{len(p)} {l}" for l, p in zip(all_langs, parts))
        print(f"Dataset: {sizes} = {len(train_dataset)} total")

    sft_cfg = build_sft_config(cfg, args.output_dir)
    # Disable mid-training checkpoint saves — we save MoE weights manually after training
    sft_cfg.save_strategy = "no"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_cfg,
    )
    trainer.train()

    # Save only the MoE-LoRA weights (not the full frozen base)
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        moe_config = {
            "base_model": args.model,
            "n_experts": args.n_experts,
            "r": args.r,
            "lora_alpha": args.lora_alpha,
            "dropout_p": float(cfg.peft.get("lora_dropout", 0.05)),
            "target_modules": TARGET_MODULES,
            "train_langs": all_langs,
            "train_samples": len(train_dataset),
        }
        save_moe(unwrapped, args.output_dir, moe_config)
        tokenizer.save_pretrained(args.output_dir)
        meta = {"method": "MoE-LoRA", **moe_config}
        with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nMoE-LoRA training complete → {args.output_dir}")


if __name__ == "__main__":
    main()
