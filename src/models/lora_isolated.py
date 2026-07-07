"""
Language-Isolated LoRA (B2 approach).

Architecture:
  - One shared LoRA adapter trained on en + target low-resource language (full data)
  - One language-specific LoRA adapter fine-tuned on the target language only
  - At inference: merge both adapters into base model

Training (per target language):
  Stage 1: shared adapter on full en + full lang data, step1_epochs (2)
  Stage 2: freeze shared, train lang adapter on full lang data, step2_epochs (1)
"""

import glob
import os
import re

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

from src.training.trainer import build_sft_config, build_trainer_kwargs


def setup_isolated_lora(base_model, lang_name: str, shared_r: int = 8, lang_r: int = 8):
    """
    Initialize isolated LoRA for one target language.
    Creates a 'shared' adapter and one lang-specific adapter.
    """
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    shared_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=shared_r,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(base_model, shared_config, adapter_name="shared")

    lang_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lang_r,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model.add_adapter(lang_name, lang_config)

    print(f"Isolated LoRA: shared (r={shared_r}) + {lang_name} adapter (r={lang_r})")
    return model


def _load_shared_from_checkpoint(model, stage1_output: str) -> bool:
    """Load shared adapter weights from the latest Stage 1 checkpoint. Returns True if loaded."""
    from safetensors.torch import load_file
    from peft.utils import set_peft_model_state_dict

    ckpts = sorted(
        glob.glob(os.path.join(stage1_output, "checkpoint-*")),
        key=lambda x: int(x.rsplit("-", 1)[-1]),
    )
    if not ckpts:
        return False
    weights_path = os.path.join(ckpts[-1], "shared", "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        return False
    state_dict = load_file(weights_path)
    set_peft_model_state_dict(model, state_dict, adapter_name="shared")
    print(f"  Loaded Stage 1 shared adapter from {ckpts[-1]}")
    return True


def _freeze_except(model, adapter_name: str):
    """
    Freeze all LoRA params except those belonging to adapter_name.
    PEFT names params as: ...lora_A.<adapter_name>.weight
    """
    marker = f".{adapter_name}."
    for name, param in model.named_parameters():
        if "lora_A." in name or "lora_B." in name:
            param.requires_grad = (marker in name)


def train_isolated_lora(model, tokenizer, en_data, lang_data, lang_name: str, cfg, output_dir: str):
    """
    Two-stage training for isolated LoRA (single target language).

    Stage 1: shared adapter on full en + full lang_data, step1_epochs.
    Stage 2: freeze shared, train lang adapter on full lang_data, step2_epochs.
    """
    from datasets import concatenate_datasets

    isolated_cfg = cfg.methods.isolated_lora

    # -------- Stage 1: shared adapter on en + lang --------
    print(f"\n=== Stage 1: Training shared adapter (en + {lang_name}) ===")
    stage1_output = os.path.join(output_dir, "stage1_shared")

    if _load_shared_from_checkpoint(model, stage1_output):
        print("  Stage 1 checkpoint found — skipping retraining.")
    else:
        model.set_adapter("shared")
        _freeze_except(model, "shared")

        stage1_data = concatenate_datasets([en_data, lang_data]).shuffle(seed=42)
        print(f"  {len(en_data)} en + {len(lang_data)} {lang_name} = {len(stage1_data)} total")

        cfg_s1 = OmegaConf.to_container(cfg, resolve=True)
        cfg_s1["training"]["num_epochs"] = isolated_cfg.step1_epochs
        cfg_s1 = OmegaConf.create(cfg_s1)
        stage1_sft_cfg = build_sft_config(cfg_s1, stage1_output)

        SFTTrainer(**build_trainer_kwargs(
            SFTTrainer,
            model=model,
            processing_class=tokenizer,
            train_dataset=stage1_data,
            args=stage1_sft_cfg,
        )).train()
        print("Stage 1 complete.")

    # -------- Stage 2: lang-specific adapter on lang only --------
    print(f"\n=== Stage 2: Training {lang_name} adapter (lang only, {isolated_cfg.step2_epochs} epoch) ===")
    model.base_model.set_adapter(["shared", lang_name])  # PeftModel.set_adapter rejects list in 0.19.1
    _freeze_except(model, lang_name)
    print(f"  {len(lang_data)} {lang_name} samples")

    cfg_s2 = OmegaConf.to_container(cfg, resolve=True)
    cfg_s2["training"]["num_epochs"] = isolated_cfg.step2_epochs
    cfg_s2 = OmegaConf.create(cfg_s2)
    stage2_sft_cfg = build_sft_config(cfg_s2, os.path.join(output_dir, f"stage2_{lang_name}"))

    SFTTrainer(**build_trainer_kwargs(
        SFTTrainer,
        model=model,
        processing_class=tokenizer,
        train_dataset=lang_data,
        args=stage2_sft_cfg,
    )).train()
    print(f"Stage 2 [{lang_name}] complete.")
