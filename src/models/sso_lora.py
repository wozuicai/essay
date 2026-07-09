"""
Shared-Specific Orthogonal LoRA (SSO-LoRA)

Two LoRA components per target linear layer:
  - "shared": trained on all 4 languages, captures cross-lingual transfer
  - lang-specific ("en"/"yo"/"so"/"ha"): trained on each language separately

Orthogonality penalty forces shared and lang-specific subspaces apart:
  L_orth = ||A_shared @ A_lang.T||_F^2 + ||B_shared.T @ B_lang||_F^2

Usage:
  stage1 -- train shared LoRA (all 4 langs, 1 epoch)
  stage2 -- freeze shared, train lang LoRA with orth penalty (1 epoch per lang)
  merge  -- load shared+lang, merge_and_unload → standard HF model
"""

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def setup_shared(model, r=16, lora_alpha=32.0, dropout_p=0.05):
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout_p,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg, adapter_name="shared")


def add_lang_adapter(peft_model, lang: str, r=8, lora_alpha=16.0, dropout_p=0.05):
    """Add a per-language LoRA adapter on ALL layers (orthogonal to 'shared')."""
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout_p,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model.add_adapter(lang, cfg)
    return peft_model


def set_trainable_for_stage2(peft_model, lang: str):
    """Freeze shared adapter, enable gradients only for lang adapter."""
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.requires_grad_("." + lang + "." in name or f"_{lang}" in name)
    # make sure shared is frozen
    for name, param in peft_model.named_parameters():
        if "lora_" in name and ".shared." in name:
            param.requires_grad_(False)
        elif "lora_" in name and f".{lang}." in name:
            param.requires_grad_(True)


def orthogonal_loss(peft_model, lang: str) -> torch.Tensor:
    """
    Penalize non-orthogonality between shared and lang-specific LoRA matrices.
    For each target module:
      L += ||A_shared @ A_lang.T||_F^2   (input projection directions)
      L += ||B_shared.T @ B_lang||_F^2   (output projection directions)
    Values are mean-normalized per layer to be scale-invariant.
    """
    loss = peft_model.parameters().__next__().new_zeros(1).squeeze()
    n = 0
    for module in peft_model.modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or not isinstance(lora_A, nn.ModuleDict):
            continue
        if "shared" not in lora_A or lang not in lora_A:
            continue
        A_s = lora_A["shared"].weight   # [r_s, d_in]
        A_l = lora_A[lang].weight       # [r_l, d_in]
        B_s = lora_B["shared"].weight   # [d_out, r_s]
        B_l = lora_B[lang].weight       # [d_out, r_l]
        loss = loss + (A_s @ A_l.T).pow(2).mean()
        loss = loss + (B_s.T @ B_l).pow(2).mean()
        n += 1
    return loss / max(n, 1)
