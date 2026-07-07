"""
Layer-wise Progressive Language Routing helpers.

Bottom layers (0 .. split_layer-1): one shared LoRA adapter, trained on all 4 languages.
Top layers   (split_layer .. n_layers-1): per-language LoRA adapters (yo, so, ha).

Inference per language L:
    base model + shared_bottom_adapter + L_top_adapter → merge_and_unload → plain HF model
"""

from peft import LoraConfig, get_peft_model

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def setup_shared_bottom(model, n_total_layers: int, split_layer: int,
                        r: int, lora_alpha: float, dropout_p: float = 0.05):
    """
    Add 'shared' LoRA adapter restricted to bottom layers (0..split_layer-1).
    Returns PeftModel.
    """
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        layers_to_transform=list(range(split_layer)),
        lora_dropout=dropout_p,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg, adapter_name="shared")
    model.print_trainable_parameters()
    return model


def add_lang_top(peft_model, lang: str, split_layer: int, n_total_layers: int,
                 r: int, lora_alpha: float, dropout_p: float = 0.05):
    """
    Add a language-specific LoRA adapter on top layers (split_layer..n_total_layers-1).
    Freezes 'shared' adapter; only lang adapter receives gradients.
    """
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        layers_to_transform=list(range(split_layer, n_total_layers)),
        lora_dropout=dropout_p,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model.add_adapter(lang, cfg)

    # Freeze shared, train only lang adapter
    marker_lang = f".{lang}."
    for name, param in peft_model.named_parameters():
        if "lora_A." in name or "lora_B." in name:
            param.requires_grad_(marker_lang in name)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[{lang}] top-layer trainable: {trainable/1e6:.1f}M params")
    return peft_model
