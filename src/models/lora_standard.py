"""
Standard LoRA setup using PEFT.
"""

from peft import LoraConfig, get_peft_model, TaskType


def setup_standard_lora(model, cfg) -> object:
    """
    Apply standard LoRA to model using config dict or OmegaConf node.
    cfg should have: r, lora_alpha, lora_dropout, target_modules, bias
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias if hasattr(cfg, "bias") else "none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
