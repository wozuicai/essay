"""
Training argument builder and trainer utilities.
"""

import os
from trl import SFTConfig


def build_sft_config(cfg, output_dir: str, max_seq_length: int = None) -> SFTConfig:
    """
    Build trl SFTConfig from OmegaConf config.
    SFTConfig subclasses TrainingArguments and adds max_seq_length / dataset_text_field.
    """
    t = cfg.training
    seq_len = max_seq_length or t.get("max_seq_length", 2048)

    # Detect whether wandb is actually logged-in to avoid hanging
    report_to = _get_report_to()

    """
Training argument builder and trainer utilities.
"""

import inspect
import os
from trl import SFTConfig


def build_sft_config(cfg, output_dir: str, max_seq_length: int = None) -> SFTConfig:
    """
    Build trl SFTConfig from OmegaConf config.
    SFTConfig subclasses TrainingArguments and adds max_seq_length / dataset_text_field.
    """
    t = cfg.training
    seq_len = max_seq_length or t.get("max_seq_length", 2048)

    # Detect whether wandb is actually logged-in to avoid hanging
    report_to = _get_report_to()

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=t.num_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size, 
        gradient_accumulation_steps=t.gradient_accumulation_steps, 
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler,
        warmup_ratio=t.warmup_ratio,
        bf16=True,
        fp16=False,
        logging_steps=t.get("logging_steps",100),
        save_steps=t.get("save_steps",500),
        save_total_limit=t.get("save_total_limit",1),
        eval_strategy="no",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=report_to,
        seed=t.get("seed",42),
        max_length=seq_len,
        dataset_text_field="text",
    )


def _get_report_to() -> str:
    """Use wandb only if WANDB_API_KEY or netrc login exists; fall back to none."""
    import os
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    netrc = os.path.expanduser("~/.netrc")
    has_netrc = os.path.exists(netrc) and "api.wandb.ai" in open(netrc).read()
    if wandb_key or has_netrc:
        return "wandb"
    return "none"

