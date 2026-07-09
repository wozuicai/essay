"""Shared training utilities for all experiment scripts.

The train scripts now follow the World20K TRL LoRA SFT contract:

- datasets are prompt/completion rows;
- TRL masks prompt tokens with `completion_only_loss=True`;
- long samples use `truncation_mode="keep_end"`;
- intermediate checkpoints are disabled by default to avoid large files.
"""

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig


DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
DEFAULT_CUDA_HOME = "/home/tiger/.local/lib/python3.11/site-packages/nvidia/cu13"
DEFAULT_LOCAL_CACHE = "/mnt/local/localcache00"


def setup_training_environment() -> None:
    """Set worker-safe defaults before importing/initializing CUDA-heavy code."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if os.path.isdir(DEFAULT_CUDA_HOME):
        os.environ.setdefault("CUDA_HOME", DEFAULT_CUDA_HOME)
        cuda_bin = os.path.join(DEFAULT_CUDA_HOME, "bin")
        cuda_lib = os.path.join(DEFAULT_CUDA_HOME, "lib")
        _prepend_env_path("PATH", "/home/tiger/.local/bin")
        _prepend_env_path("PATH", cuda_bin)
        _prepend_env_path("LD_LIBRARY_PATH", cuda_lib)
        _prepend_env_path("LIBRARY_PATH", cuda_lib)

    # Put the real driver path before stale CUDA compat libraries to avoid CUDA 803.
    _prepend_env_path("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu")

    tmpdir, triton_cache = _scratch_dirs()
    os.environ.setdefault("TMPDIR", tmpdir)
    os.environ.setdefault("TRITON_CACHE_DIR", triton_cache)


def _scratch_dirs() -> tuple[str, str]:
    """Return writable TMPDIR/TRITON cache dirs, preferring local NVMe on workers."""
    candidates = []
    if os.path.isdir(DEFAULT_LOCAL_CACHE):
        candidates.append(DEFAULT_LOCAL_CACHE)
    candidates.append("/tmp")

    user = os.environ.get("USER", "tiger")
    for root in candidates:
        tmpdir = os.path.join(root, f"tmp-{user}")
        triton_cache = os.path.join(root, f"triton-cache-{user}")
        try:
            os.makedirs(tmpdir, exist_ok=True)
            os.makedirs(triton_cache, exist_ok=True)
            probe = os.path.join(tmpdir, ".write_test")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return tmpdir, triton_cache
        except OSError:
            continue
    raise RuntimeError("Could not find a writable scratch directory for training caches.")


def _prepend_env_path(name: str, value: str) -> None:
    if not value or not os.path.exists(value):
        return
    current = os.environ.get(name, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if value not in parts:
        os.environ[name] = value if not current else value + os.pathsep + current


def strtobool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_attn_implementation() -> str:
    """Return requested attention backend; empty string disables explicit setting."""
    return os.environ.get("ATTN_IMPLEMENTATION", DEFAULT_ATTN_IMPLEMENTATION).strip()


def get_use_liger_kernel() -> bool:
    return strtobool_env("USE_LIGER_KERNEL", True)


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def model_load_kwargs(
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: Optional[str] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    attn = get_attn_implementation() if attn_implementation is None else attn_implementation
    if attn:
        kwargs["attn_implementation"] = attn
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def load_causal_lm(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: Optional[str] = None,
    use_cache: Optional[bool] = False,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """Load a causal LM with the project's FlashAttention defaults."""
    kwargs = model_load_kwargs(
        dtype=dtype,
        attn_implementation=attn_implementation,
        extra_kwargs=extra_kwargs,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception:
        if kwargs.get("attn_implementation") and strtobool_env("ALLOW_ATTN_FALLBACK", False):
            kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        else:
            raise
    if use_cache is not None and hasattr(model, "config"):
        model.config.use_cache = use_cache
    if use_cache is False:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def build_sft_config(
    cfg,
    output_dir: str,
    max_seq_length: int = None,
    *,
    no_eval: bool = True,
    packing: Optional[bool] = None,
    completion_only_loss: bool = True,
    truncation_mode: str = "keep_end",
) -> SFTConfig:
    """
    Build a TRL SFTConfig aligned with `world20k_lora_sft/scripts/trl_lora_sft.py`.

    We intentionally fail fast when the installed TRL lacks prompt-completion
    loss or keep-end truncation; silently falling back would change the loss.
    """
    t = cfg.training
    env_seq_len = os.environ.get("MAX_SEQ_LENGTH", "").strip()
    seq_len = int(env_seq_len) if env_seq_len else (max_seq_length or t.get("max_seq_length", 24000))
    report_to = _get_report_to()
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    use_packing = strtobool_env("PACKING", True) if packing is None else packing
    requested_save_strategy = os.environ.get("SAVE_STRATEGY", "no")
    save_strategy = (
        requested_save_strategy
        if strtobool_env("ALLOW_TRAIN_CHECKPOINTS", False)
        else "no"
    )
    if requested_save_strategy != "no" and save_strategy == "no":
        print(
            f"[trainer] Ignoring SAVE_STRATEGY={requested_save_strategy!r}; "
            "export ALLOW_TRAIN_CHECKPOINTS=1 to enable intermediate checkpoints."
        )

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": t.num_epochs,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "per_device_eval_batch_size": t.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "learning_rate": t.learning_rate,
        "lr_scheduler_type": t.lr_scheduler,
        "warmup_ratio": t.warmup_ratio,
        "warmup_steps": t.get("warmup_steps", 0),
        "weight_decay": t.get("weight_decay", 0.0),
        "bf16": True,
        "fp16": False,
        "logging_steps": t.get("logging_steps", 100),
        "save_steps": t.get("save_steps", 500),
        "eval_steps": t.get("eval_steps", 500),
        "save_total_limit": 1,
        "save_strategy": save_strategy,
        "gradient_checkpointing": True,
        "dataloader_num_workers": int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
        "remove_unused_columns": True,
        "report_to": report_to,
        "seed": t.get("seed", 42),
        "ddp_find_unused_parameters": False,
        "group_by_length": strtobool_env("GROUP_BY_LENGTH", True),
        "deepspeed": os.environ.get("DEEPSPEED_CONFIG") or None,
        "max_length": seq_len,
        "max_seq_length": seq_len,
        "packing": use_packing,
        "completion_only_loss": completion_only_loss,
        "truncation_mode": truncation_mode,
        "use_liger_kernel": get_use_liger_kernel(),
    }

    if "eval_strategy" in sft_params:
        kwargs["eval_strategy"] = "no" if no_eval else "steps"
    elif "evaluation_strategy" in sft_params:
        kwargs["evaluation_strategy"] = "no" if no_eval else "steps"

    if "completion_only_loss" not in sft_params:
        raise RuntimeError(
            "Installed TRL does not expose SFTConfig.completion_only_loss. "
            "Install/upgrade TRL before running these prompt-completion jobs."
        )
    if "truncation_mode" not in sft_params:
        raise RuntimeError(
            "Installed TRL does not expose SFTConfig.truncation_mode. "
            "Install/upgrade TRL or this refactor would not match World20K."
        )
    if "max_length" not in sft_params and "max_seq_length" not in sft_params:
        raise RuntimeError("Installed TRL SFTConfig has no max_length/max_seq_length parameter.")
    if use_packing and "packing" not in sft_params:
        raise RuntimeError("Installed TRL SFTConfig has no packing parameter, but packing is enabled.")
    if get_use_liger_kernel() and "use_liger_kernel" not in sft_params and "use_liger" not in sft_params:
        raise RuntimeError(
            "Installed TRL SFTConfig has no use_liger_kernel/use_liger parameter. "
            "Install liger-kernel + recent TRL, or export USE_LIGER_KERNEL=0."
        )

    if "use_liger_kernel" not in sft_params and "use_liger" in sft_params:
        kwargs["use_liger"] = kwargs.pop("use_liger_kernel")
    elif "use_liger_kernel" not in sft_params:
        kwargs.pop("use_liger_kernel", None)

    if "max_length" not in sft_params:
        kwargs.pop("max_length", None)
    if "max_seq_length" not in sft_params:
        kwargs.pop("max_seq_length", None)
    if "packing" not in sft_params:
        kwargs.pop("packing", None)

    filtered = _filter_kwargs(SFTConfig, kwargs)
    return SFTConfig(**filtered)


def build_trainer_kwargs(trainer_cls, **kwargs) -> Dict[str, Any]:
    """Filter trainer kwargs for TRL versions that use tokenizer or processing_class."""
    params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in kwargs and "tokenizer" not in kwargs and "tokenizer" in params:
        kwargs["tokenizer"] = kwargs["processing_class"]
    return _filter_kwargs(trainer_cls, kwargs)


def save_training_metadata(output_dir: str, metadata: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_run_config(output_dir: str, args: Any) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data = vars(args) if hasattr(args, "__dict__") else dict(args)
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def require_full_model_save_allowed(context: str) -> None:
    """Guard paths that would write a merged/full base model to disk."""
    if strtobool_env("ALLOW_FULL_MODEL_SAVE", False):
        return
    raise RuntimeError(
        f"{context} would save a merged/full model checkpoint. This is disabled by "
        "default to avoid writing huge weights. Use the LoRA adapter path or "
        "merge_eval in-memory path instead. To override intentionally, export "
        "ALLOW_FULL_MODEL_SAVE=1."
    )


def _filter_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _get_report_to() -> str:
    """Use wandb only if explicitly enabled and credentials exist; otherwise none."""
    if strtobool_env("WANDB_DISABLED", False):
        return "none"
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    netrc = os.path.expanduser("~/.netrc")
    has_netrc = False
    if os.path.exists(netrc):
        try:
            with open(netrc, "r", encoding="utf-8", errors="ignore") as f:
                has_netrc = "api.wandb.ai" in f.read()
        except OSError:
            has_netrc = False
    return "wandb" if (wandb_key or has_netrc) else "none"
