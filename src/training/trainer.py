"""
Shared training utilities for all experiment scripts.

The project has several experiment-specific trainers (standard LoRA, MID, DSCT,
MoE-LoRA, SSO-LoRA, layerwise LoRA).  This module centralizes the common TRL SFT
settings so every trainer consistently uses the B200 worker setup requested in
/root/action plan:

- FlashAttention 2 via attn_implementation="flash_attention_2"
- Liger kernel when the installed TRL version exposes use_liger_kernel
- bf16 + gradient checkpointing
- safe local scratch/cache directories on /mnt/local/localcache00 when present
- no answer-weighted loss customization
"""

from __future__ import annotations

import inspect
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
    return model


def build_sft_config(cfg, output_dir: str, max_seq_length: int = None) -> SFTConfig:
    """
    Build a TRL SFTConfig while filtering arguments for the installed TRL version.

    Different workers have used different TRL releases; inspect the constructor so
    newer options such as use_liger_kernel are enabled when available but do not
    break older installs.
    """
    t = cfg.training
    seq_len = max_seq_length or t.get("max_seq_length", 2048)
    report_to = _get_report_to()

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": t.num_epochs,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "learning_rate": t.learning_rate,
        "lr_scheduler_type": t.lr_scheduler,
        "warmup_ratio": t.warmup_ratio,
        "bf16": True,
        "fp16": False,
        "logging_steps": t.get("logging_steps", 100),
        "save_steps": t.get("save_steps", 500),
        "save_total_limit": t.get("save_total_limit", 1),
        "gradient_checkpointing": True,
        "dataloader_num_workers": int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
        "remove_unused_columns": False,
        "report_to": report_to,
        "seed": t.get("seed", 42),
        "max_length": seq_len,
        "max_seq_length": seq_len,
        "dataset_text_field": "text",
        "packing": strtobool_env("PACKING", False),
        "use_liger_kernel": get_use_liger_kernel(),
    }

    sig = inspect.signature(SFTConfig.__init__)
    params = sig.parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"

    # Some TRL versions support completion_only_loss only for prompt-completion
    # datasets. Our current processed data uses a single text field, so do not set it.
    filtered = _filter_kwargs(SFTConfig, kwargs)
    return SFTConfig(**filtered)


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
