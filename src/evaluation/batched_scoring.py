"""Batched scoring helpers for HF/PEFT/custom-module eval.

These helpers intentionally use plain PyTorch + Transformers forward/generate,
so they work for PEFT adapters and custom modules such as MoE-LoRA.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch


def get_eval_max_length(default: int = 2048) -> int:
    raw = os.environ.get("EVAL_MAX_LENGTH", "").strip()
    return int(raw) if raw else default


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def letter_token_ids(tokenizer, n_choices: int) -> list[int]:
    ids = []
    for i in range(n_choices):
        letter = chr(65 + i)
        for surface in (f" {letter}", letter, f"{letter}."):
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
            if token_ids:
                ids.append(token_ids[0])
                break
        else:
            raise ValueError(f"Could not tokenize MCQ letter {letter!r}.")
    return ids


def batched_next_token_predictions(
    model,
    tokenizer,
    prompts: list[str],
    choice_token_ids: list[int],
    *,
    batch_size: int,
    max_length: int | None = None,
    num_choices_per_prompt: list[int] | None = None,
) -> list[int]:
    """Predict the highest-scoring next-token choice for each prompt."""
    ensure_pad_token(tokenizer)
    max_length = max_length or get_eval_max_length()
    device = next(model.parameters()).device
    preds: list[int] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.inference_mode():
            logits = model(**enc).logits

        positions = torch.arange(enc["attention_mask"].shape[1], device=logits.device).unsqueeze(0)
        last_positions = (enc["attention_mask"].to(logits.device) * positions).max(dim=1).values
        row_idx = torch.arange(logits.shape[0], device=logits.device)
        next_logits = logits[row_idx, last_positions]
        scores = torch.stack([next_logits[:, tid] for tid in choice_token_ids], dim=1)

        if num_choices_per_prompt is not None:
            batch_counts = num_choices_per_prompt[start : start + len(batch_prompts)]
            for row, n_choices in enumerate(batch_counts):
                if n_choices < scores.shape[1]:
                    scores[row, n_choices:] = float("-inf")

        preds.extend(scores.argmax(dim=1).tolist())

    return preds


@contextmanager
def temporary_padding_side(tokenizer, side: str):
    old_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = old_side


def batched_generate_text(
    model,
    tokenizer,
    prompts: list[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    max_length: int | None = None,
) -> list[str]:
    """Batched greedy generation, returning only newly generated text."""
    ensure_pad_token(tokenizer)
    max_length = max_length or get_eval_max_length(1024)
    device = next(model.parameters()).device
    outputs: list[str] = []

    with temporary_padding_side(tokenizer, "left"):
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_width = enc["input_ids"].shape[1]
            for row in generated:
                gen_ids = row[input_width:]
                outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    return outputs
