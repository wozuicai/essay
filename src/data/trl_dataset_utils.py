#!/usr/bin/env python3
"""Dataset utilities for TRL prompt-completion SFT.

The World20K reference script trains with TRL prompt/completion examples and
`completion_only_loss=True`.  This module makes the project datasets follow the
same contract while still accepting the older local JSONL shape:

- `messages`: OpenAI-style chat rows, last message must be assistant.
- `instruction` + `response`: converted to the project SFT prompt template.
- `text`: split at `### Response:` as a compatibility fallback.
- `prompt` + `completion`: kept as-is.
"""

from __future__ import annotations

import re
import os
from typing import Any, Iterable


MODE_SUFFIX_RE = re.compile(r"/(no_think|think)\s*$")
RESPONSE_MARKERS = ("### Response:\n", "### Response:", "### 回答：\n", "### 回答：")


def set_last_user_mode(messages: Iterable[dict[str, Any]], mode: str):
    """Apply `/think` or `/no_think` to the last user turn, matching World20K."""
    if mode == "original":
        return [dict(m) for m in messages]
    rows = [dict(m) for m in messages]
    for message in reversed(rows):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        base = MODE_SUFFIX_RE.sub("", content).rstrip()
        message["content"] = f"{base}/{mode}"
        break
    return rows


def split_prompt_completion_messages(messages, thinking_mode: str = "original"):
    rows = set_last_user_mode(messages, thinking_mode)
    if not rows or rows[-1].get("role") != "assistant":
        raise ValueError("Prompt-completion SFT expects the last message to be assistant.")
    return rows[:-1], [rows[-1]]


def split_prompt_completion_text(text: str):
    for marker in RESPONSE_MARKERS:
        if marker in text:
            head, tail = text.split(marker, 1)
            return head + marker, tail
    raise ValueError("Could not split text sample: missing a supported response marker.")


def prompt_from_instruction(example: dict[str, Any]) -> tuple[str, str]:
    instruction = example.get("instruction") or example.get("input") or ""
    response = example.get("response") or example.get("output") or example.get("target") or ""
    language = example.get("language") or example.get("lang") or example.get("lang_code") or "unknown"
    prompt = f"### Instruction:\n<|tgt_lang:{language}|> {instruction}\n\n### Response:\n"
    return prompt, response


def prepare_dataset_for_trl(dataset, thinking_mode: str = "original", name: str = "train"):
    """Return a TRL-ready dataset with exactly `prompt` and `completion` columns."""
    columns = set(dataset.column_names)

    if {"prompt", "completion"}.issubset(columns):
        drop = [col for col in dataset.column_names if col not in ("prompt", "completion")]
        result = dataset.remove_columns(drop) if drop else dataset
        return filter_by_char_length(result, name)

    if "messages" in columns:
        def _map_messages(batch):
            prompts, completions = [], []
            for messages in batch["messages"]:
                prompt, completion = split_prompt_completion_messages(messages, thinking_mode)
                prompts.append(prompt)
                completions.append(completion)
            return {"prompt": prompts, "completion": completions}

        result = dataset.map(
            _map_messages,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"format_{name}_messages_prompt_completion",
        )
        return filter_by_char_length(result, name)

    if "instruction" in columns and ({"response", "output", "target"} & columns):
        def _map_instruction(batch):
            prompts, completions = [], []
            batch_size = len(batch["instruction"])
            for idx in range(batch_size):
                row = {col: batch[col][idx] for col in batch.keys()}
                prompt, completion = prompt_from_instruction(row)
                prompts.append(prompt)
                completions.append(completion)
            return {"prompt": prompts, "completion": completions}

        result = dataset.map(
            _map_instruction,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"format_{name}_instruction_prompt_completion",
        )
        return filter_by_char_length(result, name)

    if "text" in columns:
        def _map_text(batch):
            prompts, completions = [], []
            for text in batch["text"]:
                prompt, completion = split_prompt_completion_text(text)
                prompts.append(prompt)
                completions.append(completion)
            return {"prompt": prompts, "completion": completions}

        result = dataset.map(
            _map_text,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"format_{name}_text_prompt_completion",
        )
        return filter_by_char_length(result, name)

    raise ValueError(
        "Unsupported SFT dataset schema. Expected messages, prompt/completion, "
        "instruction/response, or text columns; got "
        f"{dataset.column_names}"
    )


def filter_by_char_length(dataset, name: str = "train"):
    """Drop pathological long samples before TRL tokenization.

    Exact token filtering needs the runtime tokenizer, but a character cap catches
    data corruption and multi-megabyte rows early. Set `MAX_TRAIN_CHARS<=0` to
    disable; launchers default it to 200000 so only severe corruption is dropped.
    """
    raw = os.environ.get("MAX_TRAIN_CHARS", "").strip()
    if not raw:
        return dataset
    max_chars = int(raw)
    if max_chars <= 0:
        return dataset

    before = len(dataset)
    result = dataset.filter(
        lambda ex: len((ex.get("prompt") or "")) + len((ex.get("completion") or "")) <= max_chars,
        desc=f"filter_{name}_max_chars_{max_chars}",
    )
    dropped = before - len(result)
    if dropped:
        print(f"[data] {name}: dropped {dropped}/{before} samples over {max_chars} chars")
    return result
