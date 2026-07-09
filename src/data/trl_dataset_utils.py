#!/usr/bin/env python3
"""
Dataset utilities for TRL SFT training.

Converts datasets to prompt-completion format for TRL SFTTrainer.
"""

from typing import Optional


def convert_to_prompt_completion(dataset, thinking_mode: str = "original"):
    """
    Convert a dataset with 'text' field to prompt-completion format.

    Assumes the dataset has a 'text' field with full conversation.
    For SFT datasets, typically splits at the last assistant response.

    Args:
        dataset: HuggingFace dataset with 'text' field
        thinking_mode: Mode for thinking tags (not used in our case, kept for compatibility)

    Returns:
        Dataset with 'prompt' and 'completion' fields
    """
    def split_text(example):
        text = example["text"]

        # Try to split at common markers
        if "### Response:" in text:
            parts = text.split("### Response:", 1)
            prompt = parts[0] + "### Response:"
            completion = parts[1] if len(parts) > 1 else ""
        elif "### 回答：" in text:
            parts = text.split("### 回答：", 1)
            prompt = parts[0] + "### 回答："
            completion = parts[1] if len(parts) > 1 else ""
        else:
            # Fallback: use the whole text as completion with empty prompt
            # This shouldn't happen with proper SFT data
            prompt = ""
            completion = text

        return {
            "prompt": prompt.strip(),
            "completion": completion.strip()
        }

    return dataset.map(split_text, desc="Converting to prompt-completion format")


def convert_messages_to_prompt_completion(dataset, thinking_mode: str = "original"):
    """
    Convert a dataset with 'messages' field to prompt-completion format.

    Based on /root/sft_lora/trl_lora_sft.py implementation.

    Args:
        dataset: HuggingFace dataset with 'messages' field (list of dicts with role/content)
        thinking_mode: Mode for thinking tags (not used in our case)

    Returns:
        Dataset with 'prompt' and 'completion' fields
    """
    def _map(batch):
        prompts = []
        completions = []

        for messages in batch["messages"]:
            # Extract prompt (all messages except last) and completion (last message)
            if not messages or messages[-1].get("role") != "assistant":
                # Skip invalid samples
                prompts.append([])
                completions.append([])
                continue

            prompt = messages[:-1]  # All except last
            completion = [messages[-1]]  # Last message only

            prompts.append(prompt)
            completions.append(completion)

        return {"prompt": prompts, "completion": completions}

    # Determine which columns to keep
    keep_cols = dataset.column_names
    result = dataset.map(_map, batched=True, remove_columns=keep_cols,
                        desc="Converting messages to prompt-completion")

    return result


def prepare_dataset_for_trl(dataset, format_type: str = "text"):
    """
    Prepare dataset for TRL SFTTrainer.

    Args:
        dataset: Input dataset
        format_type: Either "text" or "messages" depending on dataset structure

    Returns:
        Dataset ready for TRL SFTTrainer
    """
    if format_type == "text":
        # Dataset has 'text' field with full conversation
        # For TRL with dataset_text_field="text", we can use it directly
        # But for completion_only_loss, we need prompt-completion format
        return convert_to_prompt_completion(dataset)
    elif format_type == "messages":
        # Dataset has 'messages' field (OpenAI format)
        return convert_messages_to_prompt_completion(dataset)
    else:
        raise ValueError(f"Unknown format_type: {format_type}")
