"""IrokoBench components used in the required eval pass.

This module intentionally avoids importing `scripts/eval_extended.py` because
that script initializes GPT judge clients at import time.  The required pass only
needs open/local model scoring for AfriMMLU, AfriXNLI, and AfriMGSM.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from .batched_scoring import (
    batched_generate_text,
    batched_next_token_predictions,
    letter_token_ids,
)


IROKO_LANGS = ["yo", "ha", "so"]
AFRIMMLU_CONFIGS = {"yo": "yor", "ha": "hau"}
AFRIXNLI_CONFIGS = {"yo": "yor", "ha": "hau"}
AFRIMGSM_CONFIGS = {"yo": "yor", "ha": "hau"}
XNLI_CHOICES = ["entailment", "neutral", "contradiction"]


def _correct_idx(answer_key: Any, n_choices: int) -> int | None:
    if isinstance(answer_key, int) and 0 <= answer_key < n_choices:
        return answer_key
    if isinstance(answer_key, str):
        key = answer_key.strip().upper()
        if key in "ABCDE":
            idx = ord(key) - ord("A")
            return idx if idx < n_choices else None
        if key.isdigit():
            idx = int(key)
            return idx if 0 <= idx < n_choices else None
    return None


def _parse_choices(raw) -> list[str]:
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            return list(parsed) if isinstance(parsed, (list, tuple)) else []
        except Exception:
            return []
    return list(raw) if isinstance(raw, (list, tuple)) else []


def eval_afrimmlu(model, tokenizer, inject_lang_tag: bool = False, batch_size: int = 16) -> dict:
    results = {}
    answer_token_ids = letter_token_ids(tokenizer, 5)
    for lang in IROKO_LANGS:
        if lang not in AFRIMMLU_CONFIGS:
            results[lang] = {
                "mcq_accuracy": None,
                "gen_score": None,
                "n_mcq": 0,
                "note": "not in afrimmlu",
            }
            continue

        config = AFRIMMLU_CONFIGS[lang]
        try:
            ds = load_dataset("masakhane/afrimmlu", config, split="test")
        except Exception as exc:
            results[lang] = {"mcq_accuracy": None, "gen_score": None, "n_mcq": 0, "error": str(exc)}
            continue

        prompts = []
        labels = []
        n_choices_per_prompt = []
        for item in tqdm(ds, desc=f"IrokoBench-AfriMMLU-{lang}", leave=False):
            question = item.get("question") or ""
            choices = _parse_choices(item.get("choices"))
            answer = item.get("answer")
            if not question or not choices or answer is None:
                continue
            choice_str = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
            tag_prefix = f"<|tgt_lang:{lang}|> " if inject_lang_tag else ""
            gold_idx = _correct_idx(answer, len(choices))
            if gold_idx is not None:
                prompts.append(f"{tag_prefix}{question}\n{choice_str}\nAnswer:")
                labels.append(gold_idx)
                n_choices_per_prompt.append(len(choices))

        preds = batched_next_token_predictions(
            model,
            tokenizer,
            prompts,
            answer_token_ids,
            batch_size=batch_size,
            num_choices_per_prompt=n_choices_per_prompt,
        )
        correct = [int(pred == label) for pred, label in zip(preds, labels)]

        results[lang] = {
            "mcq_accuracy": round(sum(correct) / len(correct), 4) if correct else None,
            "gen_score": None,
            "n_mcq": len(correct),
        }
    return results


def eval_afrixnli(model, tokenizer, inject_lang_tag: bool = False, batch_size: int = 16) -> dict:
    results = {}
    choice_str = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(XNLI_CHOICES))
    answer_token_ids = letter_token_ids(tokenizer, len(XNLI_CHOICES))
    for lang in IROKO_LANGS:
        if lang not in AFRIXNLI_CONFIGS:
            results[lang] = {"afrixnli_accuracy": None, "n_afrixnli": 0, "note": "not in afrixnli"}
            continue

        config = AFRIXNLI_CONFIGS[lang]
        try:
            ds = load_dataset("masakhane/afrixnli", config, split="test")
        except Exception as exc:
            results[lang] = {"afrixnli_accuracy": None, "n_afrixnli": 0, "error": str(exc)}
            continue

        prompts = []
        labels = []
        for item in tqdm(ds, desc=f"IrokoBench-AfriXNLI-{lang}", leave=False):
            premise = item.get("premise") or ""
            hypothesis = item.get("hypothesis") or ""
            label = item.get("label")
            if not premise or not hypothesis or label is None:
                continue
            tag_prefix = f"<|tgt_lang:{lang}|> " if inject_lang_tag else ""
            prompt = (
                f"{tag_prefix}Premise: {premise}\nHypothesis: {hypothesis}\n"
                f"Question: What is the relationship between the premise and the hypothesis?\n"
                f"{choice_str}\nAnswer:"
            )
            prompts.append(prompt)
            labels.append(int(label))

        preds = batched_next_token_predictions(
            model,
            tokenizer,
            prompts,
            answer_token_ids,
            batch_size=batch_size,
        )
        correct = [int(pred == label) for pred, label in zip(preds, labels)]

        results[lang] = {
            "afrixnli_accuracy": round(sum(correct) / len(correct), 4) if correct else None,
            "n_afrixnli": len(correct),
        }
    return results


def _extract_number(text: str) -> float | None:
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def eval_afrimgsm(model, tokenizer, batch_size: int = 4) -> dict:
    results = {}
    for lang in IROKO_LANGS:
        if lang not in AFRIMGSM_CONFIGS:
            results[lang] = {"afrimgsm_accuracy": None, "n_afrimgsm": 0, "note": "not in afrimgsm"}
            continue

        config = AFRIMGSM_CONFIGS[lang]
        try:
            ds = load_dataset("masakhane/afrimgsm", config, split="test")
        except Exception as exc:
            results[lang] = {"afrimgsm_accuracy": None, "n_afrimgsm": 0, "error": str(exc)}
            continue

        prompts = []
        golds = []
        for item in tqdm(ds, desc=f"IrokoBench-AfriMGSM-{lang}", leave=False):
            question = item.get("question") or ""
            answer_number = item.get("answer_number")
            if not question or answer_number is None:
                continue
            prompts.append(f"{question}\nAnswer with the final numeric answer only.\nAnswer:")
            golds.append(float(answer_number))

        responses = batched_generate_text(
            model,
            tokenizer,
            prompts,
            batch_size=batch_size,
            max_new_tokens=64,
            max_length=1024,
        )
        correct = []
        for response, answer_number in zip(responses, golds):
            pred_num = _extract_number(response)
            correct.append(int(pred_num is not None and abs(pred_num - answer_number) < 1e-4))

        results[lang] = {
            "afrimgsm_accuracy": round(sum(correct) / len(correct), 4) if correct else None,
            "n_afrimgsm": len(correct),
        }
    return results


def run_irokobench_eval(
    model,
    tokenizer,
    inject_lang_tag: bool = False,
    batch_size: int = 16,
    generation_batch_size: int = 4,
) -> dict:
    mcq = eval_afrimmlu(model, tokenizer, inject_lang_tag=inject_lang_tag, batch_size=batch_size)
    xnli = eval_afrixnli(model, tokenizer, inject_lang_tag=inject_lang_tag, batch_size=batch_size)
    mgsm = eval_afrimgsm(model, tokenizer, batch_size=generation_batch_size)

    merged = {}
    for lang in IROKO_LANGS:
        merged[lang] = {
            **(mcq.get(lang) or {}),
            **(xnli.get(lang) or {}),
            **(mgsm.get(lang) or {}),
        }
        if lang == "so":
            merged[lang]["note"] = "not in afrimmlu/afrixnli/afrimgsm"
    return merged
