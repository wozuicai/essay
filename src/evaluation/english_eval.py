"""
English benchmark evaluation using lm-evaluation-harness.
Tasks: MMLU, HellaSwag, ARC-Challenge, TruthfulQA-MC1
"""

import json
import os
from typing import Optional

import lm_eval


ENGLISH_TASKS = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc1"]


def _get_donor_adapter(model_path: str) -> Optional[str]:
    """Return donor_adapter path from training_metadata.json if present, else None."""
    meta_path = os.path.join(os.path.abspath(model_path), "training_metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    donor = meta.get("donor_adapter")
    return donor if donor and os.path.isdir(donor) else None


def _apply_donor_adapter(base_model, donor_adapter_path: str):
    """Merge donor LoRA into base_model in-memory and return the merged model."""
    from peft import PeftModel
    base_model = PeftModel.from_pretrained(base_model, donor_adapter_path)
    return base_model.merge_and_unload()


def _lm_eval_model_args(model_path: str, batch_size: int) -> str:
    """Build lm-eval model_args string, handling PEFT adapters transparently.
    Always uses absolute paths so lm-eval doesn't confuse them with HF model IDs.
    """
    abs_path = os.path.abspath(model_path)
    adapter_cfg_path = os.path.join(abs_path, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base = adapter_cfg.get("base_model_name_or_path", abs_path)
        return f"pretrained={base},peft={abs_path},dtype=bfloat16,parallelize=True"
    return f"pretrained={abs_path},dtype=bfloat16,parallelize=True"


def _load_model_tokenizer_for_eval(model_path: str):
    """Load model + tokenizer for manual MCQ evaluation."""
    import torch, os, json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg.get("base_model_name_or_path", model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )
        donor = _get_donor_adapter(model_path)
        if donor:
            base_model = _apply_donor_adapter(base_model, donor)
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )
    model.eval()
    return model, tokenizer


_TAG_PREFIX = "<|tgt_lang:en|>"


def _run_truthfulqa_tagged(model_path: str, model=None, tokenizer=None) -> float:
    """TruthfulQA MC1: full-answer log-likelihood with lang-tag prefix.

    Context      : "<|tgt_lang:en|> Q: {question}\nA: "
    Continuation : raw choice text (no letter prefix)
    Score        : sum of log-probs over continuation tokens (same as lm-eval native)

    This is directly comparable to no-tag lm-eval scores because the scoring
    formula is identical — only the context prefix differs.
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    _owner = model is None
    if _owner:
        model, tokenizer = _load_model_tokenizer_for_eval(model_path)
    dev = next(model.parameters()).device

    correct = 0
    total = 0

    for ex in tqdm(ds, desc="  [TruthfulQA MC1 tag]", leave=False):
        question  = ex["question"]
        choices   = ex["mc1_targets"]["choices"]
        labels    = ex["mc1_targets"]["labels"]
        label_idx = labels.index(1)

        context = f"{_TAG_PREFIX} Q: {question}\nA: "
        ctx_ids = tokenizer.encode(context, return_tensors="pt",
                                   add_special_tokens=True).to(dev)
        ctx_len = ctx_ids.shape[1]

        lls = []
        for choice in choices:
            choice_ids = tokenizer.encode(
                choice, add_special_tokens=False, return_tensors="pt"
            ).to(dev)
            full_ids = torch.cat([ctx_ids, choice_ids], dim=1)

            with torch.no_grad():
                logits = model(full_ids).logits[0]  # [seq, vocab]

            log_probs  = F.log_softmax(logits, dim=-1)
            choice_len = choice_ids.shape[1]
            # logits[i] predicts token[i+1]; choice tokens at ctx_len..ctx_len+choice_len-1
            ll = sum(
                log_probs[ctx_len - 1 + j, full_ids[0, ctx_len + j].item()].item()
                for j in range(choice_len)
            )
            lls.append(ll)

        if int(torch.tensor(lls).argmax()) == label_idx:
            correct += 1
        total += 1

    if _owner:
        del model
        torch.cuda.empty_cache()
    acc = correct / total if total > 0 else 0.0
    print(f"  [TruthfulQA MC1 tag+loglik] {correct}/{total} = {acc:.4f}")
    return acc


def run_english_eval(
    model_path: str,
    tasks: Optional[list] = None,
    batch_size: int = 16,
    num_fewshot: int = 0,
    inject_lang_tag: bool = False,
    model=None,
    tokenizer=None,
) -> dict:
    """
    Run English benchmark evaluation via lm-eval harness.
    Handles both full models and PEFT adapter directories.

    When inject_lang_tag=True and truthfulqa_mc1 is requested, uses
    _run_truthfulqa_tagged (full-answer log-likelihood + tag prefix) so that
    the scoring protocol is identical to the no-tag lm-eval path.

    Returns:
        dict mapping task name -> score (accuracy or normalized accuracy)
    """
    if tasks is None:
        tasks = ENGLISH_TASKS

    scores = {}

    if inject_lang_tag and "truthfulqa_mc1" in tasks:
        scores["truthfulqa_mc1"] = _run_truthfulqa_tagged(model_path, model=model, tokenizer=tokenizer)
        tasks = [t for t in tasks if t != "truthfulqa_mc1"]
        if not tasks:
            scores["english_avg"] = scores["truthfulqa_mc1"]
            return scores

    if tasks:
        if model is not None:
            # Pre-loaded model: wrap with HFLM for lm-eval
            from lm_eval.models.huggingface import HFLM
            lm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
            results = lm_eval.simple_evaluate(
                model=lm_model,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
            )
        else:
            donor = _get_donor_adapter(model_path)
            if donor:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
                from lm_eval.models.huggingface import HFLM
                abs_path = os.path.abspath(model_path)
                with open(os.path.join(abs_path, "adapter_config.json")) as f:
                    base_name = json.load(f).get("base_model_name_or_path", abs_path)
                _tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
                _base = AutoModelForCausalLM.from_pretrained(
                    base_name, torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                _base = _apply_donor_adapter(_base, donor)
                _base = PeftModel.from_pretrained(_base, abs_path)
                lm_model = HFLM(pretrained=_base, tokenizer=_tok, batch_size=batch_size)
                results = lm_eval.simple_evaluate(
                    model=lm_model,
                    tasks=tasks,
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                )
            else:
                model_args = _lm_eval_model_args(model_path, batch_size)
                results = lm_eval.simple_evaluate(
                    model="hf",
                    model_args=model_args,
                    tasks=tasks,
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                    device="cuda",
                )
        for task in tasks:
            task_results = results["results"].get(task, {})
            if "acc_norm,none" in task_results:
                scores[task] = task_results["acc_norm,none"]
            elif "acc,none" in task_results:
                scores[task] = task_results["acc,none"]
            else:
                scores[task] = 0.0

    scores["english_avg"] = (
        sum(scores.get(t, 0.0) for t in ENGLISH_TASKS if t in scores) /
        max(1, sum(1 for t in ENGLISH_TASKS if t in scores))
    )
    return scores


def compute_english_avg(scores: dict) -> float:
    """Compute average across the four English benchmarks."""
    keys = [k for k in ENGLISH_TASKS if k in scores]
    if not keys:
        return 0.0
    return sum(scores[k] for k in keys) / len(keys)
