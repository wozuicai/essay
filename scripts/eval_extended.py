"""
扩展评测脚本：AfriQA / Aya Evaluation / IrokoBench / Uhura-TruthfulQA
- 对一个模型（base 或 SFT adapter）跑各 benchmark
- 结果写入已有 result JSON 文件的 scores.multilingual 下，不新建文件
用法：
  python3 eval_extended.py --model_path <path> --result_json <path.json>
  python3 eval_extended.py --model_path <path> --result_json <path.json> --only_uhura_truthfulqa
"""

import argparse
import json
import os
import re
import sys
import time

import ast
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, "/root")
from config import MODEL_CONFIG

BASE_MODEL_PATH = "/root/project/models/Qwen3.5-9B-Base"

# ── GPT-5.4 judge client ──────────────────────────────────────────────────────

GPT_MODEL_ID = "gpt-5.4-2026-03-05"
_gpt_cfg = MODEL_CONFIG[GPT_MODEL_ID]

try:
    from openai import AzureOpenAI
    _gpt_client = AzureOpenAI(
        api_key=_gpt_cfg["api_key"],
        azure_endpoint=_gpt_cfg["BASE_URL"],
        api_version=_gpt_cfg["API_VERSION"],
    )
except Exception as _e:
    print(f"[warn] AzureOpenAI init failed ({_e}), falling back to OpenAI client")
    from openai import OpenAI
    _gpt_client = OpenAI(
        api_key=_gpt_cfg["api_key"],
        base_url=_gpt_cfg["BASE_URL"],
    )

_N_PARALLEL_WORKERS = int(os.environ.get("EVAL_PARALLEL_WORKERS", "1"))
_QPM_DELAY = 60.0 / _gpt_cfg["qpm"] * _N_PARALLEL_WORKERS   # split shared QPM budget across concurrent processes


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str):
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.evaluation.english_eval import _get_donor_adapter, _apply_donor_adapter

    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg.get("base_model_name_or_path", BASE_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )
        donor = _get_donor_adapter(model_path)
        if donor:
            base_model = _apply_donor_adapter(base_model, donor)
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )
    model.eval()
    return model, tokenizer


# ── Generation helper ─────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ── GPT judge ─────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an impartial judge. Evaluate the quality of the AI assistant's response "
    "to the user question, considering helpfulness, relevance, accuracy, and detail. "
    "Give a short explanation, then rate on a scale of 1 to 10 using this exact format: "
    "\"Rating: [[N]]\"."
)

def gpt_judge(question: str, answer: str, retries: int = 3) -> float | None:
    user_msg = f"[Question]\n{question}\n\n[Assistant's Answer]\n{answer}"
    for attempt in range(retries):
        try:
            resp = _gpt_client.chat.completions.create(
                model=GPT_MODEL_ID,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=256,
                temperature=0,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
            if m:
                return float(m.group(1))
            print(f"[warn] judge parse failed: {text[:80]}")
        except Exception as e:
            print(f"[warn] GPT judge error attempt {attempt+1}: {e}")
            time.sleep(5 * (attempt + 1))
    return None


# ── F1 / EM helpers ───────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    import string
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_f1_em(pred: str, gold: str):
    p_norm, g_norm = _normalize(pred), _normalize(gold)
    em = int(p_norm == g_norm)
    p_toks, g_toks = p_norm.split(), g_norm.split()
    if not p_toks or not g_toks:
        return float(em), em
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0, em
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    f1   = 2 * prec * rec / (prec + rec)
    return f1, em


# ── AfriQA ────────────────────────────────────────────────────────────────────
# masakhane/afriqa — open-domain QA, F1/EM, no GPT judge
# Languages: yo (Yoruba), ha (Hausa); Somali not available
# datasets==5.0.0 removed script-based loading; use the HF-auto-converted parquet ref

AFRIQA_CONFIGS = {"yo": "yor", "ha": "hau"}

def _load_afriqa_split(lang_code: str):
    url = f"hf://datasets/masakhane/afriqa@refs%2Fconvert%2Fparquet/{lang_code}/test/0000.parquet"
    return load_dataset("parquet", data_files=url, split="train")

def eval_afriqa(model, tokenizer) -> dict:
    results = {}
    for iso, lang_code in AFRIQA_CONFIGS.items():
        print(f"\n[AfriQA] lang={iso} ({lang_code})")
        try:
            ds = _load_afriqa_split(lang_code)
        except Exception as e:
            print(f"  [error] {e}")
            results[iso] = {"f1": None, "exact_match": None, "error": str(e)}
            continue

        f1s, ems = [], []

        for item in tqdm(ds, desc=f"AfriQA-{iso}"):
            question = item.get("question") or ""
            context  = item.get("context") or item.get("passage") or ""
            answers  = item.get("answers")
            if isinstance(answers, str):
                try:
                    answers = ast.literal_eval(answers)
                except Exception:
                    answers = [answers]
            if isinstance(answers, dict):
                golds = answers.get("text") or []
            elif isinstance(answers, list):
                golds = answers
            elif answers:
                golds = [str(answers)]
            else:
                golds = []
            if not golds or not question:
                continue

            prompt = (
                f"Context: {context}\nQuestion: {question}\nAnswer:"
                if context else
                f"Question: {question}\nAnswer:"
            )
            pred = generate(model, tokenizer, prompt, max_new_tokens=64)
            pred = pred.strip().split("\n")[0]   # first line only

            best_f1 = max(compute_f1_em(pred, g)[0] for g in golds)
            best_em = max(compute_f1_em(pred, g)[1] for g in golds)
            f1s.append(best_f1)
            ems.append(best_em)

        res = {
            "f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "exact_match": round(sum(ems) / len(ems), 4) if ems else None,
            "n": len(f1s),
        }
        print(f"  F1={res['f1']}  EM={res['exact_match']}  n={res['n']}")
        results[iso] = res

    return results


# ── Aya Evaluation ────────────────────────────────────────────────────────────
# CohereLabs/aya_evaluation_suite (config: dolly_machine_translated) — generative,
# GPT-5.4 judge 1-10. Languages: en, yo, so, ha (ISO codes eng/yor/som/hau)

AYA_DATASET_ID = "CohereLabs/aya_evaluation_suite"
AYA_CONFIG = "dolly_machine_translated"
AYA_LANG_MAP = {"en": "eng", "yo": "yor", "so": "som", "ha": "hau"}

def eval_aya(model, tokenizer) -> dict:
    print("\n[Aya Evaluation] loading dataset...")
    try:
        ds_full = load_dataset(AYA_DATASET_ID, AYA_CONFIG, split="test")
    except Exception as e:
        print(f"  [error] {e}")
        return {lang: {"gpt_score": None, "error": str(e)} for lang in AYA_LANG_MAP}

    lang_col = "language" if "language" in ds_full.column_names else None
    print(f"  columns: {ds_full.column_names}  lang_col={lang_col}")
    if lang_col:
        available_langs = set(ds_full[lang_col])
        print(f"  available languages (sample): {list(available_langs)[:20]}")

    results = {}
    for iso, lang_code in AYA_LANG_MAP.items():
        print(f"\n[Aya Evaluation] lang={iso}")
        if lang_col:
            lang_ds = ds_full.filter(lambda x: x[lang_col] == lang_code)
        else:
            lang_ds = ds_full
        if len(lang_ds) == 0:
            print(f"  [skip] no data for {lang_code}")
            results[iso] = {"gpt_score": None, "n": 0, "note": "no data"}
            continue

        # Detect instruction column
        instr_col = None
        for col in ["inputs", "instruction", "prompt", "question", "input"]:
            if col in lang_ds.column_names:
                instr_col = col
                break
        if instr_col is None:
            print(f"  [skip] cannot find instruction column in {lang_ds.column_names}")
            results[iso] = {"gpt_score": None, "n": 0, "note": "no instruction column"}
            continue

        scores = []
        for item in tqdm(lang_ds, desc=f"Aya-{iso}"):
            instruction = item[instr_col] or ""
            if not instruction:
                continue
            pred = generate(model, tokenizer, instruction, max_new_tokens=256)
            score = gpt_judge(instruction, pred)
            if score is not None:
                scores.append(score)
            time.sleep(_QPM_DELAY)

        res = {
            "gpt_score": round(sum(scores) / len(scores), 4) if scores else None,
            "n": len(scores),
        }
        print(f"  GPT score={res['gpt_score']}  n={res['n']}")
        results[iso] = res

    return results


# ── IrokoBench ────────────────────────────────────────────────────────────────
# IrokoBench (Adelani et al. 2024) is not a single unified HF dataset; using its
# afrimmlu (MMLU-style MCQ) component, hosted as masakhane/afrimmlu.
# Languages: yo, ha; Somali confirmed NOT in afrimmlu/afrixnli/afrimgsm -> so=None

IROKO_LANGS = ["yo", "ha", "so"]
IROKO_DATASET_ID = "masakhane/afrimmlu"
IROKO_CONFIGS = {"yo": "yor", "ha": "hau"}   # so: not available

def _score_mcq_logprob(model, tokenizer, prompt: str, choices: list[str]) -> int:
    """Return index of highest-scoring choice via next-token log-prob."""
    device = next(model.parameters()).device
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    n_prompt = enc_prompt["input_ids"].shape[1]

    with torch.no_grad():
        out = model(**enc_prompt)
        logits_next = out.logits[0, n_prompt - 1, :]   # logits for next token

    best_idx, best_score = 0, float("-inf")
    for i, choice_letter in enumerate(["A", "B", "C", "D", "E"][: len(choices)]):
        for surface in [f" {choice_letter}", choice_letter, f"{choice_letter}."]:
            tids = tokenizer.encode(surface, add_special_tokens=False)
            if tids:
                score = logits_next[tids[0]].item()
                if score > best_score:
                    best_score = score
                    best_idx = i
                break
    return best_idx

def _get_correct_idx(answer_key, n_choices: int) -> int | None:
    if isinstance(answer_key, int) and 0 <= answer_key < n_choices:
        return answer_key
    if isinstance(answer_key, str):
        ak = answer_key.strip().upper()
        if ak in "ABCDE":
            return ord(ak) - ord("A")
        if ak.isdigit():
            return int(ak)
    return None

def eval_irokobench(model, tokenizer, inject_lang_tag: bool = False) -> dict:
    print("\n[IrokoBench/afrimmlu] loading dataset...")
    results = {}

    for iso in IROKO_LANGS:
        if iso not in IROKO_CONFIGS:
            print(f"\n[IrokoBench] lang={iso}  [skip] not available in afrimmlu (e.g. Somali)")
            results[iso] = {"mcq_accuracy": None, "gen_score": None, "note": "not in afrimmlu/afrixnli/afrimgsm"}
            continue

        config_name = IROKO_CONFIGS[iso]
        print(f"\n[IrokoBench] lang={iso} ({config_name})")
        try:
            mcq_ds = load_dataset(IROKO_DATASET_ID, config_name, split="test")
        except Exception as e:
            print(f"  [error] {e}")
            results[iso] = {"mcq_accuracy": None, "gen_score": None, "error": str(e)}
            continue

        print(f"  columns: {mcq_ds.column_names}")
        mcq_correct = []

        for item in tqdm(mcq_ds, desc=f"IrokoBench-{iso}"):
            question = item.get("question") or ""
            choices  = item.get("choices")
            if isinstance(choices, str):
                try:
                    choices = ast.literal_eval(choices)
                except Exception:
                    choices = []
            choices = choices or []
            answer  = item.get("answer")
            if not question or not choices or answer is None:
                continue

            choice_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            tag_prefix = f"<|tgt_lang:{iso}|> " if inject_lang_tag else ""
            prompt = f"{tag_prefix}{question}\n{choice_str}\nAnswer:"
            pred_idx = _score_mcq_logprob(model, tokenizer, prompt, choices)
            correct_idx = _get_correct_idx(answer, len(choices))
            if correct_idx is not None:
                mcq_correct.append(int(pred_idx == correct_idx))

        mcq_acc = round(sum(mcq_correct) / len(mcq_correct), 4) if mcq_correct else None
        print(f"  MCQ accuracy={mcq_acc}  n={len(mcq_correct)}")

        results[iso] = {
            "mcq_accuracy": mcq_acc,
            "gen_score":    None,   # afrimmlu test split is pure MCQ, no generation component
            "n_mcq":        len(mcq_correct),
        }

    return results


# ── IrokoBench: AfriXNLI (3-way NLI) ────────────────────────────────────────────
# masakhane/afrixnli, label: 0=entailment, 1=neutral, 2=contradiction
# Languages: yo, ha; Somali not present (same as afrimmlu).

IROKO_XNLI_DATASET_ID = "masakhane/afrixnli"
IROKO_XNLI_CONFIGS = {"yo": "yor", "ha": "hau"}
IROKO_XNLI_CHOICES = ["entailment", "neutral", "contradiction"]

def eval_irokobench_afrixnli(model, tokenizer, inject_lang_tag: bool = False) -> dict:
    print("\n[IrokoBench/afrixnli] loading dataset...")
    results = {}

    for iso in IROKO_LANGS:
        if iso not in IROKO_XNLI_CONFIGS:
            print(f"\n[IrokoBench/afrixnli] lang={iso}  [skip] not available")
            results[iso] = {"afrixnli_accuracy": None, "note": "not in afrixnli"}
            continue

        config_name = IROKO_XNLI_CONFIGS[iso]
        print(f"\n[IrokoBench/afrixnli] lang={iso} ({config_name})")
        try:
            ds = load_dataset(IROKO_XNLI_DATASET_ID, config_name, split="test")
        except Exception as e:
            print(f"  [error] {e}")
            results[iso] = {"afrixnli_accuracy": None, "error": str(e)}
            continue

        choice_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(IROKO_XNLI_CHOICES))
        correct = []
        for item in tqdm(ds, desc=f"AfriXNLI-{iso}"):
            premise = item.get("premise") or ""
            hypothesis = item.get("hypothesis") or ""
            label = item.get("label")
            if not premise or not hypothesis or label is None:
                continue
            tag_prefix = f"<|tgt_lang:{iso}|> " if inject_lang_tag else ""
            prompt = (
                f"{tag_prefix}Premise: {premise}\nHypothesis: {hypothesis}\n"
                f"Question: What is the relationship between the premise and the hypothesis?\n"
                f"{choice_str}\nAnswer:"
            )
            pred_idx = _score_mcq_logprob(model, tokenizer, prompt, IROKO_XNLI_CHOICES)
            correct.append(int(pred_idx == int(label)))

        acc = round(sum(correct) / len(correct), 4) if correct else None
        print(f"  AfriXNLI accuracy={acc}  n={len(correct)}")
        results[iso] = {"afrixnli_accuracy": acc, "n_afrixnli": len(correct)}

    return results


# ── IrokoBench: AfriMGSM (grade-school math, generative) ───────────────────────
# masakhane/afrimgsm, fields: question, answer_number (gold numeric answer)
# Languages: yo, ha; Somali not present (same as afrimmlu).

IROKO_MGSM_DATASET_ID = "masakhane/afrimgsm"
IROKO_MGSM_CONFIGS = {"yo": "yor", "ha": "hau"}

def _extract_number(text: str) -> float | None:
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None

def eval_irokobench_afrimgsm(model, tokenizer) -> dict:
    print("\n[IrokoBench/afrimgsm] loading dataset...")
    results = {}

    for iso in IROKO_LANGS:
        if iso not in IROKO_MGSM_CONFIGS:
            print(f"\n[IrokoBench/afrimgsm] lang={iso}  [skip] not available")
            results[iso] = {"afrimgsm_accuracy": None, "note": "not in afrimgsm"}
            continue

        config_name = IROKO_MGSM_CONFIGS[iso]
        print(f"\n[IrokoBench/afrimgsm] lang={iso} ({config_name})")
        try:
            ds = load_dataset(IROKO_MGSM_DATASET_ID, config_name, split="test")
        except Exception as e:
            print(f"  [error] {e}")
            results[iso] = {"afrimgsm_accuracy": None, "error": str(e)}
            continue

        correct = []
        for item in tqdm(ds, desc=f"AfriMGSM-{iso}"):
            question = item.get("question") or ""
            answer_number = item.get("answer_number")
            if not question or answer_number is None:
                continue
            prompt = f"{question}\nAnswer with the final numeric answer only.\nAnswer:"
            resp = generate(model, tokenizer, prompt, max_new_tokens=64)
            pred_num = _extract_number(resp)
            correct.append(int(pred_num is not None and abs(pred_num - float(answer_number)) < 1e-4))

        acc = round(sum(correct) / len(correct), 4) if correct else None
        print(f"  AfriMGSM accuracy={acc}  n={len(correct)}")
        results[iso] = {"afrimgsm_accuracy": acc, "n_afrimgsm": len(correct)}

    return results


# ── Uhura-TruthfulQA ──────────────────────────────────────────────────────────
# ebayes/uhura-truthfulqa — MC1 log-likelihood scoring, langs: yo, ha
# MC1: pick the choice with highest conditional log-likelihood; acc = fraction correct

UHURA_LANGS = ["yo", "ha"]

def _batch_loglikelihoods(
    model,
    tokenizer,
    prompts: list[str],
    continuations: list[str],
    batch_size: int = 32,
) -> list[float]:
    """Return sum-of-log-probs for each (prompt, continuation) pair (batched)."""
    device = next(model.parameters()).device
    scores = []
    for start in range(0, len(prompts), batch_size):
        batch_p = prompts[start : start + batch_size]
        batch_c = continuations[start : start + batch_size]

        full_texts = [p + c for p, c in zip(batch_p, batch_c)]
        enc_full = tokenizer(full_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(device)
        enc_prefix = tokenizer(batch_p, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)

        with torch.no_grad():
            logits = model(**enc_full).logits  # (B, L, V)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, L, V)

        for i in range(len(batch_p)):
            prefix_len = enc_prefix["attention_mask"][i].sum().item()
            full_ids   = enc_full["input_ids"][i]
            full_mask  = enc_full["attention_mask"][i]
            # continuation token positions: prefix_len .. end (masked)
            score = 0.0
            n = 0
            for pos in range(prefix_len, full_mask.sum().item()):
                token_id = full_ids[pos].item()
                score += log_probs[i, pos - 1, token_id].item()
                n += 1
            scores.append(score / max(n, 1))  # length-normalised
    return scores


def eval_uhura_truthfulqa(model, tokenizer, batch_size: int = 32) -> dict:
    """MC1 accuracy on Uhura-TruthfulQA for yo and ha."""
    results = {}
    for lang in UHURA_LANGS:
        config = f"{lang}_multiple_choice"
        print(f"\n[Uhura-TruthfulQA] lang={lang} config={config}")
        try:
            ds = load_dataset("ebayes/uhura-truthfulqa", config, split="test")
        except Exception as e:
            print(f"  [error] {e}")
            results[lang] = {"mc1_accuracy": None, "error": str(e)}
            continue

        correct = 0
        total = 0
        for item in tqdm(ds, desc=f"Uhura-TFQ-{lang}"):
            question = item["question"]
            mc1 = item["mc1_targets"]
            choices = mc1["choices"]
            labels  = mc1["labels"]
            if not choices:
                continue

            # correct answer = the unique choice where label == 1
            correct_idx = next((i for i, l in enumerate(labels) if l == 1), None)
            if correct_idx is None:
                continue

            prefix = f"Question: {question}\nAnswer:"
            prompts = [prefix] * len(choices)
            continuations = [" " + c for c in choices]

            ll_scores = _batch_loglikelihoods(model, tokenizer, prompts,
                                              continuations, batch_size=batch_size)
            pred_idx = int(max(range(len(ll_scores)), key=lambda i: ll_scores[i]))
            if pred_idx == correct_idx:
                correct += 1
            total += 1

        acc = round(correct / total, 4) if total > 0 else None
        print(f"  MC1 accuracy={acc}  n={total}")
        results[lang] = {"mc1_accuracy": acc, "n": total}

    return results


# ── JSON update ───────────────────────────────────────────────────────────────

def update_result_json_partial(json_path: str, key: str, value: dict):
    """Write a single benchmark's results into the JSON immediately, so progress
    is visible without waiting for the other benchmarks to finish."""
    with open(json_path) as f:
        data = json.load(f)
    ml = data.setdefault("scores", {}).setdefault("multilingual", {})
    ml[key] = value
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[partial-write] {key} -> {json_path}")


def update_result_json_merge_iroko(json_path: str, extra: dict):
    """Merge AfriXNLI/AfriMGSM results into the existing irokobench dict per
    language, without disturbing the AfriMMLU fields already written there."""
    with open(json_path) as f:
        data = json.load(f)
    ml = data.setdefault("scores", {}).setdefault("multilingual", {})
    iroko = ml.setdefault("irokobench", {})
    for lang, vals in extra.items():
        if not isinstance(iroko.get(lang), dict):
            iroko[lang] = {}
        iroko[lang].update(vals)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[partial-write] irokobench-extra -> {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--skip_afriqa", action="store_true")
    parser.add_argument("--skip_aya",    action="store_true")
    parser.add_argument("--skip_iroko",  action="store_true")
    parser.add_argument("--only_iroko_extra", action="store_true",
        help="只跑 AfriXNLI/AfriMGSM，合并写入已有 irokobench 结果（不覆盖 afrimmlu 部分）")
    parser.add_argument("--only_uhura_truthfulqa", action="store_true",
        help="只跑 Uhura-TruthfulQA MC1（yo+ha），写入 scores.multilingual.uhura_truthfulqa")
    parser.add_argument("--uhura_batch_size", type=int, default=32,
        help="Uhura-TruthfulQA 的 log-likelihood 计算 batch size（默认 32）")
    parser.add_argument("--inject_lang_tag", action="store_true", default=False,
        help="Inject <|tgt_lang:xx|> into MCQ prompts")
    parser.add_argument("--only_iroko_mcq", action="store_true",
        help="只跑 IrokoBench MCQ（afrimmlu yo+ha），合并覆盖已有 irokobench.mcq_accuracy 字段")
    args = parser.parse_args()

    print(f"=== eval_extended: {args.model_path} ===")
    if not Path(args.result_json).exists():
        print(f"[error] result JSON not found: {args.result_json}")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    if args.only_iroko_extra:
        xnli = eval_irokobench_afrixnli(model, tokenizer, inject_lang_tag=args.inject_lang_tag)
        mgsm = eval_irokobench_afrimgsm(model, tokenizer)
        merged = {iso: {**xnli.get(iso, {}), **mgsm.get(iso, {})} for iso in IROKO_LANGS}
        update_result_json_merge_iroko(args.result_json, merged)
        print(f"\n[done] iroko-extra finished for {args.model_path}")
        return

    if args.only_uhura_truthfulqa:
        uhura = eval_uhura_truthfulqa(model, tokenizer, batch_size=args.uhura_batch_size)
        update_result_json_partial(args.result_json, "uhura_truthfulqa",
            {lang: uhura.get(lang) for lang in UHURA_LANGS})
        print(f"\n[done] Uhura-TruthfulQA finished for {args.model_path}")
        return

    if args.only_iroko_mcq:
        mcq = eval_irokobench(model, tokenizer, inject_lang_tag=args.inject_lang_tag)
        # 只把 mcq_accuracy 和 n_mcq 合并写入，保留已有的 afrixnli/afrimgsm 字段
        update_result_json_merge_iroko(args.result_json,
            {iso: {"mcq_accuracy": mcq[iso]["mcq_accuracy"],
                   "n_mcq":        mcq[iso].get("n_mcq")}
             for iso in ["yo", "ha"] if iso in mcq})
        print(f"\n[done] IrokoBench MCQ finished for {args.model_path}")
        return

    if not args.skip_afriqa:
        afriqa = eval_afriqa(model, tokenizer)
        update_result_json_partial(args.result_json, "afriqa", {
            "yo": afriqa.get("yo"), "ha": afriqa.get("ha"), "so": None,
        })

    if not args.skip_aya:
        aya = eval_aya(model, tokenizer)
        update_result_json_partial(args.result_json, "aya_evaluation",
            {lang: aya.get(lang) for lang in ["en", "yo", "so", "ha"]})

    if not args.skip_iroko:
        iroko = eval_irokobench(model, tokenizer)
        update_result_json_partial(args.result_json, "irokobench",
            {lang: iroko.get(lang) for lang in ["yo", "ha", "so"]})

    print(f"\n[done] all benchmarks finished for {args.model_path}")


if __name__ == "__main__":
    main()
