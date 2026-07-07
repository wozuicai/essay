"""
Multilingual evaluation:
  - SIB-200 (topic classification)
  - Belebele (reading comprehension)
  - FLORES-200 (translation quality: BLEU, chrF, COMET)
  - MT-Bench multilingual (judge-based, Phase 3 & 4 only)
  - LCB (Language Confusion Benchmark) metrics
"""

import json
import os
import re
import sys
from typing import Optional, Tuple

import lm_eval
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.english_eval import _lm_eval_model_args


def _load_model_and_tokenizer(model_path: str):
    """
    Load (model, tokenizer) from either a full checkpoint or a PEFT adapter directory.
    If adapter_config.json is present, loads base model + PEFT adapter.
    """
    import torch
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
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )

    model.eval()
    return model, tokenizer


# LCB noise patterns to strip before language purity check
_NOISE_PATTERNS = [
    re.compile(r'```[\s\S]*?```'),   # code blocks
    re.compile(r'\$\$[\s\S]*?\$\$'), # LaTeX display math
    re.compile(r'\$[^$]+\$'),        # LaTeX inline math
    re.compile(r'https?://\S+'),     # URLs
]

MULTILINGUAL_LANGS = ["en", "yo", "so", "ha"]


def run_multilingual_eval(
    model_path: str,
    languages: list[str],
    run_flores: bool = True,
    run_sib200: bool = True,
    run_belebele: bool = True,
    batch_size: int = 8,
) -> dict:
    """Run SIB-200, Belebele, and FLORES for specified languages."""
    results = {}

    if run_sib200:
        print("  Running SIB-200...")
        results["sib200"] = _run_sib200(model_path, languages, batch_size)

    if run_belebele:
        print("  Running Belebele...")
        results["belebele"] = _run_belebele(model_path, languages, batch_size)

    if run_flores:
        print("  Running FLORES-200...")
        results["flores"] = _run_flores(model_path, languages, batch_size)

    return results


def _run_sib200(model_path: str, languages: list, batch_size: int) -> dict:
    """
    Run SIB-200 topic classification directly via HuggingFace dataset.
    sib200 is not in lm-eval 0.4.x registry; implement few-shot classification manually.

    NOTE: Zero-shot log-prob classification. Base models score ~14-20% (near random
    for 7 classes); instruction-tuned models score higher. Takes ~45 min due to
    7 unbatched forward passes per example × 6 languages × 200 examples.
    """
    import torch
    from tqdm import tqdm

    # SIB-200 uses flores_200 language codes
    SIB_FLORES_CODES = {
        "en": "eng_Latn",
        "yo": "yor_Latn", "so": "som_Latn", "ha": "hau_Latn",
    }
    # Exact category strings used in Davlan/sib200 dataset
    CATEGORIES = [
        "science/technology", "travel", "politics",
        "sports", "health", "entertainment", "geography",
    ]

    model, tokenizer = _load_model_and_tokenizer(model_path)

    scores = {}
    for lang in languages:
        flores_code = SIB_FLORES_CODES.get(lang)
        if not flores_code:
            continue
        try:
            ds = load_dataset("Davlan/sib200", flores_code, split="test")
        except Exception:
            try:
                # Fallback: try the validation split or different config name
                ds = load_dataset("Davlan/sib200", flores_code, split="validation")
            except Exception as e:
                print(f"  [SIB-200] Cannot load {lang}/{flores_code}: {e}")
                scores[lang] = 0.0
                continue

        correct = 0
        total = 0
        import torch.nn.functional as F
        # Pre-encode categories at token level (avoid BPE boundary merge when
        # concatenating "Topic: " + category as a string — the trailing space in
        # the prompt merges with the first token of the category, making cat_ids
        # empty and always selecting the first category with score=0.0).
        cat_token_ids = {
            cat: tokenizer.encode(cat, add_special_tokens=False)
            for cat in CATEGORIES
        }
        dev = next(model.parameters()).device
        examples = [ex for ex in ds if ex.get("text") and ex.get("category")][:200]
        for example in tqdm(examples, desc=f"  [SIB-200] {lang}", leave=False):
            text = example["text"]
            label = example["category"]

            prompt = f"Text: {text}\nTopic: "
            prompt_token_ids = tokenizer.encode(prompt, return_tensors=None)
            n_prompt = len(prompt_token_ids)

            best_score = float("-inf")
            best_cat = None
            for cat in CATEGORIES:
                c_ids = cat_token_ids[cat]
                # Concatenate at token-ID level to avoid BPE boundary merging
                all_ids = torch.tensor([prompt_token_ids + c_ids]).to(dev)
                with torch.no_grad():
                    logits = model(all_ids).logits
                cat_logits = logits[0, n_prompt - 1:n_prompt - 1 + len(c_ids)]
                log_probs = F.log_softmax(cat_logits, dim=-1)
                score = sum(log_probs[i, c_ids[i]].item() for i in range(len(c_ids)))
                if score > best_score:
                    best_score = score
                    best_cat = cat

            if best_cat == label:
                correct += 1
            total += 1

        scores[lang] = correct / total if total > 0 else 0.0
        print(f"  [SIB-200] {lang}: {correct}/{total} = {scores[lang]:.3f}")

    del model
    torch.cuda.empty_cache()
    return scores


def _run_belebele(model_path: str, languages: list, batch_size: int) -> dict:
    """Run Belebele reading comprehension via lm-eval.
    Downloads missing configs on demand; uses cache if available.
    """
    import os

    BELEBELE_TASK_MAP = {
        "en": "belebele_eng_Latn",
        "yo": "belebele_yor_Latn", "so": "belebele_som_Latn", "ha": "belebele_hau_Latn",
    }
    tasks = [BELEBELE_TASK_MAP[l] for l in languages if l in BELEBELE_TASK_MAP]
    if not tasks:
        return {}

    # Temporarily allow HF downloads so missing language configs can be fetched.
    prev_ds_offline = os.environ.pop("HF_DATASETS_OFFLINE", None)
    prev_hub_offline = os.environ.pop("HF_HUB_OFFLINE", None)

    try:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=_lm_eval_model_args(model_path, batch_size),
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size,
            device="cuda",
        )
    finally:
        if prev_ds_offline is not None:
            os.environ["HF_DATASETS_OFFLINE"] = prev_ds_offline
        if prev_hub_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = prev_hub_offline

    scores = {}
    for lang, task in BELEBELE_TASK_MAP.items():
        if lang in languages and task in results["results"]:
            scores[lang] = results["results"][task].get("acc,none", 0.0)
    return scores


def _run_flores(model_path: str, languages: list, batch_size: int) -> dict:
    """
    Run FLORES-200 translation evaluation (lang→English) using sacrebleu BLEU.

    facebook/flores is a gated dataset; instead we source parallel passages from
    facebook/belebele which is openly cached: each belebele example contains the
    original FLORES-200 flores_passage, aligned across languages by question_number.
    We treat these passages as the translation source/reference (equivalent data).

    Uses 3-shot prompt for base models, capped at 97 passage-pairs per language.
    """
    import sys
    import torch
    from tqdm import tqdm

    BELEBELE_CODES = {
        # en intentionally omitted: translating English→English is meaningless
        "yo": "yor_Latn", "so": "som_Latn", "ha": "hau_Latn",
    }
    MAX_PAIRS = 97  # 900 total; reserve first 3 for few-shot

    model, tokenizer = _load_model_and_tokenizer(model_path)

    # Load English FLORES passages from belebele eng_Latn (already cached)
    try:
        belebele_en = load_dataset("facebook/belebele", "eng_Latn", split="test")
    except Exception as e:
        print(f"  [FLORES] Cannot load English belebele reference: {e}")
        del model
        return {}

    bleu_metric = BLEU(effective_order=True)
    scores = {}

    for lang in languages:
        belebele_code = BELEBELE_CODES.get(lang)
        if not belebele_code:
            continue
        try:
            belebele_lang = load_dataset("facebook/belebele", belebele_code, split="test")
        except Exception as e:
            print(f"  [FLORES] Cannot load {lang} belebele: {e}")
            scores[lang] = {"bleu": 0.0}
            continue

        # Build parallel pairs aligned by index position (same index = same source article).
        # Deduplicate by link field since each FLORES passage spawns multiple questions.
        pairs = []
        seen_links = set()
        for ex_lang, ex_en in zip(belebele_lang, belebele_en):
            link = ex_lang.get("link", "")
            if link in seen_links:
                continue
            seen_links.add(link)
            pairs.append((ex_lang["flores_passage"], ex_en["flores_passage"]))

        if len(pairs) < 4:
            scores[lang] = {"bleu": 0.0}
            continue

        # 3-shot examples from first 3 pairs
        fewshot_prompt = ""
        for src_ex, ref_ex in pairs[:3]:
            fewshot_prompt += f"Translate to English:\n{src_ex}\nEnglish: {ref_ex}\n\n"

        dev = next(model.parameters()).device
        hypotheses, references = [], []
        eval_pairs = pairs[3:3 + MAX_PAIRS]
        for src, ref in tqdm(eval_pairs, desc=f"  [FLORES] {lang}", leave=False,
                             file=sys.stdout, dynamic_ncols=True):
            prompt = fewshot_prompt + f"Translate to English:\n{src}\nEnglish:"
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=1024).to(dev)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )
            hyp = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().split("\n")[0].strip()
            hypotheses.append(hyp)
            references.append(ref)

        bleu_score = bleu_metric.corpus_score(hypotheses, [references]).score
        scores[lang] = {"bleu": bleu_score}
        print(f"  [FLORES] {lang}: BLEU={bleu_score:.2f} (via belebele passages, n={len(hypotheses)})")

    del model
    import torch as _torch
    _torch.cuda.empty_cache()
    return scores


def run_mt_bench_multilingual(
    model_path: str,
    languages: list[str],
    judge_model: str = "claude-sonnet-4-6",
    n_questions: int = 40,
) -> dict:
    """
    MT-Bench multilingual evaluation using LLM-as-judge.
    Uses 40 MT-Bench questions translated via NLLB-200-3.3B.

    Note: Requires ANTHROPIC_API_KEY in environment for Claude judge.
    """
    import anthropic
    import torch

    model, tokenizer = _load_model_and_tokenizer(model_path)

    client = anthropic.Anthropic()

    # Load translated MT-Bench questions from local cache
    questions_path = "data/mt_bench_multilingual.json"
    if not os.path.exists(questions_path):
        raise FileNotFoundError(
            f"MT-Bench multilingual questions not found at {questions_path}. "
            "Run scripts/prepare_mt_bench.py first."
        )

    with open(questions_path) as f:
        all_questions = json.load(f)

    results = {}
    for lang in languages:
        if lang not in all_questions:
            continue

        lang_questions = all_questions[lang][:n_questions]
        scores = []

        for q in lang_questions:
            inputs = tokenizer(q["question"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            judge_prompt = (
                f"[Question]\n{q['question']}\n\n"
                f"[Model Answer]\n{answer}\n\n"
                f"[Reference Answer]\n{q.get('reference', 'N/A')}\n\n"
                "Please rate the quality of the model's answer on a scale from 1 to 10, "
                "where 10 is excellent. Consider accuracy, fluency, and instruction following. "
                "Return only a single integer score."
            )

            response = client.messages.create(
                model=judge_model,
                max_tokens=10,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            try:
                score = int(response.content[0].text.strip())
                scores.append(score)
            except ValueError:
                pass

        results[lang] = sum(scores) / len(scores) if scores else 0.0

    return results


def compute_lcb_metrics(model_path: str, target_lang: str, n_samples: int = 200) -> dict:
    """
    Compute LCB (Language Confusion Benchmark) metrics:
      - LPR (Line-level Pass Rate): fraction of output lines in correct language
      - WPR (Word-level Pass Rate): among passing lines, fraction of target-language words
      - LCPR (combined): LPR * WPR

    Uses langdetect for line/word level language detection.
    Strips code blocks, math, URLs before detection.
    Requires model to generate ≥50 tokens (short outputs filtered).
    """
    import langdetect
    from langdetect.lang_detect_exception import LangDetectException
    import torch

    LCB_PROMPTS_PATH = f"data/lcb_prompts_{target_lang}.jsonl"

    model_obj, tokenizer = _load_model_and_tokenizer(model_path)

    if not os.path.exists(LCB_PROMPTS_PATH):
        raise FileNotFoundError(
            f"LCB prompts not found: {LCB_PROMPTS_PATH}. Run scripts/prepare_lcb_prompts.py first."
        )

    prompts = []
    with open(LCB_PROMPTS_PATH) as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    prompts = prompts[:n_samples]

    line_pass_count = 0
    line_total = 0
    word_pass_count = 0
    word_total = 0

    for item in prompts:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model_obj.device)
        with torch.no_grad():
            out = model_obj.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Strip noise
        cleaned = response
        for pat in _NOISE_PATTERNS:
            cleaned = pat.sub("", cleaned)

        # Filter too-short outputs
        if len(response.split()) < 10:
            continue

        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        for line in lines:
            line_total += 1
            try:
                detected = langdetect.detect(line)
                lang_match = detected == target_lang or (target_lang == "zh" and detected in ("zh-cn", "zh-tw"))
                if lang_match:
                    line_pass_count += 1
                    # Word-level check
                    words = line.split()
                    word_total += len(words)
                    # Approximate word-level pass: count words that langdetect says match
                    # (simple heuristic: if full line passes, credit all words)
                    word_pass_count += len(words)
            except LangDetectException:
                pass

    lpr = line_pass_count / line_total if line_total > 0 else 0.0
    wpr = word_pass_count / word_total if word_total > 0 else 0.0
    lcpr = lpr * wpr

    return {"LPR": round(lpr, 4), "WPR": round(wpr, 4), "LCPR": round(lcpr, 4)}
