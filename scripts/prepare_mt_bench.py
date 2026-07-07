"""
Prepare multilingual MT-Bench question set for Phase 3/4 evaluation.

Downloads MT-Bench question bank, filters to 40 non-math/non-coding questions,
translates them to fr/zh/sw/th/bn using NLLB-200-3.3B, and saves to
data/mt_bench_multilingual.json.

Expected output format:
{
  "fr": [{"question": "...", "reference": "...", "category": "..."}, ...],
  "zh": [...],
  ...
}
"""

import json
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Questions from categories we include (no math, no coding)
INCLUDE_CATEGORIES = {"writing", "roleplay", "reasoning", "extraction", "stem", "humanities"}

NLLB_LANG_CODES = {
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "sw": "swh_Latn",
    "th": "tha_Thai",
    "bn": "ben_Beng",
}

TARGET_LANGUAGES = list(NLLB_LANG_CODES.keys())
NLLB_MODEL = "facebook/nllb-200-3.3B"
N_QUESTIONS = 40


def load_mt_bench_questions() -> list[dict]:
    """Load MT-Bench questions from HuggingFace, filter out math/coding."""
    print("Loading MT-Bench questions from HuggingFace...")
    try:
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        questions = []
        for row in ds:
            category = row.get("category", "").lower()
            if category in INCLUDE_CATEGORIES:
                # Use the first turn as the question
                turns = row.get("turns", [])
                if turns:
                    questions.append({
                        "question_id": row.get("question_id", len(questions)),
                        "category": category,
                        "question": turns[0],
                        "reference": row.get("reference_answer", turns[-1] if len(turns) > 1 else ""),
                    })
    except Exception as e:
        print(f"HuggingFaceH4/mt_bench_prompts failed ({e}), trying lm-sys/mt_bench_human_judgments...")
        ds = load_dataset("lm-sys/mt_bench_human_judgments", split="human")
        seen_ids = set()
        questions = []
        for row in ds:
            qid = row.get("question_id")
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            category = row.get("category", "").lower()
            if category in INCLUDE_CATEGORIES:
                question_text = row.get("question", row.get("prompt", ""))
                if isinstance(question_text, list):
                    question_text = question_text[0]
                if question_text:
                    questions.append({
                        "question_id": qid,
                        "category": category,
                        "question": question_text,
                        "reference": row.get("reference", ""),
                    })

    print(f"Found {len(questions)} questions in allowed categories.")
    if not questions:
        # Fallback: take all questions regardless of category (filter too strict or field missing)
        print("WARNING: 0 questions matched categories — relaxing filter to accept all.")
        try:
            ds2 = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
            questions = []
            for row in ds2:
                turns = row.get("turns", row.get("prompt", []))
                if isinstance(turns, str):
                    turns = [turns]
                category = row.get("category", row.get("category_tag", "unknown")).lower()
                if turns:
                    questions.append({
                        "question_id": row.get("question_id", len(questions)),
                        "category": category,
                        "question": turns[0] if isinstance(turns, list) else turns,
                        "reference": row.get("reference_answer", ""),
                    })
            # Filter out math/coding if we have enough
            if len(questions) > N_QUESTIONS:
                EXCLUDE = {"math", "coding"}
                filtered = [q for q in questions if q["category"] not in EXCLUDE]
                questions = filtered if filtered else questions
            print(f"  Relaxed filter: {len(questions)} questions available")
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
    # Balance categories, up to N_QUESTIONS total
    from collections import defaultdict
    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q)
    selected = []
    if by_cat:
        per_cat = max(1, N_QUESTIONS // len(by_cat))
        for cat_qs in by_cat.values():
            selected.extend(cat_qs[:per_cat])
        selected = selected[:N_QUESTIONS]
    else:
        selected = questions[:N_QUESTIONS]
    print(f"Selected {len(selected)} questions across {len(by_cat)} categories.")
    return selected


def translate_questions(questions: list[dict], target_lang: str,
                        model, tokenizer) -> list[dict]:
    """Translate question and reference fields to target_lang."""
    tgt_lang_id = tokenizer.convert_tokens_to_ids(NLLB_LANG_CODES[target_lang])

    def _translate(text: str) -> str:
        if not text or not text.strip():
            return text
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
            src_lang="eng_Latn"
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=512,
                num_beams=4,
            )
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)

    translated = []
    for i, q in enumerate(questions):
        t_question = _translate(q["question"])
        t_reference = _translate(q["reference"]) if q.get("reference") else ""
        translated.append({
            "question_id": q["question_id"],
            "category": q["category"],
            "question": t_question,
            "reference": t_reference,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{target_lang}] {i+1}/{len(questions)} translated")
    return translated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=TARGET_LANGUAGES)
    parser.add_argument("--output", default="data/mt_bench_multilingual.json")
    parser.add_argument("--n_questions", type=int, default=N_QUESTIONS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    questions = load_mt_bench_questions()
    questions = questions[:args.n_questions]

    print(f"\nLoading NLLB model: {NLLB_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("NLLB model loaded.\n")

    result = {}
    for lang in args.languages:
        if lang not in NLLB_LANG_CODES:
            print(f"Skipping {lang}: no NLLB code defined")
            continue
        print(f"Translating to {lang}...")
        result[lang] = translate_questions(questions, lang, model, tokenizer)
        print(f"  Done: {len(result[lang])} questions for {lang}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in result.values())
    print(f"\nSaved {total} translated questions to {args.output}")
    print(f"Languages: {list(result.keys())}")


if __name__ == "__main__":
    main()
