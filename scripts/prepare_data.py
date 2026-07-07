"""
Data preparation script: download and preprocess SFT data from Aya dataset,
supplement low-resource languages with NLLB translation if needed.
"""

import argparse
import json
import os
import random

from datasets import load_dataset, concatenate_datasets, Dataset

LANGUAGES = {
    'yo': 'Yoruba',
    'so': 'Somali',
    'ha': 'Hausa',
    'en': 'English',
}

# Full Aya counts — no truncation, no NLLB supplement
SAMPLE_SIZES = {
    'yo': 11758,
    'so': 7704,
    'ha': 3512,
    'en': 24926,
}

PROMPT_TEMPLATE = "### Instruction:\n<|tgt_lang:{language}|> {instruction}\n\n### Response:\n{response}"


def prepare_english_data(n_samples: int, seed: int = 42) -> Dataset:
    """Load Open-Platypus English SFT data."""
    dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    def format_sample(example):
        return {
            "instruction": example.get("instruction", ""),
            "response": example.get("output", ""),
            "language": "en",
            "source": "open_platypus",
        }

    return dataset.map(format_sample, remove_columns=dataset.column_names)


def prepare_language_data(lang_code: str, n_samples: int, seed: int = 42) -> Dataset:
    """Load Aya dataset for a given language, supplement with NLLB translation if needed."""
    if lang_code == 'en':
        return prepare_english_data(n_samples, seed)

    lang_name = LANGUAGES[lang_code]
    aya = load_dataset("CohereForAI/aya_dataset", split="train")
    lang_data = aya.filter(lambda x: x['language'] == lang_name)

    def format_aya(example):
        return {
            "instruction": example.get("inputs", ""),
            "response": example.get("targets", ""),
            "language": lang_code,
            "source": "aya",
        }

    lang_data = lang_data.map(format_aya, remove_columns=lang_data.column_names)
    print(f"[{lang_code}] Aya native: {len(lang_data)} samples (using all, no NLLB supplement)")

    lang_data = lang_data.shuffle(seed=seed)
    # n_samples=None or ≥ len → use full dataset
    if n_samples is not None and n_samples < len(lang_data):
        lang_data = lang_data.select(range(n_samples))
    return lang_data


def translate_supplement(target_lang: str, n_needed: int, seed: int = 42) -> Dataset:
    """
    Translate English data from Open-Platypus into target_lang using NLLB-200-3.3B.
    Both instruction and response are translated to maintain language consistency.
    Uses AutoModelForSeq2SeqLM directly (pipeline("translation") removed in transformers 5.x).
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    NLLB_LANG_CODES = {
        'sw': 'swh_Latn',
        'yo': 'yor_Latn',
        'th': 'tha_Thai',
        'bn': 'ben_Beng',
        'fr': 'fra_Latn',
        'zh': 'zho_Hans',
    }

    NLLB_MODEL = "facebook/nllb-200-3.3B"
    nllb_tgt = NLLB_LANG_CODES[target_lang]

    print(f"[NLLB] Loading {NLLB_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    tgt_lang_id = tokenizer.convert_tokens_to_ids(nllb_tgt)
    BATCH_SIZE = 16  # batch decode on H100 for ~10-20x speedup vs single-sample
    print(f"[NLLB] Model loaded. Translating {n_needed} samples to {nllb_tgt} (batch={BATCH_SIZE})...")

    def _translate_batch(texts: list[str]) -> list[str]:
        """Translate a batch of texts (greedy, faster than beam for this use case)."""
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return texts
        idxs, txts = zip(*non_empty)
        inputs = tokenizer(
            list(txts), return_tensors="pt", truncation=True, max_length=512,
            padding=True, src_lang="eng_Latn"
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=256,
                max_length=None,
                do_sample=False,
            )
        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        result = list(texts)
        for i, dec in zip(idxs, decoded):
            result[i] = dec
        return result

    en_data = load_dataset("garage-bAInd/Open-Platypus", split="train")
    en_data = en_data.shuffle(seed=seed).select(range(min(n_needed, len(en_data))))

    translated_samples = []
    examples = list(en_data)
    for batch_start in range(0, len(examples), BATCH_SIZE):
        batch = examples[batch_start:batch_start + BATCH_SIZE]
        instructions = [ex.get("instruction", "") for ex in batch]
        responses = [ex.get("output", "") for ex in batch]
        try:
            trans_instructions = _translate_batch(instructions)
            trans_responses = _translate_batch(responses)
            for instr, resp in zip(trans_instructions, trans_responses):
                translated_samples.append({
                    "instruction": instr,
                    "response": resp,
                    "language": target_lang,
                    "source": "nllb_translated",
                })
        except Exception as e:
            print(f"Translation error at batch {batch_start}: {e}")
            continue
        done = min(batch_start + BATCH_SIZE, len(examples))
        if done % 200 == 0 or done == len(examples):
            print(f"[NLLB] {target_lang}: {done}/{len(examples)} translated")

    del model
    torch.cuda.empty_cache()
    return Dataset.from_list(translated_samples)


def format_for_training(dataset: Dataset) -> Dataset:
    """Apply prompt template to all samples."""
    def apply_template(example):
        text = PROMPT_TEMPLATE.format(
            language=example["language"],
            instruction=example["instruction"],
            response=example["response"],
        )
        return {"text": text, **example}

    return dataset.map(apply_template)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES.keys()))
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for lang_code in args.languages:
        print(f"\n=== Preparing data for: {lang_code} ===")
        n_samples = SAMPLE_SIZES.get(lang_code, 500)
        dataset = prepare_language_data(lang_code, n_samples, seed=args.seed)
        dataset = format_for_training(dataset)

        out_path = os.path.join(args.output_dir, f"{lang_code}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Saved {len(dataset)} samples to {out_path}")


if __name__ == "__main__":
    main()
