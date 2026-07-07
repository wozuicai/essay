"""
Prepare Language Confusion Benchmark (LCB) prompt sets per language.

For each target language, builds a set of ~200 prompts that a model should answer
in the target language. Sources:
  1. Existing Aya dataset instructions (already in target language) — primary
  2. FLORES-200 source sentences used as open-ended questions (fallback)

Output: data/lcb_prompts_{lang}.jsonl, one JSON object per line:
  {"prompt": "<instruction in target language>", "language": "<lang_code>"}

Usage:
  python scripts/prepare_lcb_prompts.py --languages fr zh sw th bn yo
  python scripts/prepare_lcb_prompts.py --languages sw --n_prompts 200
"""

import json
import os
import argparse
from datasets import load_dataset

AYA_LANG_NAMES = {
    "fr": "French",
    "zh": "Simplified Chinese",
    "sw": "Swahili",
    "th": "Thai",
    "bn": "Bengali",
    "yo": "Yoruba",
}

N_PROMPTS_DEFAULT = 200


def load_aya_prompts(lang_code: str, n: int) -> list[str]:
    """Load instruction prompts from Aya dataset for the given language."""
    lang_name = AYA_LANG_NAMES.get(lang_code)
    if not lang_name:
        return []
    try:
        ds = load_dataset("CohereForAI/aya_dataset", split="train")
        lang_data = ds.filter(lambda x: x.get("language") == lang_name)
        prompts = [row["inputs"] for row in lang_data if row.get("inputs", "").strip()]
        return prompts[:n]
    except Exception as e:
        print(f"  [Aya] Failed for {lang_code}: {e}")
        return []


def load_flores_sentences(lang_code: str, n: int) -> list[str]:
    """
    Load FLORES-200 source sentences as fallback prompts.
    Wraps each sentence as 'Please continue or respond to: <sentence>'
    so the model needs to reply in the target language.
    """
    FLORES_LANG_CODES = {
        "fr": "fra_Latn",
        "zh": "zho_Hans",
        "sw": "swh_Latn",
        "th": "tha_Thai",
        "bn": "ben_Beng",
        "yo": "yor_Latn",
    }
    flores_lang = FLORES_LANG_CODES.get(lang_code)
    if not flores_lang:
        return []
    try:
        ds = load_dataset("facebook/flores", flores_lang, split="devtest")
        sentences = [row["sentence"] for row in ds if row.get("sentence", "").strip()]
        # Wrap as prompts: a native-language sentence that invites a continuation
        prompts = [s for s in sentences[:n]]
        return prompts
    except Exception as e:
        print(f"  [FLORES] Failed for {lang_code}: {e}")
        return []


def load_local_processed(lang_code: str, n: int) -> list[str]:
    """Fall back to locally-processed Aya data (already downloaded)."""
    path = f"data/processed/{lang_code}.jsonl"
    if not os.path.exists(path):
        return []
    prompts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            instr = item.get("instruction", "").strip()
            if instr:
                prompts.append(instr)
    return prompts[:n]


def build_lcb_prompts(lang_code: str, n: int) -> list[dict]:
    """Build LCB prompt set for lang_code, targeting n prompts."""
    print(f"  Loading Aya prompts for {lang_code}...")
    prompts = load_aya_prompts(lang_code, n)
    print(f"  Got {len(prompts)} from Aya online.")

    if len(prompts) < n:
        local = load_local_processed(lang_code, n - len(prompts))
        seen = set(prompts)
        new = [p for p in local if p not in seen]
        prompts.extend(new)
        print(f"  Got {len(new)} additional from local processed data.")

    if len(prompts) < n:
        flores = load_flores_sentences(lang_code, n - len(prompts))
        seen = set(prompts)
        new = [p for p in flores if p not in seen]
        prompts.extend(new)
        print(f"  Got {len(new)} additional from FLORES sentences.")

    result = [{"prompt": p, "language": lang_code} for p in prompts[:n]]
    print(f"  Final: {len(result)} prompts for {lang_code}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+",
                        default=list(AYA_LANG_NAMES.keys()))
    parser.add_argument("--n_prompts", type=int, default=N_PROMPTS_DEFAULT)
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for lang in args.languages:
        print(f"\n=== Building LCB prompts for: {lang} ===")
        prompts = build_lcb_prompts(lang, args.n_prompts)
        if not prompts:
            print(f"  WARNING: No prompts found for {lang}, skipping.")
            continue
        out_path = os.path.join(args.output_dir, f"lcb_prompts_{lang}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in prompts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {len(prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()
