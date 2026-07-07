"""
Dataset loader: reads preprocessed JSONL files from data/processed/.
Also handles on-the-fly loading from HuggingFace if local files don't exist.
"""

import json
import os
from datasets import Dataset, load_dataset


def load_sft_dataset(data_dir: str, lang_code: str, n_samples: int = None) -> Dataset:
    """
    Load SFT dataset for a given language.
    Tries local JSONL first, falls back to HuggingFace Aya/Open-Platypus.
    """
    local_path = os.path.join(data_dir, f"{lang_code}.jsonl")

    if os.path.exists(local_path):
        samples = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        dataset = Dataset.from_list(samples)
    else:
        print(f"Local data not found for [{lang_code}], loading from HuggingFace...")
        dataset = _load_from_hf(lang_code)

    if n_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

    return dataset


def _load_from_hf(lang_code: str) -> Dataset:
    """Fallback: load directly from HuggingFace datasets."""
    LANG_NAMES = {
        'fr': 'French', 'zh': 'Chinese Simplified', 'sw': 'Swahili',
        'th': 'Thai', 'bn': 'Bengali', 'yo': 'Yoruba',
    }

    if lang_code == "en":
        ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
        return ds.map(
            lambda x: {"instruction": x.get("instruction", ""), "response": x.get("output", ""),
                       "language": "en", "source": "open_platypus",
                       "text": f"### Instruction:\n<|tgt_lang:en|> {x.get('instruction','')}\n\n### Response:\n{x.get('output','')}"},
            remove_columns=ds.column_names,
        )

    lang_name = LANG_NAMES.get(lang_code, lang_code)
    ds = load_dataset("CohereForAI/aya_dataset", split="train")
    ds = ds.filter(lambda x: x["language"] == lang_name)
    return ds.map(
        lambda x: {"instruction": x.get("inputs", ""), "response": x.get("targets", ""),
                   "language": lang_code, "source": "aya",
                   "text": f"### Instruction:\n<|tgt_lang:{lang_code}|> {x.get('inputs','')}\n\n### Response:\n{x.get('targets','')}"},
        remove_columns=ds.column_names,
    )
