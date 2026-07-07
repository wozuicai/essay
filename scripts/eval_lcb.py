"""
LCB (Language Consistency Benchmark) evaluation.

Prompts: MasakhaNEWS first-2-sentence text continuation (no SFT template).
Detection: fastText lid.176.

Metrics per language:
  lc_rate  — fraction of outputs detected as target language
  en_leak  — fraction of outputs detected as English
  n        — valid generations counted
"""

import argparse
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_CACHE = "/root/project/hf_cache"
# GlotLID uses ISO 639-3 + script labels
MASAKHA_CODES = {"yo": "yor", "so": "som", "ha": "hau"}
GLOTLID_LABELS = {"yo": "yor_Latn", "so": "som_Latn", "ha": "hau_Latn"}
ENGLISH_LABEL = "eng_Latn"
MAX_NEW_TOKENS = 200
BATCH_SIZE = 8


def split_sentences(text, n=2):
    sents = re.split(r'(?<=[.!?።])\s+', text.strip())
    return " ".join(sents[:n])


def load_prompts(lang, n):
    code = MASAKHA_CODES[lang]
    ds = load_dataset("masakhane/masakhanews", code, split="test")
    prompts = []
    for item in ds:
        t = (item.get("text") or "").strip()
        if not t:
            continue
        prompt = split_sentences(t, n=2)
        if len(prompt.split()) >= 5:
            prompts.append(prompt)
    return prompts[:n]


def detect_lang(ft_model, text):
    clean = text.replace("\n", " ").strip()
    if not clean:
        return "unknown"
    pred, _ = ft_model.predict(clean)
    return pred[0].replace("__label__", "")  # returns e.g. "yor_Latn", "eng_Latn"


def eval_one_lang(model, tokenizer, prompts, lang, ft_model):
    correct = en_count = total = 0
    target_label = GLOTLID_LABELS[lang]
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for j in range(len(batch)):
            gen = tokenizer.decode(
                out[j][input_len:], skip_special_tokens=True
            ).strip()
            if len(gen.split()) < 5:
                continue
            detected = detect_lang(ft_model, gen)
            if detected == target_label:
                correct += 1
            if detected == ENGLISH_LABEL:
                en_count += 1
            total += 1
        print(f"  [{lang}] {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", flush=True)

    return {
        "lc_rate": round(correct / total, 4) if total else 0.0,
        "en_leak": round(en_count / total, 4) if total else 0.0,
        "n": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--langs", default="yo,so,ha")
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    langs = args.langs.split(",")

    import fasttext
    import warnings
    from huggingface_hub import hf_hub_download
    glotlid_path = hf_hub_download(
        repo_id=GLOTLID_REPO, filename="model.bin", cache_dir=GLOTLID_CACHE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ft = fasttext.load_model(glotlid_path)

    print(f"Loading model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    scores = {}
    for lang in langs:
        print(f"\n=== LCB: {lang} ===", flush=True)
        prompts = load_prompts(lang, n=args.n_samples)
        print(f"  {len(prompts)} prompts loaded", flush=True)
        scores[lang] = eval_one_lang(model, tokenizer, prompts, lang, ft)
        print(f"  {scores[lang]}", flush=True)

    result = {"model_path": args.model_path, "scores": scores}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
