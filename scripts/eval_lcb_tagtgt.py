"""
4×4 LCB-chat matrix evaluation — tag=output_lang variant.

tag is set to the TARGET output language (what we want the model to produce).
The English instruction text also names the same output language.
News excerpt is always in input_lang.

Prompt format:
  ### Instruction:
  <|tgt_lang:{output_lang}|> Please continue the following news article in
  {output_lang_name}: {news_excerpt}

  ### Response:

Contrast with eval_lcb_matrix.py (tag=input_lang) and eval_lcb_notag.py (no tag).
"""

import argparse
import json
import os
import random
import re
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "/root/project/models/Qwen3.5-9B-Base"
HF_CACHE = "/root/project/hf_cache/datasets"
GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_CACHE = "/root/project/hf_cache"

# MasakhaNEWS language codes
MASAKHA_CODES = {
    "en": "eng",
    "yo": "yor",
    "so": "som",
    "ha": "hau",
}

LANG_LABELS = {
    "en": "eng_Latn",
    "yo": "yor_Latn",
    "so": "som_Latn",
    "ha": "hau_Latn",
}

LANG_NAMES = {
    "en": "English",
    "yo": "Yoruba",
    "so": "Somali",
    "ha": "Hausa",
}

N_PER_CELL = 50
MAX_NEW_TOKENS = 200
BATCH_SIZE = 8
EXCERPT_CHARS = 220  # approx 1-2 sentences


def extract_excerpt(text, max_chars=EXCERPT_CHARS):
    """Return first ~1-2 sentences, capped at max_chars."""
    text = text.strip()
    # try to cut at sentence boundary
    match = re.search(r'[.!?]\s', text[:max_chars + 40])
    if match and match.end() > 30:
        return text[:match.end()].strip()
    return text[:max_chars].strip()


def load_news_excerpts(lang, n, seed=42):
    from datasets import load_dataset
    code = MASAKHA_CODES[lang]
    ds = load_dataset(
        "masakhane/masakhanews", code,
        split="test",
        cache_dir=HF_CACHE,
    )
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    excerpts = []
    for idx in indices:
        text = ds[idx]["text"]
        exc = extract_excerpt(text)
        if len(exc.split()) >= 8:
            excerpts.append(exc)
        if len(excerpts) == n:
            break
    return excerpts


def make_prompt(input_lang, output_lang, excerpt):
    out_name = LANG_NAMES[output_lang]
    return (
        f"### Instruction:\n"
        f"<|tgt_lang:{output_lang}|> Please continue the following news article "
        f"in {out_name}: {excerpt}\n\n"
        f"### Response:\n"
    )


def detect_lang(ft_model, text):
    clean = text.replace("\n", " ").strip()
    if not clean:
        return "unknown"
    pred, _ = ft_model.predict(clean)
    return pred[0].replace("__label__", "")


def load_model_and_tokenizer(model_path):
    is_peft = os.path.isfile(os.path.join(model_path, "adapter_config.json"))
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL if is_peft else model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL if is_peft else model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if is_peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        model = base
    model.eval()
    return model, tokenizer


def eval_cell(model, tokenizer, ft_model, input_lang, output_lang, excerpts):
    prompts = [make_prompt(input_lang, output_lang, exc) for exc in excerpts]
    target_label = LANG_LABELS[output_lang]
    tokenizer.padding_side = "left"

    correct = en_count = total = 0
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
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
            gen = tokenizer.decode(out[j][input_len:], skip_special_tokens=True).strip()
            if len(gen.split()) < 5:
                continue
            detected = detect_lang(ft_model, gen)
            if detected == target_label:
                correct += 1
            if detected == LANG_LABELS["en"]:
                en_count += 1
            total += 1
    return {
        "lc_rate": round(correct / total, 4) if total else 0.0,
        "en_leak": round(en_count / total, 4) if total else 0.0,
        "n": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input_langs", default="en,yo,so,ha")
    parser.add_argument("--output_langs", default="en,yo,so,ha")
    parser.add_argument("--n", type=int, default=N_PER_CELL)
    args = parser.parse_args()

    input_langs = args.input_langs.split(",")
    output_langs = args.output_langs.split(",")

    import fasttext
    from huggingface_hub import hf_hub_download
    glotlid_path = hf_hub_download(
        repo_id=GLOTLID_REPO, filename="model.bin", cache_dir=GLOTLID_CACHE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ft = fasttext.load_model(glotlid_path)

    print(f"Loading model: {args.model_path}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    print(f"Pre-loading MasakhaNEWS excerpts for {input_langs}", flush=True)
    excerpt_cache = {lang: load_news_excerpts(lang, args.n) for lang in input_langs}
    for lang, excs in excerpt_cache.items():
        print(f"  {lang}: {len(excs)} excerpts loaded, e.g. {excs[0][:80]!r}", flush=True)

    matrix = {}
    for in_lang in input_langs:
        matrix[in_lang] = {}
        excerpts = excerpt_cache[in_lang]
        for out_lang in output_langs:
            print(f"\n=== cell ({in_lang} → {out_lang}) ===", flush=True)
            result = eval_cell(model, tokenizer, ft, in_lang, out_lang, excerpts)
            matrix[in_lang][out_lang] = result
            print(
                f"  lc_rate={result['lc_rate']:.4f}  "
                f"en_leak={result['en_leak']:.4f}  n={result['n']}",
                flush=True,
            )

    output_data = {"model_path": args.model_path, "matrix": matrix}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}", flush=True)

    print("\n=== lc_rate matrix ===")
    header = "instr\\tgt  " + "  ".join(f"{l:>6}" for l in output_langs)
    print(header)
    for in_lang in input_langs:
        row = f"{in_lang:>10}  " + "  ".join(
            f"{matrix[in_lang][out_lang]['lc_rate']:>6.3f}" for out_lang in output_langs
        )
        print(row)


if __name__ == "__main__":
    main()
