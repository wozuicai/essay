#!/usr/bin/env python3
"""Audit SFT data length distribution.

If transformers is installed and --model is provided, reports exact tokenizer
lengths. Otherwise reports character lengths only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--langs", default="en,yo,so,ha")
    p.add_argument("--model", default="")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_train_chars", type=int, default=12000)
    return p.parse_args()


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def text_from_example(ex: dict, fallback_lang: str) -> str:
    if "prompt" in ex and "completion" in ex:
        return (ex.get("prompt") or "") + (ex.get("completion") or "")
    if "text" in ex:
        return ex.get("text") or ""
    lang = ex.get("language") or ex.get("lang") or fallback_lang
    response = ex.get("response") or ex.get("output") or ex.get("target") or ""
    return f"### Instruction:\n<|tgt_lang:{lang}|> {ex.get('instruction','')}\n\n### Response:\n{response}"


def maybe_load_tokenizer(model_path: str):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        print(f"[warn] tokenizer unavailable: {exc}")
        return None


def summarize(values, limit):
    return {
        "n": len(values),
        "avg": round(sum(values) / len(values), 2) if values else None,
        "p50": pct(values, 0.50),
        "p90": pct(values, 0.90),
        "p95": pct(values, 0.95),
        "p99": pct(values, 0.99),
        "max": max(values) if values else None,
        f"over_{limit}": sum(1 for value in values if limit > 0 and value > limit),
    }


def main():
    args = parse_args()
    tokenizer = maybe_load_tokenizer(args.model)
    data_dir = Path(args.data_dir)

    for lang in [x for x in args.langs.split(",") if x]:
        path = data_dir / f"{lang}.jsonl"
        if not path.exists():
            print(f"{lang}: missing {path}")
            continue
        chars = []
        tokens = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                text = text_from_example(json.loads(line), lang)
                chars.append(len(text))
                if tokenizer is not None:
                    tokens.append(len(tokenizer.encode(text, add_special_tokens=True)))

        print(f"\n== {lang} ==")
        print("chars ", summarize(chars, args.max_train_chars))
        if tokens:
            print("tokens", summarize(tokens, args.max_length))


if __name__ == "__main__":
    main()
