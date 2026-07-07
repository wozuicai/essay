#!/usr/bin/env python3
"""Preflight checks for the required train/eval runs.

No external packages are imported, so this can run before the Python training
environment is fully set up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_COLUMNS = (
    ("prompt", "completion"),
    ("messages",),
    ("instruction", "response"),
    ("instruction", "output"),
    ("instruction", "target"),
    ("text",),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--langs", default="en,yo,so,ha")
    p.add_argument("--max_train_chars", type=int, default=12000)
    p.add_argument("--allow_missing_model_weights", action="store_true")
    return p.parse_args()


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def has_supported_schema(keys: set[str]) -> bool:
    return any(all(col in keys for col in cols) for cols in REQUIRED_COLUMNS)


def check_model(model_path: Path, allow_missing_weights: bool) -> list[str]:
    errors = []
    for name in ("config.json", "tokenizer_config.json"):
        if not (model_path / name).exists():
            errors.append(f"missing model file: {model_path / name}")

    index_path = model_path / "model.safetensors.index.json"
    weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            index = json.load(f)
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shards if not (model_path / name).exists()]
        if missing and not allow_missing_weights:
            errors.append(f"missing model shards under {model_path}: {missing[:5]}")
    elif not weight_files and not allow_missing_weights:
        errors.append(f"missing model weights under {model_path}")
    return errors


def read_text_for_length(ex: dict) -> str:
    if "prompt" in ex and "completion" in ex:
        return (ex.get("prompt") or "") + (ex.get("completion") or "")
    if "text" in ex:
        return ex.get("text") or ""
    lang = ex.get("language") or ex.get("lang") or "unknown"
    return f"### Instruction:\n<|tgt_lang:{lang}|> {ex.get('instruction','')}\n\n### Response:\n{ex.get('response','')}"


def check_data_file(path: Path, max_train_chars: int) -> tuple[list[str], dict]:
    errors = []
    lengths = []
    keys_seen = None
    bad_json = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            keys_seen = keys_seen or set(ex.keys())
            lengths.append(len(read_text_for_length(ex)))

    if bad_json:
        errors.append(f"{path}: {bad_json} bad JSON lines")
    if keys_seen is None:
        errors.append(f"{path}: empty file")
    elif not has_supported_schema(keys_seen):
        errors.append(f"{path}: unsupported schema {sorted(keys_seen)}")

    over = sum(1 for n in lengths if max_train_chars > 0 and n > max_train_chars)
    stats = {
        "n": len(lengths),
        "p50_chars": pct(lengths, 0.50),
        "p90_chars": pct(lengths, 0.90),
        "p95_chars": pct(lengths, 0.95),
        "p99_chars": pct(lengths, 0.99),
        "max_chars": max(lengths) if lengths else None,
        "over_max_train_chars": over,
    }
    return errors, stats


def main():
    args = parse_args()
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    langs = [lang for lang in args.langs.split(",") if lang]

    errors = []
    errors.extend(check_model(model_path, args.allow_missing_model_weights))

    print("=== Data preflight ===")
    for lang in langs:
        path = data_dir / f"{lang}.jsonl"
        if not path.exists():
            errors.append(f"missing data file: {path}")
            continue
        file_errors, stats = check_data_file(path, args.max_train_chars)
        errors.extend(file_errors)
        print(f"{lang}: {stats}")

    if errors:
        print("\n=== Preflight failed ===")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("Preflight OK.")


if __name__ == "__main__":
    main()
