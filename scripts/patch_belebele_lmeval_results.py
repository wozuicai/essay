#!/usr/bin/env python3
"""Recompute Belebele with the current lm-eval implementation and patch result JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.multilingual_eval import _run_belebele


LANGS = ["en", "yo", "so", "ha"]


def resolve_model_path(result_path: Path, payload: dict) -> str:
    model_path = payload.get("model_path")
    if not model_path:
        stem = result_path.name.removesuffix("_eval.json")
        sibling = result_path.with_name(stem)
        if sibling.exists():
            model_path = str(sibling)
    if not model_path:
        raise ValueError(f"Cannot infer model_path for {result_path}")
    if not os.path.isabs(model_path):
        model_path = str((Path.cwd() / model_path).resolve())
    return model_path


def patch_one(path: Path, batch_size: int) -> dict:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    old = payload.get("scores", {}).get("multilingual", {}).get("belebele")
    model_path = resolve_model_path(path, payload)
    print(f"\n=== {path} ===", flush=True)
    print(f"model_path={model_path}", flush=True)
    print(f"old={old}", flush=True)
    new = _run_belebele(model_path, LANGS, batch_size=batch_size)
    print(f"new={new}", flush=True)
    payload.setdefault("scores", {}).setdefault("multilingual", {})["belebele"] = new
    payload.setdefault("metadata", {}).setdefault("patches", []).append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "field": "scores.multilingual.belebele",
            "method": "lm-eval harness",
            "script": "scripts/patch_belebele_lmeval_results.py",
            "old": old,
            "new": new,
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return new


def update_phase2_belebele_csv() -> None:
    base = Path("results/phase2_v2")
    rows = []
    mapping = {
        "baseline": base / "Qwen3.5-9B-Base_baseline.json",
        "train_en": base / "lis_Qwen3.5-9B-Base_train_en_eval.json",
        "train_yo": base / "lis_Qwen3.5-9B-Base_train_yo_eval.json",
        "train_so": base / "lis_Qwen3.5-9B-Base_train_so_eval.json",
        "train_ha": base / "lis_Qwen3.5-9B-Base_train_ha_eval.json",
    }
    for name, path in mapping.items():
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            bele = json.load(f).get("scores", {}).get("multilingual", {}).get("belebele", {})
        rows.append({"model": name, **{lang: bele.get(lang, "") for lang in LANGS}})
    if not rows:
        return
    out = base / "lis_matrix_Qwen3.5-9B-Base_belebele.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *LANGS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nUpdated {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for raw in args.paths:
        patch_one(Path(raw), args.batch_size)
    update_phase2_belebele_csv()


if __name__ == "__main__":
    main()
