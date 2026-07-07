#!/usr/bin/env bash
# Run SIB-200 only. Writes result directly to baseline.json.
set -e
export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /root/project

python3 -u - << 'PYEOF'
import json, sys, torch
sys.path.insert(0, "/root/project")
from src.evaluation.multilingual_eval import _run_sib200

MODEL    = "/root/project/models/Qwen3.5-9B-Base"
OUT_JSON = "/root/project/results/phase1_baseline/Qwen3.5-9B-Base_baseline.json"
LANGUAGES = ["fr", "zh", "sw", "th", "bn", "yo"]

print("=== SIB-200 zero-shot ===", flush=True)
scores = _run_sib200(MODEL, LANGUAGES, batch_size=8)
print("SIB-200 result:", scores, flush=True)

with open(OUT_JSON) as f:
    baseline = json.load(f)

baseline.setdefault("scores", {}).setdefault("multilingual", {})["sib200"] = scores

with open(OUT_JSON, "w") as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)

print("Saved to", OUT_JSON, flush=True)
print("Done.", flush=True)
PYEOF
