#!/usr/bin/env bash
set -e
export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /root/project

python3 -u - << 'PYEOF'
import json, sys, torch
sys.path.insert(0, "/root/project")
from src.evaluation.multilingual_eval import _run_flores

MODEL    = "/root/project/models/Qwen3.5-9B-Base"
OUT_JSON = "/root/project/results/phase1_baseline/Qwen3.5-9B-Base_baseline.json"
LANGUAGES = ["fr", "zh", "sw", "th", "bn", "yo"]

print("=== FLORES zero-shot (via belebele passages) ===", flush=True)
scores = _run_flores(MODEL, LANGUAGES, batch_size=1)
print("FLORES result:", scores, flush=True)

with open(OUT_JSON) as f:
    baseline = json.load(f)

baseline.setdefault("scores", {}).setdefault("multilingual", {})["flores"] = scores

with open(OUT_JSON, "w") as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)

print("Saved to", OUT_JSON, flush=True)
print("Done.", flush=True)
PYEOF
