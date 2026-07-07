#!/usr/bin/env bash
# Patch the phase1 baseline to add English SIB-200 + Belebele scores.
# Run AFTER run_baseline_sib200_flores.sh completes.
set -e

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false

MODEL=/root/project/models/Qwen3.5-9B-Base
FINAL_OUT=/root/project/results/phase1_baseline/Qwen3.5-9B-Base_baseline.json

cd /root/project
echo "[$(date)] Adding English SIB-200 + Belebele to baseline..."

python3 - << 'PYEOF'
import json, os, sys, torch
sys.path.insert(0, "/root/project")

from src.evaluation.multilingual_eval import _run_sib200, _run_belebele

MODEL = "/root/project/models/Qwen3.5-9B-Base"
FINAL_OUT = "/root/project/results/phase1_baseline/Qwen3.5-9B-Base_baseline.json"

print("=== SIB-200 English zero-shot ===")
sib200_en = _run_sib200(MODEL, ["en"], batch_size=8)
print("SIB-200 en:", sib200_en)
torch.cuda.empty_cache()

print("\n=== Belebele English zero-shot ===")
belebele_en = _run_belebele(MODEL, ["en"], batch_size=16)
print("Belebele en:", belebele_en)
torch.cuda.empty_cache()

# Merge into existing baseline JSON
with open(FINAL_OUT) as f:
    baseline = json.load(f)

ml = baseline.setdefault("scores", {}).setdefault("multilingual", {})
ml.setdefault("sib200", {}).update(sib200_en)
ml.setdefault("belebele", {}).update(belebele_en)

with open(FINAL_OUT, "w") as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)
print(f"\nBaseline updated with English scores: {FINAL_OUT}")
print("SIB-200:", ml["sib200"])
print("Belebele:", ml["belebele"])
PYEOF

echo "[$(date)] Done."
