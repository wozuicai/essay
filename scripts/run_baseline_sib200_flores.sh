#!/usr/bin/env bash
# Re-run zero-shot SIB-200 and FLORES for the base model baseline.
# Each benchmark is written into baseline.json immediately after it completes.
set -e

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASELINE_JSON=/root/project/results/phase1_baseline/Qwen3.5-9B-Base_baseline.json

mkdir -p /root/project/results/phase1_baseline
cd /root/project

echo "[$(date)] Starting zero-shot SIB-200 + FLORES baseline patch..."

python3 -u - << 'PYEOF'
import json, os, sys, torch
sys.path.insert(0, "/root/project")

from src.evaluation.multilingual_eval import _run_sib200, _run_flores

LANGUAGES = ["fr", "zh", "sw", "th", "bn", "yo"]
MODEL    = "/root/project/models/Qwen3.5-9B-Base"
OUT_JSON = "/root/project/results/phase1_baseline/Qwen3.5-9B-Base_baseline.json"

def load_baseline():
    with open(OUT_JSON) as f:
        return json.load(f)

def save_baseline(data):
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[saved] {OUT_JSON}", flush=True)

baseline = load_baseline()
ml = baseline.setdefault("scores", {}).setdefault("multilingual", {})

# ── SIB-200 ──────────────────────────────────────────────────────────────────
# Re-run if values are all 0.2 (old BPE bug) or missing
sib_vals = ml.get("sib200", {})
if not sib_vals or all(abs(v - 0.2) < 1e-6 for v in sib_vals.values()):
    print("\n=== SIB-200 zero-shot ===", flush=True)
    sib200_scores = _run_sib200(MODEL, LANGUAGES, batch_size=8)
    print("SIB-200:", sib200_scores, flush=True)
    torch.cuda.empty_cache()
    ml["sib200"] = sib200_scores
    save_baseline(baseline)          # write immediately
else:
    print(f"SIB-200 looks correct, skipping: {sib_vals}", flush=True)

# ── FLORES ───────────────────────────────────────────────────────────────────
flores_vals = ml.get("flores", {})
if not flores_vals:
    print("\n=== FLORES zero-shot (via belebele passages) ===", flush=True)
    flores_scores = _run_flores(MODEL, LANGUAGES, batch_size=8)
    print("FLORES:", flores_scores, flush=True)
    torch.cuda.empty_cache()
    ml["flores"] = flores_scores
    save_baseline(baseline)          # write immediately
else:
    print(f"FLORES already present, skipping: {flores_vals}", flush=True)

print("\nDone. Final multilingual scores:", flush=True)
print(json.dumps(ml, indent=2, ensure_ascii=False), flush=True)
PYEOF

echo "[$(date)] Done."
