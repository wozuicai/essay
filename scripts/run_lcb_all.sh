#!/bin/bash
# LCB evaluation for all 8 models.
# Round 1 (GPUs 0-3): baseline, train_en, train_yo, train_so
# Round 2 (GPUs 0-3): train_ha, mix_en_yo, mix_en_so, mix_en_ha
#
# Usage: nohup bash scripts/run_lcb_all.sh > logs/lcb_master.log 2>&1 &

set -euo pipefail
export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0

cd /root/project
mkdir -p logs results/lcb

declare -A MODELS
MODELS[baseline]="/root/project/models/Qwen3.5-9B-Base"
MODELS[train_en]="/root/project/results/phase2_v2/lis_Qwen3.5-9B-Base_train_en"
MODELS[train_yo]="/root/project/results/phase2_v2/lis_Qwen3.5-9B-Base_train_yo"
MODELS[train_so]="/root/project/results/phase2_v2/lis_Qwen3.5-9B-Base_train_so"
MODELS[train_ha]="/root/project/results/phase2_v2/lis_Qwen3.5-9B-Base_train_ha"
MODELS[mix_en_yo]="/root/project/results/mix_en/mix_Qwen3.5-9B-Base_en_yo"
MODELS[mix_en_so]="/root/project/results/mix_en/mix_Qwen3.5-9B-Base_en_so"
MODELS[mix_en_ha]="/root/project/results/mix_en/mix_Qwen3.5-9B-Base_en_ha"

run_round() {
    local names=("$@")
    local pids=()
    local gpu=0
    for name in "${names[@]}"; do
        out="results/lcb/${name}_lcb.json"
        if [[ -f "$out" ]]; then
            echo "[$(date -u)] Skip $name (exists)"
            gpu=$((gpu + 1))
            continue
        fi
        echo "[$(date -u)] GPU $gpu <- $name"
        CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_lcb.py \
            --model_path "${MODELS[$name]}" \
            --output "$out" \
            > "logs/lcb_${name}.log" 2>&1 &
        pids+=($!)
        gpu=$((gpu + 1))
    done
    for pid in "${pids[@]}"; do
        wait "$pid" && echo "[$(date -u)] pid $pid done" || echo "[$(date -u)] pid $pid FAILED"
    done
}

echo "[$(date -u)] === Round 1 ==="
run_round baseline train_en train_yo train_so

echo "[$(date -u)] === Round 2 ==="
run_round train_ha mix_en_yo mix_en_so mix_en_ha

echo "[$(date -u)] === Summary ==="
python3 - << 'EOF'
import json, os, math

RESULTS = "results/lcb"
NAMES = ["baseline","train_en","train_yo","train_so","train_ha","mix_en_yo","mix_en_so","mix_en_ha"]

def g(d, lang, k):
    v = d.get(lang, {}).get(k, float("nan"))
    return f"{v:.3f}" if not math.isnan(v) else " N/A"

print(f"{'model':<14} | {'yo_lc':>6} {'yo_en':>6} | {'so_lc':>6} {'so_en':>6} | {'ha_lc':>6} {'ha_en':>6}")
print("-" * 65)
for name in NAMES:
    path = f"{RESULTS}/{name}_lcb.json"
    if not os.path.exists(path):
        print(f"{name:<14} | {'N/A (missing)':>41}")
        continue
    d = json.load(open(path))["scores"]
    print(f"{name:<14} | {g(d,'yo','lc_rate'):>6} {g(d,'yo','en_leak'):>6} | "
          f"{g(d,'so','lc_rate'):>6} {g(d,'so','en_leak'):>6} | "
          f"{g(d,'ha','lc_rate'):>6} {g(d,'ha','en_leak'):>6}")
EOF
