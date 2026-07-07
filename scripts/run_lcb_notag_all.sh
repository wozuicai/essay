#!/bin/bash
# Run 4x4 LCB no-tag matrix evaluation on all 8 models sequentially.
# No <|tgt_lang|> tag; instruction written in input language.

set -e
export PATH=/home/tiger/.local/bin:$PATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

BASE=/root/project
RESULTS=$BASE/results/lcb_notag
mkdir -p $RESULTS

EVAL="python $BASE/scripts/eval_lcb_notag.py"

run_model() {
    local name=$1
    local path=$2
    local out="$RESULTS/${name}_lcb_notag.json"
    if [ -f "$out" ]; then
        echo "SKIP (exists): $name"
        return
    fi
    echo ""
    echo "=========================================="
    echo "MODEL: $name"
    echo "PATH:  $path"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    $EVAL --model_path "$path" --output "$out"
    echo "DONE: $name at $(date '+%H:%M:%S')"
}

run_model "baseline"    "$BASE/models/Qwen3.5-9B-Base"
run_model "train_en"    "$BASE/results/phase2_v2/lis_Qwen3.5-9B-Base_train_en"
run_model "train_yo"    "$BASE/results/phase2_v2/lis_Qwen3.5-9B-Base_train_yo"
run_model "train_so"    "$BASE/results/phase2_v2/lis_Qwen3.5-9B-Base_train_so"
run_model "train_ha"    "$BASE/results/phase2_v2/lis_Qwen3.5-9B-Base_train_ha"
run_model "mix_en_yo"   "$BASE/results/mix_en/mix_Qwen3.5-9B-Base_en_yo"
run_model "mix_en_so"   "$BASE/results/mix_en/mix_Qwen3.5-9B-Base_en_so"
run_model "mix_en_ha"   "$BASE/results/mix_en/mix_Qwen3.5-9B-Base_en_ha"

echo ""
echo "=========================================="
echo "ALL DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results in: $RESULTS"
echo "=========================================="
