#!/bin/bash
# Run 4x4 LCB-chat matrix evaluation on all 9 models sequentially.
# Each model: 4x4=16 cells x 50 samples = 800 generations -> ~8-12 min on 4xH100.

set -e
export PATH=/home/tiger/.local/bin:$PATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0   # GlotLID still needs hub for model.bin check
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

BASE=/root/project
RESULTS=$BASE/results/lcb_matrix
mkdir -p $RESULTS

EVAL="python $BASE/scripts/eval_lcb_matrix.py"

run_model() {
    local name=$1
    local path=$2
    local out="$RESULTS/${name}_lcb_matrix.json"
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
# tag_routing still training on A100 worker — will be added after training completes

echo ""
echo "=========================================="
echo "ALL DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results in: $RESULTS"
echo "=========================================="
