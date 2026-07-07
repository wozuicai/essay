#!/bin/bash
# run_eval_extended.sh
# Phase 2 v2 跑完后自动调用，对 5 个模型（baseline + 4 SFT）跑扩展评测
# 结果写入已有 JSON 文件，不新建
#
# 每个模型单独 pin 到一张 GPU 上（不再用 device_map=auto 跨卡模型并行），
# 5 个模型分批在 4 张卡上并行跑，显著提速。

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1
export EVAL_PARALLEL_WORKERS=4

RESULTS_DIR="/root/project/results/phase2_v2"
SCRIPT="/root/project/scripts/eval_extended.py"
LOG="/root/project/logs/eval_extended.log"
N_GPUS=4

mkdir -p /root/project/logs

declare -A MODELS=(
    ["baseline"]="/root/project/models/Qwen3.5-9B-Base"
    ["en"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_en"
    ["yo"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_yo"
    ["so"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_so"
    ["ha"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_ha"
)

declare -A JSONS=(
    ["baseline"]="$RESULTS_DIR/Qwen3.5-9B-Base_baseline.json"
    ["en"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_en_eval.json"
    ["yo"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_yo_eval.json"
    ["so"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_so_eval.json"
    ["ha"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_ha_eval.json"
)

echo "[$(date)] === run_eval_extended.sh 启动（并行模式，$N_GPUS 张卡）===" | tee -a "$LOG"

keys=(baseline en yo so ha)
gpu=0
pids=()

run_one() {
    local key="$1" gpu_id="$2"
    local model_path="${MODELS[$key]}" json_path="${JSONS[$key]}"
    local per_log="/root/project/logs/eval_extended_${key}.log"

    if [ ! -f "$json_path" ]; then
        echo "[$(date)] [skip] $json_path 不存在，跳过 $key" >> "$LOG"
        return
    fi

    echo "[$(date)] --- 开始评测 model=$key (GPU $gpu_id) ---" >> "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$SCRIPT" \
        --model_path  "$model_path" \
        --result_json "$json_path" \
        > "$per_log" 2>&1
    echo "[$(date)] --- 完成 model=$key (GPU $gpu_id) ---" >> "$LOG"
    cat "$per_log" >> "$LOG"
}

# 第一批：4 个模型同时占满 4 张卡
for i in 0 1 2 3; do
    key="${keys[$i]}"
    run_one "$key" "$i" &
    pids+=($!)
done
wait "${pids[@]}"

# 第二批：剩下 1 个模型（ha）单独跑
key="${keys[4]}"
run_one "$key" 0

echo "[$(date)] === 全部扩展评测完成 ===" | tee -a "$LOG"
