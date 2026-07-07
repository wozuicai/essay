#!/bin/bash
# run_eval_iroko_extra.sh
# 在 run_eval_extended.sh 全部跑完后执行：补跑 IrokoBench 的另外两个子测试集
# AfriXNLI / AfriMGSM，结果合并写入已有 JSON 的 irokobench 字段（不覆盖 afrimmlu 部分）

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1
export EVAL_PARALLEL_WORKERS=4

RESULTS_DIR="/root/project/results/phase2_v2"
SCRIPT="/root/project/scripts/eval_extended.py"
LOG="/root/project/logs/eval_iroko_extra.log"

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

echo "[$(date)] === run_eval_iroko_extra.sh 启动（AfriXNLI/AfriMGSM 补跑，并行模式）===" | tee -a "$LOG"

keys=(baseline en yo so ha)
pids=()

run_one() {
    local key="$1" gpu_id="$2"
    local model_path="${MODELS[$key]}" json_path="${JSONS[$key]}"
    local per_log="/root/project/logs/eval_iroko_extra_${key}.log"

    if [ ! -f "$json_path" ]; then
        echo "[$(date)] [skip] $json_path 不存在，跳过 $key" >> "$LOG"
        return
    fi

    echo "[$(date)] --- 开始 iroko-extra model=$key (GPU $gpu_id) ---" >> "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$SCRIPT" \
        --model_path  "$model_path" \
        --result_json "$json_path" \
        --only_iroko_extra \
        > "$per_log" 2>&1
    echo "[$(date)] --- 完成 iroko-extra model=$key (GPU $gpu_id) ---" >> "$LOG"
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

echo "[$(date)] === IrokoBench 补跑（AfriXNLI/AfriMGSM）全部完成 ===" | tee -a "$LOG"
