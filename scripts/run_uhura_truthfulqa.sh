#!/bin/bash
# run_uhura_truthfulqa.sh
# 对 5 个模型（baseline + train_en/yo/so/ha）跑 Uhura-TruthfulQA MC1（yo + ha）
# 并行：一张卡跑一个模型；第一批 4 卡同时，第二批 1 卡单独

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1

RESULTS_DIR="/root/project/results/phase2_v2"
SCRIPT="/root/project/scripts/eval_extended.py"
LOG="/root/project/logs/uhura_truthfulqa.log"
BATCH_SIZE=32

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

echo "[$(date)] === run_uhura_truthfulqa.sh 启动（Uhura-TruthfulQA MC1，yo+ha，并行）===" | tee -a "$LOG"

keys=(baseline en yo so ha)
pids=()

run_one() {
    local key="$1" gpu_id="$2"
    local model_path="${MODELS[$key]}" json_path="${JSONS[$key]}"
    local per_log="/root/project/logs/uhura_truthfulqa_${key}.log"

    if [ ! -f "$json_path" ]; then
        echo "[$(date)] [skip] $json_path 不存在，跳过 $key" | tee -a "$LOG"
        return
    fi

    echo "[$(date)] --- 开始 Uhura-TFQ model=$key (GPU $gpu_id) ---" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$SCRIPT" \
        --model_path         "$model_path" \
        --result_json        "$json_path" \
        --only_uhura_truthfulqa \
        --uhura_batch_size   "$BATCH_SIZE" \
        > "$per_log" 2>&1
    local ret=$?
    echo "[$(date)] --- 完成 Uhura-TFQ model=$key (GPU $gpu_id, exit=$ret) ---" | tee -a "$LOG"
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
run_one "${keys[4]}" 0

echo "[$(date)] === Uhura-TruthfulQA 全部完成，结果写入各 result JSON ===" | tee -a "$LOG"

# 打印汇总
echo ""
echo "=== 结果汇总 ==="
python3 - << 'EOF'
import json, os

RESULTS_DIR = "/root/project/results/phase2_v2"
models = {
    "baseline": f"{RESULTS_DIR}/Qwen3.5-9B-Base_baseline.json",
    "train_en": f"{RESULTS_DIR}/lis_Qwen3.5-9B-Base_train_en_eval.json",
    "train_yo": f"{RESULTS_DIR}/lis_Qwen3.5-9B-Base_train_yo_eval.json",
    "train_so": f"{RESULTS_DIR}/lis_Qwen3.5-9B-Base_train_so_eval.json",
    "train_ha": f"{RESULTS_DIR}/lis_Qwen3.5-9B-Base_train_ha_eval.json",
}
print(f"{'model':<12} {'yo_mc1':>8} {'ha_mc1':>8}")
print("-" * 32)
for name, path in models.items():
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    uhura = data.get("scores", {}).get("multilingual", {}).get("uhura_truthfulqa", {})
    yo = uhura.get("yo", {}).get("mc1_accuracy", "N/A") if isinstance(uhura.get("yo"), dict) else "N/A"
    ha = uhura.get("ha", {}).get("mc1_accuracy", "N/A") if isinstance(uhura.get("ha"), dict) else "N/A"
    print(f"{name:<12} {str(yo):>8} {str(ha):>8}")
EOF
