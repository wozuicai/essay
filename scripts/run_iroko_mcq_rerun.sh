#!/bin/bash
# run_iroko_mcq_rerun.sh
# 对 train_yo 和 train_ha 重跑 IrokoBench MCQ（afrimmlu yo+ha）
# 两张卡并行，结果合并写入已有 JSON 的 irokobench.mcq_accuracy 字段

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1

RESULTS_DIR="/root/project/results/phase2_v2"
SCRIPT="/root/project/scripts/eval_extended.py"
LOG="/root/project/logs/iroko_mcq_rerun.log"

mkdir -p /root/project/logs

declare -A MODELS=(
    ["yo"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_yo"
    ["ha"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_ha"
)
declare -A JSONS=(
    ["yo"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_yo_eval.json"
    ["ha"]="$RESULTS_DIR/lis_Qwen3.5-9B-Base_train_ha_eval.json"
)

echo "[$(date)] === run_iroko_mcq_rerun.sh 启动（train_yo GPU0 / train_ha GPU1 并行）===" | tee -a "$LOG"

run_one() {
    local key="$1" gpu_id="$2"
    local per_log="/root/project/logs/iroko_mcq_rerun_${key}.log"
    echo "[$(date)] --- 开始 IrokoBench MCQ model=train_${key} (GPU $gpu_id) ---" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$SCRIPT" \
        --model_path  "${MODELS[$key]}" \
        --result_json "${JSONS[$key]}" \
        --only_iroko_mcq \
        > "$per_log" 2>&1
    local ret=$?
    echo "[$(date)] --- 完成 train_${key} (GPU $gpu_id, exit=$ret) ---" | tee -a "$LOG"
    cat "$per_log" >> "$LOG"
}

run_one yo 0 &
run_one ha 1 &
wait

echo "[$(date)] === IrokoBench MCQ 重跑完成 ===" | tee -a "$LOG"

echo ""
echo "=== 结果汇总 ==="
python3 - << 'EOF'
import json

RESULTS_DIR = "/root/project/results/phase2_v2"
for key in ["yo", "ha"]:
    path = f"{RESULTS_DIR}/lis_Qwen3.5-9B-Base_train_{key}_eval.json"
    with open(path) as f:
        data = json.load(f)
    iroko = data["scores"]["multilingual"].get("irokobench", {})
    print(f"train_{key}:")
    for lang in ["yo", "ha"]:
        mcq = iroko.get(lang, {}).get("mcq_accuracy", "N/A")
        print(f"  {lang} MCQ = {mcq}")
EOF
