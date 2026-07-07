#!/bin/bash
# B200 GPU 6+7 顺序训练 mid_so_normfix → mid_ha_normfix
# H100 libcuda.so.580 为 0 字节（驱动安装不完整），改到 B200 跑
set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# Fix CUDA 803: compat dir (575.x) in ldconfig overrides the real driver (580.x)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08

cd /root/project
mkdir -p logs results/mid

MODEL="/root/project/models/Qwen3.5-9B-Base"
TEACHER="results/phase2_v2/lis_Qwen3.5-9B-Base_train_en"
CFG="configs/experiments/lis_matrix.yaml"
ACCEL="configs/accelerate_2gpu.yaml"

for LANG in so ha; do
    OUT_DIR="results/mid/mid_Qwen3.5-9B-Base_${LANG}_normfix"
    LOG="logs/mid_${LANG}_normfix_train.log"

    if [[ -f "${OUT_DIR}/adapter_config.json" ]]; then
        echo "[$(date -u '+%H:%M:%S UTC')] ${LANG}_normfix adapter already exists, skipping."
        continue
    fi

    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === Starting mid_${LANG}_normfix (GPU 6+7) ==="

    CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
        --config_file "$ACCEL" \
        --main_process_port 29504 \
        scripts/train_mid.py \
        --model       "$MODEL" \
        --teacher_adapter "$TEACHER" \
        --train_lang  "$LANG" \
        --output_dir  "$OUT_DIR" \
        --config      "$CFG" \
        --alpha 0.1 --beta 0.05 --top_n_layers 4 --n_pos2 3 \
        2>&1 | tee "$LOG"

    echo "[$(date -u '+%H:%M:%S UTC')] ${LANG}_normfix done → ${OUT_DIR}"
done

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === 全部完成: so_normfix + ha_normfix ==="
