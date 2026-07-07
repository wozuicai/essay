#!/bin/bash
# H100 上用 4 GPU 顺序训练 mid_so_normfix → mid_ha_normfix
set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /root/project
mkdir -p logs results/mid

MODEL="/root/project/models/Qwen3.5-9B-Base"
TEACHER="results/phase2_v2/lis_Qwen3.5-9B-Base_train_en"
CFG="configs/experiments/lis_matrix.yaml"
ACCEL="/tmp/accel_normfix_4gpu_h100.yaml"

# 创建 4-GPU accelerate 配置（H100 本地 /tmp）
cat > "$ACCEL" << 'YAML'
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: auto
  gradient_clipping: auto
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero3_save_16bit_model: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: "no"
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

for LANG in so ha; do
    OUT_DIR="results/mid/mid_Qwen3.5-9B-Base_${LANG}_normfix"
    LOG="logs/mid_${LANG}_normfix_train.log"

    if [[ -f "${OUT_DIR}/adapter_config.json" ]]; then
        echo "[$(date -u '+%H:%M:%S UTC')] ${LANG}_normfix adapter already exists, skipping."
        continue
    fi

    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === Starting mid_${LANG}_normfix (4 GPU) ==="

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file "$ACCEL" \
        --main_process_port 29500 \
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
