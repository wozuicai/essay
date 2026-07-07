#!/bin/bash
# Phase 2 launcher for H100 worker (en, fr, zh, sw)
# Sequential execution - each job uses both H100 GPUs
# Run via: nohup bash scripts/launch_phase2_h100.sh > logs/phase2_h100_master.log 2>&1 &

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /root/project
mkdir -p logs results/phase2_lis_matrix

echo "[$(date)] Setting up accelerate on H100..."
bash scripts/setup_accelerate.sh

ACCEL_CFG="configs/accelerate_fullgpu.yaml"
MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"

for LANG in en fr zh sw; do
    EXP_NAME="lis_${MODEL_SHORT}_train_${LANG}"
    OUT_DIR="results/phase2_lis_matrix/${EXP_NAME}"
    EVAL_OUT="results/phase2_lis_matrix/${EXP_NAME}_eval.json"

    # Skip if already done
    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] Skipping $EXP_NAME (eval already exists)"
        continue
    fi

    echo "[$(date)] === Training: $EXP_NAME ==="
    accelerate launch \
        --config_file "$ACCEL_CFG" \
        scripts/train.py \
        --model "$MODEL" \
        --train_lang "$LANG" \
        --train_samples 500 \
        --method standard_lora \
        --output_dir "$OUT_DIR" \
        --config configs/experiments/lis_matrix.yaml \
        --no_wandb \
        2>&1 | tee "logs/${EXP_NAME}_train.log"

    echo "[$(date)] === Evaluating: $EXP_NAME ==="
    python scripts/evaluate.py \
        --model_path "$OUT_DIR" \
        --tasks all \
        --languages fr,zh,sw,th,bn,yo \
        --skip_flores \
        --output "$EVAL_OUT" \
        2>&1 | tee "logs/${EXP_NAME}_eval.log"

    echo "[$(date)] === Done: $EXP_NAME ==="
done

echo "[$(date)] H100 Phase 2 COMPLETE (en, fr, zh, sw)"
