#!/bin/bash
set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME="${HF_HOME:-/home/tiger/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
export LOG_DIR="${LOG_DIR:-/root/result_new/logs/fe3462a_sft_eval}"

cd /root/project
mkdir -p "$LOG_DIR" /root/result_new/phase2_fe3462a

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
EXP_CFG="configs/experiments/lis_matrix.yaml"
OUT_ROOT="/root/result_new/phase2_fe3462a"
LANGS="en,yo,so,ha"

echo "[$(date)] HEAD: $(git rev-parse --short HEAD)"
echo "[$(date)] Setting up accelerate config..."
bash scripts/setup_accelerate.sh

for LANG in en yo so ha; do
    EXP_NAME="lis_${MODEL_SHORT}_train_${LANG}"
    OUT_DIR="${OUT_ROOT}/${EXP_NAME}"
    EVAL_OUT="${OUT_ROOT}/${EXP_NAME}_eval.json"

    if [[ -f "$OUT_DIR/adapter_model.safetensors" ]]; then
        echo "[$(date)] === Training: $LANG already has adapter, skipping train ==="
    else
        echo "[$(date)] === Training: $LANG ==="
        accelerate launch \
            --config_file "$ACCEL_CFG" \
            scripts/train.py \
            --model "$MODEL" \
            --train_lang "$LANG" \
            --method standard_lora \
            --output_dir "$OUT_DIR" \
            --config "$EXP_CFG" \
            --no_wandb \
            2>&1 | tee "$LOG_DIR/${EXP_NAME}_train.log"
    fi

    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] === Eval: $LANG already exists, skipping eval ==="
    else
        echo "[$(date)] === Evaluating: $LANG ==="
        python scripts/evaluate.py \
            --model_path "$OUT_DIR" \
            --tasks all \
            --languages "$LANGS" \
            --skip_flores \
            --batch_size "$EVAL_BATCH_SIZE" \
            --output "$EVAL_OUT" \
            2>&1 | tee "$LOG_DIR/${EXP_NAME}_eval.log"
    fi

    echo "[$(date)] === Done: $LANG ==="
done

echo "[$(date)] COMPLETE. Results in $OUT_ROOT"
