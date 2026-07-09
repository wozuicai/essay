#!/bin/bash
set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME="${HF_HOME:-/root/project/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
export LOG_DIR="${LOG_DIR:-logs/phase2_v3}"

cd /root/project
mkdir -p "$LOG_DIR" results/phase2_v3

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
EXP_CFG="configs/experiments/lis_matrix_phase2_v3.yaml"
LANGS="en,yo,ha,so"

echo "[$(date)] Setting up accelerate config..."
bash scripts/setup_accelerate.sh
python scripts/preflight_required.py --model "$MODEL" --data_dir data/processed --langs "$LANGS" --max_train_chars "$MAX_TRAIN_CHARS"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

for LANG in en yo ha so; do
    EXP_NAME="lis_${MODEL_SHORT}_train_${LANG}"
    OUT_DIR="results/phase2_v3/${EXP_NAME}"
    EVAL_OUT="results/phase2_v3/${EXP_NAME}_eval.json"

    if [[ -f "$OUT_DIR/adapter_model.safetensors" ]]; then
        echo "[$(date)] === Training: $LANG already has adapter, skipping train ==="
    else
        echo "[$(date)] === Training: $LANG ==="
        accelerate launch --config_file "$ACCEL_CFG" scripts/train.py --model "$MODEL" --train_lang "$LANG" --method standard_lora --output_dir "$OUT_DIR" --config "$EXP_CFG" --no_wandb 2>&1 | tee "$LOG_DIR/phase2_v3_${LANG}_train.log"
    fi

    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] === Eval: $LANG already exists, skipping eval ==="
    else
        echo "[$(date)] === Evaluating: $LANG ==="
        python scripts/eval_required.py --model_path "$OUT_DIR" --languages en,yo,ha,so --output "$EVAL_OUT" --batch_size "$EVAL_BATCH_SIZE" --generation_batch_size "$GENERATION_BATCH_SIZE" 2>&1 | tee "$LOG_DIR/phase2_v3_${LANG}_eval.log"
    fi

    bash scripts/cleanup_large_artifacts.sh "$OUT_DIR"
    echo "[$(date)] === Done: $LANG ==="
done

echo "[$(date)] Phase 2 v3 train+eval COMPLETE. Results in results/phase2_v3/"
