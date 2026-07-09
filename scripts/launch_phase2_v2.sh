#!/bin/bash
# Phase 2 v2: baseline + SFT LIS matrix for en+yo+so+ha
# Languages: English (24926), Yoruba (11758), Somali (7704), Hausa (3512)
# Config: rank=16, 2 epochs, full data, 4×H100 ZeRO-2 per job (sequential)
#
# Usage: nohup bash scripts/launch_phase2_v2.sh > logs/phase2_v2_master.log 2>&1 &

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"

cd /root/project
mkdir -p logs results/phase2_v2

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
EXP_CFG="configs/experiments/lis_matrix.yaml"
BASELINE_OUT="results/phase2_v2/${MODEL_SHORT}_baseline.json"

echo "[$(date)] Setting up accelerate config..."
bash scripts/setup_accelerate.sh
python scripts/preflight_required.py \
    --model "$MODEL" \
    --data_dir data/processed \
    --langs en,yo,so,ha \
    --max_train_chars "$MAX_TRAIN_CHARS"

# ── Step 1: Baseline (base model, no SFT) ─────────────────────────────────────

if [[ -f "$BASELINE_OUT" ]]; then
    echo "[$(date)] Baseline already exists, skipping."
else
    echo "[$(date)] === Baseline: base model on en+yo+so+ha ==="
    python scripts/eval_required.py \
        --model_path "$MODEL" \
        --languages  en,yo,so,ha \
        --output     "$BASELINE_OUT" \
        2>&1 | tee logs/phase2_v2_baseline.log
    echo "[$(date)] Baseline done."
fi

# ── Step 2: SFT + eval for each language (sequential, 4 GPUs each) ────────────

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

for LANG in en yo so ha; do
    EXP_NAME="lis_${MODEL_SHORT}_train_${LANG}"
    OUT_DIR="results/phase2_v2/${EXP_NAME}"
    EVAL_OUT="results/phase2_v2/${EXP_NAME}_eval.json"

    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] Skipping $LANG — eval already exists."
        continue
    fi

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
        2>&1 | tee "logs/phase2_v2_${LANG}_train.log"

    echo "[$(date)] === Evaluating: $LANG ==="
    python scripts/eval_required.py \
        --model_path "$OUT_DIR" \
        --languages  en,yo,so,ha \
        --output     "$EVAL_OUT" \
        2>&1 | tee "logs/phase2_v2_${LANG}_eval.log"

    bash scripts/cleanup_large_artifacts.sh "$OUT_DIR"

    echo "[$(date)] === Done: $LANG ==="
done

# ── Step 3: LIS matrix ────────────────────────────────────────────────────────

echo "[$(date)] === Computing 4×4 LIS matrix ==="
python scripts/compute_lis.py \
    --results_dir results/phase2_v2 \
    --baseline_dir results/phase2_v2 \
    --output_dir results/phase2_v2

echo "[$(date)] Phase 2 v2 COMPLETE. Results in results/phase2_v2/"
