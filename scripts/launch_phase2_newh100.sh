#!/bin/bash
# Phase 2 completion script for new H100 worker (2×H100 80GB)
# - Evaluates already-trained adapters (en, fr, zh, th)
# - Trains + evaluates missing adapters (sw, yo, bn)
# Sequential execution; each job uses both H100 GPUs via DeepSpeed ZeRO-2
#
# Usage: nohup bash scripts/launch_phase2_newh100.sh > logs/phase2_newh100_master.log 2>&1 &

set -e

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache

cd /root/project
mkdir -p logs results/phase2_lis_matrix

echo "[$(date)] Setting up accelerate config for H100..."
bash scripts/setup_accelerate.sh

ACCEL_CFG="configs/accelerate_fullgpu.yaml"
MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"

run_eval() {
    local LANG="$1"
    local OUT_DIR="$2"
    local EVAL_OUT="$3"

    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] Skipping eval $LANG (eval JSON already exists)"
        return 0
    fi

    echo "[$(date)] === Evaluating train_${LANG} ==="
    python scripts/evaluate.py \
        --model_path "$OUT_DIR" \
        --tasks all \
        --languages en,fr,zh,sw,th,bn,yo \
        --skip_flores \
        --batch_size 16 \
        --output "$EVAL_OUT" \
        2>&1 | tee "logs/lis_${MODEL_SHORT}_train_${LANG}_eval_newh100.log"
    echo "[$(date)] === Eval done: train_${LANG} ==="
}

# ── Phase A: eval already-trained adapters (no training needed) ──────────────
for LANG in en fr zh th; do
    EXP_NAME="lis_${MODEL_SHORT}_train_${LANG}"
    OUT_DIR="results/phase2_lis_matrix/${EXP_NAME}"
    EVAL_OUT="results/phase2_lis_matrix/${EXP_NAME}_eval.json"

    if [[ ! -d "$OUT_DIR" ]] || [[ ! -f "$OUT_DIR/adapter_config.json" ]]; then
        echo "[$(date)] WARNING: adapter missing for $LANG, skipping"
        continue
    fi

    run_eval "$LANG" "$OUT_DIR" "$EVAL_OUT"
done

# ── Phase B: train + eval missing adapters ────────────────────────────────────
for LANG in sw yo bn; do
    EXP_NAME="lis_${MODEL_SHORT}_train_${LANG}"
    OUT_DIR="results/phase2_lis_matrix/${EXP_NAME}"
    EVAL_OUT="results/phase2_lis_matrix/${EXP_NAME}_eval.json"

    # Clean empty directory if it exists (failed previous run)
    if [[ -d "$OUT_DIR" ]] && [[ ! -f "$OUT_DIR/adapter_config.json" ]]; then
        echo "[$(date)] Removing empty/incomplete adapter dir for $LANG..."
        rm -rf "$OUT_DIR"
    fi

    if [[ ! -f "$EVAL_OUT" ]]; then
        if [[ ! -f "$OUT_DIR/adapter_config.json" ]]; then
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
                2>&1 | tee "logs/${EXP_NAME}_train_newh100.log"
            echo "[$(date)] === Training done: $EXP_NAME ==="
        else
            echo "[$(date)] Adapter already exists for $LANG, skipping training"
        fi

        run_eval "$LANG" "$OUT_DIR" "$EVAL_OUT"
    else
        echo "[$(date)] Skipping $EXP_NAME (eval already exists)"
    fi
done

echo "[$(date)] Phase 2 complete. All 7 language adapters trained and evaluated."
echo "[$(date)] Run: python scripts/compute_lis.py"
