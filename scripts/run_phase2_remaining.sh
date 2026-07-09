#!/usr/bin/env bash
# Phase 2 completion: eval fr/zh/th, then train+eval sw/yo/bn
set -e
export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /root/project
mkdir -p logs results/phase2_lis_matrix

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"

run_eval() {
    local LANG="$1"
    local ADAPTER_DIR="$2"
    local EVAL_OUT="$3"
    local LOG="logs/lis_${MODEL_SHORT}_train_${LANG}_eval.log"

    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] Skipping eval $LANG (already exists)"
        return 0
    fi

    echo "[$(date)] === Eval: train_${LANG} ==="
    python3 -u scripts/evaluate.py \
        --model_path "$ADAPTER_DIR" \
        --tasks all \
        --languages en,fr,zh,sw,th,bn,yo \
        --skip_flores \
        --batch_size 8 \
        --output "$EVAL_OUT" \
        2>&1 | tee "$LOG"
    echo "[$(date)] === Eval done: train_${LANG} ==="
}

run_train() {
    local LANG="$1"
    local OUT_DIR="$2"
    local LOG="logs/lis_${MODEL_SHORT}_train_${LANG}_train.log"

    if [[ -f "$OUT_DIR/adapter_config.json" ]]; then
        echo "[$(date)] Adapter already exists for $LANG, skipping training"
        return 0
    fi

    # Remove incomplete dir if exists
    [[ -d "$OUT_DIR" ]] && rm -rf "$OUT_DIR"

    echo "[$(date)] === Train: train_${LANG} ==="
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
        2>&1 | tee "$LOG"
    echo "[$(date)] === Train done: train_${LANG} ==="
}

# ── Already-trained adapters: eval only ──────────────────────────────────────
for LANG in fr zh th; do
    EXP="lis_${MODEL_SHORT}_train_${LANG}"
    ADAPTER="results/phase2_lis_matrix/${EXP}"
    EVAL_OUT="results/phase2_lis_matrix/${EXP}_eval.json"

    if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
        echo "[$(date)] WARNING: adapter missing for $LANG, skipping"
        continue
    fi
    run_eval "$LANG" "$ADAPTER" "$EVAL_OUT"
done

# ── Missing adapters: train then eval ────────────────────────────────────────
for LANG in sw yo bn; do
    EXP="lis_${MODEL_SHORT}_train_${LANG}"
    ADAPTER="results/phase2_lis_matrix/${EXP}"
    EVAL_OUT="results/phase2_lis_matrix/${EXP}_eval.json"

    run_train "$LANG" "$ADAPTER"
    run_eval "$LANG" "$ADAPTER" "$EVAL_OUT"
done

echo "[$(date)] Phase 2 complete. All 7 adapters evaluated."
echo "[$(date)] Run: python scripts/compute_lis.py"
