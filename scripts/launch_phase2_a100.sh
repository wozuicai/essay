#!/bin/bash
# Phase 2 launcher for A100 worker: prepare_mt_bench first, then th, bn, yo
# Sequential execution - each job uses both A100 GPUs
# Run via: nohup bash scripts/launch_phase2_a100.sh > logs/phase2_a100_master.log 2>&1 &

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TRITON_CACHE_DIR=/tmp/triton_cache
# libnvidia-ml.so.1 symlink in container points to empty 0-byte placeholder (535.129.03).
# Real library is 535.161.08. Copy to /tmp/nv_libs and add to LD_LIBRARY_PATH.
# This lets NCCL find the library without shadowing libcuda.so.1 (which breaks CUDA).
mkdir -p /tmp/nv_libs
if [[ ! -s /tmp/nv_libs/libnvidia-ml.so.1 ]]; then
    cp /lib/x86_64-linux-gnu/libnvidia-ml.so.535.161.08 /tmp/nv_libs/libnvidia-ml.so.1
fi
export LD_LIBRARY_PATH=/tmp/nv_libs:${LD_LIBRARY_PATH}
# Use default HF cache (has NLLB + eval datasets)

cd /root/project
mkdir -p logs results/phase2_lis_matrix data

echo "[$(date)] Setting up accelerate on A100..."
bash scripts/setup_accelerate.sh

# Step 1: Prepare MT-Bench multilingual prompts
if [[ ! -f "data/mt_bench_multilingual.json" ]]; then
    echo "[$(date)] Preparing MT-Bench multilingual prompts..."
    python scripts/prepare_mt_bench.py
    echo "[$(date)] MT-Bench prep done."
else
    echo "[$(date)] MT-Bench prompts already exist, skipping."
fi

# Step 2: Phase 2 training for remaining languages
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"

for LANG in th bn yo; do
    EXP_NAME="lis_${MODEL_SHORT}_train_${LANG}"
    OUT_DIR="results/phase2_lis_matrix/${EXP_NAME}"
    EVAL_OUT="results/phase2_lis_matrix/${EXP_NAME}_eval.json"

    # Skip if already done
    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] Skipping $EXP_NAME (eval already exists)"
        continue
    fi

    echo "[$(date)] === Training: $EXP_NAME ==="
    # Inline LD_LIBRARY_PATH so NCCL finds libnvidia-ml.so.1 even if parent env is stale
    mkdir -p /tmp/nv_libs
    [[ -s /tmp/nv_libs/libnvidia-ml.so.1 ]] || cp /lib/x86_64-linux-gnu/libnvidia-ml.so.535.161.08 /tmp/nv_libs/libnvidia-ml.so.1
    LD_LIBRARY_PATH=/tmp/nv_libs:${LD_LIBRARY_PATH} accelerate launch \
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

echo "[$(date)] A100 Phase 2 COMPLETE (th, bn, yo)"
