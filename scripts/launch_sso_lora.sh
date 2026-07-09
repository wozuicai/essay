#!/bin/bash
# SSO-LoRA：Shared-Specific Orthogonal LoRA
#
# Stage 1: shared LoRA (all layers, r=16, 4-lang mixed, 2 epochs)
# Stage 2: per-lang LoRA (all layers, r=8) with orth penalty, sequential yo/so/ha
# Merge + Eval per language
#
# 用法：
#   # Stage 1（B200 GPU 4,5）
#   CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/launch_sso_lora.sh stage1 \
#       > logs/sso_lora_stage1.log 2>&1 &
#
#   # Stage 2（Stage 1 完成后）
#   CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/launch_sso_lora.sh stage2 \
#       > logs/sso_lora_stage2.log 2>&1 &

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# Fix CUDA 803: compat dir (575.x) in ldconfig overrides the real driver (580.x)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08
export TRITON_CACHE_DIR=/tmp/triton_cache

cd /root/project
mkdir -p logs results/sso_lora

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_2gpu.yaml"
EXP_CFG="configs/experiments/lis_matrix.yaml"
OUT_DIR="results/sso_lora"
STAGE1_DIR="${OUT_DIR}/stage1_shared"

STEP="${1:-stage1}"

N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-0,1}" | tr ',' '\n' | wc -l)
if [[ "$N_GPUS" -ge 4 ]]; then
    ACCEL_CFG="configs/accelerate_4gpu.yaml"
    ACCEL_CFG_STAGE2="configs/accelerate_4gpu_ddp.yaml"
else
    ACCEL_CFG_STAGE2="configs/accelerate_2gpu_ddp.yaml"
fi
echo "[$(date -u '+%H:%M:%S UTC')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}, accel=${ACCEL_CFG}, step=${STEP}"
python scripts/preflight_required.py \
    --model "$MODEL" \
    --data_dir data/processed \
    --langs en,yo,so,ha \
    --max_train_chars "$MAX_TRAIN_CHARS"

# ── Stage 1: 共享 LoRA（全层，4-lang，2 epochs）────────────────────────────
if [[ "$STEP" == "stage1" ]]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === SSO-LoRA Stage 1: Shared LoRA ==="

    if [[ -f "${STAGE1_DIR}/adapter_config.json" ]]; then
        echo "Stage 1 adapter exists, skipping."
    else
        accelerate launch --config_file "$ACCEL_CFG" --main_process_port 29503 \
            scripts/train_sso_lora.py \
            --model      "$MODEL" \
            --output_dir "$OUT_DIR" \
            --config     "$EXP_CFG" \
            --mode       stage1 \
            --r_shared   16 \
            --lora_alpha_shared 32.0 \
            --no_wandb \
            2>&1 | tee logs/sso_lora_stage1_train.log
        echo "[$(date -u '+%H:%M:%S UTC')] Stage 1 done → ${STAGE1_DIR}"
    fi
    echo "接下来运行: CUDA_VISIBLE_DEVICES=... bash scripts/launch_sso_lora.sh stage2"
fi

# ── Stage 2: 各语言特异 LoRA（全层，r=8，orth_weight=0.1，顺序执行）────────
if [[ "$STEP" == "stage2" ]]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === SSO-LoRA Stage 2: Lang-specific LoRA ==="

    if [[ ! -f "${STAGE1_DIR}/adapter_config.json" && ! -f "${STAGE1_DIR}/shared/adapter_config.json" ]]; then
        echo "ERROR: Stage 1 adapter not found at ${STAGE1_DIR}"
        exit 1
    fi

    for LANG in yo so ha; do
        STAGE2_DIR="${OUT_DIR}/stage2_${LANG}"
        EVAL_OUT="${OUT_DIR}/sso_${MODEL_SHORT}_${LANG}_eval.json"

        echo ""
        echo "[$(date -u '+%H:%M:%S UTC')] ─── Stage 2: ${LANG} ───"

        if [[ ! -f "${STAGE2_DIR}/${LANG}/adapter_config.json" ]]; then
            accelerate launch --config_file "$ACCEL_CFG_STAGE2" --main_process_port 29503 \
                scripts/train_sso_lora.py \
                --model       "$MODEL" \
                --output_dir  "$OUT_DIR" \
                --config      "$EXP_CFG" \
                --mode        stage2 \
                --train_lang  "$LANG" \
                --stage1_dir  "$STAGE1_DIR" \
                --r_lang      8 \
                --lora_alpha_lang 16.0 \
                --orth_weight 0.1 \
                --no_wandb \
                2>&1 | tee "logs/sso_lora_stage2_${LANG}.log"
            echo "[$(date -u '+%H:%M:%S UTC')] ${LANG} Stage 2 done."
        fi

        if [[ ! -f "$EVAL_OUT" ]]; then
            echo "[$(date -u '+%H:%M:%S UTC')] Merge+Eval ${LANG} (in-memory, no disk save) ..."
            python scripts/train_sso_lora.py \
                --model       "$MODEL" \
                --output_dir  "$OUT_DIR" \
                --config      "$EXP_CFG" \
                --mode        merge_eval \
                --train_lang  "$LANG" \
                --stage1_dir  "$STAGE1_DIR" \
                --eval_output "$EVAL_OUT" \
                2>&1 | tee "logs/sso_lora_eval_${LANG}.log"
            bash scripts/cleanup_large_artifacts.sh "$STAGE2_DIR"
            echo "[$(date -u '+%H:%M:%S UTC')] ${LANG} eval done → ${EVAL_OUT}"
        fi
    done

    echo ""
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === SSO-LoRA 全部完成 ==="
fi
