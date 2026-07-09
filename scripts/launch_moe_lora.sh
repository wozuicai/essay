#!/bin/bash
# launch_moe_lora.sh
# Soft MoE-LoRA (LA-MoA) 训练 + 评测
#
# 架构：K=4 LoRA expert + token-level soft router，所有 target 线性层替换
# 数据：en+yo+so+ha 全量 concat，1 epoch（同 tag_routing 4-lang）
#
# 运行机器：B200 worker（GPU 0-1，前两张空闲）
# 用法：
#   nohup bash scripts/launch_moe_lora.sh > logs/moe_lora.log 2>&1 &
#
# GPU 说明：默认使用 GPU 0-1（CUDA_VISIBLE_DEVICES=0,1）
# 若需要使用 GPU 2-3，在运行前：
#   export CUDA_VISIBLE_DEVICES=2,3

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# Fix CUDA 803: compat dir (575.x) in ldconfig overrides the real driver (580.x)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08

# 默认 GPU 0-1，若外部已设置 CUDA_VISIBLE_DEVICES 则保留
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd /root/project
mkdir -p logs results/moe_lora

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_2gpu.yaml"
EXP_CFG="configs/experiments/lis_matrix.yaml"
OUT_DIR="results/moe_lora/moe_lora_${MODEL_SHORT}"
EVAL_OUT="results/moe_lora/moe_lora_${MODEL_SHORT}_eval.json"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === MoE-LoRA Launch ==="
echo "[$(date -u '+%H:%M:%S UTC')] GPU: $CUDA_VISIBLE_DEVICES"
echo "[$(date -u '+%H:%M:%S UTC')] Output: $OUT_DIR"
python scripts/preflight_required.py \
    --model "$MODEL" \
    --data_dir data/processed \
    --langs en,yo,so,ha \
    --max_train_chars "$MAX_TRAIN_CHARS"

# ── Training ──────────────────────────────────────────────────────────────
if [[ -f "${OUT_DIR}/moe_config.json" ]]; then
    echo "[$(date -u '+%H:%M:%S UTC')] MoE weights found, skipping training."
else
    echo "[$(date -u '+%H:%M:%S UTC')] Starting MoE-LoRA training ..."
    accelerate launch --config_file "$ACCEL_CFG" --main_process_port 29501 scripts/train_moe_lora.py \
        --model       "$MODEL" \
        --output_dir  "$OUT_DIR" \
        --config      "$EXP_CFG" \
        --n_experts   4 \
        --r           8 \
        --lora_alpha  16.0 \
        --no_wandb \
        2>&1 | tee logs/moe_lora_train.log
    echo "[$(date -u '+%H:%M:%S UTC')] Training done."
fi

# ── Evaluation ────────────────────────────────────────────────────────────
if [[ -f "$EVAL_OUT" ]]; then
    echo "[$(date -u '+%H:%M:%S UTC')] Eval JSON found, skipping eval."
else
    echo "[$(date -u '+%H:%M:%S UTC')] Starting MoE-LoRA evaluation ..."
    # 单 GPU 评测（使用第一张可见 GPU）
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
    python scripts/eval_required.py \
        --moe_dir   "$OUT_DIR" \
        --languages en,yo,so,ha \
        --output    "$EVAL_OUT" \
        2>&1 | tee logs/moe_lora_eval.log
    bash scripts/cleanup_large_artifacts.sh "$OUT_DIR"
    echo "[$(date -u '+%H:%M:%S UTC')] Evaluation done → $EVAL_OUT"
fi

echo ""
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === MoE-LoRA 完成 ==="
echo ""
python3 - << 'PYEOF'
import json, os

path = "results/moe_lora/moe_lora_Qwen3.5-9B-Base_eval.json"
if not os.path.exists(path):
    print("Eval JSON not found yet.")
else:
    with open(path) as f:
        r = json.load(f)
    sc = r.get("scores", {})
    eng = sc.get("english", {})
    ml  = sc.get("multilingual", {})
    bele = ml.get("belebele", {})
    iroko = ml.get("irokobench", {})
    lcb   = ml.get("lcb_chat", {})
    print(f"TruthfulQA MC1 : {eng.get('truthfulqa_mc1', 'N/A'):.4f}")
    for lang in ["en", "yo", "so", "ha"]:
        v = bele.get(lang)
        print(f"Belebele {lang:2s}    : {v:.4f}" if v else f"Belebele {lang}: N/A")
    for lang in ["yo", "ha"]:
        v = (iroko.get(lang) or {}).get("mcq_accuracy")
        print(f"Iroko MCQ {lang}  : {v:.4f}" if v else f"Iroko MCQ {lang}: N/A")
    for lang in ["yo", "so", "ha"]:
        v = (lcb.get(lang) or {}).get("lc_rate")
        print(f"LCB-chat {lang}   : {v:.3f}" if v else f"LCB-chat {lang}: N/A")
PYEOF
