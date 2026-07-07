#!/bin/bash
# Tag Routing 评测脚本
# TruthfulQA MC1 + Belebele + SIB200 + IrokoBench MCQ + LCB-chat 4x4矩阵
set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0

cd /root/project

TAG_DIR=results/tag_routing/tag_routing_Qwen3.5-9B-Base
EVAL_OUT=results/tag_routing/tag_routing_Qwen3.5-9B-Base_eval.json
LCB_OUT=results/tag_routing/tag_routing_Qwen3.5-9B-Base_lcb_matrix.json

echo "[$(date -u)] === tag_routing eval: TruthfulQA+Belebele+SIB200 ==="
python scripts/evaluate.py \
    --model_path "$TAG_DIR" \
    --tasks all \
    --en_tasks truthfulqa_mc1 \
    --languages en,yo,so,ha \
    --skip_flores \
    --inject_lang_tag \
    --output "$EVAL_OUT" \
    2>&1 | tee logs/tag_routing_eval.log

echo "[$(date -u)] === tag_routing eval: IrokoBench MCQ ==="
python scripts/eval_extended.py \
    --model_path "$TAG_DIR" \
    --result_json "$EVAL_OUT" \
    --only_iroko_mcq \
    2>&1 | tee logs/tag_routing_iroko.log

echo "[$(date -u)] === tag_routing eval: LCB-chat 4x4 matrix ==="
python scripts/eval_lcb_matrix.py \
    --model_path "$TAG_DIR" \
    --output "$LCB_OUT" \
    2>&1 | tee logs/tag_routing_lcb.log

echo "[$(date -u)] tag_routing eval ALL DONE."
