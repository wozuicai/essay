#!/bin/bash
# launch_mix_en.sh
# Phase 3 数据混合实验：en+yo / en+so / en+ha 全量 concat shuffle SFT
# 顺序训练（4×H100 ZeRO-2），训练完即评测，评测后补跑 IrokoBench MCQ
#
# 评测内容：
#   - Belebele（en/yo/so/ha 四语言）
#   - TruthfulQA MC1（英文）
#   - IrokoBench AfriMMLU MCQ（yo/ha）
#
# 用法：nohup bash scripts/launch_mix_en.sh > logs/mix_en_master.log 2>&1 &

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-12000}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

cd /root/project
mkdir -p logs results/mix_en

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
EXP_CFG="configs/experiments/lis_matrix.yaml"
RESULTS="results/mix_en"

bash scripts/setup_accelerate.sh
python scripts/preflight_required.py \
    --model "$MODEL" \
    --data_dir data/processed \
    --langs en,yo,so,ha \
    --max_train_chars "$MAX_TRAIN_CHARS"

for LANG in yo so ha; do
    EXP_NAME="mix_${MODEL_SHORT}_en_${LANG}"
    OUT_DIR="${RESULTS}/${EXP_NAME}"
    EVAL_OUT="${RESULTS}/${EXP_NAME}_eval.json"

    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] Skipping $LANG — eval already exists."
        continue
    fi

    # ── 训练 ──────────────────────────────────────────────────────────────────
    echo "[$(date)] === Training: mix(en+${LANG}) ==="
    accelerate launch \
        --config_file "$ACCEL_CFG" \
        scripts/train.py \
        --model       "$MODEL" \
        --train_lang  "$LANG" \
        --method      standard_lora \
        --mix_all \
        --output_dir  "$OUT_DIR" \
        --config      "$EXP_CFG" \
        --no_wandb \
        2>&1 | tee "logs/mix_en_${LANG}_train.log"

    # ── 评测：English + Belebele + IrokoBench ───────────────────────────────
    echo "[$(date)] === Evaluating: mix(en+${LANG}) — required suite ==="
    python scripts/eval_required.py \
        --model_path  "$OUT_DIR" \
        --languages   en,yo,so,ha \
        --output      "$EVAL_OUT" \
        2>&1 | tee "logs/mix_en_${LANG}_eval.log"

    bash scripts/cleanup_large_artifacts.sh "$OUT_DIR"

    echo "[$(date)] === Done: mix(en+${LANG}) ==="
done

# ── 汇总 ──────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === 全部完成，结果汇总 ==="
python3 - << 'EOF'
import json, os

RESULTS = "/root/project/results/mix_en"
MODEL_SHORT = "Qwen3.5-9B-Base"
LANGS = ["yo", "so", "ha"]

# 读 baseline 作对比
BASE_PATH = "/root/project/results/phase2_v2/Qwen3.5-9B-Base_baseline.json"
with open(BASE_PATH) as f:
    base = json.load(f)["scores"]

print(f"{'model':<20} {'tqa_mc1':>9} | {'bele_en':>8} {'bele_yo':>8} {'bele_so':>8} {'bele_ha':>8} | {'mcq_yo':>7} {'mcq_ha':>7}")
print("-" * 85)

# baseline row
b_tqa = base["english"].get("truthfulqa_mc1", "N/A")
b_bele = base["multilingual"]["belebele"]
b_mcq  = base["multilingual"]["irokobench"]
print(f"{'baseline':<20} {b_tqa:>9.4f} | {b_bele['en']:>8.4f} {b_bele['yo']:>8.4f} {b_bele['so']:>8.4f} {b_bele['ha']:>8.4f} | {b_mcq['yo']['mcq_accuracy']:>7.4f} {b_mcq['ha']['mcq_accuracy']:>7.4f}")

for lang in LANGS:
    path = f"{RESULTS}/mix_{MODEL_SHORT}_en_{lang}_eval.json"
    if not os.path.exists(path):
        print(f"{'mix_en_'+lang:<20} {'N/A (not done)':>9}")
        continue
    with open(path) as f:
        d = json.load(f)["scores"]
    tqa  = d.get("english", {}).get("truthfulqa_mc1", float("nan"))
    bele = d.get("multilingual", {}).get("belebele", {})
    mcq  = d.get("multilingual", {}).get("irokobench", {})
    print(f"{'mix_en_'+lang:<20} {tqa:>9.4f} | {bele.get('en',float('nan')):>8.4f} {bele.get('yo',float('nan')):>8.4f} {bele.get('so',float('nan')):>8.4f} {bele.get('ha',float('nan')):>8.4f} | {mcq.get('yo',{}).get('mcq_accuracy',float('nan')):>7.4f} {mcq.get('ha',{}).get('mcq_accuracy',float('nan')):>7.4f}")
EOF
