#!/bin/bash
# launch_dsct.sh
# DSCT (Dual-Space Constrained Tuning) 实验
#
# Teacher  : Base + LoRA_donor (merged, frozen)
# Student  : Base + LoRA_donor (merged, frozen) + LoRA_spec (trainable)
# Loss     : L_CE + α·L_MID + λ·L_ortho
#
# 顺序训练：yo → so → ha
# 评测：TruthfulQA MC1 (带 tag) + Belebele + SIB200 + IrokoBench MCQ
#
# 用法：
#   nohup bash scripts/launch_dsct.sh > logs/dsct_master.log 2>&1 &

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

cd /root/project
mkdir -p logs results/dsct

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
DONOR_ADAPTER="results/phase2_v2/lis_${MODEL_SHORT}_train_en"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
EXP_CFG="configs/experiments/lis_matrix.yaml"
RESULTS="results/dsct"

bash scripts/setup_accelerate.sh
python scripts/preflight_required.py \
    --model "$MODEL" \
    --data_dir data/processed \
    --langs yo,so,ha \
    --max_train_chars "$MAX_TRAIN_CHARS"

for LANG in yo so ha; do
    EXP_NAME="dsct_${MODEL_SHORT}_${LANG}"
    OUT_DIR="${RESULTS}/${EXP_NAME}"
    EVAL_OUT="${RESULTS}/${EXP_NAME}_eval.json"

    echo ""
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ============================================"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === DSCT Training: ${LANG} ==="
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ============================================"

    if [[ -f "${OUT_DIR}/adapter_config.json" ]]; then
        echo "[$(date -u '+%H:%M:%S UTC')] Skipping training — adapter already exists."
    else
        accelerate launch --config_file "$ACCEL_CFG" scripts/train_dsct.py \
            --model              "$MODEL" \
            --donor_adapter      "$DONOR_ADAPTER" \
            --train_lang         "$LANG" \
            --output_dir         "$OUT_DIR" \
            --config             "$EXP_CFG" \
            --alpha              0.1 \
            --beta               0.05 \
            --lambda_ortho       0.01 \
            --top_n_layers       4 \
            --n_pos2             3 \
            2>&1 | tee "logs/dsct_${LANG}_train.log"
    fi

    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date -u '+%H:%M:%S UTC')] Skipping eval — ${EVAL_OUT} already exists."
    else
        echo "[$(date -u '+%H:%M:%S UTC')] === Eval: DSCT-${LANG} ==="
        # 注意：model_path 指向 spec adapter；eval 脚本从 training_metadata.json
        # 读取 donor_adapter，在加载时自动 merge donor + spec。
        python scripts/eval_required.py \
            --model_path  "$OUT_DIR" \
            --languages   en,yo,so,ha \
            --output      "$EVAL_OUT" \
            2>&1 | tee "logs/dsct_${LANG}_eval.log"
    fi

    bash scripts/cleanup_large_artifacts.sh "$OUT_DIR"

    echo "[$(date -u '+%H:%M:%S UTC')] === Done: DSCT-${LANG} ==="
done

echo ""
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] =============================="
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === DSCT 全部完成，结果汇总 ==="
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] =============================="

python3 - << 'PYEOF'
import json, os

RESULTS     = "/root/project/results"
MODEL_SHORT = "Qwen3.5-9B-Base"
LANGS       = ["yo", "so", "ha"]

def safe_get(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("scores", {})

def metrics(sc):
    if sc is None:
        return {}
    eng  = sc.get("english", {})
    bele = sc.get("multilingual", {}).get("belebele", {})
    iroko = sc.get("multilingual", {}).get("irokobench", {})
    return dict(
        tqa     = eng.get("truthfulqa_mc1"),
        bele_en = bele.get("en"),
        bele_yo = bele.get("yo"),
        bele_so = bele.get("so"),
        bele_ha = bele.get("ha"),
        mcq_yo  = (iroko.get("yo") or {}).get("mcq_accuracy"),
        mcq_ha  = (iroko.get("ha") or {}).get("mcq_accuracy"),
    )

def fmt(v):
    return f"{v:.4f}" if v is not None else "  N/A"

rows = [
    ("baseline",  f"{RESULTS}/phase2_v2/{MODEL_SHORT}_baseline.json"),
    ("MID_yo",    f"{RESULTS}/mid/mid_{MODEL_SHORT}_yo_eval.json"),
    ("MID_so",    f"{RESULTS}/mid/mid_{MODEL_SHORT}_so_eval.json"),
    ("MID_ha",    f"{RESULTS}/mid/mid_{MODEL_SHORT}_ha_eval.json"),
    ("DSCT_yo",   f"{RESULTS}/dsct/dsct_{MODEL_SHORT}_yo_eval.json"),
    ("DSCT_so",   f"{RESULTS}/dsct/dsct_{MODEL_SHORT}_so_eval.json"),
    ("DSCT_ha",   f"{RESULTS}/dsct/dsct_{MODEL_SHORT}_ha_eval.json"),
]

hdr = f"{'model':<12} {'tqa':>6} | {'bele_en':>7} {'bele_yo':>7} {'bele_so':>7} {'bele_ha':>7} | {'mcq_yo':>6} {'mcq_ha':>6}"
print(hdr)
print("-" * len(hdr))
for name, path in rows:
    m = metrics(safe_get(path))
    print(f"{name:<12} {fmt(m.get('tqa')):>6} | "
          f"{fmt(m.get('bele_en')):>7} {fmt(m.get('bele_yo')):>7} "
          f"{fmt(m.get('bele_so')):>7} {fmt(m.get('bele_ha')):>7} | "
          f"{fmt(m.get('mcq_yo')):>6} {fmt(m.get('mcq_ha')):>6}")
PYEOF

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] launch_dsct.sh complete."
