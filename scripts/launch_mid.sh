#!/bin/bash
# launch_mid.sh
# MID (Mechanistic Interface Distillation) 实验
#
# Teacher: Base + LoRA_en (merged, frozen)
# Student: Base + LoRA_spec (纯目标语言 CE + top-K层 CosDist 约束)
# 核心 claim: 无需英文数据，仅靠潜空间约束维持指令接口
#
# 顺序训练：yo → so → ha（各自独立 LoRA_spec）
# 评测：TruthfulQA MC1 + Belebele + SIB200 + IrokoBench MCQ + LCB-chat 4×4矩阵
#
# 用法：
#   nohup bash scripts/launch_mid.sh > logs/mid_master.log 2>&1 &

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-24000}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

cd /root/project
mkdir -p logs results/mid

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
TEACHER_ADAPTER="results/phase2_v2/lis_${MODEL_SHORT}_train_en"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
EXP_CFG="configs/experiments/lis_matrix.yaml"
RESULTS="results/mid"

bash scripts/setup_accelerate.sh
python scripts/preflight_required.py \
    --model "$MODEL" \
    --data_dir data/processed \
    --langs yo,so,ha \
    --max_train_chars "$MAX_TRAIN_CHARS"

# ── Step 0: Probe（单 GPU 快速验证 teacher 信号质量）──────────────────────────
echo "[$(date)] === Probe: teacher hidden-state consistency on yo/so/ha ==="
python scripts/train_mid.py \
    --model           "$MODEL" \
    --teacher_adapter "$TEACHER_ADAPTER" \
    --train_lang      yo \
    --output_dir      /tmp/mid_probe_unused \
    --config          "$EXP_CFG" \
    --probe_only \
    --probe_langs     yo,so,ha \
    2>&1 | tee logs/mid_probe.log
echo "[$(date)] Probe done. See logs/mid_probe.log"

# ── Step 1–N: 每种语言训练 + 评测 ────────────────────────────────────────────
for LANG in yo so ha; do
    EXP_NAME="mid_${MODEL_SHORT}_${LANG}"
    OUT_DIR="${RESULTS}/${EXP_NAME}"
    EVAL_OUT="${RESULTS}/${EXP_NAME}_eval.json"
    echo ""
    echo "[$(date)] =================================================="
    echo "[$(date)] === MID Training: ${LANG} (no English data) ==="
    echo "[$(date)] =================================================="

    # ── 训练 ──────────────────────────────────────────────────────────────
    if [[ -f "${OUT_DIR}/adapter_config.json" ]]; then
        echo "[$(date)] Skipping training — adapter_config.json already exists."
    else
        accelerate launch --config_file "$ACCEL_CFG" scripts/train_mid.py \
            --model           "$MODEL" \
            --teacher_adapter "$TEACHER_ADAPTER" \
            --train_lang      "$LANG" \
            --output_dir      "$OUT_DIR" \
            --config          "$EXP_CFG" \
            --alpha           0.1 \
            --beta            0.05 \
            --top_n_layers    4 \
            --n_pos2          3 \
            2>&1 | tee "logs/mid_${LANG}_train.log"
    fi

    # ── 评测：English + Belebele + IrokoBench ────────────────────────────
    if [[ -f "$EVAL_OUT" ]]; then
        echo "[$(date)] Skipping eval — ${EVAL_OUT} already exists."
    else
        echo "[$(date)] === Eval: MID-${LANG} — required suite ==="
        python scripts/eval_required.py \
            --model_path  "$OUT_DIR" \
            --languages   en,yo,so,ha \
            --output      "$EVAL_OUT" \
            2>&1 | tee "logs/mid_${LANG}_eval.log"
    fi

    bash scripts/cleanup_large_artifacts.sh "$OUT_DIR"

    echo "[$(date)] === Done: MID-${LANG} ==="
done

# ── 汇总 ──────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] ======================================================"
echo "[$(date)] === 全部完成，MID 实验结果汇总 ==="
echo "[$(date)] ======================================================"

python3 - << 'PYEOF'
import json, os

RESULTS     = "/root/project/results/mid"
MIX_DIR     = "/root/project/results/mix_en"
MODEL_SHORT = "Qwen3.5-9B-Base"
LANGS       = ["yo", "so", "ha"]

BASE_PATH = f"/root/project/results/phase2_v2/{MODEL_SHORT}_baseline.json"

def safe_load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("scores", {})

def get_metrics(scores):
    if scores is None:
        return {}
    tqa  = scores.get("english",       {}).get("truthfulqa_mc1", float("nan"))
    bele = scores.get("multilingual",  {}).get("belebele",        {})
    mcq  = scores.get("multilingual",  {}).get("irokobench",      {})
    return {
        "tqa":     tqa,
        "bele_en": bele.get("en",   float("nan")),
        "bele_yo": bele.get("yo",   float("nan")),
        "bele_so": bele.get("so",   float("nan")),
        "bele_ha": bele.get("ha",   float("nan")),
        "mcq_yo":  mcq.get("yo",    {}).get("mcq_accuracy", float("nan")),
        "mcq_ha":  mcq.get("ha",    {}).get("mcq_accuracy", float("nan")),
    }

def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and v == v else " N/A "

hdr = f"{'model':<22} {'tqa':>6} | {'bele_en':>7} {'bele_yo':>7} {'bele_so':>7} {'bele_ha':>7} | {'mcq_yo':>7} {'mcq_ha':>7}"
print(hdr)
print("-" * len(hdr))

def print_row(name, path):
    m = get_metrics(safe_load(path))
    print(f"{name:<22} {fmt(m.get('tqa',float('nan'))):>6} | "
          f"{fmt(m.get('bele_en',float('nan'))):>7} {fmt(m.get('bele_yo',float('nan'))):>7} "
          f"{fmt(m.get('bele_so',float('nan'))):>7} {fmt(m.get('bele_ha',float('nan'))):>7} | "
          f"{fmt(m.get('mcq_yo',float('nan'))):>7} {fmt(m.get('mcq_ha',float('nan'))):>7}")

print_row("baseline", BASE_PATH)
for lang in LANGS:
    print_row(f"mix_en_{lang}", f"{MIX_DIR}/mix_{MODEL_SHORT}_en_{lang}_eval.json")
for lang in LANGS:
    print_row(f"MID_{lang}",    f"{RESULTS}/mid_{MODEL_SHORT}_{lang}_eval.json")

PYEOF

echo "[$(date)] launch_mid.sh complete."
