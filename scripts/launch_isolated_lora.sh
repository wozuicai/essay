#!/bin/bash
# launch_isolated_lora.sh
# Isolated-LoRA 实验：yo / so / ha 各训练一组 (shared + lang adapter)
#
# 设计：
#   Stage 1: shared adapter on full en + full lang, 1 epoch
#   Stage 2: lang adapter on full lang only, 1 epoch
#   训练完自动 merge 两个 adapter，保存为标准 HF 模型供 evaluate.py 使用
#
# 评测内容：
#   - TruthfulQA MC1（英文）
#   - Belebele（en/yo/so/ha）
#   - IrokoBench AfriMMLU MCQ（yo/ha）
#   - LCB-chat（lc_rate / en_leak，yo/so/ha，n=50）
#
# 用法：nohup bash scripts/launch_isolated_lora.sh > logs/isolated_lora_master.log 2>&1 &

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0

cd /root/project
mkdir -p logs results/isolated_lora

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
EXP_CFG="configs/experiments/lora_comparison.yaml"
RESULTS="results/isolated_lora"

bash scripts/setup_accelerate.sh

for LANG in yo so ha; do
    EXP_NAME="isolated_${MODEL_SHORT}_${LANG}"
    OUT_DIR="${RESULTS}/${EXP_NAME}"
    EVAL_OUT="${RESULTS}/${EXP_NAME}_eval.json"
    LCB_OUT="${RESULTS}/${EXP_NAME}_lcb_chat.json"

    if [[ -f "$EVAL_OUT" && -f "$LCB_OUT" ]]; then
        echo "[$(date)] Skipping $LANG — eval already exists."
        continue
    fi

    # ── 训练（两阶段，merge 后保存到 OUT_DIR）────────────────────────────────
    echo "[$(date)] === Training: isolated_lora (${LANG}) ==="
    accelerate launch \
        --config_file "$ACCEL_CFG" \
        scripts/train.py \
        --model       "$MODEL" \
        --train_lang  "$LANG" \
        --method      isolated_lora \
        --output_dir  "$OUT_DIR" \
        --config      "$EXP_CFG" \
        --no_wandb \
        2>&1 | tee "logs/isolated_lora_${LANG}_train.log"

    # ── 评测：Belebele + TruthfulQA MC1 ─────────────────────────────────────
    echo "[$(date)] === Evaluating: isolated_lora (${LANG}) — Belebele + TruthfulQA ==="
    python scripts/evaluate.py \
        --model_path  "$OUT_DIR" \
        --tasks       all \
        --en_tasks    truthfulqa_mc1 \
        --languages   en,yo,so,ha \
        --skip_flores \
        --output      "$EVAL_OUT" \
        2>&1 | tee "logs/isolated_lora_${LANG}_eval.log"

    # ── 评测：IrokoBench MCQ ─────────────────────────────────────────────────
    echo "[$(date)] === Evaluating: isolated_lora (${LANG}) — IrokoBench MCQ ==="
    python scripts/eval_extended.py \
        --model_path  "$OUT_DIR" \
        --result_json "$EVAL_OUT" \
        --only_iroko_mcq \
        2>&1 | tee "logs/isolated_lora_${LANG}_iroko.log"

    # ── 评测：LCB-chat ───────────────────────────────────────────────────────
    echo "[$(date)] === Evaluating: isolated_lora (${LANG}) — LCB-chat ==="
    python scripts/eval_lcb_chat.py \
        --model_path  "$OUT_DIR" \
        --langs       yo,so,ha \
        --output      "$LCB_OUT" \
        2>&1 | tee "logs/isolated_lora_${LANG}_lcb_chat.log"

    echo "[$(date)] === Done: isolated_lora (${LANG}) ==="
done

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === 全部完成，结果汇总 ==="
python3 - << 'EOF'
import json, os

RESULTS = "/root/project/results/isolated_lora"
MODEL_SHORT = "Qwen3.5-9B-Base"
LANGS = ["yo", "so", "ha"]

BASE_PATH = "/root/project/results/phase2_v2/Qwen3.5-9B-Base_baseline.json"
MIX_DIR   = "/root/project/results/mix_en"

def load_scores(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["scores"]

print(f"\n{'model':<28} {'tqa_mc1':>9} | {'bele_en':>8} {'bele_yo':>8} {'bele_so':>8} {'bele_ha':>8} | {'mcq_yo':>7} {'mcq_ha':>7}")
print("-" * 95)

# baseline
base = load_scores(BASE_PATH)
if base:
    b = base["multilingual"]["belebele"]
    m = base["multilingual"]["irokobench"]
    print(f"{'baseline':<28} {base['english']['truthfulqa_mc1']:>9.4f} | {b['en']:>8.4f} {b['yo']:>8.4f} {b['so']:>8.4f} {b['ha']:>8.4f} | {m['yo']['mcq_accuracy']:>7.4f} {m['ha']['mcq_accuracy']:>7.4f}")

for lang in LANGS:
    # mix 对照
    mix_s = load_scores(f"{MIX_DIR}/mix_{MODEL_SHORT}_en_{lang}_eval.json")
    if mix_s:
        b = mix_s["multilingual"]["belebele"]
        m = mix_s["multilingual"].get("irokobench", {})
        print(f"{'mix_en_'+lang:<28} {mix_s['english']['truthfulqa_mc1']:>9.4f} | {b['en']:>8.4f} {b['yo']:>8.4f} {b['so']:>8.4f} {b['ha']:>8.4f} | {m.get('yo',{}).get('mcq_accuracy',float('nan')):>7.4f} {m.get('ha',{}).get('mcq_accuracy',float('nan')):>7.4f}")

    # isolated
    iso_s = load_scores(f"{RESULTS}/isolated_{MODEL_SHORT}_{lang}_eval.json")
    if iso_s:
        b = iso_s["multilingual"]["belebele"]
        m = iso_s["multilingual"].get("irokobench", {})
        print(f"{'isolated_lora_'+lang:<28} {iso_s['english']['truthfulqa_mc1']:>9.4f} | {b['en']:>8.4f} {b['yo']:>8.4f} {b['so']:>8.4f} {b['ha']:>8.4f} | {m.get('yo',{}).get('mcq_accuracy',float('nan')):>7.4f} {m.get('ha',{}).get('mcq_accuracy',float('nan')):>7.4f}")
    else:
        print(f"{'isolated_lora_'+lang:<28} {'N/A (not done)':>9}")

print("\n--- LCB-chat (lc_rate / en_leak) ---")
print(f"{'model':<28} {'yo_lc':>7} {'yo_en':>7} {'so_lc':>7} {'so_en':>7} {'ha_lc':>7} {'ha_en':>7}")
print("-" * 65)

for lang in LANGS:
    lcb_path = f"{RESULTS}/isolated_{MODEL_SHORT}_{lang}_lcb_chat.json"
    if not os.path.exists(lcb_path):
        print(f"{'isolated_lora_'+lang:<28} N/A")
        continue
    with open(lcb_path) as f:
        lcb = json.load(f)["scores"]
    row = [lcb.get(l, {}) for l in ["yo", "so", "ha"]]
    vals = [f"{r.get('lc_rate', float('nan')):>7.2f} {r.get('en_leak', float('nan')):>7.2f}" for r in row]
    print(f"{'isolated_lora_'+lang:<28} {' '.join(vals)}")
EOF

# ── Phase 5：Tag Routing（Isolated-LoRA 全部完成后自动触发）────────────────
echo ""
echo "[$(date)] === Isolated-LoRA 全部完成，自动启动 Phase 5 Tag Routing ==="
bash scripts/launch_tag_routing.sh 2>&1 | tee "logs/tag_routing_master.log"
