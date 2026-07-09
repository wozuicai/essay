#!/bin/bash
# launch_tag_routing.sh
# Phase 5：全语言 SFT + Tag Hard Routing
#
# 训练：en + yo + so + ha 全量 concat shuffle，标准 LoRA r=16，1 epoch（en adapter + 对照基线）
# yo/so/ha adapter：直接复用 results/mix_en/ 已有 adapter，无需重新训练
# 推理时解析 <|tgt_lang:xx|> tag，硬切换到对应语言 adapter
#
# 评测内容：
#   - TruthfulQA MC1（英文）
#   - Belebele（en/yo/so/ha）
#   - SIB-200（en/yo/so/ha）
#   - IrokoBench AfriMMLU MCQ（yo/ha）
#   - LCB-chat（lc_rate / en_leak，yo/so/ha，n=50）
#
# 用法（直接启动）：
#   nohup bash scripts/launch_tag_routing.sh > logs/tag_conditioning_master.log 2>&1 &
# 或由 launch_isolated_lora.sh 末尾自动调用。

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-24000}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

cd /root/project
mkdir -p logs results/tag_routing

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_CFG="configs/accelerate_fullgpu.yaml"
EXP_CFG="configs/experiments/lis_matrix.yaml"
RESULTS="results/tag_routing"
EXP_NAME="tag_routing_${MODEL_SHORT}"
OUT_DIR="${RESULTS}/${EXP_NAME}"
EVAL_OUT="${RESULTS}/${EXP_NAME}_eval.json"
LCB_OUT="${RESULTS}/${EXP_NAME}_lcb_chat.json"

bash scripts/setup_accelerate.sh
python scripts/preflight_required.py \
    --model "$MODEL" \
    --data_dir data/processed \
    --langs en,yo,so,ha \
    --max_train_chars "$MAX_TRAIN_CHARS"

if [[ -f "$EVAL_OUT" ]]; then
    echo "[$(date)] Tag routing eval already exists, skipping training."
else
    # ── 训练 ──────────────────────────────────────────────────────────────────
    echo "[$(date)] === Training: tag_routing (en+yo+so+ha, all langs) ==="
    accelerate launch \
        --config_file "$ACCEL_CFG" \
        scripts/train.py \
        --model       "$MODEL" \
        --train_lang  "en" \
        --method      standard_lora \
        --mix_all_langs \
        --output_dir  "$OUT_DIR" \
        --config      "$EXP_CFG" \
        --no_wandb \
        2>&1 | tee "logs/tag_routing_train.log"

    # ── 评测：English + Belebele + IrokoBench ───────────────────────────────
    echo "[$(date)] === Evaluating: tag_routing — required suite ==="
    python scripts/eval_required.py \
        --model_path  "$OUT_DIR" \
        --languages   en,yo,so,ha \
        --inject_lang_tag \
        --output      "$EVAL_OUT" \
        2>&1 | tee "logs/tag_routing_eval.log"

    bash scripts/cleanup_large_artifacts.sh "$OUT_DIR"
fi

# ── 汇总（与其他实验对比）────────────────────────────────────────────────────
echo ""
echo "[$(date)] === 全部完成，结果汇总 ==="
python3 - << 'EOF'
import json, os

BASE_PATH    = "/root/project/results/phase2_v2/Qwen3.5-9B-Base_baseline.json"
PHASE2_DIR   = "/root/project/results/phase2_v2"
MIX_DIR      = "/root/project/results/mix_en"
ISO_DIR      = "/root/project/results/isolated_lora"
TAG_DIR      = "/root/project/results/tag_routing"
MODEL_SHORT  = "Qwen3.5-9B-Base"

def load_scores(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f).get("scores")

def fmt_row(name, s, lcb=None):
    if s is None:
        print(f"  {name:<30} N/A")
        return
    tqa  = s.get("english", {}).get("truthfulqa_mc1", float("nan"))
    b    = s.get("multilingual", {}).get("belebele", {})
    m    = s.get("multilingual", {}).get("irokobench", {})
    row  = f"  {name:<30} {tqa:>7.4f} | {b.get('en',float('nan')):>6.3f} {b.get('yo',float('nan')):>6.3f} {b.get('so',float('nan')):>6.3f} {b.get('ha',float('nan')):>6.3f} | {m.get('yo',{}).get('mcq_accuracy',float('nan')):>6.3f} {m.get('ha',{}).get('mcq_accuracy',float('nan')):>6.3f}"
    if lcb:
        yo = lcb.get("yo", {}); so = lcb.get("so", {}); ha = lcb.get("ha", {})
        row += f" | {yo.get('lc_rate',float('nan')):>5.2f} {so.get('lc_rate',float('nan')):>5.2f} {ha.get('lc_rate',float('nan')):>5.2f}"
    print(row)

header = f"  {'model':<30} {'tqa_mc1':>7} | {'bele_en':>6} {'bele_yo':>6} {'bele_so':>6} {'bele_ha':>6} | {'mcq_yo':>6} {'mcq_ha':>6} | {'lc_yo':>5} {'lc_so':>5} {'lc_ha':>5}"
print(header)
print("  " + "-" * (len(header) - 2))

fmt_row("baseline",     load_scores(BASE_PATH))
fmt_row("train_en",     load_scores(f"{PHASE2_DIR}/lis_{MODEL_SHORT}_train_en_eval.json"),
        json.load(open(f"/root/project/results/lcb_chat/lcb_chat_{MODEL_SHORT}_train_en.json"))["scores"] if os.path.exists(f"/root/project/results/lcb_chat/lcb_chat_{MODEL_SHORT}_train_en.json") else None)

for lang in ["yo", "so", "ha"]:
    fmt_row(f"mix_en_{lang}", load_scores(f"{MIX_DIR}/mix_{MODEL_SHORT}_en_{lang}_eval.json"))
    iso_lcb_path = f"{ISO_DIR}/isolated_{MODEL_SHORT}_{lang}_lcb_chat.json"
    iso_lcb = json.load(open(iso_lcb_path))["scores"] if os.path.exists(iso_lcb_path) else None
    fmt_row(f"isolated_lora_{lang}", load_scores(f"{ISO_DIR}/isolated_{MODEL_SHORT}_{lang}_eval.json"), iso_lcb)

tag_lcb_path = f"{TAG_DIR}/tag_routing_{MODEL_SHORT}_lcb_chat.json"
tag_lcb = json.load(open(tag_lcb_path))["scores"] if os.path.exists(tag_lcb_path) else None
fmt_row("tag_routing", load_scores(f"{TAG_DIR}/tag_routing_{MODEL_SHORT}_eval.json"), tag_lcb)
EOF
