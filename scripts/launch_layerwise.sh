#!/bin/bash
# launch_layerwise.sh
# Layer-wise Progressive Language Routing 训练 + 评测
#
# 架构：底层 0-15（共享 LoRA，4-lang混训）+ 顶层 16-31（per-lang LoRA）
# 总流程：Stage 1（B200 GPU 2-3）→ Stage 2 × yo/so/ha（H100 4 GPU）→ Merge → Eval
#
# === 分两步运行 ===
#
# 步骤 1：在 B200 上运行 Stage 1（GPU 2-3，2 张卡）
#   export CUDA_VISIBLE_DEVICES=2,3
#   nohup bash scripts/launch_layerwise.sh stage1 > logs/layerwise_stage1.log 2>&1 &
#
# 步骤 2：Stage 1 完成后，在 H100 上运行 Stage 2（4 张卡）
#   SSH 到 H100 worker，然后：
#   nohup bash scripts/launch_layerwise.sh stage2 > logs/layerwise_stage2.log 2>&1 &
#
# 若只有 B200 可用（Stage 2 也在 B200），可用：
#   export CUDA_VISIBLE_DEVICES=2,3
#   bash scripts/launch_layerwise.sh stage2

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-24000}"
export MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-200000}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# Fix CUDA 803: compat dir (575.x) in ldconfig overrides the real driver (580.x)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08
export TRITON_CACHE_DIR=/tmp/triton_cache

cd /root/project
mkdir -p logs results/layerwise

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
EXP_CFG="configs/experiments/lis_matrix.yaml"
OUT_DIR="results/layerwise"
STAGE1_DIR="${OUT_DIR}/stage1_shared"

STEP="${1:-stage1}"   # stage1 | stage2 | merge_eval

# ── 选择 accelerate 配置 ──────────────────────────────────────────────────
# Stage2 只训练顶层（底层冻结），ZeRO-2 gradient bucket 在部分参数冻结时会 IndexError
# 改用 ZeRO-1（只 partition optimizer states，不 partition gradients）
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" | tr ',' '\n' | wc -l)
if [[ "$N_GPUS" -ge 4 ]]; then
    ACCEL_CFG_STAGE1="configs/accelerate_4gpu.yaml"
    ACCEL_CFG_STAGE2="configs/accelerate_4gpu_ddp.yaml"
else
    ACCEL_CFG_STAGE1="configs/accelerate_2gpu.yaml"
    ACCEL_CFG_STAGE2="configs/accelerate_2gpu_ddp.yaml"
fi
ACCEL_CFG="${ACCEL_CFG_STAGE1}"
echo "[$(date -u '+%H:%M:%S UTC')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}, accel=$ACCEL_CFG"
python scripts/preflight_required.py \
    --model "$MODEL" \
    --data_dir data/processed \
    --langs en,yo,so,ha \
    --max_train_chars "$MAX_TRAIN_CHARS"

# ── Stage 1: 共享底层 LoRA（4-lang mixed, 1 epoch, 底层 0-15）────────────
if [[ "$STEP" == "stage1" ]]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === Stage 1: Shared bottom LoRA ==="

    if [[ -f "${STAGE1_DIR}/adapter_config.json" ]]; then
        echo "[$(date -u '+%H:%M:%S UTC')] Stage 1 adapter already exists, skipping."
    else
        accelerate launch --config_file "$ACCEL_CFG" --main_process_port 29502 scripts/train_layerwise.py \
            --model      "$MODEL" \
            --output_dir "$OUT_DIR" \
            --config     "$EXP_CFG" \
            --mode       stage1 \
            --r          16 \
            --lora_alpha 32.0 \
            --no_wandb \
            2>&1 | tee logs/layerwise_stage1_train.log
        echo "[$(date -u '+%H:%M:%S UTC')] Stage 1 done → ${STAGE1_DIR}"
    fi
    echo "[$(date -u '+%H:%M:%S UTC')] Stage 1 complete."
    echo "接下来在 H100 运行: bash scripts/launch_layerwise.sh stage2"
fi

# ── Stage 2: 三个语言的顶层 LoRA（顺序执行，各 1 epoch, 顶层 16-31）────
if [[ "$STEP" == "stage2" ]]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === Stage 2: Language-specific top LoRA ==="

    if [[ ! -f "${STAGE1_DIR}/adapter_config.json" && ! -f "${STAGE1_DIR}/shared/adapter_config.json" ]]; then
        echo "ERROR: Stage 1 adapter not found at ${STAGE1_DIR}. Run stage1 first."
        exit 1
    fi

    for LANG in yo so ha; do
        STAGE2_DIR="${OUT_DIR}/stage2_${LANG}"

        echo ""
        echo "[$(date -u '+%H:%M:%S UTC')] ─── Stage 2: ${LANG} ───"

        if [[ -f "${STAGE2_DIR}/${LANG}/adapter_config.json" ]]; then
            echo "[$(date -u '+%H:%M:%S UTC')] ${LANG} stage2 adapter exists, skipping training."
        else
            accelerate launch --config_file "$ACCEL_CFG_STAGE2" --main_process_port 29502 scripts/train_layerwise.py \
                --model       "$MODEL" \
                --output_dir  "$OUT_DIR" \
                --config      "$EXP_CFG" \
                --mode        stage2 \
                --train_lang  "$LANG" \
                --stage1_dir  "$STAGE1_DIR" \
                --r           16 \
                --lora_alpha  32.0 \
                --no_wandb \
                2>&1 | tee "logs/layerwise_stage2_${LANG}.log"
            echo "[$(date -u '+%H:%M:%S UTC')] ${LANG} Stage 2 done."
        fi

        # Merge in memory + eval (no disk save of merged weights)
        EVAL_OUT="${OUT_DIR}/layerwise_${MODEL_SHORT}_${LANG}_eval.json"
        if [[ -f "$EVAL_OUT" ]]; then
            echo "[$(date -u '+%H:%M:%S UTC')] ${LANG} eval exists, skipping."
        else
            echo "[$(date -u '+%H:%M:%S UTC')] Merge+Eval ${LANG} (in-memory, no disk save) ..."
            python scripts/train_layerwise.py \
                --model       "$MODEL" \
                --output_dir  "$OUT_DIR" \
                --config      "$EXP_CFG" \
                --mode        merge_eval \
                --train_lang  "$LANG" \
                --stage1_dir  "$STAGE1_DIR" \
                --eval_output "$EVAL_OUT" \
                2>&1 | tee "logs/layerwise_eval_${LANG}.log"
            bash scripts/cleanup_large_artifacts.sh "$STAGE2_DIR"
            echo "[$(date -u '+%H:%M:%S UTC')] ${LANG} eval done → ${EVAL_OUT}"
        fi
    done

    echo ""
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === Layer-wise 全部完成 ==="
    echo ""

    python3 - << 'PYEOF'
import json, os

MODEL_SHORT = "Qwen3.5-9B-Base"
RESULTS = "results/layerwise"

def safe_scores(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("scores", {})

rows = [("baseline", "results/phase2_v2/Qwen3.5-9B-Base_baseline.json")]
for lang in ["yo", "so", "ha"]:
    rows.append((f"layerwise_{lang}", f"{RESULTS}/layerwise_{MODEL_SHORT}_{lang}_eval.json"))

hdr = f"{'model':<18} {'tqa':>6} | {'bele_en':>7} {'bele_yo':>7} {'bele_so':>7} {'bele_ha':>7}"
print(hdr)
print("-" * len(hdr))
for name, path in rows:
    sc = safe_scores(path)
    eng  = sc.get("english", {})
    bele = sc.get("multilingual", {}).get("belebele", {})
    tqa  = eng.get("truthfulqa_mc1")
    def f(v): return f"{v:.4f}" if v is not None else "  N/A"
    print(f"{name:<18} {f(tqa):>6} | {f(bele.get('en')):>7} {f(bele.get('yo')):>7} "
          f"{f(bele.get('so')):>7} {f(bele.get('ha')):>7}")
PYEOF
fi
