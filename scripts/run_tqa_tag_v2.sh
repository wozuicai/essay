#!/bin/bash
# run_tqa_tag_v2.sh
# 为 12 个模型重跑 TruthfulQA MC1（带 tag，全句 log-likelihood 方法）
# 保留无 tag 原始分至 truthfulqa_mc1_notag
# 包含 tag_routing（单独的结果 JSON）
set -e
cd /root/project
export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# DSCT 训练占用 GPU 2/3；评测只用 GPU 0+1（各有 21/27GB 空余，足够 9B 推理）
export CUDA_VISIBLE_DEVICES=0,1

declare -A MODELS
MODELS["baseline"]="models/Qwen3.5-9B-Base|results/phase2_v2/Qwen3.5-9B-Base_baseline.json"
MODELS["train_en"]="results/phase2_v2/lis_Qwen3.5-9B-Base_train_en|results/phase2_v2/lis_Qwen3.5-9B-Base_train_en_eval.json"
MODELS["train_yo"]="results/phase2_v2/lis_Qwen3.5-9B-Base_train_yo|results/phase2_v2/lis_Qwen3.5-9B-Base_train_yo_eval.json"
MODELS["train_so"]="results/phase2_v2/lis_Qwen3.5-9B-Base_train_so|results/phase2_v2/lis_Qwen3.5-9B-Base_train_so_eval.json"
MODELS["train_ha"]="results/phase2_v2/lis_Qwen3.5-9B-Base_train_ha|results/phase2_v2/lis_Qwen3.5-9B-Base_train_ha_eval.json"
MODELS["mix_en_yo"]="results/mix_en/mix_Qwen3.5-9B-Base_en_yo|results/mix_en/mix_Qwen3.5-9B-Base_en_yo_eval.json"
MODELS["mix_en_so"]="results/mix_en/mix_Qwen3.5-9B-Base_en_so|results/mix_en/mix_Qwen3.5-9B-Base_en_so_eval.json"
MODELS["mix_en_ha"]="results/mix_en/mix_Qwen3.5-9B-Base_en_ha|results/mix_en/mix_Qwen3.5-9B-Base_en_ha_eval.json"
MODELS["MID_yo"]="results/mid/mid_Qwen3.5-9B-Base_yo|results/mid/mid_Qwen3.5-9B-Base_yo_eval.json"
MODELS["MID_so"]="results/mid/mid_Qwen3.5-9B-Base_so|results/mid/mid_Qwen3.5-9B-Base_so_eval.json"
MODELS["MID_ha"]="results/mid/mid_Qwen3.5-9B-Base_ha|results/mid/mid_Qwen3.5-9B-Base_ha_eval.json"
MODELS["tag_routing"]="results/tag_routing/tag_routing_Qwen3.5-9B-Base|results/tag_routing/tag_routing_Qwen3.5-9B-Base_eval.json"

ORDER="baseline train_en train_yo train_so train_ha mix_en_yo mix_en_so mix_en_ha MID_yo MID_so MID_ha tag_routing"

for NAME in $ORDER; do
    IFS='|' read -r MODEL_PATH EVAL_JSON <<< "${MODELS[$NAME]}"
    TMP="/tmp/tqa_tag_v2_${NAME}.json"
    echo "[$(date -u '+%H:%M:%S UTC')] === $NAME ==="

    python scripts/evaluate.py \
        --model_path "$MODEL_PATH" \
        --tasks english_main \
        --en_tasks truthfulqa_mc1 \
        --inject_lang_tag \
        --output "$TMP"

    python3 - "$TMP" "$EVAL_JSON" << 'PYEOF'
import json, sys

tmp_path, eval_path = sys.argv[1], sys.argv[2]

with open(tmp_path) as f:
    tmp = json.load(f)
new_score = tmp["scores"]["english"]["truthfulqa_mc1"]

with open(eval_path) as f:
    ev = json.load(f)

eng = ev.setdefault("scores", {}).setdefault("english", {})

# 保留无 tag 原始分（只在第一次运行时备份）
if "truthfulqa_mc1_notag" not in eng and "truthfulqa_mc1" in eng:
    eng["truthfulqa_mc1_notag"] = eng["truthfulqa_mc1"]

eng["truthfulqa_mc1"] = new_score

# 更新 english_avg（4 项均值）
keys = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc1"]
vals = [eng[k] for k in keys if k in eng]
if len(vals) == 4:
    eng["english_avg"] = sum(vals) / 4

with open(eval_path, "w") as f:
    json.dump(ev, f, indent=2, ensure_ascii=False)

notag = eng.get("truthfulqa_mc1_notag")
notag_str = f"{notag:.4f}" if notag is not None else "N/A"
print(f"  notag={notag_str}  tag={new_score:.4f}  avg={eng.get('english_avg', '?')}")
PYEOF

    echo "[$(date -u '+%H:%M:%S UTC')] Done: $NAME"
done

echo ""
echo "=== All 12 models done ==="
