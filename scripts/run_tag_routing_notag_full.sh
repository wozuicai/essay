#!/bin/bash
# 对 tag_routing 做完整无-inject-tag 评测，覆盖原 eval JSON
set -euo pipefail
cd /root/project
export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="results/tag_routing/tag_routing_Qwen3.5-9B-Base"
EVAL_JSON="results/tag_routing/tag_routing_Qwen3.5-9B-Base_eval.json"
TMP="/tmp/tag_routing_notag_full.json"

echo "[$(date -u +%H:%M:%S)] === tag_routing 全量 no-tag 评测 ==="

# 全量评测，不加 inject_lang_tag
python scripts/evaluate.py \
    --model_path "$MODEL" \
    --tasks all \
    --languages en,yo,so,ha \
    --skip_flores \
    --batch_size 32 \
    --output "$TMP"

# IrokoBench MCQ
python scripts/eval_extended.py \
    --model_path "$MODEL" \
    --result_json "$TMP" \
    --only_iroko_mcq

# 合并回原 JSON，保留 truthfulqa_mc1_notag 字段
python3 - "$TMP" "$EVAL_JSON" << 'PYEOF'
import json, sys

tmp_path, eval_path = sys.argv[1], sys.argv[2]
new = json.load(open(tmp_path))
old = json.load(open(eval_path))

# 保留旧 JSON 的 notag 备份（run_tqa_tag_v2 写入的，值为 0.3758）
notag_backup = old.get("scores", {}).get("english", {}).get("truthfulqa_mc1_notag")

merged = new.copy()
eng = merged.setdefault("scores", {}).setdefault("english", {})
if notag_backup is not None:
    eng["truthfulqa_mc1_notag"] = notag_backup

with open(eval_path, "w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("=== tag_routing 最终分数（no-tag 全量）===")
scores = merged["scores"]
eng = scores.get("english", {})
ml  = scores.get("multilingual", {})
bele  = ml.get("belebele", {})
sib   = ml.get("sib200", {})
iroko = ml.get("irokobench", {})
print("English:")
print("  mmlu=%.4f  hellaswag=%.4f  arc=%.4f  tqa(notag)=%.4f" % (
    eng.get("mmlu", 0), eng.get("hellaswag", 0),
    eng.get("arc_challenge", 0), eng.get("truthfulqa_mc1", 0)))
print("Belebele:  en=%.4f  yo=%.4f  so=%.4f  ha=%.4f" % (
    bele.get("en",0), bele.get("yo",0), bele.get("so",0), bele.get("ha",0)))
print("SIB-200:   en=%.3f  yo=%.3f  so=%.3f  ha=%.3f" % (
    sib.get("en",0), sib.get("yo",0), sib.get("so",0), sib.get("ha",0)))
print("IrokoBench: yo=%.4f  ha=%.4f" % (
    iroko.get("yo",{}).get("mcq_accuracy",0),
    iroko.get("ha",{}).get("mcq_accuracy",0)))
PYEOF

echo "[$(date -u +%H:%M:%S)] 完成"
