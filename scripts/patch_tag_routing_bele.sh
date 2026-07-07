#!/bin/bash
# patch_tag_routing_bele.sh
# 补跑 tag_routing 模型的 belebele so+ha（yo 和 so 结果一致，so 数据来源有误）
set -euo pipefail
cd /root/project
export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="results/tag_routing/tag_routing_Qwen3.5-9B-Base"
EVAL_JSON="results/tag_routing/tag_routing_Qwen3.5-9B-Base_eval.json"
TMP="/tmp/tag_routing_soha_patch.json"

echo "[$(date -u +%H:%M:%S)] === patching tag_routing belebele so+ha ==="

python scripts/evaluate.py \
    --model_path "$MODEL" \
    --tasks target_lang \
    --languages yo,so,ha \
    --skip_flores \
    --batch_size 32 \
    --output "$TMP"

python3 - "$TMP" "$EVAL_JSON" << 'PYEOF'
import json, sys

tmp_path, eval_path = sys.argv[1], sys.argv[2]

with open(tmp_path) as f:
    tmp = json.load(f)
with open(eval_path) as f:
    ev = json.load(f)

ml_new = tmp.get("scores", {}).get("multilingual", {})
ml_old = ev.setdefault("scores", {}).setdefault("multilingual", {})

for task in ("sib200", "belebele"):
    if task not in ml_new:
        continue
    if task not in ml_old:
        ml_old[task] = {}
    for lang, val in ml_new[task].items():
        old_val = ml_old[task].get(lang, "N/A")
        ml_old[task][lang] = val
        print(f"  {task}[{lang}]: {old_val} -> {val:.4f}")

with open(eval_path, "w") as f:
    json.dump(ev, f, indent=2, ensure_ascii=False)
print(f"  -> updated {eval_path}")
PYEOF

echo "[$(date -u +%H:%M:%S)] Done."
echo ""
echo "=== tag_routing 最终 belebele ==="
python3 -c "
import json
d = json.load(open('$EVAL_JSON'))
bele = d['scores']['multilingual']['belebele']
sib  = d['scores']['multilingual']['sib200']
for lang in ('en','yo','so','ha'):
    print(f'  belebele[{lang}]={bele.get(lang,\"?\"):.4f}  sib200[{lang}]={sib.get(lang,\"?\")}')
"
