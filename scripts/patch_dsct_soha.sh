#!/bin/bash
# patch_dsct_soha.sh
# 补跑 DSCT 三个模型的 belebele/sib200 so+ha 评测，合并回原 eval JSON
set -euo pipefail
cd /root/project
export PATH=/home/tiger/.local/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for LANG in yo so ha; do
    EXP="dsct_Qwen3.5-9B-Base_${LANG}"
    MODEL="results/dsct/${EXP}"
    EVAL_JSON="results/dsct/${EXP}_eval.json"
    TMP="/tmp/dsct_${LANG}_soha_patch.json"

    echo "[$(date -u +%H:%M:%S)] === patching ${EXP} ==="

    python scripts/evaluate.py \
        --model_path "$MODEL" \
        --tasks target_lang \
        --languages so,ha \
        --skip_flores \
        --batch_size 32 \
        --output "$TMP"

    # 合并 so/ha 的 belebele 和 sib200 回原 JSON
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
        ml_old[task][lang] = val
        print(f"  {task}[{lang}] = {val:.4f}")

with open(eval_path, "w") as f:
    json.dump(ev, f, indent=2, ensure_ascii=False)
print(f"  -> updated {eval_path}")
PYEOF

    echo "[$(date -u +%H:%M:%S)] Done: ${EXP}"
done

echo ""
echo "=== 全部完成，最终分数 ==="
python3 - << 'PYEOF'
import json

for lang in ("yo", "so", "ha"):
    path = f"results/dsct/dsct_Qwen3.5-9B-Base_{lang}_eval.json"
    d = json.load(open(path))
    ml = d["scores"]["multilingual"]
    bele = ml.get("belebele", {})
    sib  = ml.get("sib200", {})
    print(f"DSCT_{lang}: bele_en={bele.get('en',0):.4f} bele_yo={bele.get('yo',0):.4f} "
          f"bele_so={bele.get('so',0):.4f} bele_ha={bele.get('ha',0):.4f} | "
          f"sib_so={sib.get('so',0):.4f} sib_ha={sib.get('ha',0):.4f}")
PYEOF
