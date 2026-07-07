#!/bin/bash
# 补跑 DSCT 三个模型的 sib200 so+ha（之前跑时缓存缺失导致输出0）
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
    TMP="/tmp/dsct_${LANG}_sib200_patch.json"

    echo "[$(date -u +%H:%M:%S)] === sib200 patch: ${EXP} ==="

    python scripts/evaluate.py \
        --model_path "$MODEL" \
        --tasks target_lang \
        --languages so,ha \
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
# 只更新 sib200（belebele 已经修好了）
for lang, val in ml_new.get("sib200", {}).items():
    if ml_old.get("sib200", {}).get(lang, 1.0) == 0.0:
        ml_old.setdefault("sib200", {})[lang] = val
        print(f"  sib200[{lang}] = {val:.4f}")
    else:
        print(f"  sib200[{lang}] already {ml_old['sib200'].get(lang):.4f}, skipping")
with open(eval_path, "w") as f:
    json.dump(ev, f, indent=2, ensure_ascii=False)
PYEOF

    echo "[$(date -u +%H:%M:%S)] Done: ${EXP}"
done

echo ""
echo "=== DSCT 最终分数 ==="
python3 - << 'PYEOF'
import json
for lang in ("yo", "so", "ha"):
    path = f"results/dsct/dsct_Qwen3.5-9B-Base_{lang}_eval.json"
    d = json.load(open(path))
    ml = d["scores"]["multilingual"]
    bele = ml.get("belebele", {})
    sib  = ml.get("sib200", {})
    print(f"DSCT_{lang}: bele_yo={bele.get('yo',0):.4f} bele_so={bele.get('so',0):.4f} bele_ha={bele.get('ha',0):.4f} | "
          f"sib_yo={sib.get('yo',0):.3f} sib_so={sib.get('so',0):.3f} sib_ha={sib.get('ha',0):.3f}")
PYEOF
