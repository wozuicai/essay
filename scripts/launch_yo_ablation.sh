#!/bin/bash
# Yoruba ablation — 3 conditions in parallel, 1 GPU each
#
#  GPU 0 — yo-500-e2  : 500 Aya samples, 2 epochs   (H2: overfitting)
#  GPU 1 — yo-full-e2 : all ~11k Aya samples, 2 epochs (H1: data quantity)
#  GPU 2 — yo-bel-mix : 500 Aya + 200 Belebele, 2 epochs (H3: task mismatch)
#
# Already done: Phase 2 Yoruba baseline, used for comparison
#
# Usage: nohup bash scripts/launch_yo_ablation.sh > logs/yo_ablation_master.log 2>&1 &

set -euo pipefail
export PATH=/home/tiger/.local/bin:$PATH
export HF_HOME=/root/project/hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /root/project
mkdir -p logs results/yo_ablation data/processed_yo_full data/processed_yo_belmix

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
ACCEL_1GPU="configs/accelerate_1gpu.yaml"
CFG_FULL="configs/experiments/yo_ablation_1gpu.yaml"
CFG_SMALL="configs/experiments/yo_ablation_1gpu_e1.yaml"

# ─── Step 1: Data preparation (sequential, fast) ──────────────────────────────

echo "[$(date)] === Data prep: yo-full (all Aya yo, no NLLB) ==="
python3 - <<'PYEOF'
import sys, os, json
sys.path.insert(0, '/root/project')

OUT = 'data/processed_yo_full/yo.jsonl'
if os.path.exists(OUT):
    n = sum(1 for _ in open(OUT))
    print(f"  already exists ({n} lines), skipping.")
    sys.exit(0)

from datasets import load_dataset

TMPL = "### Instruction:\n<|tgt_lang:{language}|> {instruction}\n\n### Response:\n{response}"

aya = load_dataset("CohereForAI/aya_dataset", split="train")
lang_data = aya.filter(lambda x: x['language'] == 'Yoruba')
print(f"  Aya yo raw: {len(lang_data)} samples")

with open(OUT, 'w', encoding='utf-8') as f:
    for ex in lang_data:
        instr = ex.get('inputs', '')
        resp  = ex.get('targets', '')
        text  = TMPL.format(language='yo', instruction=instr, response=resp)
        f.write(json.dumps({"instruction": instr, "response": resp,
                            "language": "yo", "source": "aya", "text": text},
                           ensure_ascii=False) + '\n')
print(f"  saved {len(lang_data)} samples -> {OUT}")
PYEOF

echo "[$(date)] === Data prep: yo-bel-mix (500 Aya + 200 Belebele) ==="
unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE
python3 - <<'PYEOF'
import sys, os, json, random
sys.path.insert(0, '/root/project')

OUT = 'data/processed_yo_belmix/yo.jsonl'
if os.path.exists(OUT):
    n = sum(1 for _ in open(OUT))
    print(f"  already exists ({n} lines), skipping.")
    sys.exit(0)

from datasets import load_dataset

TMPL = "### Instruction:\n<|tgt_lang:{language}|> {instruction}\n\n### Response:\n{response}"
KEYS = {1: 'mc_answer1', 2: 'mc_answer2', 3: 'mc_answer3', 4: 'mc_answer4'}
LTRS = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

def fmt_bel(ex):
    idx = int(ex['correct_choice'])
    instr = (f"Read the passage and answer the question.\n\n"
             f"Passage: {ex['flores_passage']}\n\nQuestion: {ex['question']}\n\n"
             f"A. {ex['mc_answer1']}\nB. {ex['mc_answer2']}\n"
             f"C. {ex['mc_answer3']}\nD. {ex['mc_answer4']}")
    resp = f"{LTRS[idx]}. {ex[KEYS[idx]]}"
    return {"instruction": instr, "response": resp, "language": "yo",
            "source": "belebele_oracle",
            "text": TMPL.format(language='yo', instruction=instr, response=resp)}

bel = load_dataset("facebook/belebele", "yor_Latn", split="test")
rng = random.Random(42)
bel_samples = [fmt_bel(bel[i]) for i in rng.sample(range(len(bel)), 200)]

aya_samples = [json.loads(l) for l in open('data/processed/yo.jsonl')]
mixed = aya_samples + bel_samples
rng.shuffle(mixed)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    for s in mixed:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f"  saved {len(mixed)} samples -> {OUT}  (aya={len(aya_samples)}, belebele={len(bel_samples)})")
PYEOF

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
N_FULL=$(wc -l < data/processed_yo_full/yo.jsonl)
echo "[$(date)] Data ready — yo-full: ${N_FULL} samples"

# ─── Step 2: Launch 3 training jobs in parallel ───────────────────────────────

echo "[$(date)] === Launching 3 parallel training jobs ==="

_train_eval() {
    local GPU="$1" LABEL="$2" DATA_DIR="$3" N_SAMPLES="$4" CFG="$5"
    local OUT_DIR="results/yo_ablation/${LABEL}"
    local EVAL_OUT="results/yo_ablation/${LABEL}_eval.json"
    local TRAIN_LOG="logs/yo_abl_${LABEL}_train.log"
    local EVAL_LOG="logs/yo_abl_${LABEL}_eval.log"

    if [[ -f "$EVAL_OUT" ]]; then
        echo "  [GPU${GPU}] ${LABEL}: eval already exists, skipping."
        return
    fi

    echo "  [GPU${GPU}] ${LABEL}: training (${N_SAMPLES} samples)..."
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch \
        --config_file "$ACCEL_1GPU" \
        scripts/train.py \
        --model "$MODEL" \
        --train_lang yo \
        --train_samples "$N_SAMPLES" \
        --method standard_lora \
        --output_dir "$OUT_DIR" \
        --config "$CFG" \
        --data_dir "$DATA_DIR" \
        --no_wandb \
        > "$TRAIN_LOG" 2>&1

    echo "  [GPU${GPU}] ${LABEL}: evaluating..."
    CUDA_VISIBLE_DEVICES=$GPU python scripts/evaluate.py \
        --model_path "$OUT_DIR" \
        --tasks all \
        --languages fr,zh,sw,th,bn,yo \
        --skip_flores \
        --output "$EVAL_OUT" \
        > "$EVAL_LOG" 2>&1

    echo "  [GPU${GPU}] ${LABEL}: DONE"
}

export -f _train_eval
export MODEL ACCEL_1GPU CFG_FULL CFG_SMALL MODEL_SHORT N_FULL

_train_eval 0 "yo_500_e2"  "data/processed"       500      "$CFG_SMALL" &
PID_A=$!

_train_eval 1 "yo_full_e2" "data/processed_yo_full" "$N_FULL" "$CFG_FULL" &
PID_B=$!

_train_eval 2 "yo_bel_mix" "data/processed_yo_belmix" 700   "$CFG_FULL" &
PID_C=$!

echo "[$(date)] PIDs — yo_500_e2:${PID_A}  yo_full_e2:${PID_B}  yo_bel_mix:${PID_C}"
echo "[$(date)] Waiting for all 3 jobs..."
wait $PID_A && echo "[$(date)] yo_500_e2 finished" || echo "[$(date)] yo_500_e2 FAILED"
wait $PID_B && echo "[$(date)] yo_full_e2 finished" || echo "[$(date)] yo_full_e2 FAILED"
wait $PID_C && echo "[$(date)] yo_bel_mix finished" || echo "[$(date)] yo_bel_mix FAILED"

# ─── Step 3: Compare results ──────────────────────────────────────────────────

echo ""
echo "[$(date)] === Results ==="
python3 - <<'PYEOF'
import json, os

def get_ml(path):
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        d = json.load(f)
    ml = d['scores'].get('multilingual', {})
    return ml.get('belebele', {}), ml.get('sib200', {})

base_bel, base_sib = get_ml('results/phase1_baseline/Qwen3.5-9B-Base_baseline.json')
orig_bel, orig_sib = get_ml('results/phase2_lis_matrix/lis_Qwen3.5-9B-Base_train_yo_eval.json')
e2_bel,   e2_sib   = get_ml('results/yo_ablation/yo_500_e2_eval.json')
ful_bel,  ful_sib  = get_ml('results/yo_ablation/yo_full_e2_eval.json')
bel_bel,  bel_sib  = get_ml('results/yo_ablation/yo_bel_mix_eval.json')

langs = ['en', 'fr', 'zh', 'sw', 'th', 'bn', 'yo']

for name, base_d, orig_d, e2_d, ful_d, bel_d in [
    ('Belebele', base_bel, orig_bel, e2_bel, ful_bel, bel_bel),
    ('SIB-200',  base_sib, orig_sib, e2_sib, ful_sib, bel_sib),
]:
    print(f"\n=== {name} ===")
    print(f"{'lang':<5} {'base':>7} {'orig':>8} {'500-e2':>8} {'full-e2':>8} {'bel-mix*':>9}")
    print("-" * 48)
    for lang in langs:
        b   = base_d.get(lang, 0)
        o   = orig_d.get(lang, 0)
        e2  = e2_d.get(lang, 0)
        ful = ful_d.get(lang, 0)
        bel = bel_d.get(lang, 0)
        mk  = " <--" if lang == 'yo' else ""
        print(f"{lang:<5} {b:>7.4f} {o:>8.4f} {e2:>8.4f} {ful:>8.4f} {bel:>9.4f}{mk}")

print("\n* bel-mix = ORACLE (200 samples from Belebele test split). Diagnostic only.")
print("\nyo行解读:")
print("  full-e2 > 500-e2? → H1成立 (数据量不足)")
print("  bel-mix >> others? → H3成立 (任务格式错位是根本)")
PYEOF

echo ""
echo "[$(date)] yo ablation COMPLETE"
