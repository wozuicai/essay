#!/bin/bash
# launch_mid_yo_normfix.sh
#
# Train mid_yo with FIXED _mid_loss normalisation (÷valid_b instead of ÷n).
# Same hyper-params as original mid_yo (α=0.1, β=0.05, K=4, P2=3).
#
# Diagnostic purpose: verify that the per-sample normalisation actually makes
# the MID constraint strong enough to maintain English TruthfulQA after yo SFT.
#
# Uses GPU 4-7 so it can run in parallel with Experiment A (GPU 0-3 eval).
#
# Usage:
#   nohup bash scripts/launch_mid_yo_normfix.sh > logs/mid_yo_normfix.log 2>&1 &

set -euo pipefail

export PATH=/home/tiger/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
# H100 container: libnvidia-ml.so.1 is a 0-byte placeholder; preload the real 535 lib
_NV_ML=$(find /usr/lib/x86_64-linux-gnu -name "libnvidia-ml.so.535*" 2>/dev/null | head -1)
[[ -n "$_NV_ML" ]] && export LD_PRELOAD="$_NV_ML${LD_PRELOAD:+:$LD_PRELOAD}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /root/project
mkdir -p logs results/mid

MODEL="/root/project/models/Qwen3.5-9B-Base"
MODEL_SHORT="Qwen3.5-9B-Base"
TEACHER_ADAPTER="results/phase2_v2/lis_${MODEL_SHORT}_train_en"
EXP_CFG="configs/experiments/lis_matrix.yaml"
RESULTS="results/mid"
LANG="yo"
EXP_NAME="mid_${MODEL_SHORT}_${LANG}_normfix"
OUT_DIR="${RESULTS}/${EXP_NAME}"
EVAL_OUT="${RESULTS}/${EXP_NAME}_eval.json"

# Write a 4-GPU accelerate config (CUDA_VISIBLE_DEVICES=4,5,6,7 → visible as 0-3)
ACCEL_CFG="/tmp/accel_normfix.yaml"
cat > "$ACCEL_CFG" << 'ACCEL_EOF'
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: auto
  gradient_clipping: auto
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero3_save_16bit_model: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
ACCEL_EOF

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === MID yo normfix training ==="
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] GPUs: all available"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Output: $OUT_DIR"

if [[ -f "${OUT_DIR}/adapter_config.json" ]]; then
    echo "[$(date)] Skipping training — adapter already exists."
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
        2>&1 | tee "logs/mid_${LANG}_normfix_train.log"
fi

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === Eval: mid_yo_normfix ==="
if [[ -f "$EVAL_OUT" ]]; then
    echo "[$(date)] Skipping eval — $EVAL_OUT already exists."
else
    python scripts/evaluate.py \
        --model_path  "$OUT_DIR" \
        --tasks       english \
        --languages   en \
        --skip_flores \
        --output      "$EVAL_OUT" \
        2>&1 | tee "logs/mid_${LANG}_normfix_eval.log"

    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Summary:"
    python3 -c "
import json
p = '$EVAL_OUT'
with open(p) as f: d = json.load(f)
eng = d.get('scores', {}).get('english', {})
print('  TruthfulQA:', eng.get('truthfulqa_mc1'))
print('  MMLU:      ', eng.get('mmlu'))
print('  HellaSwag: ', eng.get('hellaswag'))
print('  ARC:       ', eng.get('arc_challenge'))
print('  en_avg:    ', eng.get('english_avg'))
"
fi

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === mid_yo_normfix complete ==="
