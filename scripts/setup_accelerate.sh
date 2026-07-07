#!/bin/bash
# Auto-generate accelerate configs based on actual GPU count.
# Call this once before running experiments.
# Usage: bash scripts/setup_accelerate.sh

set -euo pipefail

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "Detected $GPU_COUNT GPU(s)"

if [[ $GPU_COUNT -lt 1 ]]; then
  echo "ERROR: No GPUs detected. Please check nvidia-smi."
  exit 1
fi

# Determine per-experiment GPU allocation
# Phase 2 & Full-FT: use half the GPUs (minimum 2)
HALF_GPU=$(( GPU_COUNT / 2 ))
[[ $HALF_GPU -lt 1 ]] && HALF_GPU=1

# LoRA variants: use 2 GPUs (or all if < 2)
LORA_GPU=$(( GPU_COUNT < 2 ? GPU_COUNT : 2 ))

write_accel_cfg() {
  local n_gpus=$1
  local out=$2
  if [[ $n_gpus -le 1 ]]; then
    # Single GPU: no DeepSpeed overhead, avoids gradient_accumulation_steps mismatch
    cat > "$out" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
  else
    # Multi-GPU: DeepSpeed ZeRO-2 with auto values to avoid TrainingArguments mismatch
    cat > "$out" <<EOF
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
num_processes: $n_gpus
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
  fi
  echo "  Written: $out (num_processes=$n_gpus)"
}

write_accel_cfg "$GPU_COUNT"  configs/accelerate_fullgpu.yaml
write_accel_cfg "$HALF_GPU"   configs/accelerate_halfgpu.yaml
write_accel_cfg "$LORA_GPU"   configs/accelerate_loragpu.yaml

# Also rewrite the legacy named configs
write_accel_cfg "$HALF_GPU"   configs/accelerate_4gpu.yaml
write_accel_cfg "$LORA_GPU"   configs/accelerate_2gpu.yaml

# Patch ds_zero2.json micro-batch for smaller GPU configs
# If only 2 GPUs, gradient_accumulation can stay at 4, effective batch stays ~64
echo ""
echo "GPU allocation plan:"
echo "  Full-FT / phase1 evals : $GPU_COUNT GPUs  (accelerate_fullgpu.yaml)"
echo "  Phase 2 / half-GPU jobs : $HALF_GPU GPUs   (accelerate_halfgpu.yaml)"
echo "  LoRA training jobs      : $LORA_GPU GPUs   (accelerate_loragpu.yaml)"
echo ""
echo "Run 'bash scripts/setup_accelerate.sh' again after adding GPUs."
