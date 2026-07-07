#!/usr/bin/env bash
# Remove large transient artifacts after an experiment is trained/evaluated.
# Keeps trained adapter/MoE weights by default because downstream experiments may
# need them. Set DELETE_TRAINED_WEIGHTS=1 to also delete adapter_model.safetensors
# and moe_weights.safetensors after the final eval.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/cleanup_large_artifacts.sh <path> [<path> ...]" >&2
  exit 2
fi

for target in "$@"; do
  if [[ ! -e "$target" ]]; then
    echo "[cleanup] skip missing: $target"
    continue
  fi
  echo "[cleanup] $target"
  find "$target" -type d -name 'checkpoint-*' -prune -exec rm -rf {} +
  find "$target" -type f \( \
    -name '*.bin' -o \
    -name '*.pt' -o \
    -name '*.pth' -o \
    -name '*.ckpt' -o \
    -name 'model.safetensors' -o \
    -name 'model-*.safetensors' -o \
    -name 'pytorch_model*.bin' -o \
    -name 'optimizer.pt' -o \
    -name 'scheduler.pt' -o \
    -name 'trainer_state.json' -o \
    -name 'rng_state*.pth' \
  \) -delete
  if [[ "${DELETE_TRAINED_WEIGHTS:-0}" == "1" ]]; then
    find "$target" -type f \( \
      -name 'adapter_model.safetensors' -o \
      -name 'moe_weights.safetensors' \
    \) -delete
  fi
done
