#!/usr/bin/env bash
set -euo pipefail

cd /root/project

RUN_ID="${RUN_ID:-refactor_flash_liger_20260707}"
ROOT="results/${RUN_ID}"
LOG_ROOT="logs/${RUN_ID}"
MODEL="${MODEL:-/root/project/models/Qwen3.5-9B-Base}"
MODEL_SHORT="Qwen3.5-9B-Base"
EXP_CFG="${EXP_CFG:-configs/experiments/lis_matrix.yaml}"
ACCEL8="${ACCEL8:-configs/accelerate_fullgpu.yaml}"
ACCEL2="${ACCEL2:-configs/accelerate_2gpu.yaml}"

mkdir -p "$ROOT" "$LOG_ROOT"

export CUDA_HOME="${CUDA_HOME:-/home/tiger/.local/lib/python3.11/site-packages/nvidia/cu13}"
export PATH="/home/tiger/.local/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export TMPDIR="${TMPDIR:-/mnt/local/localcache00/tmp-tiger}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/local/localcache00/triton-cache-tiger}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-1}"
export HF_HOME="${HF_HOME:-/root/project/hf_cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

# Keep training/eval manageable by default; user allowed hyperparams to change.
export PACKING="${PACKING:-0}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

log(){ echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

append_progress(){
  printf '\n[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" >> /root/progress.md
}

run_cmd(){
  local name="$1"; shift
  log "START $name"
  append_progress "START $name"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    log "FAIL $name rc=$rc"
    append_progress "FAIL $name rc=$rc (see $LOG_ROOT/${name}.log)"
    return $rc
  fi
  log "DONE $name"
  append_progress "DONE $name"
}

cleanup_large_files(){
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  log "cleanup large files under $dir"
  # Preserve JSON/config/tokenizer/readme metadata; remove only large training artifacts in this run directory.
  find "$dir" -type f \( \
    -name '*.pt' -o -name '*.pth' -o -name 'optimizer.pt' -o -name 'scheduler.pt' -o \
    -name 'rng_state*.pth' -o -name '*optim_states.pt' -o -name 'mp_rank_*_model_states.pt' \
  \) -print -delete || true
  find "$dir" -type d -name 'checkpoint-*' -print -exec rm -rf {} + || true
}

std_eval(){
  local model_path="$1"; local out_json="$2"; local tag_arg="${3:-}"
  if [[ -f "$out_json" ]]; then log "skip eval exists $out_json"; return 0; fi
  run_cmd "eval_$(basename "$out_json" .json)" \
    python scripts/evaluate.py \
      --model_path "$model_path" \
      --tasks all \
      --languages en,yo,so,ha \
      --skip_flores \
      --skip_sib200 \
      --output "$out_json" \
      $tag_arg
  run_cmd "iroko_mcq_$(basename "$out_json" .json)" \
    python scripts/eval_extended.py --model_path "$model_path" --result_json "$out_json" --only_iroko_mcq $tag_arg
  run_cmd "iroko_extra_$(basename "$out_json" .json)" \
    python scripts/eval_extended.py --model_path "$model_path" --result_json "$out_json" --only_iroko_extra $tag_arg
}

train_standard(){
  local lang="$1"; local out_dir="$2"; shift 2
  if [[ -f "$out_dir/adapter_config.json" ]]; then log "skip train exists $out_dir"; return 0; fi
  run_cmd "train_standard_${lang}_$(basename "$out_dir")" \
    accelerate launch --config_file "$ACCEL8" scripts/train.py \
      --model "$MODEL" --train_lang "$lang" --method standard_lora \
      --output_dir "$out_dir" --config "$EXP_CFG" --no_wandb "$@"
}

log "Run ID: $RUN_ID"
log "Root: $ROOT"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
python3 - <<'PY'
import torch, flash_attn, inspect
from trl import SFTConfig
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())
print('flash_attn', flash_attn.__version__)
print('use_liger_kernel_param', 'use_liger_kernel' in inspect.signature(SFTConfig.__init__).parameters)
PY

# 1. phase2_v2 equivalent: train en/yo/so/ha, eval all. English adapter is donor for MID/DSCT.
PHASE2="$ROOT/phase2_v2"
mkdir -p "$PHASE2"
for LANG in en yo so ha; do
  OUT="$PHASE2/lis_${MODEL_SHORT}_train_${LANG}"
  EVAL="$PHASE2/lis_${MODEL_SHORT}_train_${LANG}_eval.json"
  train_standard "$LANG" "$OUT"
  std_eval "$OUT" "$EVAL"
  # Do not clean train_en until MID/DSCT finish; it is the donor.
  if [[ "$LANG" != "en" ]]; then cleanup_large_files "$OUT"; fi
done
DONOR="$PHASE2/lis_${MODEL_SHORT}_train_en"

# 2. mix_en: en+target full concat
MIX="$ROOT/mix_en"; mkdir -p "$MIX"
for LANG in yo so ha; do
  OUT="$MIX/mix_${MODEL_SHORT}_en_${LANG}"
  EVAL="$MIX/mix_${MODEL_SHORT}_en_${LANG}_eval.json"
  train_standard "$LANG" "$OUT" --mix_all
  std_eval "$OUT" "$EVAL"
  cleanup_large_files "$OUT"
done

# 3. tag_routing: all 4 languages + tag injected eval
TAG="$ROOT/tag_routing"; mkdir -p "$TAG"
TAG_OUT="$TAG/tag_routing_${MODEL_SHORT}"
train_standard en "$TAG_OUT" --mix_all_langs
std_eval "$TAG_OUT" "$TAG/tag_routing_${MODEL_SHORT}_eval.json" "--inject_lang_tag"
cleanup_large_files "$TAG_OUT"

# 4. MID
MID="$ROOT/mid"; mkdir -p "$MID"
for LANG in yo so ha; do
  OUT="$MID/mid_${MODEL_SHORT}_${LANG}"
  EVAL="$MID/mid_${MODEL_SHORT}_${LANG}_eval.json"
  if [[ ! -f "$OUT/adapter_config.json" ]]; then
    run_cmd "train_mid_${LANG}" accelerate launch --config_file "$ACCEL8" scripts/train_mid.py \
      --model "$MODEL" --teacher_adapter "$DONOR" --train_lang "$LANG" \
      --output_dir "$OUT" --config "$EXP_CFG" --alpha 0.1 --beta 0.05 --top_n_layers 4 --n_pos2 3 --no_wandb
  fi
  std_eval "$OUT" "$EVAL"
  cleanup_large_files "$OUT"
done

# 5. DSCT
DSCT="$ROOT/dsct"; mkdir -p "$DSCT"
for LANG in yo so ha; do
  OUT="$DSCT/dsct_${MODEL_SHORT}_${LANG}"
  EVAL="$DSCT/dsct_${MODEL_SHORT}_${LANG}_eval.json"
  if [[ ! -f "$OUT/adapter_config.json" ]]; then
    run_cmd "train_dsct_${LANG}" accelerate launch --config_file "$ACCEL8" scripts/train_dsct.py \
      --model "$MODEL" --donor_adapter "$DONOR" --train_lang "$LANG" \
      --output_dir "$OUT" --config "$EXP_CFG" --alpha 0.1 --beta 0.05 --lambda_ortho 0.01 --top_n_layers 4 --n_pos2 3 --no_wandb
  fi
  std_eval "$OUT" "$EVAL"
  cleanup_large_files "$OUT"
done

# Donor no longer needed after MID/DSCT have finished.
cleanup_large_files "$DONOR"

# 6. MoE-LoRA
MOE="$ROOT/moe_lora"; mkdir -p "$MOE"
MOE_OUT="$MOE/moe_lora_${MODEL_SHORT}"
MOE_EVAL="$MOE/moe_lora_${MODEL_SHORT}_eval.json"
if [[ ! -f "$MOE_OUT/moe_config.json" ]]; then
  run_cmd "train_moe_lora" accelerate launch --config_file "$ACCEL2" --main_process_port 29511 scripts/train_moe_lora.py \
    --model "$MODEL" --output_dir "$MOE_OUT" --config "$EXP_CFG" --n_experts 4 --r 8 --lora_alpha 16.0 --no_wandb
fi
if [[ ! -f "$MOE_EVAL" ]]; then
  run_cmd "eval_moe_lora" python scripts/eval_moe_lora.py --moe_dir "$MOE_OUT" --output "$MOE_EVAL" --device cuda
  # eval_moe_lora has its own benchmark scope; if it lacks iroko extra, supplement if possible is non-trivial for custom MoE loader.
fi
cleanup_large_files "$MOE_OUT"

# 7. SSO-LoRA
SSO="$ROOT/sso_lora"; mkdir -p "$SSO"
SSO_STAGE1="$SSO/stage1_shared"
if [[ ! -f "$SSO_STAGE1/adapter_config.json" && ! -f "$SSO_STAGE1/shared/adapter_config.json" ]]; then
  run_cmd "train_sso_stage1" accelerate launch --config_file "$ACCEL8" --main_process_port 29512 scripts/train_sso_lora.py \
    --model "$MODEL" --output_dir "$SSO" --config "$EXP_CFG" --mode stage1 --r_shared 16 --lora_alpha_shared 32.0 --no_wandb
fi
for LANG in yo so ha; do
  EVAL="$SSO/sso_${MODEL_SHORT}_${LANG}_eval.json"
  if [[ ! -f "$SSO/stage2_${LANG}/${LANG}/adapter_config.json" ]]; then
    run_cmd "train_sso_stage2_${LANG}" accelerate launch --config_file "$ACCEL8" --main_process_port 29512 scripts/train_sso_lora.py \
      --model "$MODEL" --output_dir "$SSO" --config "$EXP_CFG" --mode stage2 --train_lang "$LANG" --stage1_dir "$SSO_STAGE1" \
      --r_lang 8 --lora_alpha_lang 16.0 --orth_weight 0.1 --no_wandb
  fi
  if [[ ! -f "$EVAL" ]]; then
    run_cmd "eval_sso_${LANG}" python scripts/train_sso_lora.py \
      --model "$MODEL" --output_dir "$SSO" --config "$EXP_CFG" --mode merge_eval --train_lang "$LANG" --stage1_dir "$SSO_STAGE1" --eval_output "$EVAL"
    # supplement Iroko extra with in-memory merged model is not supported by eval_extended without disk save; keep generated eval JSON.
  fi
  cleanup_large_files "$SSO/stage2_${LANG}"
done
cleanup_large_files "$SSO_STAGE1"

if [[ "${RUN_LAYERWISE:-0}" == "1" ]]; then
  LAYER="$ROOT/layerwise"; mkdir -p "$LAYER"
  LAYER_STAGE1="$LAYER/stage1_shared"
  if [[ ! -f "$LAYER_STAGE1/adapter_config.json" && ! -f "$LAYER_STAGE1/shared/adapter_config.json" ]]; then
    run_cmd "train_layerwise_stage1" accelerate launch --config_file "$ACCEL8" --main_process_port 29513 scripts/train_layerwise.py \
      --model "$MODEL" --output_dir "$LAYER" --config "$EXP_CFG" --mode stage1 --r 16 --lora_alpha 32.0 --no_wandb
  fi
  for LANG in yo so ha; do
    EVAL="$LAYER/layerwise_${MODEL_SHORT}_${LANG}_eval.json"
    if [[ ! -f "$LAYER/stage2_${LANG}/${LANG}/adapter_config.json" ]]; then
      run_cmd "train_layerwise_stage2_${LANG}" accelerate launch --config_file "$ACCEL8" --main_process_port 29513 scripts/train_layerwise.py \
        --model "$MODEL" --output_dir "$LAYER" --config "$EXP_CFG" --mode stage2 --train_lang "$LANG" --stage1_dir "$LAYER_STAGE1" --r 16 --lora_alpha 32.0 --no_wandb
    fi
    if [[ ! -f "$EVAL" ]]; then
      run_cmd "eval_layerwise_${LANG}" python scripts/train_layerwise.py \
        --model "$MODEL" --output_dir "$LAYER" --config "$EXP_CFG" --mode merge_eval --train_lang "$LANG" --stage1_dir "$LAYER_STAGE1" --eval_output "$EVAL"
    fi
    cleanup_large_files "$LAYER/stage2_${LANG}"
  done
  cleanup_large_files "$LAYER_STAGE1"
fi

log "ALL DONE $RUN_ID"
append_progress "ALL DONE $RUN_ID"
