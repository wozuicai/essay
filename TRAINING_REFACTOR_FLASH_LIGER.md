# Training refactor for FlashAttention 2 + Liger

Date: 2026-07-07

## What changed

The training entry points now share a common utility module:

- `src/training/trainer.py`

This module centralizes:

- worker-safe environment defaults (`setup_training_environment`)
- tokenizer loading (`load_tokenizer`)
- model loading with `attn_implementation="flash_attention_2"` (`load_causal_lm`)
- TRL `SFTConfig` creation with version-aware filtering (`build_sft_config`)
- Liger kernel enablement when supported by the installed TRL (`USE_LIGER_KERNEL=1`, default)

The following training scripts were updated to use the shared path:

- `scripts/train.py`
- `scripts/train_dsct.py`
- `scripts/train_mid.py`
- `scripts/train_layerwise.py`
- `scripts/train_moe_lora.py`
- `scripts/train_sso_lora.py`

## Runtime defaults

The refactor defaults to:

```bash
ATTN_IMPLEMENTATION=flash_attention_2
USE_LIGER_KERNEL=1
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
WANDB_DISABLED=true
```

On B200 workers, if present, it also prefers:

```bash
CUDA_HOME=/home/tiger/.local/lib/python3.11/site-packages/nvidia/cu13
LD_LIBRARY_PATH=$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
TMPDIR=/mnt/local/localcache00/tmp-tiger
TRITON_CACHE_DIR=/mnt/local/localcache00/triton-cache-tiger
```

If `/mnt/local/localcache00` exists but is not writable, the utility falls back to `/tmp` instead of failing.

## Scope and non-goals

- The custom `answer_weighted_loss` code from `/root/sft_lora` was not ported, per request.
- Existing experiment-specific losses are preserved:
  - MID hidden-state distillation
  - DSCT MID + orthogonality loss
  - SSO orthogonal penalty
  - Layerwise staged adapters
  - MoE-LoRA router/expert modules
- Current processed datasets still use the existing `text` field path in `SFTTrainer`. The code does not force prompt-completion conversion yet, because that would change loss masking semantics across all existing experiments.

## Verification performed locally

```bash
python3 -m py_compile \
  src/training/trainer.py \
  scripts/train.py \
  scripts/train_mid.py \
  scripts/train_dsct.py \
  scripts/train_layerwise.py \
  scripts/train_moe_lora.py \
  scripts/train_sso_lora.py
```

This passed on 2026-07-07.

A local `SFTConfig` constructor smoke test reached TRL argument validation, but failed on this controller because CUDA/bf16 is unavailable locally. This is expected for the controller and should be rechecked on the GPU worker.

## Operational notes

To disable FlashAttention for debugging:

```bash
export ATTN_IMPLEMENTATION=
```

To allow automatic fallback if FlashAttention import/loading fails:

```bash
export ALLOW_ATTN_FALLBACK=1
```

To disable Liger for debugging:

```bash
export USE_LIGER_KERNEL=0
```
