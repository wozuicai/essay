#!/usr/bin/env python3
"""
Layer-wise Progressive Language Routing training.

--mode stage1  : Train shared bottom LoRA (layers 0..split-1) on all 4 languages.
--mode stage2  : Train lang-specific top LoRA (layers split..31) on target language.
                 Requires --train_lang and --stage1_dir.
--mode merge   : Merge stage1 + stage2 into a plain HF model (CPU, no GPU needed).
                 Requires --train_lang and --stage1_dir.
"""

import argparse
import json
import os
import sys

import torch
from datasets import concatenate_datasets
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import load_sft_dataset
from src.data.trl_dataset_utils import prepare_dataset_for_trl
from src.models.layerwise_lora import add_lang_top, setup_shared_bottom
from src.training.trainer import (
    build_sft_config,
    build_trainer_kwargs,
    load_causal_lm,
    load_tokenizer,
    require_full_model_save_allowed,
    setup_training_environment,
)

N_LAYERS = 32  # Qwen3.5-9B has 32 transformer layers
SPLIT = 16  # bottom 0-15 = shared, top 16-31 = lang-specific


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument(
        "--mode", required=True, choices=["stage1", "stage2", "merge", "merge_eval"]
    )
    p.add_argument(
        "--train_lang", default=None, help="Target language for stage2/merge"
    )
    p.add_argument("--stage1_dir", default=None, help="Path to stage1 shared adapter")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument(
        "--final_dir",
        default=None,
        help="Override merged model output dir (e.g. /tmp/...). "
        "If not set, defaults to output_dir/layerwise_{lang}.",
    )
    p.add_argument(
        "--eval_output",
        default=None,
        help="Path for eval JSON (merge_eval mode). Skips if file already exists.",
    )
    return p.parse_args()


def run_stage1(args, cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    stage1_dir = os.path.join(args.output_dir, "stage1_shared")

    if os.path.exists(os.path.join(stage1_dir, "adapter_config.json")) or os.path.exists(
        os.path.join(stage1_dir, "shared", "adapter_config.json")
    ):
        if local_rank == 0:
            print(f"Stage 1 adapter found at {stage1_dir}, skipping training.")
        return

    if local_rank == 0:
        print(
            f"\n=== Stage 1: Shared bottom LoRA (layers 0-{SPLIT-1}, 4-lang, "
            f"r={args.r}) ===\n"
        )

    tokenizer = load_tokenizer(args.model)

    base = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)

    model = setup_shared_bottom(
        base,
        N_LAYERS,
        SPLIT,
        r=args.r,
        lora_alpha=args.lora_alpha,
        dropout_p=float(cfg.peft.get("lora_dropout", 0.05)),
    )

    all_langs = ["en", "yo", "so", "ha"]
    parts = [load_sft_dataset(args.data_dir, l) for l in all_langs]
    train_dataset = prepare_dataset_for_trl(
        concatenate_datasets(parts).shuffle(seed=42),
        name="layerwise_stage1_all_langs",
    )
    if local_rank == 0:
        sizes = " + ".join(f"{len(p)} {l}" for l, p in zip(all_langs, parts))
        print(f"Stage 1 dataset: {sizes} = {len(train_dataset)}")

    sft_cfg = build_sft_config(cfg, stage1_dir)
    trainer = SFTTrainer(**build_trainer_kwargs(
        SFTTrainer,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_cfg,
    ))
    trainer.train()

    if local_rank == 0:
        model.save_pretrained(stage1_dir, selected_adapters=["shared"])
        tokenizer.save_pretrained(stage1_dir)
        print(f"Stage 1 shared adapter saved → {stage1_dir}")


def run_stage2(args, cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    lang = args.train_lang
    assert lang, "--train_lang required for stage2"

    stage1_dir = os.path.abspath(
        args.stage1_dir or os.path.join(args.output_dir, "stage1_shared")
    )
    assert os.path.exists(
        os.path.join(stage1_dir, "adapter_config.json")
    ) or os.path.exists(
        os.path.join(stage1_dir, "shared", "adapter_config.json")
    ), f"Stage 1 adapter not found at {stage1_dir}"

    stage2_dir = os.path.join(args.output_dir, f"stage2_{lang}")

    if os.path.exists(os.path.join(stage2_dir, lang, "adapter_config.json")):
        if local_rank == 0:
            print(f"[{lang}] Stage 2 adapter found at {stage2_dir}, skipping training.")
        return

    if local_rank == 0:
        print(
            f"\n=== Stage 2: [{lang}] top LoRA (layers {SPLIT}-{N_LAYERS-1}, "
            f"r={args.r}, 1 epoch) ===\n"
        )

    tokenizer = load_tokenizer(args.model)

    from peft import PeftModel

    base = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)
    model = PeftModel.from_pretrained(
        base, os.path.join(stage1_dir, "shared"), adapter_name="shared"
    )
    model = add_lang_top(
        model,
        lang,
        SPLIT,
        N_LAYERS,
        r=args.r,
        lora_alpha=args.lora_alpha,
        dropout_p=float(cfg.peft.get("lora_dropout", 0.05)),
    )

    lang_data = prepare_dataset_for_trl(
        load_sft_dataset(args.data_dir, lang),
        name=f"layerwise_stage2_{lang}",
    )
    if local_rank == 0:
        print(f"Stage 2 [{lang}]: {len(lang_data)} samples")

    cfg_s2 = OmegaConf.to_container(cfg, resolve=True)
    cfg_s2["training"]["num_epochs"] = 1
    cfg_s2 = OmegaConf.create(cfg_s2)
    sft_cfg = build_sft_config(cfg_s2, stage2_dir)

    trainer = SFTTrainer(**build_trainer_kwargs(
        SFTTrainer,
        model=model,
        processing_class=tokenizer,
        train_dataset=lang_data,
        args=sft_cfg,
    ))
    trainer.train()

    if local_rank == 0:
        model.save_pretrained(stage2_dir, selected_adapters=[lang])
        tokenizer.save_pretrained(stage2_dir)
        print(f"[{lang}] Stage 2 adapter saved → {stage2_dir}")


def run_merge(args):
    """Merge shared + lang adapter → plain HF model. CPU-only, no DDP."""
    require_full_model_save_allowed("train_layerwise.py --mode merge")
    lang = args.train_lang
    assert lang, "--train_lang required for merge"

    stage1_dir = os.path.abspath(
        args.stage1_dir or os.path.join(args.output_dir, "stage1_shared")
    )
    stage2_dir = os.path.abspath(os.path.join(args.output_dir, f"stage2_{lang}"))
    final_dir = os.path.abspath(
        args.final_dir or os.path.join(args.output_dir, f"layerwise_{lang}")
    )

    if os.path.exists(os.path.join(final_dir, "config.json")):
        print(f"[{lang}] Merged model already exists at {final_dir}, skipping.")
        return

    print(
        f"\n=== Merging [{lang}]: shared (layers 0-{SPLIT-1}) + lang (layers {SPLIT}-{N_LAYERS-1}) ==="
    )

    tokenizer = load_tokenizer(args.model)

    from peft import PeftModel

    base = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)

    # Step 1: merge shared adapter (layers 0-15) into base
    model = PeftModel.from_pretrained(
        base, os.path.join(stage1_dir, "shared"), adapter_name="shared"
    )
    merged1 = model.merge_and_unload()
    # Step 2: load lang adapter (layers 16-31) on top and merge
    merged1_peft = PeftModel.from_pretrained(
        merged1, os.path.join(stage2_dir, lang), adapter_name=lang
    )
    merged = merged1_peft.merge_and_unload()

    os.makedirs(final_dir, exist_ok=True)
    merged.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    meta = {
        "method": "layerwise",
        "lang": lang,
        "split_layer": SPLIT,
        "n_layers": N_LAYERS,
        "r": args.r,
        "lora_alpha": args.lora_alpha,
        "stage1_dir": stage1_dir,
        "stage2_dir": stage2_dir,
    }
    with open(os.path.join(final_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{lang}] Merged model → {final_dir}")


def run_merge_eval(args):
    """Merge adapters in memory, run eval on the live model, save only JSON. No disk save."""
    lang = args.train_lang
    assert lang, "--train_lang required for merge_eval"
    eval_out = args.eval_output
    assert eval_out, "--eval_output required for merge_eval"

    if os.path.exists(eval_out):
        print(f"[{lang}] Eval already exists at {eval_out}, skipping.")
        return

    stage1_dir = os.path.abspath(
        args.stage1_dir or os.path.join(args.output_dir, "stage1_shared")
    )
    stage2_dir = os.path.abspath(os.path.join(args.output_dir, f"stage2_{lang}"))

    print(f"\n=== Merge+Eval [{lang}] in memory (no disk save) ===")

    tokenizer = load_tokenizer(args.model)

    from peft import PeftModel

    base = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)
    # Step 1: merge shared adapter (layers 0-15) into base
    model = PeftModel.from_pretrained(
        base, os.path.join(stage1_dir, "shared"), adapter_name="shared"
    )
    merged1 = model.merge_and_unload()
    # Step 2: load lang adapter (layers 16-31) on top and merge
    merged1_peft = PeftModel.from_pretrained(
        merged1, os.path.join(stage2_dir, lang), adapter_name=lang
    )
    merged = merged1_peft.merge_and_unload()
    merged.eval()
    merged = merged.cuda()
    print(f"[{lang}] Merge complete (shared→base, {lang}→base), model on GPU.")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.evaluation.english_eval import run_english_eval
    from src.evaluation.irokobench_eval import run_irokobench_eval
    from src.evaluation.multilingual_eval import run_multilingual_eval

    results = {"model_path": f"layerwise_{lang}_in_memory", "scores": {}}

    print("=== English eval ===")
    results["scores"]["english"] = run_english_eval(
        model_path=args.model, model=merged, tokenizer=tokenizer
    )

    print("=== Multilingual eval ===")
    results["scores"]["multilingual"] = run_multilingual_eval(
        model_path=args.model,
        model=merged,
        tokenizer=tokenizer,
        languages=["en", "yo", "so", "ha"],
        run_flores=False,
        run_sib200=False,
        run_belebele=True,
    )
    print("=== IrokoBench eval ===")
    results["scores"]["multilingual"]["irokobench"] = run_irokobench_eval(
        merged, tokenizer
    )

    os.makedirs(os.path.dirname(os.path.abspath(eval_out)) or ".", exist_ok=True)
    with open(eval_out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[{lang}] Eval saved → {eval_out}")

    del merged
    torch.cuda.empty_cache()


def main():
    setup_training_environment()
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "stage1":
        run_stage1(args, cfg)
    elif args.mode == "stage2":
        run_stage2(args, cfg)
    elif args.mode == "merge":
        run_merge(args)
    elif args.mode == "merge_eval":
        run_merge_eval(args)


if __name__ == "__main__":
    main()
