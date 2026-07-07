#!/usr/bin/env python3
"""
Custom evaluation for MoE-LoRA model.

Since MoE-LoRA has custom layers, it cannot be loaded with AutoModelForCausalLM.from_pretrained.
This script:
  1. Loads base model + reconstructs MoE layers + loads saved weights
  2. Evaluates: TruthfulQA MC1, Belebele (yo/so/ha/en), IrokoBench MCQ (yo/ha), LCB-chat (yo/so/ha)
  3. Saves results in the same JSON format as other experiments

Usage:
  python scripts/eval_moe_lora.py \\
      --moe_dir results/moe_lora/moe_lora_Qwen3.5-9B-Base \\
      --output  results/moe_lora/moe_lora_Qwen3.5-9B-Base_eval.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.moe_lora import freeze_base, load_moe, setup_moe_lora

LANG_FLORES = {"en": "eng_Latn", "yo": "yor_Latn", "so": "som_Latn", "ha": "hau_Latn"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moe_dir",  required=True, help="MoE model directory (has moe_config.json)")
    p.add_argument("--output",   required=True, help="Output JSON path")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_moe_model(moe_dir: str, device: str):
    with open(os.path.join(moe_dir, "moe_config.json")) as f:
        cfg = json.load(f)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_path = cfg["base_model"]
    print(f"Loading base model from {base_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(moe_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    setup_moe_lora(model, cfg["target_modules"], cfg["r"], cfg["lora_alpha"],
                   cfg["n_experts"], cfg.get("dropout_p", 0.05))
    freeze_base(model)
    load_moe(model, moe_dir)
    model.eval()
    model = model.to(device)
    print("MoE model ready.")
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Log-likelihood MCQ scorer
# ---------------------------------------------------------------------------

def lm_score(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    """Return sum log-prob of completion tokens given prompt."""
    full = prompt + completion
    enc_full   = tokenizer(full,   return_tensors="pt", add_special_tokens=True).to(device)
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    n_prompt = enc_prompt["input_ids"].shape[1]

    with torch.no_grad():
        logits = model(**enc_full).logits[0]        # [T, vocab]

    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)  # [T-1, vocab]
    labels    = enc_full["input_ids"][0, 1:]                    # [T-1]

    # Positions n_prompt-1 .. T-2 predict completion tokens
    if n_prompt - 1 >= len(labels):
        return -1e9
    comp_lp = log_probs[n_prompt - 1:]   # [T_comp, vocab]
    comp_lb = labels[n_prompt - 1:]      # [T_comp]
    return comp_lp[torch.arange(len(comp_lb)), comp_lb].sum().item()


def mcq_accuracy(model, tokenizer, examples, prompt_fn, choices_fn, label_fn, device, desc=""):
    correct = total = 0
    for ex in tqdm(examples, desc=desc, leave=False):
        prompt  = prompt_fn(ex)
        choices = choices_fn(ex)
        label   = label_fn(ex)
        scores  = [lm_score(model, tokenizer, prompt, c, device) for c in choices]
        pred    = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == label)
        total   += 1
    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# TruthfulQA MC1
# ---------------------------------------------------------------------------

def eval_truthfulqa(model, tokenizer, device):
    from datasets import load_dataset
    print("  [TruthfulQA MC1] loading ...")
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    except Exception:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

    def prompt_fn(ex):
        return f"### Instruction:\n<|tgt_lang:en|> {ex['question']}\n\n### Response:\n"

    def choices_fn(ex):
        return ex["mc1_targets"]["choices"]

    def label_fn(ex):
        labels = ex["mc1_targets"]["labels"]
        return next(i for i, l in enumerate(labels) if l == 1)

    acc = mcq_accuracy(model, tokenizer, list(ds), prompt_fn, choices_fn, label_fn,
                       device, desc="TruthfulQA")
    print(f"  TruthfulQA MC1: {acc:.4f}")
    return acc


# ---------------------------------------------------------------------------
# Belebele
# ---------------------------------------------------------------------------

BELEBELE_TEMPLATE = (
    "The following are reading comprehension questions.\n\n"
    "Passage: {passage}\n"
    "Question: {question}\n"
    "A: {a1}\nB: {a2}\nC: {a3}\nD: {a4}\n"
    "Answer:"
)

def eval_belebele(model, tokenizer, device, langs=None):
    from datasets import load_dataset
    if langs is None:
        langs = ["en", "yo", "so", "ha"]

    results = {}
    for lang in langs:
        flores_code = LANG_FLORES[lang]
        print(f"  [Belebele] loading {lang}/{flores_code} ...")
        try:
            ds = load_dataset("facebook/belebele", flores_code, split="test")
        except Exception:
            try:
                ds = load_dataset("Muennighoff/belebele", flores_code, split="test")
            except Exception as e:
                print(f"  Belebele {lang} failed: {e}")
                results[lang] = None
                continue

        def prompt_fn(ex):
            return BELEBELE_TEMPLATE.format(
                passage=ex["flores_passage"],
                question=ex["question"],
                a1=ex["mc_answer1"], a2=ex["mc_answer2"],
                a3=ex["mc_answer3"], a4=ex["mc_answer4"],
            )

        choices = [" A", " B", " C", " D"]

        def label_fn(ex):
            return int(ex["correct_answer_num"]) - 1  # 1-indexed → 0-indexed

        acc = mcq_accuracy(model, tokenizer, list(ds),
                           prompt_fn, lambda _: choices, label_fn,
                           device, desc=f"Belebele-{lang}")
        print(f"  Belebele {lang}: {acc:.4f}")
        results[lang] = acc

    return results


# ---------------------------------------------------------------------------
# IrokoBench MCQ (AfriMMLU)
# ---------------------------------------------------------------------------

IROKO_TEMPLATE = (
    "The following is a multiple choice question. Choose the best answer.\n\n"
    "Question: {question}\n"
    "A: {a}\nB: {b}\nC: {c}\nD: {d}\n"
    "Answer:"
)
IROKO_CHOICE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}

def eval_irokobench(model, tokenizer, device, langs=None):
    from datasets import load_dataset
    if langs is None:
        langs = ["yo", "ha"]

    results = {}
    for lang in langs:
        print(f"  [IrokoBench MCQ] loading {lang} ...")
        try:
            ds = load_dataset("masakhane/afrimmlu", lang, split="test")
        except Exception as e:
            print(f"  IrokoBench {lang} failed: {e}")
            results[lang] = None
            continue

        ds = list(ds)[:500]  # cap at 500 to match existing evals

        def prompt_fn(ex):
            return IROKO_TEMPLATE.format(
                question=ex["question"],
                a=ex["choice_a"], b=ex["choice_b"],
                c=ex["choice_c"], d=ex["choice_d"],
            )

        choices = [" A", " B", " C", " D"]

        def label_fn(ex):
            ans = ex.get("answer", "A").strip().upper()
            return IROKO_CHOICE_MAP.get(ans, 0)

        acc = mcq_accuracy(model, tokenizer, ds,
                           prompt_fn, lambda _: choices, label_fn,
                           device, desc=f"IrokoBench-{lang}")
        print(f"  IrokoBench MCQ {lang}: {acc:.4f}")
        results[lang] = {"mcq_accuracy": acc}

    return results


# ---------------------------------------------------------------------------
# LCB-chat (language consistency, generation + GlotLID)
# ---------------------------------------------------------------------------

LCB_CHAT_TEMPLATE = (
    "### Instruction:\n"
    "<|tgt_lang:en|> Please respond to the following in {lang_name}: {instr}\n\n"
    "### Response:\n"
)

LANG_NAMES = {"yo": "Yoruba", "so": "Somali", "ha": "Hausa"}


def eval_lcb_chat(model, tokenizer, device, n_prompts=50):
    from datasets import load_dataset

    print("  [LCB-chat] loading instructions ...")
    try:
        en_ds = load_dataset("garage-bAInd/Open-Platypus", split="train[:200]")
        instructions = [ex["instruction"] for ex in en_ds if ex.get("instruction")][:n_prompts]
    except Exception:
        instructions = [f"Explain what is happening in the world today. Example {i}."
                        for i in range(n_prompts)]

    # GlotLID detector
    try:
        from transformers import pipeline as hf_pipeline
        glotlid = hf_pipeline("text-classification", model="cis-lmu/glotlid",
                               device=0 if "cuda" in device else -1,
                               truncation=True, max_length=128)
        def detect_lang(text):
            if not text.strip():
                return "und"
            result = glotlid(text[:512])[0]["label"]
            return result.split("_")[0] if "_" in result else result
    except Exception as e:
        print(f"  GlotLID unavailable ({e}), using langdetect fallback")
        try:
            from langdetect import detect
            def detect_lang(t):
                if not t.strip():
                    return "und"
                try:
                    return detect(t)
                except Exception:
                    return "und"
        except Exception:
            detect_lang = lambda t: "und"

    results = {}
    for lang in ["yo", "so", "ha"]:
        lang_name = LANG_NAMES[lang]
        lc_count = 0

        for instr in tqdm(instructions, desc=f"LCB-chat-{lang}", leave=False):
            prompt = LCB_CHAT_TEMPLATE.format(lang_name=lang_name, instr=instr)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out[0, enc["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            detected = detect_lang(gen_text)
            if detected == lang:
                lc_count += 1

        lc_rate = lc_count / len(instructions)
        print(f"  LCB-chat {lang}: lc_rate={lc_rate:.3f}")
        results[lang] = {"lc_rate": lc_rate, "n_prompts": len(instructions)}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model, tokenizer, moe_cfg = build_moe_model(args.moe_dir, args.device)

    scores = {}

    # English: TruthfulQA MC1
    print("\n=== TruthfulQA MC1 ===")
    scores["english"] = {"truthfulqa_mc1": eval_truthfulqa(model, tokenizer, args.device)}

    # Multilingual: Belebele
    print("\n=== Belebele ===")
    bele = eval_belebele(model, tokenizer, args.device)
    scores.setdefault("multilingual", {})["belebele"] = bele

    # IrokoBench MCQ
    print("\n=== IrokoBench MCQ ===")
    iroko = eval_irokobench(model, tokenizer, args.device)
    scores["multilingual"]["irokobench"] = iroko

    # LCB-chat
    print("\n=== LCB-chat (cross-lingual instruction following) ===")
    lcb = eval_lcb_chat(model, tokenizer, args.device)
    scores["multilingual"]["lcb_chat"] = lcb

    result = {
        "model_path": args.moe_dir,
        "method": "MoE-LoRA",
        "moe_config": moe_cfg,
        "scores": scores,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {args.output}")

    # Quick summary
    tqa  = scores["english"].get("truthfulqa_mc1", 0)
    bele = scores["multilingual"].get("belebele", {})
    iroko = scores["multilingual"].get("irokobench", {})
    print("\n=== Summary ===")
    print(f"TruthfulQA MC1: {tqa:.4f}")
    for lang in ["en", "yo", "so", "ha"]:
        print(f"Belebele {lang}: {bele.get(lang, 'N/A')}")
    for lang in ["yo", "ha"]:
        mcq = (iroko.get(lang) or {}).get("mcq_accuracy", "N/A")
        print(f"IrokoBench MCQ {lang}: {mcq}")
    lcb = scores["multilingual"].get("lcb_chat", {})
    for lang in ["yo", "so", "ha"]:
        lc = (lcb.get(lang) or {}).get("lc_rate", "N/A")
        print(f"LCB-chat {lang}: {lc}")


if __name__ == "__main__":
    main()
