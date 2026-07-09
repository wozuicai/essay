#!/usr/bin/env python3
"""
DSCT (Dual-Space Constrained Tuning) training.

Teacher  : Base + LoRA_donor (merged, frozen)  — same as MID
Student  : Base + LoRA_donor (merged, frozen) + LoRA_spec (trainable)
           i.e. student starts from donor-merged base, spec is layered on top

Loss:
    L_total = L_CE
            + α · Σ CosDist(h_stu, h_tea) @ Pos1, top-K layers   [MID term]
            + β · Σ CosDist(h_stu, h_tea) @ Pos2, top-K layers   [MID term]
            + λ · L_ortho(A_donor, A_spec, B_donor, B_spec)       [new term]

L_ortho enforces that LoRA_spec matrices occupy a subspace orthogonal to
LoRA_donor matrices, preventing spec from overwriting donor's parameter space.

Orthogonality metric (per layer):
    cos²(A_donor, A_spec) + cos²(B_donor, B_spec)
where cos² is the squared cosine between the matrices viewed as flattened vectors.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset_loader import load_sft_dataset
from src.training.trainer import (
    build_sft_config,
    load_causal_lm,
    load_tokenizer,
    setup_training_environment,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Base model path")
    p.add_argument(
        "--donor_adapter", required=True, help="LoRA_en adapter dir (train_en result)"
    )
    p.add_argument("--train_lang", required=True, help="Target language (yo / so / ha)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", required=True, help="Experiment YAML")
    p.add_argument("--data_dir", default="data/processed")
    # MID hyper-params (same defaults as train_mid.py)
    p.add_argument("--alpha", type=float, default=0.1, help="Pos1 CosDist weight")
    p.add_argument("--beta", type=float, default=0.05, help="Pos2 CosDist weight")
    p.add_argument("--top_n_layers", type=int, default=4, help="Top-K layers for MID")
    p.add_argument("--n_pos2", type=int, default=3, help="Response-start tokens (Pos2)")
    p.add_argument("--sep_str", default="### Response:\n")
    # DSCT-specific
    p.add_argument(
        "--lambda_ortho", type=float, default=0.01, help="Ortho regularization weight"
    )
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def load_teacher(base_path: str, adapter_path: str, device: torch.device):
    """Merge LoRA_donor into base → frozen teacher. Identical to MID."""
    print(f"  [teacher] loading base ...")
    t = load_causal_lm(base_path, dtype=torch.bfloat16, use_cache=False)
    print(f"  [teacher] merging donor adapter ...")
    t = PeftModel.from_pretrained(t, adapter_path)
    t = t.merge_and_unload()
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    t = t.to(device)
    print(
        f"  [teacher] ready — {sum(p.numel() for p in t.parameters())/1e9:.2f}B params, frozen"
    )
    return t


def load_student(
    base_path: str, donor_adapter_path: str, cfg_peft, device: torch.device
):
    """
    Student = donor-merged base (in-memory only) + fresh LoRA_spec (trainable).

    We load base, apply donor LoRA, extract donor matrices for ortho loss,
    then merge_and_unload — no disk write.
    """
    print(f"  [student] loading base + donor adapter ...")
    base = load_causal_lm(base_path, dtype=torch.bfloat16, use_cache=False)
    peft_donor = PeftModel.from_pretrained(
        base, donor_adapter_path, adapter_name="donor"
    )

    donor_ref = {}
    for name, module in peft_donor.named_modules():
        if hasattr(module, "lora_A") and "donor" in module.lora_A:
            donor_ref[name] = (
                module.lora_A["donor"].weight.detach().clone(),
                module.lora_B["donor"].weight.detach().clone(),
            )
    print(f"  [student] extracted {len(donor_ref)} donor LoRA layers")

    print(f"  [student] merging donor into base (in-memory, no save) ...")
    merged_base = peft_donor.merge_and_unload()
    merged_base.config.use_cache = False

    # Add LoRA_spec on top of merged base
    lora_cfg = LoraConfig(
        r=cfg_peft.r,
        lora_alpha=cfg_peft.lora_alpha,
        target_modules=list(cfg_peft.target_modules),
        lora_dropout=cfg_peft.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=base_path,  # original base; donor_adapter recorded in metadata
    )
    student = get_peft_model(merged_base, lora_cfg)
    student.print_trainable_parameters()

    return student, donor_ref


# ---------------------------------------------------------------------------
# DSCT Trainer
# ---------------------------------------------------------------------------


class DSCTTrainer(SFTTrainer):
    """
    SFTTrainer with:
      1. MID loss  — CosDist at top-K layers, Pos1 & Pos2
      2. Ortho loss — squared cosine between LoRA_spec and donor_ref matrices
    """

    def set_dsct_config(
        self,
        teacher,
        donor_ref,
        tokenizer,
        alpha,
        beta,
        lambda_ortho,
        top_n_layers,
        n_pos2,
        sep_str,
    ):
        self._tea = teacher
        self._donor_ref = (
            donor_ref  # {name: (A, B)} — will be moved to device on first use
        )
        self._alpha = alpha
        self._beta = beta
        self._lambda_ortho = lambda_ortho
        self._K = top_n_layers
        self._P2 = n_pos2
        self._sep_ids = tokenizer.encode(sep_str, add_special_tokens=False)
        self._sep_len = len(self._sep_ids)
        self._first_log = True
        self._n_batches = 0
        self._n_sep_miss = 0
        self._donor_on_dev = False  # lazy device move
        print(
            f"[DSCT] α={alpha}  β={beta}  λ_ortho={lambda_ortho}  "
            f"top-K={top_n_layers}  Pos2={n_pos2}"
        )
        print(f"[DSCT] sep_str={repr(sep_str)} => ids {self._sep_ids}")
        print(f"[DSCT] donor_ref layers: {len(donor_ref)}")

    # ------------------------------------------------------------------ #
    #  compute_loss                                                        #
    # ------------------------------------------------------------------ #

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "labels")
        }

        stu_out = model(**model_inputs, output_hidden_states=True)
        ce_loss = stu_out.loss

        with torch.no_grad():
            tea_out = self._tea(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
            )

        mid_loss = self._mid_loss(
            stu_out.hidden_states, tea_out.hidden_states, inputs["input_ids"]
        )
        ortho_loss = self._ortho_loss(model)

        if self._first_log:
            print(
                f"[DSCT] 1st batch → "
                f"ce={ce_loss.item():.4f}  "
                f"mid={mid_loss.item():.6f}  "
                f"ortho={ortho_loss.item():.6f}"
            )
            self._first_log = False

        total = ce_loss + mid_loss + self._lambda_ortho * ortho_loss
        return (total, stu_out) if return_outputs else total

    # ------------------------------------------------------------------ #
    #  MID loss  (identical to MIDTrainer)                                #
    # ------------------------------------------------------------------ #

    def _mid_loss(self, stu_hs, tea_hs, input_ids: torch.Tensor) -> torch.Tensor:
        top_idxs = list(range(len(stu_hs) - self._K, len(stu_hs)))
        sep = self._sep_ids
        slen = self._sep_len
        ids_cpu = input_ids.cpu().tolist()

        total = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        n = 0
        self._n_batches += 1

        for b, ids in enumerate(ids_cpu):
            T = len(ids)
            resp_start = -1
            for i in range(T - slen + 1):
                if ids[i : i + slen] == sep:
                    resp_start = i + slen
                    break

            if resp_start <= 0 or resp_start >= T:
                self._n_sep_miss += 1
                continue

            pos1 = resp_start - 1
            pos2_pts = list(range(resp_start, min(resp_start + self._P2, T)))

            for li in top_idxs:
                sh = stu_hs[li][b]
                th = tea_hs[li][b]

                cos1 = 1.0 - F.cosine_similarity(
                    sh[pos1].float().unsqueeze(0),
                    th[pos1].float().unsqueeze(0),
                )
                total = total + self._alpha * cos1.squeeze()
                n += 1

                for p in pos2_pts:
                    cos2 = 1.0 - F.cosine_similarity(
                        sh[p].float().unsqueeze(0),
                        th[p].float().unsqueeze(0),
                    )
                    total = total + self._beta * cos2.squeeze()
                    n += 1

        if n == 0:
            return torch.zeros((), device=input_ids.device)

        if self._n_batches % 200 == 0:
            miss_rate = self._n_sep_miss / (self._n_batches * len(ids_cpu))
            print(f"[DSCT] batch {self._n_batches}: sep-miss={miss_rate:.1%}")

        return total / n

    # ------------------------------------------------------------------ #
    #  Ortho loss                                                         #
    # ------------------------------------------------------------------ #

    def _ortho_loss(self, model) -> torch.Tensor:
        """
        For each LoRA layer in student (LoRA_spec), compute squared cosine
        similarity between spec's A/B matrices and the frozen donor reference.

        ortho_A = (vec(A_donor) · vec(A_spec))² / (‖A_donor‖² ‖A_spec‖²)
        ortho_B = same for B

        Sum over all layers, return mean. Target: 0 (perfectly orthogonal).
        """
        # Lazy: move donor_ref tensors to model device once
        if not self._donor_on_dev:
            dev = next(model.parameters()).device
            self._donor_ref = {
                k: (A.to(dev), B.to(dev)) for k, (A, B) in self._donor_ref.items()
            }
            self._donor_on_dev = True

        total = torch.zeros(
            (), device=next(model.parameters()).device, dtype=torch.float32
        )
        n = 0

        for name, module in model.named_modules():
            if name not in self._donor_ref:
                continue
            if not hasattr(module, "lora_A"):
                continue

            # LoRA_spec matrices (trainable)
            # In get_peft_model the default adapter name is "default"
            adapter_name = list(module.lora_A.keys())[0]
            A_spec = module.lora_A[adapter_name].weight.float()  # (r, d_in)
            B_spec = module.lora_B[adapter_name].weight.float()  # (d_out, r)

            A_don, B_don = self._donor_ref[name]
            A_don = A_don.float()
            B_don = B_don.float()

            total = total + _cos_sq(A_don, A_spec) + _cos_sq(B_don, B_spec)
            n += 2

        return total / n if n > 0 else total


def _cos_sq(M1: torch.Tensor, M2: torch.Tensor) -> torch.Tensor:
    """Squared cosine similarity between two matrices viewed as flat vectors."""
    v1 = M1.flatten()
    v2 = M2.flatten()
    dot = (v1 * v2).sum()
    denom = (v1.norm() * v2.norm()).clamp(min=1e-12)
    return (dot / denom).pow(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_training_environment()
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    cfg = OmegaConf.load(args.config)

    if local_rank == 0:
        print(
            f"\n=== DSCT Training: [{args.train_lang}] | "
            f"α={args.alpha} β={args.beta} λ={args.lambda_ortho} "
            f"K={args.top_n_layers} P2={args.n_pos2} ===\n"
        )

    tokenizer = load_tokenizer(args.model)

    # Teacher (same as MID)
    if local_rank == 0:
        print("[DSCT] Loading teacher ...")
    teacher = load_teacher(args.model, args.donor_adapter, device)

    # Student: donor-merged base + LoRA_spec
    if local_rank == 0:
        print("[DSCT] Loading student ...")
    student, donor_ref = load_student(args.model, args.donor_adapter, cfg.peft, device)

    # Dataset: pure target language, no English
    dataset = load_sft_dataset(args.data_dir, args.train_lang)
    if local_rank == 0:
        print(f"[DSCT] Dataset: {len(dataset)} × [{args.train_lang}]  (NO English)")

    sft_cfg = build_sft_config(cfg, args.output_dir)

    trainer = DSCTTrainer(
        model=student,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_cfg,
    )
    trainer.set_dsct_config(
        teacher=teacher,
        donor_ref=donor_ref,
        tokenizer=tokenizer,
        alpha=args.alpha,
        beta=args.beta,
        lambda_ortho=args.lambda_ortho,
        top_n_layers=args.top_n_layers,
        n_pos2=args.n_pos2,
        sep_str=args.sep_str,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "method": "DSCT",
        "model": args.model,
        "donor_adapter": args.donor_adapter,
        "train_lang": args.train_lang,
        "train_samples": len(dataset),
        "alpha": args.alpha,
        "beta": args.beta,
        "lambda_ortho": args.lambda_ortho,
        "top_n_layers": args.top_n_layers,
        "n_pos2": args.n_pos2,
    }
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if local_rank == 0:
        print(f"\n[DSCT] Done. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
