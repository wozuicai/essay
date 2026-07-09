#!/usr/bin/env python3
"""
MID (Mechanistic Interface Distillation) training.

Teacher  : Base + LoRA_en  (merged, frozen)
Student  : Base + LoRA_spec (trainable, pure target-lang CE only)
Extra loss: CosDist at top-K layers on:
  Pos1 = instruction-end token (last token before "### Response:\n")
  Pos2 = first N response tokens

Claim: student borrows the teacher's instruction-control direction
       without ever seeing English data in training.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from peft import PeftModel
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset_loader import load_sft_dataset
from src.data.trl_dataset_utils import prepare_dataset_for_trl
from src.models.lora_standard import setup_standard_lora
from src.training.trainer import (
    build_sft_config,
    build_trainer_kwargs,
    load_causal_lm,
    load_tokenizer,
    save_run_config,
    save_training_metadata,
    setup_training_environment,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Base model path")
    p.add_argument(
        "--teacher_adapter", required=True, help="LoRA_en adapter dir (train_en result)"
    )
    p.add_argument("--train_lang", required=True, help="Target language (yo / so / ha)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", required=True, help="Experiment YAML (lis_matrix.yaml)")
    p.add_argument("--data_dir", default="data/processed")
    # MID hyper-params
    p.add_argument("--alpha", type=float, default=0.1, help="Pos1 CosDist weight")
    p.add_argument("--beta", type=float, default=0.05, help="Pos2 CosDist weight")
    p.add_argument(
        "--top_n_layers", type=int, default=4, help="Top-K layers to distill"
    )
    p.add_argument("--n_pos2", type=int, default=3, help="Response-start tokens (Pos2)")
    p.add_argument(
        "--sep_str",
        default="### Response:\n",
        help="Separator between instr and response",
    )
    # Probe
    p.add_argument("--probe_only", action="store_true")
    p.add_argument(
        "--probe_langs",
        default=None,
        help="Comma-sep langs to probe; default=train_lang",
    )
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------


def load_teacher(base_path: str, adapter_path: str, device: torch.device):
    """Merge LoRA_en into base model → single frozen inference model."""
    print(f"  [teacher] loading base from {base_path} ...")
    t = load_causal_lm(base_path, dtype=torch.bfloat16, use_cache=False)
    print(f"  [teacher] loading adapter from {adapter_path} ...")
    t = PeftModel.from_pretrained(t, adapter_path)
    t = t.merge_and_unload()
    t.eval()
    for param in t.parameters():
        param.requires_grad_(False)
    t = t.to(device)
    nparam = sum(p.numel() for p in t.parameters()) / 1e9
    print(f"  [teacher] ready on {device}, {nparam:.2f}B params, all frozen")
    return t


# ---------------------------------------------------------------------------
# Probe: hidden-state consistency check
# ---------------------------------------------------------------------------


def run_probe(
    teacher,
    tokenizer,
    data_dir: str,
    lang: str,
    device: torch.device,
    n_samples: int = 50,
    top_n_layers: int = 4,
):
    """
    Check whether teacher encodes a consistent 'instruction-control' direction
    when processing target-language text it was never trained on.

    Metric: mean pairwise cosine-sim of last-layer hidden states at Pos1.
    > 0.7 → reliable signal; 0.4–0.7 → moderate; < 0.4 → noisy.
    """
    print(
        f"\n=== PROBE [{lang}] teacher consistency at instruction-end (last layer) ==="
    )
    dataset = load_sft_dataset(data_dir, lang, n_samples=n_samples)
    sep_str = "### Response:\n"
    vecs = []

    for sample in dataset:
        text = sample["text"]
        if sep_str not in text:
            continue
        instr_text = text[: text.find(sep_str)]
        instr_ids = tokenizer(instr_text, return_tensors="pt")["input_ids"]
        pos1 = instr_ids.shape[1] - 1
        if pos1 <= 0:
            continue

        full_enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if pos1 >= full_enc["input_ids"].shape[1]:
            continue

        with torch.no_grad():
            out = teacher(
                input_ids=full_enc["input_ids"].to(device),
                attention_mask=full_enc["attention_mask"].to(device),
                output_hidden_states=True,
            )

        h = out.hidden_states[-1][0, pos1, :].float().cpu()
        vecs.append(h)

    if len(vecs) < 2:
        print(f"  Not enough valid samples ({len(vecs)}).")
        return None

    mat = torch.stack(vecs)
    normed = F.normalize(mat, dim=1)
    sim_mat = normed @ normed.T
    N = len(vecs)
    off_diag = sim_mat[~torch.eye(N, dtype=torch.bool)].mean().item()

    if off_diag > 0.7:
        verdict = "✅ HIGH — consistent control direction; MID signal reliable"
    elif off_diag > 0.4:
        verdict = "⚠️  MODERATE — proceed with caution, monitor mid_loss"
    else:
        verdict = "❌ LOW — teacher reps on this language are noisy"

    print(f"  N={N}, mean pairwise cosine-sim (last layer, Pos1) = {off_diag:.4f}")
    print(f"  Verdict: {verdict}")
    return off_diag


# ---------------------------------------------------------------------------
# MID Trainer
# ---------------------------------------------------------------------------


class MIDTrainer(SFTTrainer):
    """
    SFTTrainer with CosDist distillation on top-K layers at Pos1 and Pos2.

    total_loss = CE(student) + α·ΣCosDist(Pos1, top-K) + β·ΣCosDist(Pos2, top-K)
    """

    def set_mid_config(
        self, teacher, tokenizer, alpha, beta, top_n_layers, n_pos2, sep_str
    ):
        self._tea = teacher
        self._alpha = alpha
        self._beta = beta
        self._K = top_n_layers
        self._P2 = n_pos2
        self._sep_ids = tokenizer.encode(sep_str, add_special_tokens=False)
        self._sep_len = len(self._sep_ids)
        self._first_log = True
        self._n_batches = 0
        self._n_sep_miss = 0
        print(f"[MID] response start: labels!=-100, fallback sep_str={repr(sep_str)}")
        print(f"[MID] α={alpha}  β={beta}  top-K={top_n_layers}  Pos2={n_pos2}")

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
            stu_out.hidden_states,
            tea_out.hidden_states,
            inputs["input_ids"],
            inputs.get("labels"),
        )

        if self._first_log:
            print(
                f"[MID] 1st batch → ce={ce_loss.item():.4f}  mid={mid_loss.item():.6f}"
            )
            self._first_log = False

        total = ce_loss + mid_loss
        return (total, stu_out) if return_outputs else total

    def _mid_loss(self, stu_hs, tea_hs, input_ids: torch.Tensor, labels=None) -> torch.Tensor:
        top_idxs = list(range(len(stu_hs) - self._K, len(stu_hs)))
        sep = self._sep_ids
        slen = self._sep_len
        ids_cpu = input_ids.cpu().tolist()
        labels_cpu = labels.detach().cpu().tolist() if labels is not None else None

        total = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        valid_b = 0
        self._n_batches += 1

        for b, ids in enumerate(ids_cpu):
            T = len(ids)

            resp_start = -1
            for i in range(T - slen + 1):
                if ids[i : i + slen] == sep:
                    resp_start = i + slen
                    break
            if resp_start < 0 and labels_cpu is not None:
                for i, label_id in enumerate(labels_cpu[b]):
                    if label_id != -100:
                        # Prompt-completion fallback only. In full-sequence loss
                        # mode this would be 0, which is not a response boundary.
                        if i > 0:
                            resp_start = i
                        break

            if resp_start <= 0 or resp_start >= T:
                self._n_sep_miss += 1
                continue

            valid_b += 1
            pos1 = resp_start - 1
            pos2_pts = list(range(resp_start, min(resp_start + self._P2, T)))

            for li in top_idxs:
                sh = stu_hs[li][b]  # (T, H) — has grad
                th = tea_hs[li][b]  # (T, H) — no grad (teacher frozen)

                # Pos1: instruction-end token
                cos1 = 1.0 - F.cosine_similarity(
                    sh[pos1].float().unsqueeze(0),
                    th[pos1].float().unsqueeze(0),
                )
                total = total + self._alpha * cos1.squeeze()

                # Pos2: first N response-start tokens
                for p in pos2_pts:
                    cos2 = 1.0 - F.cosine_similarity(
                        sh[p].float().unsqueeze(0),
                        th[p].float().unsqueeze(0),
                    )
                    total = total + self._beta * cos2.squeeze()

        if valid_b == 0:
            # Entire batch had no valid separator — MID contributes nothing this step
            return torch.zeros((), device=input_ids.device)

        if self._n_batches % 200 == 0 and self._n_batches > 0:
            B = len(ids_cpu)
            miss_rate = self._n_sep_miss / (self._n_batches * B)
            print(f"[MID] batch {self._n_batches}: sep-miss rate = {miss_rate:.1%}")

        # Normalise per sample (not per token×layer): preserves intended α/β magnitude
        return total / valid_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_training_environment()
    args = parse_args()

    # ── Probe-only mode (single GPU, no distributed) ────────────────────────
    if args.probe_only:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tok = load_tokenizer(args.model)
        tea = load_teacher(args.model, args.teacher_adapter, device)
        langs = args.probe_langs.split(",") if args.probe_langs else [args.train_lang]
        for lang in langs:
            run_probe(
                tea, tok, args.data_dir, lang, device, top_n_layers=args.top_n_layers
            )
        return

    # ── Training mode ────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    cfg = OmegaConf.load(args.config)

    if local_rank == 0:
        print(
            f"\n=== MID Training: [{args.train_lang}] | "
            f"α={args.alpha} β={args.beta} K={args.top_n_layers} P2={args.n_pos2} ==="
        )

    tokenizer = load_tokenizer(args.model)

    # Each GPU process loads its own copy of the teacher (not DeepSpeed-wrapped)
    teacher = load_teacher(args.model, args.teacher_adapter, device)

    if local_rank == 0:
        run_probe(
            teacher,
            tokenizer,
            args.data_dir,
            args.train_lang,
            device,
            n_samples=50,
            top_n_layers=args.top_n_layers,
        )

    if local_rank == 0:
        print(f"\n[rank {local_rank}] Loading student base model ...")
    student = load_causal_lm(args.model, dtype=torch.bfloat16, use_cache=False)
    student.config.use_cache = False
    student = setup_standard_lora(student, cfg.peft)

    # Pure target-language data — NO English
    dataset = prepare_dataset_for_trl(
        load_sft_dataset(args.data_dir, args.train_lang),
        name=f"mid_{args.train_lang}",
    )
    if local_rank == 0:
        print(f"Dataset: {len(dataset)} × [{args.train_lang}]  (NO English)")

    # MID positions are sample-local, so do not pack multiple examples together.
    sft_cfg = build_sft_config(cfg, args.output_dir, packing=False)

    trainer = MIDTrainer(**build_trainer_kwargs(
        MIDTrainer,
        model=student,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_cfg,
    ))
    trainer.set_mid_config(
        teacher=teacher,
        tokenizer=tokenizer,
        alpha=args.alpha,
        beta=args.beta,
        top_n_layers=args.top_n_layers,
        n_pos2=args.n_pos2,
        sep_str=args.sep_str,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "method": "MID",
        "model": args.model,
        "teacher_adapter": args.teacher_adapter,
        "train_lang": args.train_lang,
        "train_samples": len(dataset),
        "alpha": args.alpha,
        "beta": args.beta,
        "top_n_layers": args.top_n_layers,
        "n_pos2": args.n_pos2,
    }
    meta.update({
        "trainer_backend": "trl_text_full_sequence",
        "completion_only_loss": False,
        "packing": False,
    })
    save_training_metadata(args.output_dir, meta)
    save_run_config(args.output_dir, args)

    if local_rank == 0:
        print(f"\n[MID] Done. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
