"""
Soft Mixture-of-LoRA-Experts (MoE-LoRA / LA-MoA).

Each target linear layer is replaced with MoELoRALinear, which has:
  - K LoRA expert pairs (A_i, B_i), each rank r
  - A token-level learnable router: softmax(W_r @ x) → gate weights [K]
  - Output = base(x) + Σ_i gate_i * B_i(A_i(dropout(x))) * (alpha/r)

All base model parameters are frozen; only lora_A, lora_B, router are trained.
"""

import json
import math
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


class MoELoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, lora_alpha: float,
                 n_experts: int, dropout_p: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.n_experts = n_experts
        self.scaling = lora_alpha / r
        d_in, d_out = base_layer.in_features, base_layer.out_features

        self.lora_A = nn.ModuleList([nn.Linear(d_in, r, bias=False) for _ in range(n_experts)])
        self.lora_B = nn.ModuleList([nn.Linear(r, d_out, bias=False) for _ in range(n_experts)])
        self.router = nn.Linear(d_in, n_experts, bias=False)
        self.dropout = nn.Dropout(dropout_p)

        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i].weight)
        nn.init.zeros_(self.router.weight)  # start uniform gate
        # Match dtype/device of the wrapped layer. This keeps eval robust if the
        # base model was loaded directly onto GPU or with a non-default device.
        target_dtype = base_layer.weight.dtype
        target_device = base_layer.weight.device
        self.lora_A.to(device=target_device, dtype=target_dtype)
        self.lora_B.to(device=target_device, dtype=target_dtype)
        self.router.to(device=target_device, dtype=target_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        gate = torch.softmax(self.router(x), dim=-1)          # [..., n_experts]
        x_d = self.dropout(x)
        lora = sum(gate[..., i:i+1] * self.lora_B[i](self.lora_A[i](x_d))
                   for i in range(self.n_experts))
        return base + lora * self.scaling


def setup_moe_lora(model, target_module_names, r, lora_alpha, n_experts, dropout_p=0.05):
    """Replace target nn.Linear layers with MoELoRALinear. Returns list of replaced paths."""
    replaced = []
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf not in target_module_names or not isinstance(module, nn.Linear):
            continue
        parent_path, child = (name.rsplit(".", 1) if "." in name else ("", name))
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child, MoELoRALinear(module, r, lora_alpha, n_experts, dropout_p))
        replaced.append(name)
    return replaced


def freeze_base(model):
    """Freeze all parameters except MoE-LoRA (lora_A, lora_B, router)."""
    for name, p in model.named_parameters():
        p.requires_grad_(any(k in name for k in ("lora_A", "lora_B", "router")))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"MoE trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")
    return model


def save_moe(model, save_dir: str, config: dict):
    """Save only MoE-LoRA trainable weights + config (not the frozen base)."""
    os.makedirs(save_dir, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    save_file(state, os.path.join(save_dir, "moe_weights.safetensors"))
    with open(os.path.join(save_dir, "moe_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {len(state)} MoE tensors → {save_dir}/moe_weights.safetensors")


def load_moe(model, save_dir: str):
    """
    Load MoE weights into a model that has already had MoE layers applied.
    Call setup_moe_lora + freeze_base first, then call this.
    """
    state = load_file(os.path.join(save_dir, "moe_weights.safetensors"))
    param_dict = {n: p for n, p in model.named_parameters()}
    loaded, unexpected = 0, 0
    for k, v in state.items():
        if k in param_dict:
            param_dict[k].data.copy_(v.to(param_dict[k].device))
            loaded += 1
        else:
            unexpected += 1
    if unexpected:
        print(f"Warning: {unexpected} unexpected keys skipped")
    print(f"Loaded {loaded} MoE tensors from {save_dir}")
    return model
