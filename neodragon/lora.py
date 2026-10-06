# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Neodragon-X: minimal, dependency-free LoRA for the Pyramid MMDiT.

Used by `distill/train_stage_a.py` (distilling Wan2.2 into the multistep DiT) and by
`scripts/inference_neodragon.py --lora` (loading the result). There is no `peft` dependency, so the
same code runs on CPU, a Kaggle T4 and any CUDA card.

File format: a `.safetensors` file with tensors `<module path>.lora_A` [r, in] and
`<module path>.lora_B` [out, r]. The metadata holds `rank`, `alpha` and `targets` (comma-separated).
"""
import math
from typing import Dict, Iterable, List

import torch
from torch import nn

# Attention projections of every JointTransformerBlock: image and text streams, q/k/v/out.
DEFAULT_TARGETS = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
)


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x.  B starts at zero, so the wrapped model is unchanged at step 0."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        a = self.lora_A.to(device=x.device, dtype=x.dtype)
        b = self.lora_B.to(device=x.device, dtype=x.dtype)
        return out + (x @ a.t() @ b.t()) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        w = self.base.weight
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scale
        self.base.weight.copy_((w.float() + delta.to(w.device)).to(w.dtype))
        return self.base


def _matches(name: str, targets: Iterable[str]) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def apply_lora(model: nn.Module, rank: int = 32, alpha: float = 32.0,
               targets: Iterable[str] = DEFAULT_TARGETS) -> List[nn.Parameter]:
    """Wrap matching nn.Linear layers in place. Returns the trainable LoRA parameters."""
    targets = tuple(targets)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and _matches(n, targets)]
    if not names:
        raise ValueError(f"no nn.Linear matched LoRA targets {targets}")
    params: List[nn.Parameter] = []
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child) if not child.isdigit() else parent[int(child)]
        wrapped = LoRALinear(base, rank, alpha).to(base.weight.device)
        if child.isdigit():
            parent[int(child)] = wrapped
        else:
            setattr(parent, child, wrapped)
        params += [wrapped.lora_A, wrapped.lora_B]
    model._neodragon_lora = {"rank": rank, "alpha": alpha, "targets": targets}
    return params


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {f"{n}.{k}": getattr(m, k).detach().float().cpu().contiguous()
            for n, m in model.named_modules() if isinstance(m, LoRALinear) for k in ("lora_A", "lora_B")}


def save_lora(model: nn.Module, path: str, extra_meta: Dict[str, str] = None) -> None:
    from safetensors.torch import save_file

    cfg = model._neodragon_lora
    meta = {"rank": str(cfg["rank"]), "alpha": str(cfg["alpha"]), "targets": ",".join(cfg["targets"])}
    meta.update({k: str(v) for k, v in (extra_meta or {}).items()})
    save_file(lora_state_dict(model), path, metadata=meta)


def load_lora(model: nn.Module, path: str, merge: bool = True, strength: float = 1.0) -> nn.Module:
    """Load a LoRA file into `model`. merge=True folds it into the weights (zero inference overhead)."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}
    rank, alpha = int(meta["rank"]), float(meta["alpha"]) * strength
    apply_lora(model, rank=rank, alpha=alpha, targets=tuple(meta["targets"].split(",")))
    missing = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)
               and f"{n}.lora_A" not in state]
    if missing:
        raise ValueError(f"LoRA file {path} is missing {len(missing)} layers, e.g. {missing[0]}")
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            m.lora_A.data.copy_(state[f"{n}.lora_A"])
            m.lora_B.data.copy_(state[f"{n}.lora_B"])
    if merge:
        merge_lora(model)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    for name, m in list(model.named_modules()):
        if isinstance(m, LoRALinear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            if child.isdigit():
                parent[int(child)] = m.merged()
            else:
                setattr(parent, child, m.merged())
    return model
