"""Unified unlearning objective: L = L_retain + λ₁ L_forget + λ₂ L_sep."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
from torch import nn

from src.components.unlearning.hvp import batch_size, batch_to_device
from src.components.unlearning.local_repair import apply_local_repair_losses

log = logging.getLogger(__name__)
TigerBatch = Any


def unified_unlearn(
    model: nn.Module,
    forget_batches: Sequence[TigerBatch],
    retain_batches: Sequence[TigerBatch],
    *,
    steps: int = 500,
    lr: float = 1e-4,
    lambda_forget: float = 1.0,
    lambda_sep: float = 0.1,
    forget_loss_level: str = "token",
    sep_temperature: float = 0.07,
    deletion_spec: str = "session",
    forget_item_ids: Optional[Set[int]] = None,
    neighbor_item_ids: Optional[Set[int]] = None,
    local_repair_cfg: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Optimize unified objective for ``steps`` mini-batch updates."""
    device = device or next(model.parameters()).device
    if not retain_batches:
        raise ValueError("retain_batches is empty")
    if not forget_batches:
        raise ValueError("forget_batches is empty")
    if not hasattr(model, "compute_unified_loss"):
        raise TypeError(
            "model must implement compute_unified_loss (SemanticIDEncoderDecoder subclass)"
        )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    model.train()

    totals: Dict[str, List[float]] = {
        "total": [],
        "retain": [],
        "forget": [],
        "sep": [],
    }

    forget_ids = set(forget_item_ids or [])
    neighbor_ids = set(neighbor_item_ids or [])

    for step in range(int(steps)):
        retain_batch = batch_to_device(
            retain_batches[step % len(retain_batches)], device
        )
        forget_batch = batch_to_device(
            forget_batches[step % len(forget_batches)], device
        )
        opt.zero_grad(set_to_none=True)
        losses = model.compute_unified_loss(
            retain_batch=retain_batch,
            forget_batch=forget_batch,
            lambda_forget=float(lambda_forget),
            lambda_sep=float(lambda_sep),
            forget_loss_level=str(forget_loss_level),
            sep_temperature=float(sep_temperature),
            deletion_spec=str(deletion_spec),
            forget_item_ids=forget_ids,
            neighbor_item_ids=neighbor_ids,
        )
        total = losses["total"]
        total = apply_local_repair_losses(
            model,
            base_loss=total,
            local_repair_cfg=local_repair_cfg or {},
            neighbor_item_ids=neighbor_ids,
            batch=retain_batch,
        )
        total.backward()
        opt.step()
        for k in totals:
            if k in losses:
                totals[k].append(float(losses[k].detach().cpu()))
        totals["total"][-1] = float(total.detach().cpu())
        if step % max(1, steps // 10) == 0:
            log.info(
                "[unified] step=%d total=%.4f retain=%.4f forget=%.4f sep=%.4f",
                step,
                totals["total"][-1],
                totals["retain"][-1] if totals["retain"] else 0.0,
                totals["forget"][-1] if totals["forget"] else 0.0,
                totals["sep"][-1] if totals["sep"] else 0.0,
            )

    def _mean(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / max(1, len(xs))) if xs else None

    return {
        "algorithm": "unified",
        "steps": int(steps),
        "lr": float(lr),
        "lambda_forget": float(lambda_forget),
        "lambda_sep": float(lambda_sep),
        "forget_loss_level": str(forget_loss_level),
        "deletion_spec": str(deletion_spec),
        "mean_total_loss": _mean(totals["total"]),
        "mean_retain_loss": _mean(totals["retain"]),
        "mean_forget_loss": _mean(totals["forget"]),
        "mean_sep_loss": _mean(totals["sep"]),
        "n_forget_batches": len(forget_batches),
        "n_retain_batches": len(retain_batches),
        "n_forget_rows": sum(batch_size(b) for b in forget_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
    }
