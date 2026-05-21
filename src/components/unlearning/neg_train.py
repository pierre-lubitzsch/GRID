"""Negative training baseline: gradient ascent on forget set."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from src.components.unlearning.hvp import batch_size, batch_to_device

log = logging.getLogger(__name__)
TigerBatch = Any


def neg_train_unlearn(
    model: nn.Module,
    forget_batches: Sequence[TigerBatch],
    retain_batches: Sequence[TigerBatch],
    *,
    steps: int = 200,
    lr: float = 1e-3,
    neg_retain_every: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Gradient ascent on forget batches with optional retain CE every k steps."""
    device = device or next(model.parameters()).device
    if not forget_batches:
        raise ValueError("forget_batches is empty")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    model.train()
    forget_losses: List[float] = []
    retain_losses: List[float] = []
    for step in range(int(steps)):
        if neg_retain_every > 0 and step % int(neg_retain_every) == 0 and retain_batches:
            batch = retain_batches[step % len(retain_batches)]
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model.model_step(*batch)
            loss.backward()
            opt.step()
            retain_losses.append(float(loss.detach().cpu()))
        else:
            batch = forget_batches[step % len(forget_batches)]
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model.model_step(*batch)
            (-loss).backward()
            opt.step()
            forget_losses.append(float(loss.detach().cpu()))
        if step % max(1, steps // 10) == 0:
            log.info("[neg_train] step=%d", step)
    return {
        "algorithm": "neg_train",
        "steps": int(steps),
        "lr": float(lr),
        "neg_retain_every": int(neg_retain_every),
        "mean_forget_loss": (
            float(sum(forget_losses) / max(1, len(forget_losses)))
            if forget_losses
            else None
        ),
        "mean_retain_loss": (
            float(sum(retain_losses) / max(1, len(retain_losses)))
            if retain_losses
            else None
        ),
        "n_forget_batches": len(forget_batches),
        "n_retain_batches": len(retain_batches),
    }
