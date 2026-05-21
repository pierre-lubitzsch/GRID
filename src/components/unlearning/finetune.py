"""Fine-tune baseline: continue training on retain (cleaned) data."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from src.components.unlearning.hvp import batch_size, batch_to_device

log = logging.getLogger(__name__)
TigerBatch = Any


def finetune_unlearn(
    model: nn.Module,
    retain_batches: Sequence[TigerBatch],
    *,
    steps: int = 500,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Fine-tune ``model`` on retain batches for ``steps`` optimizer steps."""
    device = device or next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    model.train()
    losses: List[float] = []
    if not retain_batches:
        raise ValueError("retain_batches is empty")
    for step in range(int(steps)):
        batch = retain_batches[step % len(retain_batches)]
        batch = batch_to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model.model_step(*batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step % max(1, steps // 10) == 0:
            log.info("[finetune] step=%d loss=%.4f", step, losses[-1])
    return {
        "algorithm": "finetune",
        "steps": int(steps),
        "lr": float(lr),
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(sum(losses) / max(1, len(losses))),
        "n_retain_batches": len(retain_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
    }
