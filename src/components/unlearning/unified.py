"""Unified unlearning objective: L = L_retain + λ₁ L_forget + λ₂ L_sep."""

from __future__ import annotations

import logging
import math
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
    steps: Optional[int] = 500,
    n_batch_passes: Optional[int] = None,
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
    """Optimize unified objective.

    The number of optimizer steps is set either directly via ``steps`` or
    indirectly via ``n_batch_passes`` (full passes through the batches). With
    balanced accumulation, one pass through the batches equals
    ``min(n_forget_batches, n_retain_batches)`` optimizer steps, so
    ``n_batch_passes=N`` ⇒ ``steps = N * min(n_forget, n_retain)``.
    If both are given, ``n_batch_passes`` wins.

    Each optimizer step accumulates gradients across ``q_forget`` forget
    mini-batches and ``q_retain`` retain mini-batches, where

        q_retain = ceil(n_retain / n_forget)
        q_forget = ceil(n_forget / n_retain)

    (one of the two is always 1). This balances per-sample exposure: every
    forget sample and every retain sample contributes to the gradient the same
    number of times, regardless of how many batches each side has.
    """
    device = device or next(model.parameters()).device
    if not retain_batches:
        raise ValueError("retain_batches is empty")
    if not forget_batches:
        raise ValueError("forget_batches is empty")
    if not hasattr(model, "compute_sep_loss"):
        raise TypeError(
            "model must expose compute_sep_loss / _batch_loss_from_model_step "
            "(SemanticIDEncoderDecoder subclass)"
        )

    n_forget = len(forget_batches)
    n_retain = len(retain_batches)
    q_retain = max(1, math.ceil(n_retain / n_forget))
    q_forget = max(1, math.ceil(n_forget / n_retain))
    optim_steps_per_pass = min(n_forget, n_retain)

    if n_batch_passes is not None:
        n_batch_passes = int(n_batch_passes)
        if n_batch_passes <= 0:
            raise ValueError("n_batch_passes must be > 0")
        steps = n_batch_passes * optim_steps_per_pass
    else:
        if steps is None:
            raise ValueError("Either steps or n_batch_passes must be set")
        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be > 0")

    log.info(
        "[unified] n_forget_batches=%d n_retain_batches=%d "
        "→ q_forget=%d q_retain=%d (per optim step: %d forget + %d retain mini-batches); "
        "optim_steps_per_pass=%d, total_steps=%d, n_batch_passes=%s",
        n_forget,
        n_retain,
        q_forget,
        q_retain,
        q_forget,
        q_retain,
        optim_steps_per_pass,
        steps,
        n_batch_passes if n_batch_passes is not None else "(unset)",
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
    sequence_forget = str(forget_loss_level).lower() == "sequence"

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # --- Forget side: q_forget mini-batches, each scaled by 1/q_forget ---
        l_forget_avg = 0.0
        for j in range(q_forget):
            idx = (step * q_forget + j) % n_forget
            forget_batch = batch_to_device(forget_batches[idx], device)
            if sequence_forget:
                l_forget = model._sequence_log_prob(*forget_batch)
            else:
                l_forget = -model._batch_loss_from_model_step(forget_batch)
            forget_term = (float(lambda_forget) * l_forget) / float(q_forget)
            forget_term.backward()
            l_forget_avg += float(l_forget.detach().cpu()) / float(q_forget)

        # --- Retain side: q_retain mini-batches, each scaled by 1/q_retain ---
        l_retain_avg = 0.0
        l_sep_avg = 0.0
        last_retain_batch = None
        for j in range(q_retain):
            idx = (step * q_retain + j) % n_retain
            retain_batch = batch_to_device(retain_batches[idx], device)
            last_retain_batch = retain_batch
            l_retain = model._batch_loss_from_model_step(retain_batch)
            l_sep = model.compute_sep_loss(
                retain_batch,
                neighbor_item_ids=neighbor_ids,
                forget_item_ids=forget_ids,
                temperature=float(sep_temperature),
            )
            retain_side = l_retain + float(lambda_sep) * l_sep
            retain_side = apply_local_repair_losses(
                model,
                base_loss=retain_side,
                local_repair_cfg=local_repair_cfg or {},
                neighbor_item_ids=neighbor_ids,
                batch=retain_batch,
            )
            (retain_side / float(q_retain)).backward()
            l_retain_avg += float(l_retain.detach().cpu()) / float(q_retain)
            l_sep_avg += float(l_sep.detach().cpu()) / float(q_retain)

        opt.step()

        total_avg = (
            l_retain_avg
            + float(lambda_forget) * l_forget_avg
            + float(lambda_sep) * l_sep_avg
        )
        totals["total"].append(total_avg)
        totals["retain"].append(l_retain_avg)
        totals["forget"].append(l_forget_avg)
        totals["sep"].append(l_sep_avg)

        if step % max(1, steps // 10) == 0:
            log.info(
                "[unified] step=%d/%d total=%.4f retain=%.4f forget=%.4f sep=%.4f",
                step,
                steps,
                total_avg,
                l_retain_avg,
                l_forget_avg,
                l_sep_avg,
            )

        del last_retain_batch  # free reference

    def _mean(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / max(1, len(xs))) if xs else None

    return {
        "algorithm": "unified",
        "steps": steps,
        "n_batch_passes": n_batch_passes,
        "optim_steps_per_pass": optim_steps_per_pass,
        "q_forget": q_forget,
        "q_retain": q_retain,
        "lr": float(lr),
        "lambda_forget": float(lambda_forget),
        "lambda_sep": float(lambda_sep),
        "forget_loss_level": str(forget_loss_level),
        "deletion_spec": str(deletion_spec),
        "mean_total_loss": _mean(totals["total"]),
        "mean_retain_loss": _mean(totals["retain"]),
        "mean_forget_loss": _mean(totals["forget"]),
        "mean_sep_loss": _mean(totals["sep"]),
        "n_forget_batches": n_forget,
        "n_retain_batches": n_retain,
        "n_forget_rows": sum(batch_size(b) for b in forget_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
    }
