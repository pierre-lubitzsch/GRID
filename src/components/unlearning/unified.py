"""Unified unlearning objective: L = L_retain + λ₁ L_forget + λ₂ L_sep."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
from torch import nn

from src.components.unlearning.hvp import batch_size, batch_to_device
from src.components.unlearning.local_repair import apply_local_repair_losses
from src.components.unlearning.target_params import select_adaptive_code_params

log = logging.getLogger(__name__)
TigerBatch = Any


def unified_unlearn(
    model: nn.Module,
    forget_batches: Sequence[TigerBatch],
    retain_batches: Sequence[TigerBatch],
    *,
    steps: Optional[int] = 500,
    n_epochs: Optional[int] = None,
    lr: float = 1e-4,
    lambda_forget: float = 1.0,
    lambda_sep: float = 0.1,
    forget_loss_level: str = "token",
    sep_temperature: float = 0.07,
    deletion_spec: str = "session",
    forget_item_ids: Optional[Set[int]] = None,
    neighbor_item_ids: Optional[Set[int]] = None,
    sep_negative_item_ids: Optional[Set[int]] = None,
    sep_negatives_mode: str = "forget",
    local_repair_cfg: Optional[Dict[str, Any]] = None,
    restrict_adaptive_codes: bool = False,
    stable_codes: int = 2,
    adaptive_update_backbone: bool = False,
    adaptive_adapter: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Optimize unified objective.

    The number of optimizer steps is set either directly via ``steps`` or
    indirectly via ``n_epochs`` (full passes through the batches). With
    balanced accumulation, one pass through the batches equals
    ``min(n_forget_batches, n_retain_batches)`` optimizer steps, so
    ``n_epochs=N`` ⇒ ``steps = N * min(n_forget, n_retain)``.
    If both are given, ``n_epochs`` wins.

    Each optimizer step accumulates gradients across ``q_forget`` forget
    mini-batches and ``q_retain`` retain mini-batches, where

        q_retain = ceil(n_retain / n_forget)
        q_forget = ceil(n_forget / n_retain)

    (one of the two is always 1). This balances per-sample exposure: every
    forget sample and every retain sample contributes to the gradient the same
    number of times, regardless of how many batches each side has.

    The ``L_sep`` negatives default to the forget items ``I_f`` (slide form; no
    neighbors). ``sep_negative_item_ids``, when set, fully replaces them with a
    fixed set — the ``forget_target_only`` mode (just the ``n_target`` spam
    targets) and the random-retain ablation both flow through it.
    ``sep_negatives_mode`` is the originating mode string, recorded for
    metadata only. Local repair still uses ``neighbor_item_ids``.
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

    if n_epochs is not None:
        n_epochs = int(n_epochs)
        if n_epochs <= 0:
            raise ValueError("n_epochs must be > 0")
        steps = n_epochs * optim_steps_per_pass
    else:
        if steps is None:
            raise ValueError("Either steps or n_epochs must be set")
        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be > 0")

    log.info(
        "[unified] n_forget_batches=%d n_retain_batches=%d "
        "→ q_forget=%d q_retain=%d (per optim step: %d forget + %d retain mini-batches); "
        "optim_steps_per_pass=%d, total_steps=%d, n_epochs=%s",
        n_forget,
        n_retain,
        q_forget,
        q_retain,
        q_forget,
        q_retain,
        optim_steps_per_pass,
        steps,
        n_epochs if n_epochs is not None else "(unset)",
    )

    # Stable-Adaptive Semantic IDs: optionally confine the update to the
    # adaptive (fine-grained) code positions. grad_masks holds per-parameter
    # masks applied to .grad each step before opt.step().
    if restrict_adaptive_codes and adaptive_adapter:
        # Option 2 (per-item): freeze the shared table & heads (left out of the
        # optimizer) and train only a per-item, per-adaptive-position offset.
        # The offset gradient is masked to the deletion-relevant items
        # (forget ∪ neighbors), so updates are genuinely item-local.
        if getattr(model, "adaptive_item_offset", None) is None:
            if not hasattr(model, "enable_adaptive_item_offset"):
                raise TypeError(
                    "adaptive_adapter=True requires a model exposing "
                    "enable_adaptive_item_offset (SemanticIDEncoderDecoder)"
                )
            model.enable_adaptive_item_offset(int(stable_codes))
        offset = model.adaptive_item_offset
        params = [offset]
        grad_masks = {}
        adapter_items = set(forget_item_ids or []) | set(neighbor_item_ids or [])
        n_rows = offset.shape[0]
        valid_items = sorted(i for i in adapter_items if 0 <= int(i) < n_rows)
        if valid_items:
            row_mask = torch.zeros((n_rows, 1, 1), device=offset.device)
            row_mask[torch.tensor(valid_items, device=offset.device)] = 1.0
            grad_masks[id(offset)] = row_mask
        log.info(
            "[unified] per-item adaptive adapter ON: stable_codes=%d, "
            "%d / %d offset rows trainable (others frozen via grad mask)",
            int(stable_codes),
            len(valid_items) if valid_items else n_rows,
            n_rows,
        )
    elif restrict_adaptive_codes:
        # Option 1 (shared): move the adaptive shared embedding rows + adaptive
        # heads. grad_masks zeroes the stable embedding rows on .grad.
        params, grad_masks = select_adaptive_code_params(
            model,
            stable_codes=int(stable_codes),
            update_backbone=bool(adaptive_update_backbone),
        )
        log.info(
            "[unified] adaptive-code restriction ON: stable_codes=%d "
            "update_backbone=%s → %d param tensors (%d params), %d masked",
            int(stable_codes),
            bool(adaptive_update_backbone),
            len(params),
            int(sum(p.numel() for p in params)),
            len(grad_masks),
        )
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        grad_masks = {}
    opt = torch.optim.Adam(params, lr=float(lr))
    model.train()

    totals: Dict[str, List[float]] = {
        "total": [],
        "retain": [],
        "forget": [],
        "sep": [],
    }

    forget_ids = set(forget_item_ids or [])
    neighbor_ids = set(neighbor_item_ids or [])  # used only for local-repair losses
    # L_sep negatives (slide form): the forget items I_f only — no neighbors.
    # The random_retain ablation replaces them with random retain item ids.
    if sep_negative_item_ids is not None:
        sep_negatives_set: Set[int] = set(sep_negative_item_ids)
    else:
        sep_negatives_set = forget_ids
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
                negative_item_ids=sep_negatives_set,
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

        # Mask stable-code rows out of the embedding gradient before stepping.
        if grad_masks:
            for p in params:
                m = grad_masks.get(id(p))
                if m is not None and p.grad is not None:
                    p.grad.mul_(m)

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
        "n_epochs": n_epochs,
        "optim_steps_per_pass": optim_steps_per_pass,
        "q_forget": q_forget,
        "q_retain": q_retain,
        "lr": float(lr),
        "restrict_adaptive_codes": bool(restrict_adaptive_codes),
        "stable_codes": int(stable_codes) if restrict_adaptive_codes else None,
        "adaptive_update_backbone": (
            bool(adaptive_update_backbone) if restrict_adaptive_codes else None
        ),
        "adaptive_adapter": bool(restrict_adaptive_codes and adaptive_adapter),
        "lambda_forget": float(lambda_forget),
        "lambda_sep": float(lambda_sep),
        "sep_negatives": str(sep_negatives_mode),
        "n_sep_negatives": len(sep_negatives_set),
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
