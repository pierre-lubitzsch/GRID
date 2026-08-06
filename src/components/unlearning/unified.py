"""Unified unlearning objective: L = L_retain + λ₁ L_forget + λ₂ L_sep."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
from torch import nn

from src.components.unlearning.hvp import batch_size, batch_to_device
from src.components.unlearning.local_repair import apply_local_repair_losses
from src.components.unlearning.slot_selection import select_top_t_slots
from src.components.unlearning.target_params import (
    select_adaptive_code_params,
    select_code_position_params,
    select_pkm_params,
)

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
    lambda_neighborhood: float = 0.0,
    coherence_neighbors: Optional[Sequence[Any]] = None,
    coherence_loss_type: str = "nll",
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
    update_positions: Optional[List[int]] = None,
    update_positions_backbone: bool = False,
    update_scope: str = "all",
    pkm_update_keys: bool = True,
    pkm_update_query: bool = True,
    slot_selection: str = "none",
    slot_top_t: int = 32,
    slot_lambda: float = 1.0,
    slot_mu: float = 5.0,
    slot_dot_abs: bool = False,
    optimizer: str = "adam",
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

    When ``lambda_neighborhood > 0``, the coherence term ``L_n`` is added to the
    forget side of each step: for each eligible forget sample the model is scored
    (teacher-forced) on the semantic-id codes of its target's prefix-neighbours,
    conditioned on the forget history, and this negative log-probability is
    minimised so suppressed mass flows to coherent neighbours.
    ``coherence_neighbors`` is a per-forget-batch sequence aligned to
    ``forget_batches``; each element is ``(neighbor_sids[B, C, H],
    neighbor_mask[B, C])`` (or ``None`` when a batch has no eligible neighbours)
    as produced by the caller from the codebook. ``coherence_loss_type`` selects
    the ``nll`` (TRACER Eq. 9, per-neighbour, infeasible optimum) or ``mass``
    (logsumexp over the neighbourhood, bounded and satisfiable) form — see
    ``compute_coherence_loss``.
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
    slot_select_info = None
    if str(update_scope).lower() == "pkm_only":
        # "Modular stabilizer" scope: only the Product-Key Memory is in the
        # optimizer; backbone, SID embeddings and decoder heads stay frozen, so
        # any forgetting has to be expressible as an edit to sparse memory.
        # Takes precedence over the position/adaptive restrictions below.
        params, pkm_names = select_pkm_params(
            model,
            include_keys=bool(pkm_update_keys),
            include_query=bool(pkm_update_query),
        )
        grad_masks = {}
        log.info(
            "[unified] PKM-ONLY update scope: %d param tensors (%d params) "
            "across %d memory modules (keys=%s query=%s); backbone FROZEN",
            len(params),
            int(sum(p.numel() for p in params)),
            len({n.rsplit(".", 1)[0] for n in pkm_names}),
            bool(pkm_update_keys),
            bool(pkm_update_query),
        )
        # Optional top-t slot restriction on top of pkm_only: score every value
        # slot from the forget/retain data and mask the gradient of all but the
        # best t, so only those memory rows move.
        if str(slot_selection).strip().lower() not in ("none", "", "off"):
            masks, sel_info = select_top_t_slots(
                model,
                forget_batches,
                retain_batches,
                criterion=str(slot_selection),
                top_t=int(slot_top_t),
                lam=float(slot_lambda),
                mu=float(slot_mu),
                dot_abs=bool(slot_dot_abs),
            )
            name_to_mod = dict(model.named_modules())
            n_masked = 0
            for mem_name, mask in masks.items():
                w = name_to_mod[mem_name].values.weight
                grad_masks[id(w)] = mask
                n_masked += 1
            slot_select_info = sel_info
            log.info(
                "[unified] slot selection=%s top_t=%d -> gradient masks on %d "
                "value tables",
                str(slot_selection), int(slot_top_t), n_masked,
            )
    elif update_positions:
        # Position-wise intervention (same knob as SCIF's
        # unlearning.update_positions, generalizing adaptive_codes to ANY
        # subset of code positions — e.g. [0] = only the coarsest code c1
        # moves, all other positions + backbone frozen). Takes precedence over
        # the adaptive_codes prefix modes.
        params, grad_masks = select_code_position_params(
            model,
            positions=list(update_positions),
            update_backbone=bool(update_positions_backbone),
        )
        log.info(
            "[unified] position-wise restriction ON: update_positions=%s "
            "update_backbone=%s → %d param tensors (%d params), %d masked",
            list(update_positions),
            bool(update_positions_backbone),
            len(params),
            int(sum(p.numel() for p in params)),
            len(grad_masks),
        )
    elif restrict_adaptive_codes and adaptive_adapter:
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
    # SGD is the Sparse Memory Finetuning choice: with a sparse/selected update
    # Adam's moment estimates get diluted on the steps where a slot receives no
    # gradient, distorting its effective step size.
    if str(optimizer).lower() == "sgd":
        opt = torch.optim.SGD(params, lr=float(lr), momentum=0.9)
    elif str(optimizer).lower() == "adam":
        opt = torch.optim.Adam(params, lr=float(lr))
    else:
        raise ValueError(f"optimizer must be adam|sgd, got {optimizer!r}")
    model.train()

    totals: Dict[str, List[float]] = {
        "total": [],
        "retain": [],
        "forget": [],
        "sep": [],
        "coh": [],
    }

    use_coherence = float(lambda_neighborhood) != 0.0 and coherence_neighbors is not None
    n_coherence_batches = (
        sum(1 for cn in coherence_neighbors if cn is not None)
        if coherence_neighbors is not None
        else 0
    )

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
        l_coh_avg = 0.0
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

            # --- Coherence term L_n (TRACER Eq. 9), on the same forget batch ---
            if use_coherence:
                cn = coherence_neighbors[idx]
                if cn is not None:
                    neighbor_sids, neighbor_mask = cn
                    l_coh = model.compute_coherence_loss(
                        forget_batch,
                        neighbor_sids,
                        neighbor_mask,
                        loss_type=coherence_loss_type,
                    )
                    if l_coh.requires_grad:
                        coh_term = (float(lambda_neighborhood) * l_coh) / float(q_forget)
                        coh_term.backward()
                    l_coh_avg += float(l_coh.detach().cpu()) / float(q_forget)

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
            + float(lambda_neighborhood) * l_coh_avg
        )
        totals["total"].append(total_avg)
        totals["retain"].append(l_retain_avg)
        totals["forget"].append(l_forget_avg)
        totals["sep"].append(l_sep_avg)
        totals["coh"].append(l_coh_avg)

        if step % max(1, steps // 10) == 0:
            log.info(
                "[unified] step=%d/%d total=%.4f retain=%.4f forget=%.4f "
                "sep=%.4f coh=%.4f",
                step,
                steps,
                total_avg,
                l_retain_avg,
                l_forget_avg,
                l_sep_avg,
                l_coh_avg,
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
        "slot_selection": slot_select_info,
        "update_positions": list(update_positions) if update_positions else None,
        "update_positions_backbone": (
            bool(update_positions_backbone) if update_positions else None
        ),
        "stable_codes": int(stable_codes) if restrict_adaptive_codes else None,
        "adaptive_update_backbone": (
            bool(adaptive_update_backbone) if restrict_adaptive_codes else None
        ),
        "adaptive_adapter": bool(restrict_adaptive_codes and adaptive_adapter),
        "lambda_forget": float(lambda_forget),
        "lambda_sep": float(lambda_sep),
        "lambda_neighborhood": float(lambda_neighborhood),
        "coherence_enabled": bool(use_coherence),
        "coherence_loss_type": str(coherence_loss_type) if use_coherence else None,
        "n_coherence_batches": int(n_coherence_batches),
        "sep_negatives": str(sep_negatives_mode),
        "n_sep_negatives": len(sep_negatives_set),
        "forget_loss_level": str(forget_loss_level),
        "deletion_spec": str(deletion_spec),
        "mean_total_loss": _mean(totals["total"]),
        "mean_retain_loss": _mean(totals["retain"]),
        "mean_forget_loss": _mean(totals["forget"]),
        "mean_sep_loss": _mean(totals["sep"]),
        "mean_coh_loss": _mean(totals["coh"]),
        "n_forget_batches": n_forget,
        "n_retain_batches": n_retain,
        "n_forget_rows": sum(batch_size(b) for b in forget_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
    }
