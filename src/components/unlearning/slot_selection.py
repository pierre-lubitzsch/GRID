"""Top-t Product-Key-Memory slot selection for sparse-memory unlearning.

Adapts Sparse Memory Finetuning (Lin et al., 2025) to unlearning: instead of
updating the whole memory, score every value slot, keep the best ``t``, and mask
the gradient of all the others so only those rows move.

Three scoring families, all computed from the SAME forget/retain passes:

``af``
    Raw access frequency on the forget set -- the access-count baseline (SMF's
    TF term). Cheap: no backward pass needed.
``af_ihf``
    ``AF(s) * log((T_r + 1) / (HF(s) + 1))``. The recommender-side analogue of
    TF-IDF, with the retain split as the "history": a slot scores highly when the
    forget data reads it often and the retain data rarely does. Retain is the
    right history (it is definitionally what must be preserved) and the counts
    are additive, so they can be cached and updated incrementally.
``grad``
    Per-slot gradient criteria on ``values.weight``:

    * magnitude-only ``||g_f|| - lambda * ||g_r||``
    * combined ``gf_hat - lambda * gr_hat - mu * dot_hat`` where
      ``dot_i = <g_f,i , g_r,i>``

    The dot term is the one that matters: the unlearning update moves along
    ``+g_f``, so the first-order change in the RETAIN loss from editing slot
    ``i`` is ``<g_f,i , g_r,i>`` -- not ``||g_r,i||``. It is signed, so a
    negative value means editing for forgetting also *improves* retain, which a
    norm-only score cannot express. Restricted to the selected coordinates the
    objective is additively separable over slots, so exact top-t is plain
    ``topk`` -- no greedy approximation is needed.

NOTE (see WORKFLOW.md section H): on a memory that has COLLAPSED to a handful of
slots, ``af`` and ``af_ihf`` are degenerate -- forget and retain read the same
slots, so IHF is constant and ``af_ihf == af``. The gradient criteria can still
discriminate, because magnitudes differ on shared slots. Selection is only a
real experiment once slot utilisation is healthy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

log = logging.getLogger(__name__)

VALID_CRITERIA = ("af", "af_ihf", "grad", "grad_combined")


def _memories(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    from src.models.components.network_blocks.product_key_memory import (
        HashingMemory,
    )

    return [(n, m) for n, m in model.named_modules() if isinstance(m, HashingMemory)]


def _access_sweep(
    model: nn.Module, mems: Sequence[Tuple[str, nn.Module]], batches: Sequence[Any]
) -> Dict[str, torch.Tensor]:
    """Per-slot read counts over ``batches`` (forward only, no grad)."""
    for _, m in mems:
        m.enable_access_counting()
        m.reset_access_counts()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for b in batches:
            model.model_step(*b)
    out = {n: m.get_access_counts()[0].double() for n, m in mems}
    for _, m in mems:
        m.disable_access_counting()
    if was_training:
        model.train()
    return out


def _grad_sweep(
    model: nn.Module, mems: Sequence[Tuple[str, nn.Module]], batches: Sequence[Any]
) -> Dict[str, torch.Tensor]:
    """Accumulated gradient of the summed loss w.r.t. each memory's values."""
    acc = {n: torch.zeros_like(m.values.weight) for n, m in mems}
    was_training = model.training
    model.eval()
    for b in batches:
        model.zero_grad(set_to_none=True)
        _, loss = model.model_step(*b)
        loss.backward()
        for n, m in mems:
            if m.values.weight.grad is not None:
                acc[n] += m.values.weight.grad.detach()
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return acc


def _maxnorm(v: torch.Tensor) -> torch.Tensor:
    m = v.abs().max()
    return v / m if float(m) > 0 else v


def select_top_t_slots(
    model: nn.Module,
    forget_batches: Sequence[Any],
    retain_batches: Sequence[Any],
    *,
    criterion: str = "grad_combined",
    top_t: int = 32,
    lam: float = 1.0,
    mu: float = 5.0,
    dot_abs: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Return ``(row_masks, info)``.

    ``row_masks[memory_name]`` is a float mask of shape ``(size, 1)`` that is 1
    on the selected slots and 0 elsewhere, ready to multiply into
    ``values.weight.grad``.
    """
    crit = str(criterion).strip().lower()
    if crit not in VALID_CRITERIA:
        raise ValueError(f"criterion must be one of {VALID_CRITERIA}, got {crit!r}")
    mems = _memories(model)
    if not mems:
        raise ValueError(
            "select_top_t_slots requires a PKM-bearing model (no HashingMemory found)"
        )

    need_grad = crit in ("grad", "grad_combined")
    af = _access_sweep(model, mems, forget_batches)
    hf = _access_sweep(model, mems, retain_batches)
    gf = _grad_sweep(model, mems, forget_batches) if need_grad else {}
    gr = _grad_sweep(model, mems, retain_batches) if need_grad else {}

    masks: Dict[str, torch.Tensor] = {}
    info: Dict[str, Any] = {
        "criterion": crit,
        "top_t": int(top_t),
        "lambda": float(lam),
        "mu": float(mu),
        "dot_abs": bool(dot_abs),
        "per_memory": {},
    }
    for name, mem in mems:
        n_slots = int(mem.values.weight.shape[0])
        a, h = af[name], hf[name]
        if crit == "af":
            score = a
        elif crit == "af_ihf":
            score = a * torch.log((float(h.sum().item()) + 1.0) / (h + 1.0))
        else:
            g_f, g_r = gf[name], gr[name]
            gf_n, gr_n = g_f.norm(dim=1).double(), g_r.norm(dim=1).double()
            if crit == "grad":
                score = gf_n - float(lam) * gr_n
            else:
                dot = (g_f * g_r).sum(dim=1).double()
                d = dot.abs() if dot_abs else dot
                score = (
                    _maxnorm(gf_n)
                    - float(lam) * _maxnorm(gr_n)
                    - float(mu) * _maxnorm(d)
                )

        # ELIGIBILITY MASK. A slot the forget data never touches has zero forget
        # gradient, so editing it cannot possibly help — yet it can still WIN the
        # selection: on a collapsed memory 'grad' with lambda=1 scores live slots
        # NEGATIVE (when ||g_r|| > ||g_f||) while dead slots score exactly 0, so
        # topk returns dead slots and the "selection" is a silent no-op.
        # Observed on E23D01/D1/D2 (jobs 10303835/48/49): top-25 under lam=1.0
        # had mean_gf_selected == mean_gr_selected == 0.
        eligible = a > 0
        if need_grad:
            eligible = eligible | (gf[name].norm(dim=1) > 0)
        n_eligible = int(eligible.sum().item())
        if n_eligible == 0:
            raise ValueError(
                f"{name}: no slot has any forget access or forget gradient — "
                "the forget batches never reach this memory."
            )
        score = torch.where(
            eligible, score, torch.full_like(score, float("-inf"))
        )

        k = max(1, min(int(top_t), n_slots))
        if k > n_eligible:
            log.warning(
                "[slot-select] %s: top_t=%d exceeds %d eligible slots — "
                "selecting all eligible ones instead of padding with dead slots",
                name, k, n_eligible,
            )
            k = n_eligible
        idx = torch.topk(score, k).indices
        mask = torch.zeros((n_slots, 1), device=mem.values.weight.device)
        mask[idx] = 1.0
        masks[name] = mask

        # Slots that are LIVE at all (read by either split). On a collapsed
        # memory this is tiny, and selecting t > live is a no-op dressed up as
        # a selection — surface it rather than let it pass silently.
        live = int(((a > 0) | (h > 0)).sum().item())
        info["per_memory"][name] = {
            "n_slots": n_slots,
            "selected": k,
            "live_slots": live,
            "selected_live": int(((a[idx] > 0) | (h[idx] > 0)).sum().item()),
            "selected_retain_unused": int((h[idx] == 0).sum().item()),
            "eligible_slots": n_eligible,
        }
        if k >= live:
            log.warning(
                "[slot-select] %s: top_t=%d >= live slots=%d — the selection is "
                "not restricting anything (memory collapse? see WORKFLOW (H))",
                name, k, live,
            )

    tot = sum(v["selected"] for v in info["per_memory"].values())
    live_tot = sum(v["live_slots"] for v in info["per_memory"].values())
    log.info(
        "[slot-select] criterion=%s top_t=%d -> %d slots selected across %d "
        "memories (%d live slots total)",
        crit, int(top_t), tot, len(mems), live_tot,
    )
    return masks, info
