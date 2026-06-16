"""Kookmin unlearning, ported from ``def kookmin`` in
https://github.com/deem-data/erase-bench/blob/main/recbole/trainer/trainer.py
and adapted to TIGER's ``(SequentialModelInputData, SequentialModuleLabelData)``
batches and ``model.model_step(...)`` loss.

Recipe (matches ERASE, restated):

    1. ``grads_forget`` := Σ_{b in D_f}      ∂L_b/∂θ   (averaged by ``neg_grad_sample_size``)
    2. ``grads_retain`` := Σ_{b in D_retain} ∂L_b/∂θ   (averaged by ``neg_grad_sample_size``,
       capped at ~``neg_grad_sample_size`` rows like the reference's ``k_more`` slice)
    3. ``signed_grads`` := grads_retain - grads_forget
    4. For *each* parameter tensor, reinitialise the ``init_rate`` fraction of
       entries with the smallest ``|signed_grad|`` (per-layer threshold, as in
       the original Kookmin paper). These are the slots the retain data cares
       about least and the forget data cares about most.
    5. Reset optimiser state for the touched tensors (we use a *fresh* Adam, so
       its state starts at zero — equivalent to the reference's
       ``_reset_adam_state``).
    6. Retain-repair round: fine-tune on the retain corpus, scaling the
       gradient of reinitialised slots by ``scale_for_reinit_params`` so the
       freshly-randomised weights relearn faster than the rest of the network.

The spam scenario deletes every forget user's data (no "clean-forget" subset),
so the optional clean-forget gradient path is empty by default, matching
``scif.py`` and the rest of the GRID baselines.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from src.components.unlearning.hvp import (
    batch_grad,
    batch_size,
    batch_to_device,
)
from src.components.unlearning.target_params import select_target_params

log = logging.getLogger(__name__)
TigerBatch = Any  # noqa: N816


def _accumulate_grad(
    model: nn.Module,
    batches: Sequence["TigerBatch"],
    params: Sequence[nn.Parameter],
    average_scale: float,
    *,
    max_rows: Optional[int] = None,
) -> List[torch.Tensor]:
    """Sum per-batch gradients (each divided by ``average_scale``).

    When ``max_rows`` is set we stop once that many post-collate rows have been
    consumed, mirroring the reference's ``interaction[:k_more]`` truncation of
    the retain stream to ``neg_grad_retain_sample_size`` rows.
    """
    acc = [torch.zeros_like(p) for p in params]
    rows = 0
    for batch in batches:
        g = batch_grad(model, batch, params, average_scale=average_scale)
        acc = [a + gi for a, gi in zip(acc, g)]
        rows += batch_size(batch)
        if max_rows is not None and rows >= max_rows:
            break
    return acc


def kookmin_unlearn(
    model: nn.Module,
    forget_batches: Sequence["TigerBatch"],
    retain_batches: Sequence["TigerBatch"],
    *,
    init_rate: float = 0.01,
    neg_grad_sample_size: int = 128,
    retain_epochs: int = 1,
    retain_lr: float = 1e-3,
    scale_for_reinit_params: float = 10.0,
    target_params_policy: str = "all",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run Kookmin gradient-guided reinitialisation + retain repair in-place.

    Parameters
    ----------
    model
        TIGER model exposing ``model_step(input, label_data) -> (output, loss)``.
    forget_batches, retain_batches
        Pre-collected TIGER batches already on ``device``.
    init_rate
        Per-layer fraction of weights to reinitialise (ERASE ``kookmin_init_rate``).
    neg_grad_sample_size
        Normaliser for the gradient passes and row cap on the retain gradient
        pass (ERASE ``neg_grad_retain_sample_size``).
    retain_epochs
        Number of passes over ``retain_batches`` in the repair round.
    retain_lr
        Learning rate for the repair Adam optimiser.
    scale_for_reinit_params
        Gradient multiplier applied to reinitialised slots during repair
        (ERASE ``scale_for_reinit_params``, default 10).
    target_params_policy
        Which parameters to consider — see :func:`select_target_params`.
    """
    if not forget_batches:
        raise ValueError("kookmin_unlearn: no forget batches were provided")
    if not retain_batches:
        raise ValueError("kookmin_unlearn: no retain batches were provided")

    device = device or next(model.parameters()).device
    model.train()
    params = select_target_params(model, policy=target_params_policy)
    log.info(
        "[kookmin] policy=%s touches %d tensors / %d params; init_rate=%.4g, "
        "neg_grad_sample_size=%d, retain_epochs=%d, scale_for_reinit=%.3g",
        target_params_policy,
        len(params),
        int(sum(p.numel() for p in params)),
        init_rate,
        neg_grad_sample_size,
        retain_epochs,
        scale_for_reinit_params,
    )

    # --- 1/2. forget & retain gradient passes --------------------------------
    grads_forget = _accumulate_grad(
        model, forget_batches, params, average_scale=float(neg_grad_sample_size)
    )
    grads_retain = _accumulate_grad(
        model,
        retain_batches,
        params,
        average_scale=float(neg_grad_sample_size),
        max_rows=int(neg_grad_sample_size),
    )

    # --- 3. signed gradients -------------------------------------------------
    signed_grads = [gr - gf for gr, gf in zip(grads_retain, grads_forget)]

    # --- 4. per-layer reinitialisation of low-|signed_grad| slots ------------
    reinit_masks: List[Optional[torch.Tensor]] = [None] * len(params)
    total_params_reset = 0
    with torch.no_grad():
        for i, (p, g) in enumerate(zip(params, signed_grads)):
            g_abs = g.abs()
            total_in_layer = g_abs.numel()
            k_in_layer = max(1, int(total_in_layer * init_rate))
            # smallest |signed_grad| positions (negate so topk-largest = smallest)
            _, indices = torch.topk(-g_abs.view(-1), k=k_in_layer, largest=True)
            mask = torch.zeros_like(g_abs, dtype=torch.bool)
            mask.view(-1)[indices] = True
            if not mask.any():
                continue
            total_params_reset += int(mask.sum().item())

            new_p = torch.empty_like(p.data)
            if p.dim() == 4:  # conv-like
                nn.init.kaiming_normal_(new_p, mode="fan_out", nonlinearity="relu")
            elif p.dim() == 2:  # linear / projection weights
                nn.init.kaiming_uniform_(new_p, a=math.sqrt(5))
            else:  # embeddings, biases, layernorm, ...
                new_p.normal_(0.0, 0.02)
            p.data[mask] = new_p[mask]
            reinit_masks[i] = mask

    total_params = int(sum(p.numel() for p in params))
    n_layers_reset = sum(1 for m in reinit_masks if m is not None)
    log.info(
        "[kookmin] reset %d/%d params (%.4f%%) across %d/%d tensors",
        total_params_reset,
        total_params,
        100.0 * total_params_reset / max(1, total_params),
        n_layers_reset,
        len(params),
    )

    # --- 5/6. retain-repair round with scaled grads on reinit slots ----------
    # A fresh optimiser starts with zero state, equivalent to ERASE's
    # `_reset_adam_state` on the reinitialised tensors.
    opt = torch.optim.Adam(params, lr=float(retain_lr))
    repair_losses: List[float] = []
    for epoch in range(int(retain_epochs)):
        for batch in retain_batches:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            _, loss = model.model_step(*batch)
            loss.backward()
            if scale_for_reinit_params != 1.0:
                for p, mask in zip(params, reinit_masks):
                    if mask is not None and p.grad is not None:
                        p.grad[mask] *= float(scale_for_reinit_params)
            opt.step()
            repair_losses.append(float(loss.detach().cpu()))
        if repair_losses:
            log.info(
                "[kookmin] repair epoch %d/%d mean_loss=%.4f",
                epoch + 1,
                int(retain_epochs),
                sum(repair_losses[-len(retain_batches):])
                / max(1, len(retain_batches)),
            )

    return {
        "algorithm": "kookmin",
        "init_rate": float(init_rate),
        "neg_grad_sample_size": int(neg_grad_sample_size),
        "retain_epochs": int(retain_epochs),
        "retain_lr": float(retain_lr),
        "scale_for_reinit_params": float(scale_for_reinit_params),
        "target_params_policy": target_params_policy,
        "n_param_tensors": len(params),
        "n_param_tensors_reset": n_layers_reset,
        "n_params_total": total_params,
        "n_params_reset": int(total_params_reset),
        "frac_params_reset": float(total_params_reset) / max(1, total_params),
        "mean_repair_loss": (
            float(sum(repair_losses) / len(repair_losses)) if repair_losses else None
        ),
        "final_repair_loss": repair_losses[-1] if repair_losses else None,
        "n_forget_batches": len(forget_batches),
        "n_retain_batches": len(retain_batches),
        "n_forget_rows": sum(batch_size(b) for b in forget_batches),
        "n_retain_rows": sum(batch_size(b) for b in retain_batches),
    }
