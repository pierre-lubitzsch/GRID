"""TRACER token reassignment -- the tokenizer side (arXiv:2606.07688).

TRACER unlearns a concept by *reassigning* its items to different codewords
rather than by suppressing their logits. The reassignment is made differentiable
with a per-item, per-level, per-codeword score ``phi`` that perturbs the
quantizer's distances:

    (Eq. 5)  q(s_i^l = k)      = softmax_k( -||r_i^l - c_k^l||^2 / tau )
    (Eq. 6)  q_phi(s_i^l = k)  = softmax_k( (-||r_i^l - c_k^l||^2 + phi_{i,k}^l) / tau )
             L_reg             = sum_{i,l,k} |phi_{i,k}^l|

with soft token embedding  e~_i^l = sum_k q_phi(s_i^l = k) e_k^l .

This module owns everything that depends on the *quantizer* (residuals r_i^l and
codewords c_k^l); the training loop and the loss terms live in ``tracer.py``.

WHY THE RESIDUAL RECIPE MATTERS. ``phi = 0`` must reproduce each item's stored
semantic id exactly -- otherwise TRACER silently reassigns part of the catalog
before any unlearning happens. The repo's RQ-KMeans normalizes both the input
and every residual (``residual_quantization.normalize_inputs`` /
``normalize_residuals``, both default True), so the assignment recursion is

    r_i^1     = normalize(z_i)
    s_i^l     = argmin_k ||r_i^l - c_k^l||^2
    r_i^(l+1) = normalize(r_i^l - c_{s_i^l}^l)

Verified on beauty w16: this reproduces 100.00% of stored codes at every level,
while dropping either normalization drops it to 31-84%. ``assert_reproduces_sids``
pins that, and callers should run it before trusting a codebook.

NOTE ON AVAILABILITY. The centroids come from the RQ-KMeans training checkpoint.
The original width-256 / L=4 beauty codebook checkpoint no longer exists, so
faithful TRACER cannot run on those models -- centroids cannot be recovered from
(z, codes) alone because the quantizer applies a learned normalization/encoder
before matching (best-effort reconstruction tops out at ~95/89/86% per level).
Use an identifier space whose codebook checkpoint survives (w16, w8l6, L8, L16).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def load_rq_centroids(
    ckpt_path: str,
    n_levels: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Load ``[K, D]`` codeword tensors from an RQ-KMeans training checkpoint.

    Keys look like ``quantization_layer_list.<l>.centroids``. Returns them in
    level order. ``n_levels`` truncates (the semantic levels are ``H - 1``; the
    final id digit is a dedup counter with no codebook).
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj)
    prefix, suffix = "quantization_layer_list.", ".centroids"
    idx = sorted(
        int(k[len(prefix) : -len(suffix)])
        for k in state
        if k.startswith(prefix) and k.endswith(suffix)
    )
    if not idx:
        raise ValueError(f"no '{prefix}*{suffix}' entries in {ckpt_path}")
    if n_levels is not None:
        idx = idx[:n_levels]
    cents = [state[f"{prefix}{i}{suffix}"].float() for i in idx]
    if device is not None:
        cents = [c.to(device) for c in cents]
    log.info(
        "[tracer] loaded %d codebook levels from %s (K=%d, D=%d)",
        len(cents),
        ckpt_path,
        cents[0].shape[0],
        cents[0].shape[1],
    )
    return cents


def compute_residuals(
    z: torch.Tensor,
    centroids: Sequence[torch.Tensor],
    codes: torch.Tensor,
) -> List[torch.Tensor]:
    """Per-level residuals ``r_i^l`` under the stored assignment.

    ``z`` is ``[N, D]`` (pre-quantization item embeddings), ``codes`` is
    ``[L, N]`` (or ``[H, N]``; only the first ``len(centroids)`` rows are used).
    Returns ``L`` tensors of shape ``[N, D]``.

    Follows the normalize-in / normalize-residual recursion documented above --
    the residual fed to level ``l`` is what ``phi`` perturbs, so getting this
    wrong silently changes every distance in Eq. 6.
    """
    # float64 throughout: the recursion compounds, and the entrypoints set
    # matmul precision to "medium" (see assignment_logits). Residuals are frozen
    # buffers, so the extra precision costs nothing at training time.
    r = F.normalize(z.double(), dim=-1)
    out: List[torch.Tensor] = []
    for lvl, c in enumerate(centroids):
        out.append(r)
        s = codes[lvl].long().to(r.device)
        r = F.normalize(r - c.to(r.device).double()[s], dim=-1)
    return out


def assignment_logits(
    residual: torch.Tensor,
    centroids: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``-||r_i - c_k||^2 (+ phi_{i,k})`` -- the pre-temperature scores of Eq. 6.

    ``residual`` ``[N, D]``, ``centroids`` ``[K, D]``, ``phi`` ``[N, K]`` or None.
    Returns ``[N, K]``. Temperature is applied by the caller so the same scores
    can drive both the soft assignment and the hard argmax.

    PRECISION. The distance is computed in float64. ``cdist`` is matmul-backed,
    and the unlearning entrypoints call
    ``torch.set_float32_matmul_precision("medium")`` (``src/unlearn_sequential.py:43``),
    which drops enough mantissa that the argmin flips for items whose two nearest
    codewords are near-tied -- measured at 1.41% of the beauty w16 catalog, i.e.
    ``phi=0`` reproduced only 98.59% of stored codes instead of 100%. Since the
    whole method is "perturb an existing assignment", that is a silent
    reassignment of ~170 items before any unlearning happens.
    """
    d2 = torch.cdist(residual.double(), centroids.double()).pow(2)
    scores = (-d2).to(residual.dtype)
    if phi is not None:
        scores = scores + phi
    return scores


def soft_assignment(
    residual: torch.Tensor,
    centroids: torch.Tensor,
    phi: Optional[torch.Tensor],
    tau: float,
) -> torch.Tensor:
    """``q_phi(s_i = k)`` of Eq. 6 -- a ``[N, K]`` simplex per item."""
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    return torch.softmax(assignment_logits(residual, centroids, phi) / float(tau), dim=-1)


def hard_assignment(
    residual: torch.Tensor,
    centroids: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inference-time ``argmax_k q_phi`` -- ``[N]`` codes.

    Temperature-free: softmax is monotone, so the argmax of ``q_phi`` is the
    argmax of the scores for any ``tau > 0``.
    """
    return assignment_logits(residual, centroids, phi).argmax(dim=-1)


def phi_regularizer(phis: Sequence[torch.Tensor]) -> torch.Tensor:
    """``L_reg = sum_{i,l,k} |phi_{i,k}^l|`` (Eq. 6)."""
    return sum(p.abs().sum() for p in phis)


def retain_code_usage(
    codes: torch.Tensor,
    retain_item_ids: torch.Tensor,
    n_codes: int,
    level: int,
) -> torch.Tensor:
    """``rho_l(k)`` of Eq. 11: the fraction of RETAIN items using code ``k``.

    Returns ``[K]`` summing to 1 over used codes.
    """
    s = codes[level].long()[retain_item_ids.long()]
    counts = torch.bincount(s, minlength=n_codes).float()
    return counts / counts.sum().clamp(min=1.0)


def selective_update_mask(
    q_phi: torch.Tensor,
    rho: torch.Tensor,
    grad_phi_forget: torch.Tensor,
) -> torch.Tensor:
    """The selective-update mask ``M_{i,k}^l`` of Eq. 11.

        M = 1[ rho_l(k) > rho_bar_{i,l} ] * 1[ grad_{phi} L_F > 0 ]

    where ``rho_bar_{i,l} = E_{q_phi} rho_l(k)`` is the usage the item currently
    expects under its own soft assignment. So phi only moves on codewords that
    are MORE shared with the retain set than the item's present assignment is --
    which is TRACER's point: reassign away from entangled tokens, and leave
    everything else alone.

    ``q_phi`` ``[N, K]``, ``rho`` ``[K]``, ``grad_phi_forget`` ``[N, K]``.
    """
    rho_bar = (q_phi * rho.unsqueeze(0)).sum(dim=-1, keepdim=True)   # [N, 1]
    overlap = (rho.unsqueeze(0) > rho_bar)                            # [N, K]
    conflicting = grad_phi_forget > 0
    return (overlap & conflicting).to(q_phi.dtype)


def assert_reproduces_sids(
    z: torch.Tensor,
    centroids: Sequence[torch.Tensor],
    codes: torch.Tensor,
    tol: float = 1.0,
) -> List[float]:
    """Check that ``phi = 0`` reproduces the stored semantic ids.

    This is the correctness anchor for the whole method: TRACER perturbs an
    assignment, so if the unperturbed assignment already disagrees with the
    codes the model was trained on, every downstream number is meaningless.

    Returns the per-level agreement fraction; raises if any level is below
    ``tol`` (default 1.0 == exact).
    """
    residuals = compute_residuals(z, centroids, codes)
    agree: List[float] = []
    for lvl, (r, c) in enumerate(zip(residuals, centroids)):
        pred = hard_assignment(r, c.to(r.device))
        a = (pred == codes[lvl].long().to(pred.device)).float().mean().item()
        agree.append(a)
        log.info("[tracer] level %d: phi=0 reproduces %.2f%% of stored codes", lvl, a * 100)
    worst = min(agree)
    if worst < tol:
        raise ValueError(
            f"phi=0 reproduces only {worst * 100:.2f}% of stored codes "
            f"(per level: {[round(a * 100, 2) for a in agree]}). The centroids do "
            "not match this SID tensor, so TRACER would reassign items before any "
            "unlearning. Check that the RQ-KMeans checkpoint is the one this "
            "semantic_id_path was generated from."
        )
    return agree
