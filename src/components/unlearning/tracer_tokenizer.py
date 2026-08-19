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

TWO QUANTIZERS, TWO RECIPES. ``load_rq_quantizer`` reads the codebooks AND the
residual recipe off the checkpoint, auto-detecting which quantizer produced it:
RQ-KMeans matches in the raw 2048-d space with both normalizations on, RQ-VAE
matches in its 64-d encoder latent with ``normalize_residuals: false``. Never
pick those flags by hand -- pass the returned :class:`RQFrontEnd` through.

NOTE ON AVAILABILITY. The centroids come from the quantizer's training
checkpoint. The original width-256 / L=4 beauty RQ-KMeans checkpoint no longer
exists, so faithful TRACER cannot run on the models built from it -- centroids
cannot be recovered from (z, codes) alone (best-effort reconstruction tops out at
~95/89/86% per level). Use an identifier space whose codebook checkpoint
survives: any RQ-VAE space (``embeddings/<ds>_rqvae``, checkpoints under
``logs/train/runs/codebook/<ds>_rqvae_<jobid>/``), or w16 / w8l6 / L8 / L16.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class RQFrontEnd:
    """Everything the residual recursion needs from a quantizer checkpoint.

    The two quantizers in this repo share ``ResidualQuantization`` but sit in
    DIFFERENT spaces, and the residual recipe differs with them:

    * ``rkmeans`` -- codebooks in the raw 2048-d embedding space, no encoder,
      ``normalize_inputs`` and ``normalize_residuals`` both on.
    * ``rqvae``  -- codebooks in the 64-d ENCODER LATENT space, reached through
      ``normalization_layer`` (BatchNorm1d + L2 normalize) then ``encoder``
      (MLP 2048-768-256-128-64), and ``normalize_residuals: false``.

    Getting this wrong does not raise: it silently changes every distance in
    Eq. 6, so ``phi=0`` stops reproducing the stored ids and TRACER reassigns
    part of the catalog before any unlearning happens. That is what
    :func:`assert_reproduces_sids` exists to catch.
    """

    centroids: List[torch.Tensor]
    quantizer: str
    normalize_inputs: bool = True
    normalize_residuals: bool = True
    project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


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
    # The middle segment must be JUST the level index. RQ-VAE checkpoints also
    # carry 'quantization_layer_list.<l>.initializer.clustering_module.centroids'
    # (the k-means seeding state, not the live codebook), which matches the same
    # prefix/suffix pair and would otherwise crash the int() or, worse, be picked
    # up as the codebook.
    idx = sorted(
        int(mid)
        for k in state
        if k.startswith(prefix)
        and k.endswith(suffix)
        and (mid := k[len(prefix) : -len(suffix)]).isdigit()
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


def _has_frontend(state: dict) -> bool:
    """True when the checkpoint carries a learned normalization/encoder front end.

    RQ-KMeans leaves both as ``nn.Identity`` (no parameters), so the absence of
    these keys is what distinguishes the two quantizers without having to trust
    a config flag.
    """
    return any(
        k.startswith("normalization_layer.") or k.startswith("encoder.") for k in state
    )


def load_rq_quantizer(
    ckpt_path: str,
    n_levels: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> RQFrontEnd:
    """Load the codebooks AND the residual recipe from a quantizer checkpoint.

    Auto-detects rkmeans vs rqvae from the checkpoint contents rather than from a
    caller-supplied flag. For rqvae the ``normalization_layer`` + ``encoder``
    modules are rebuilt by instantiating the run's own ``.hydra/config.yaml``
    (``ResidualQuantization.save_hyperparameters`` ignores both modules, so they
    cannot be recovered from ``hyper_parameters`` alone) and loading the
    checkpoint weights into them.

    The front end runs under ``eval()``: its BatchNorm1d must use the running
    statistics fitted on the frozen catalog, exactly as the quantizer's own
    ``assign()`` did. In train mode it would use batch statistics and produce a
    different latent for every call.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj)
    centroids = load_rq_centroids(ckpt_path, n_levels=n_levels, device=device)

    if not _has_frontend(state):
        log.info("[tracer] quantizer=rkmeans (no encoder in %s)", ckpt_path)
        return RQFrontEnd(
            centroids=centroids,
            quantizer="rkmeans",
            normalize_inputs=True,
            normalize_residuals=True,
            project=None,
        )

    import hydra
    from omegaconf import OmegaConf

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        raise ValueError(
            f"{ckpt_path} has an encoder front end (quantizer=rqvae) but its "
            f"Hydra config is missing at {cfg_path}. The encoder cannot be "
            "rebuilt from the checkpoint alone, because ResidualQuantization "
            "excludes normalization_layer/encoder from save_hyperparameters. "
            "Point tracer at a codebook run that kept its .hydra/ directory."
        )
    model_cfg = OmegaConf.load(cfg_path).model
    module = hydra.utils.instantiate(model_cfg)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if any(k.startswith(("normalization_layer.", "encoder.")) for k in missing):
        raise ValueError(
            f"{ckpt_path} is missing front-end weights after instantiate: "
            f"{[k for k in missing if k.startswith(('normalization_layer.', 'encoder.'))][:5]}"
        )
    module.eval()
    norm_layer, encoder = module.normalization_layer, module.encoder
    if device is not None:
        norm_layer = norm_layer.to(device)
        encoder = encoder.to(device)

    def project(x: torch.Tensor) -> torch.Tensor:
        # float32, matching the precision the quantizer itself assigned in; the
        # caller casts to float64 for the residual recursion afterwards.
        with torch.no_grad():
            p = next(encoder.parameters())
            return encoder(norm_layer(x.to(device=p.device, dtype=p.dtype)))

    normalize_residuals = bool(
        obj.get("hyper_parameters", {}).get("normalize_residuals", False)
    )
    log.info(
        "[tracer] quantizer=rqvae (encoder front end from %s, "
        "normalize_residuals=%s, codebook dim=%d)",
        cfg_path,
        normalize_residuals,
        centroids[0].shape[1],
    )
    return RQFrontEnd(
        centroids=centroids,
        quantizer="rqvae",
        # normalization_layer already ends in NormalizeLayer, so an extra
        # F.normalize on the latent would be a second, different projection.
        normalize_inputs=False,
        normalize_residuals=normalize_residuals,
        project=project,
    )


def compute_residuals(
    z: torch.Tensor,
    centroids: Sequence[torch.Tensor],
    codes: torch.Tensor,
    project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    normalize_inputs: bool = True,
    normalize_residuals: bool = True,
) -> List[torch.Tensor]:
    """Per-level residuals ``r_i^l`` under the stored assignment.

    ``z`` is ``[N, D]`` (pre-quantization item embeddings), ``codes`` is
    ``[L, N]`` (or ``[H, N]``; only the first ``len(centroids)`` rows are used).
    Returns ``L`` tensors of shape ``[N, D]``.

    The defaults are the RQ-KMeans recipe (normalize the input, normalize every
    residual). ``project`` maps the raw embeddings into the codebook's own space
    first, which is what RQ-VAE needs; use :func:`load_rq_quantizer` to get the
    three settings from the checkpoint instead of choosing them by hand. Getting
    them wrong silently changes every distance in Eq. 6.
    """
    if project is not None:
        z = project(z)
    # float64 throughout: the recursion compounds, and the entrypoints set
    # matmul precision to "medium" (see assignment_logits). Residuals are frozen
    # buffers, so the extra precision costs nothing at training time.
    r = z.double()
    if normalize_inputs:
        r = F.normalize(r, dim=-1)
    out: List[torch.Tensor] = []
    for lvl, c in enumerate(centroids):
        out.append(r)
        s = codes[lvl].long().to(r.device)
        r = r - c.to(r.device).double()[s]
        if normalize_residuals:
            r = F.normalize(r, dim=-1)
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
    project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    normalize_inputs: bool = True,
    normalize_residuals: bool = True,
) -> List[float]:
    """Check that ``phi = 0`` reproduces the stored semantic ids.

    This is the correctness anchor for the whole method: TRACER perturbs an
    assignment, so if the unperturbed assignment already disagrees with the
    codes the model was trained on, every downstream number is meaningless.

    Returns the per-level agreement fraction; raises if any level is below
    ``tol`` (default 1.0 == exact).
    """
    residuals = compute_residuals(
        z,
        centroids,
        codes,
        project=project,
        normalize_inputs=normalize_inputs,
        normalize_residuals=normalize_residuals,
    )
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
            "unlearning. Check that the codebook checkpoint is the one this "
            "semantic_id_path was generated from, and that the residual recipe "
            "came from load_rq_quantizer rather than being chosen by hand."
        )
    return agree
