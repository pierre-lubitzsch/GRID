"""Pick which parameters of a TIGER ``SemanticIDEncoderDecoder`` SCIF should
update. Mirrors ERASE's ``Trainer.target_params`` fallback logic but spelled
out for TIGER's actual module names.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from torch import nn


log = logging.getLogger(__name__)

_VALID_POLICIES = ("all", "sid_embeddings", "encoder_only", "tiger")

# Parameters that carry no item/user knowledge and should be excluded from
# unlearning updates for TIGER (structural sequence scaffolding).
_TIGER_EXCLUDED_PARAM_NAMES = frozenset(["decoder.bos_token", "sep_token"])


def select_target_params(model: nn.Module, policy: str = "all") -> List[nn.Parameter]:
    """Return the list of parameters SCIF will update.

    Parameters
    ----------
    model
        A ``SemanticIDEncoderDecoder`` instance (or any ``nn.Module``).
    policy
        One of:

        * ``all`` -- every trainable named parameter (mirrors ERASE's fallback,
          which is the right default for TIGER since ``num_user_bins=null`` so
          there is no per-user embedding to single out).
        * ``sid_embeddings`` -- only the SID embedding table on the encoder side
          plus the per-hierarchy decoder linear heads. Cheapest HVP, narrowest
          influence.
        * ``encoder_only`` -- all parameters of the encoder sub-module.
        * ``tiger`` -- all trainable parameters of a ``SemanticIDEncoderDecoder``
          except ``decoder.bos_token`` and ``sep_token`` (structural scaffolding
          that carries no item/user knowledge). Raises ``TypeError`` for any
          other model type.

    Returns
    -------
    list[nn.Parameter]
        Parameters to differentiate against. Always a non-empty list.
    """
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"Unknown target_params policy={policy!r}; expected one of {_VALID_POLICIES}"
        )

    if policy == "all":
        params = [p for _, p in model.named_parameters() if p.requires_grad]
    elif policy == "sid_embeddings":
        params = []
        sid_table = getattr(model, "item_sid_embedding_table_encoder", None)
        if sid_table is not None:
            params.extend(p for p in sid_table.parameters() if p.requires_grad)
        decoder = getattr(model, "decoder", None)
        decoder_mlp = getattr(decoder, "decoder_mlp", None) if decoder is not None else None
        if decoder_mlp is not None:
            params.extend(p for p in decoder_mlp.parameters() if p.requires_grad)
    elif policy == "encoder_only":
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            raise ValueError(
                "policy='encoder_only' requested but model has no .encoder attribute"
            )
        params = [p for p in encoder.parameters() if p.requires_grad]
    else:
        from src.models.modules.semantic_id.tiger_generation_model import (
            SemanticIDEncoderDecoder,
        )
        if not isinstance(model, SemanticIDEncoderDecoder):
            raise TypeError(
                f"policy='tiger' requires a SemanticIDEncoderDecoder model, "
                f"got {type(model).__name__}"
            )
        params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and n not in _TIGER_EXCLUDED_PARAM_NAMES
        ]

    if not params:
        raise ValueError(
            f"select_target_params(policy={policy!r}) returned 0 trainable params; "
            f"check the model layout."
        )
    return params


def select_adaptive_code_params(
    model: nn.Module,
    *,
    stable_codes: int,
    update_backbone: bool = False,
) -> Tuple[List[nn.Parameter], Dict[int, torch.Tensor]]:
    """Restrict updates to the *adaptive* (fine-grained) tail of the semantic ID.

    The Stable-Adaptive Semantic ID design: an RQ item id
    ``[c_0, ..., c_{H-1}]`` is split at ``stable_codes`` into a stable (coarse,
    frozen) segment ``[c_0, ..., c_{stable_codes-1}]`` and an adaptive (fine,
    trainable) segment ``[c_{stable_codes}, ..., c_{H-1}]``. Unlearning then
    moves only the adaptive segment, localizing deletions.

    Trainable surface returned:
      * ``item_sid_embedding_table_encoder`` — included as a whole tensor, but
        with a row mask zeroing the stable hierarchies' rows
        ``[0, stable_codes * K)`` so only adaptive rows
        ``[stable_codes * K, H * K)`` receive updates (the table is laid out as
        ``H`` contiguous ``K``-row hierarchy blocks).
      * Per-hierarchy decoder heads ``decoder.decoder_mlp[stable_codes:]``.
      * The shared transformer backbone (encoder + decoder T5 stack) only when
        ``update_backbone`` is True (no mask).

    Parameters
    ----------
    stable_codes
        Number of leading (coarse) hierarchies to freeze. Must satisfy
        ``1 <= stable_codes < num_hierarchies`` (at least one adaptive code).
    update_backbone
        If True, also update the shared transformer backbone.

    Returns
    -------
    (params, grad_masks)
        ``params`` is the non-empty list handed to the optimizer. ``grad_masks``
        maps ``id(param) -> float mask`` (same shape as the param) for params
        that need a partial-gradient mask applied to ``.grad`` before the
        optimizer step; params absent from the dict are updated in full.
    """
    num_hierarchies = getattr(model, "num_hierarchies", None)
    codebook_size = getattr(model, "num_embeddings_per_hierarchy", None)
    if num_hierarchies is None or codebook_size is None:
        raise TypeError(
            "select_adaptive_code_params requires a SemanticIDEncoderDecoder with "
            "num_hierarchies / num_embeddings_per_hierarchy attributes"
        )
    num_hierarchies = int(num_hierarchies)
    codebook_size = int(codebook_size)
    stable_codes = int(stable_codes)
    if not 1 <= stable_codes < num_hierarchies:
        raise ValueError(
            f"stable_codes={stable_codes} must satisfy 1 <= stable_codes < "
            f"num_hierarchies={num_hierarchies} (need at least one adaptive code)"
        )

    params: List[nn.Parameter] = []
    grad_masks: Dict[int, torch.Tensor] = {}
    seen: set = set()

    def _add(p: nn.Parameter) -> None:
        if p.requires_grad and id(p) not in seen:
            params.append(p)
            seen.add(id(p))

    # SID embedding table: whole tensor + adaptive-row mask.
    sid_table = getattr(model, "item_sid_embedding_table_encoder", None)
    if sid_table is None:
        raise ValueError("model has no item_sid_embedding_table_encoder")
    weight = sid_table.weight
    expected_rows = num_hierarchies * codebook_size
    if weight.shape[0] != expected_rows:
        log.warning(
            "SID table has %d rows but num_hierarchies*codebook_size=%d; the "
            "adaptive row mask assumes a hierarchy-blocked layout",
            weight.shape[0],
            expected_rows,
        )
    mask = torch.zeros_like(weight)
    mask[stable_codes * codebook_size :, :] = 1.0  # adaptive hierarchies only
    _add(weight)
    grad_masks[id(weight)] = mask

    # Adaptive per-hierarchy decoder heads.
    decoder = getattr(model, "decoder", None)
    decoder_mlp = getattr(decoder, "decoder_mlp", None) if decoder is not None else None
    if decoder_mlp is None:
        raise ValueError("model.decoder has no decoder_mlp")
    for h in range(stable_codes, num_hierarchies):
        for p in decoder_mlp[h].parameters():
            _add(p)

    # Optionally the shared transformer backbone.
    if update_backbone:
        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            for p in encoder.parameters():
                _add(p)
        backbone = getattr(decoder, "decoder", None)
        if backbone is not None:
            for p in backbone.parameters():
                _add(p)

    if not params:
        raise ValueError(
            "select_adaptive_code_params returned 0 trainable params; check the "
            "model layout."
        )
    return params, grad_masks


def select_code_position_params(
    model: nn.Module,
    *,
    positions: List[int],
    update_backbone: bool = False,
) -> Tuple[List[nn.Parameter], Dict[int, torch.Tensor]]:
    """Confine updates to an *arbitrary subset* of RQ semantic-ID code positions.

    Generalizes :func:`select_adaptive_code_params` (which only freezes a
    contiguous *prefix* of codes) to any subset ``positions ⊆ {0, .., H-1}`` of
    hierarchy indices. This is the "position-wise intervention" knob for the
    RQ-ID diagnosis: it lets the SCIF update touch ONLY the parameters that are
    specific to the selected code positions, so we can test whether any single
    code level (``c1``, ``c4``, …) or pair (``[c1,c2]``, ``[c3,c4]``) provides a
    local unlearning interface.

    Trainable surface returned for ``positions = S``:
      * ``item_sid_embedding_table_encoder`` — included as a whole tensor, with a
        row mask that is 1 only on the hierarchy blocks ``[h*K, (h+1)*K)`` for
        ``h in S`` (the table is laid out as ``H`` contiguous ``K``-row blocks)
        and 0 elsewhere, so only those hierarchies' input embeddings move.
      * Per-hierarchy decoder heads ``decoder.decoder_mlp[h]`` for ``h in S``
        (the output projection that maps the decoder state to position ``h``'s
        code logits — TIGER's only genuinely position-specific output weights).
      * The shared transformer backbone (encoder + decoder T5 stack) only when
        ``update_backbone`` is True (no mask).

    Because ``decoder_mlp[h]`` participates only in hierarchy ``h``'s loss term
    (``model_step`` sums independent per-hierarchy CE heads), restricting the
    updated parameters to position ``h`` also restricts the *learning signal*
    that reaches those heads to position ``h`` — the update is local in both the
    parameter and the gradient sense.

    Parameters
    ----------
    positions
        Iterable of 0-based hierarchy indices to update. Must be a non-empty
        subset of ``range(num_hierarchies)``.
    update_backbone
        If True, also update the shared transformer backbone (un-masked).

    Returns
    -------
    (params, grad_masks)
        ``params`` is the non-empty list handed to the optimizer / SCIF.
        ``grad_masks`` maps ``id(param) -> float mask`` (same shape as the
        param) for params that need a partial-gradient/-update mask applied;
        params absent from the dict are updated in full. Mirrors the contract of
        :func:`select_adaptive_code_params` so the same downstream masking code
        applies.
    """
    num_hierarchies = getattr(model, "num_hierarchies", None)
    codebook_size = getattr(model, "num_embeddings_per_hierarchy", None)
    if num_hierarchies is None or codebook_size is None:
        raise TypeError(
            "select_code_position_params requires a SemanticIDEncoderDecoder with "
            "num_hierarchies / num_embeddings_per_hierarchy attributes"
        )
    num_hierarchies = int(num_hierarchies)
    codebook_size = int(codebook_size)
    positions = sorted({int(p) for p in positions})
    if not positions:
        raise ValueError(
            "positions must be a non-empty subset of range(num_hierarchies)"
        )
    for p in positions:
        if not 0 <= p < num_hierarchies:
            raise ValueError(
                f"position index {p} out of range [0, {num_hierarchies})"
            )

    params: List[nn.Parameter] = []
    grad_masks: Dict[int, torch.Tensor] = {}
    seen: set = set()

    def _add(p: nn.Parameter) -> None:
        if p.requires_grad and id(p) not in seen:
            params.append(p)
            seen.add(id(p))

    # SID embedding table: whole tensor + per-position row mask.
    sid_table = getattr(model, "item_sid_embedding_table_encoder", None)
    if sid_table is None:
        raise ValueError("model has no item_sid_embedding_table_encoder")
    weight = sid_table.weight
    expected_rows = num_hierarchies * codebook_size
    if weight.shape[0] != expected_rows:
        log.warning(
            "SID table has %d rows but num_hierarchies*codebook_size=%d; the "
            "per-position row mask assumes a hierarchy-blocked layout",
            weight.shape[0],
            expected_rows,
        )
    mask = torch.zeros_like(weight)
    for h in positions:
        mask[h * codebook_size : (h + 1) * codebook_size, :] = 1.0
    _add(weight)
    grad_masks[id(weight)] = mask

    # Per-hierarchy decoder heads for the selected positions.
    decoder = getattr(model, "decoder", None)
    decoder_mlp = getattr(decoder, "decoder_mlp", None) if decoder is not None else None
    if decoder_mlp is None:
        raise ValueError("model.decoder has no decoder_mlp")
    for h in positions:
        for p in decoder_mlp[h].parameters():
            _add(p)

    # Optionally the shared transformer backbone.
    if update_backbone:
        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            for p in encoder.parameters():
                _add(p)
        backbone = getattr(decoder, "decoder", None)
        if backbone is not None:
            for p in backbone.parameters():
                _add(p)

    if not params:
        raise ValueError(
            "select_code_position_params returned 0 trainable params; check the "
            "model layout."
        )
    return params, grad_masks


def named_target_params(
    model: nn.Module, policy: str = "all"
) -> List[tuple]:
    """Same as :func:`select_target_params` but returns ``(name, param)`` pairs.

    Useful for logging which parameters were touched.
    """
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"Unknown target_params policy={policy!r}; expected one of {_VALID_POLICIES}"
        )
    selected_ids = {id(p) for p in select_target_params(model, policy=policy)}
    return [(n, p) for n, p in model.named_parameters() if id(p) in selected_ids]
