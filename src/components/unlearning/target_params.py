"""Pick which parameters of a TIGER ``SemanticIDEncoderDecoder`` SCIF should
update. Mirrors ERASE's ``Trainer.target_params`` fallback logic but spelled
out for TIGER's actual module names.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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


def select_pkm_params(
    model: nn.Module, *, include_query: bool = True, include_keys: bool = True
) -> Tuple[List[nn.Parameter], List[str]]:
    """Return only the Product-Key-Memory parameters (everything else frozen).

    This is the "modular stabilizer" update scope: the backbone, SID embeddings
    and decoder heads are left out of the optimizer entirely, so unlearning can
    only edit what lives in the sparse memory.

    Selection is by MODULE TYPE (``HashingMemory``), not by parameter name — the
    PKM wrappers sit at different paths depending on whether they replaced the
    FFN (``T5LayerPKM``) or run beside it (``T5LayerFFWithPKM``), and encoder
    FFNs are registered under ``model.encoder.*`` rather than ``encoder.*``.

    ``include_keys`` / ``include_query`` allow editing only the value table
    (``values.weight``, the "what to output" side per Geva et al. 2021) while
    freezing the routing, which is the more surgical variant.

    Returns ``(params, names)``; ``names`` is for logging what was selected.
    """
    from src.models.components.network_blocks.product_key_memory import (
        HashingMemory,
    )

    params: List[nn.Parameter] = []
    names: List[str] = []
    seen: set = set()
    for mod_name, module in model.named_modules():
        if not isinstance(module, HashingMemory):
            continue
        for p_name, p in module.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            # ``keys`` is the product-key routing table; ``query_proj`` (or
            # whatever the query net is called) maps the hidden state to a
            # query. Both are optional so callers can edit values only.
            is_keys = "keys" in p_name
            is_query = "query" in p_name
            if is_keys and not include_keys:
                continue
            if is_query and not include_query:
                continue
            seen.add(id(p))
            params.append(p)
            names.append(f"{mod_name}.{p_name}")

    if not params:
        raise ValueError(
            "select_pkm_params found no HashingMemory parameters — the "
            "checkpoint/model was built without PKM layers. Pass "
            "model.pkm_layers=... (and model.pkm_mode) so the memory exists "
            "before requesting update_scope='pkm_only'."
        )
    return params, names


def resolve_scope_params(
    model: nn.Module,
    update_scope: str,
    *,
    fallback: List[nn.Parameter],
    include_keys: bool = True,
    include_query: bool = True,
    algo: str = "",
) -> Tuple[List[nn.Parameter], Optional[List[str]]]:
    """Resolve an algorithm's trainable parameter list for ``update_scope``.

    Shared by every gradient-based unlearning algorithm so ``pkm_only`` means the
    same thing everywhere.

    * ``all`` (default) -> ``fallback`` unchanged (the algorithm's own choice).
    * ``pkm_only``      -> only Product-Key-Memory params; with
      ``include_keys=False, include_query=False`` this narrows further to the
      VALUE table only (routing frozen), which is the "values_only" variant.

    Returns ``(params, pkm_names)``; ``pkm_names`` is ``None`` for scope ``all``.
    """
    scope = str(update_scope or "all").strip().lower()
    if scope in ("", "all"):
        return fallback, None
    if scope == "ffn_only":
        # CONTROL for the post-hoc PKM recipe: train only the FFN sub-layers that
        # model.reinit_ffn_layers() freshly re-initialised, so "reinit + retrain
        # this layer" is measured with an ordinary FFN instead of a PKM.
        names = list(getattr(model, "_reinit_ffn_module_names", []) or [])
        if not names:
            raise ValueError(
                "update_scope='ffn_only' requires model.reinit_ffn_layers(...) to "
                "have run first (set model.ffn_reinit_layers)."
            )
        name_to_mod = dict(model.named_modules())
        params, seen = [], set()
        for nm in names:
            for p in name_to_mod[nm].parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p)); params.append(p)
        log.info(
            "[%s] FFN-ONLY update scope: %d tensors (%d params) across %d "
            "re-initialised FFNs %s; everything else FROZEN [PKM CONTROL]",
            algo or "scope", len(params),
            int(sum(p.numel() for p in params)), len(names), names,
        )
        return params, names
    if scope != "pkm_only":
        raise ValueError(
            f"unlearning.update_scope must be 'all', 'pkm_only' or 'ffn_only', "
            f"got {scope!r}"
        )
    params, names = select_pkm_params(
        model, include_keys=include_keys, include_query=include_query
    )
    log.info(
        "[%s] PKM-ONLY update scope: %d tensors (%d params) across %d memories "
        "(keys=%s query=%s); everything else FROZEN",
        algo or "scope",
        len(params),
        int(sum(p.numel() for p in params)),
        len({n.rsplit(".", 1)[0] for n in names}),
        bool(include_keys),
        bool(include_query),
    )
    return params, names


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


def split_code_params(
    model: nn.Module,
    params: List[nn.Parameter],
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split ``params`` into (semantic-ID *code* params, everything else).

    "Code" params are the ones that define what a semantic-ID token MEANS:

      * ``item_sid_embedding_table_encoder`` -- the shared SID embedding table
        (``H`` contiguous ``K``-row blocks, hierarchy ``h`` in rows
        ``[h*K, (h+1)*K)``);
      * ``decoder.decoder_mlp[*]`` -- the per-hierarchy output heads.

    Used to give the code parameters their own, lower learning rate so that
    unlearning perturbs the identifier space only slightly while the rest of the
    model absorbs the update. This is the soft counterpart of the
    stable/adaptive split, which *hard-freezes* the stable codes instead: here
    every code can still move, just slowly, and the two compose (the adaptive
    grad mask is applied to ``.grad`` regardless of which group a tensor is in).

    Membership is decided by identity against ``model.named_parameters()``, so a
    tensor that appears in ``params`` under a restriction policy keeps its
    classification.
    """
    wanted = {id(p) for p in params}
    code_ids: set = set()

    sid_table = getattr(model, "item_sid_embedding_table_encoder", None)
    if sid_table is not None:
        code_ids.update(id(p) for p in sid_table.parameters())
    decoder = getattr(model, "decoder", None)
    decoder_mlp = getattr(decoder, "decoder_mlp", None) if decoder is not None else None
    if decoder_mlp is not None:
        code_ids.update(id(p) for p in decoder_mlp.parameters())

    code = [p for p in params if id(p) in code_ids]
    other = [p for p in params if id(p) not in code_ids]
    assert len(code) + len(other) == len(params)
    del wanted
    return code, other


def split_adaptive_code_params(
    model: nn.Module,
    params: List[nn.Parameter],
    *,
    stable_codes: int,
) -> Tuple[List[nn.Parameter], Dict[int, torch.Tensor]]:
    """Identify the *adaptive-tail* subset of the semantic-ID code parameters.

    Position-aware refinement of :func:`split_code_params`: instead of treating
    the identifier space as one block, separate the codes belonging to the
    adaptive (fine, item-specific) hierarchies ``[stable_codes,
    num_hierarchies)`` from the stable (coarse, shared) prefix ``[0,
    stable_codes)``. Used to give the adaptive tail its OWN learning rate, so
    the coarse codes every neighbor shares and the fine codes that are nearly
    item-unique can move at different speeds.

    Two return channels, because the two surfaces are shaped differently:

      * ``tensors`` -- whole parameters that belong exclusively to adaptive
        hierarchies (the per-hierarchy decoder heads
        ``decoder.decoder_mlp[stable_codes:]``). These can simply go into their
        own optimizer group.
      * ``row_masks`` -- ``id(param) -> float mask`` over rows, for parameters
        whose rows span BOTH segments and which therefore cannot be split
        across optimizer groups: the shared SID embedding table
        (``H`` contiguous ``K``-row hierarchy blocks). The mask is 1.0 on the
        adaptive rows ``[stable_codes * K, H * K)`` and 0.0 on the stable rows.
        The caller applies the per-row rate by rescaling the *applied update*
        after ``opt.step()``; scaling ``.grad`` instead would do nothing under
        Adam, whose per-parameter normalization cancels any constant gradient
        factor.

    Only parameters present in ``params`` are returned, so a restriction policy
    that already excluded a tensor keeps it excluded.
    """
    num_hierarchies = getattr(model, "num_hierarchies", None)
    codebook_size = getattr(model, "num_embeddings_per_hierarchy", None)
    if num_hierarchies is None or codebook_size is None:
        raise TypeError(
            "split_adaptive_code_params requires a SemanticIDEncoderDecoder with "
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

    in_update = {id(p) for p in params}
    tensors: List[nn.Parameter] = []
    row_masks: Dict[int, torch.Tensor] = {}

    # Adaptive decoder heads: whole tensors, one head per hierarchy.
    decoder = getattr(model, "decoder", None)
    decoder_mlp = getattr(decoder, "decoder_mlp", None) if decoder is not None else None
    if decoder_mlp is not None:
        for h in range(stable_codes, min(num_hierarchies, len(decoder_mlp))):
            for p in decoder_mlp[h].parameters():
                if id(p) in in_update:
                    tensors.append(p)

    # SID embedding table: one tensor, mixed rows -> row mask.
    sid_table = getattr(model, "item_sid_embedding_table_encoder", None)
    if sid_table is not None:
        weight = sid_table.weight
        if id(weight) in in_update:
            expected_rows = num_hierarchies * codebook_size
            if weight.shape[0] != expected_rows:
                log.warning(
                    "SID table has %d rows but num_hierarchies*codebook_size=%d; "
                    "the adaptive row mask assumes a hierarchy-blocked layout",
                    weight.shape[0],
                    expected_rows,
                )
            mask = torch.zeros_like(weight)
            mask[stable_codes * codebook_size :, :] = 1.0
            row_masks[id(weight)] = mask

    return tensors, row_masks
