"""Pick which parameters of a TIGER ``SemanticIDEncoderDecoder`` SCIF should
update. Mirrors ERASE's ``Trainer.target_params`` fallback logic but spelled
out for TIGER's actual module names.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


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
