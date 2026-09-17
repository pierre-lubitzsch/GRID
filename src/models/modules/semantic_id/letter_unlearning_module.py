"""Unlearning entrypoint for LETTER checkpoints.

``LetterUnlearningModule`` is to :class:`LetterEncoderDecoder` what
``TigerUnlearningModule`` is to ``SemanticIDEncoderDecoder``: the same model plus
the unlearning algorithms. All of them (scif, seif, kookmin, fanchuan, unified,
finetune, neg_train, filter, tracer) apply unchanged, because they operate on
parameters and batches, not on how the identifiers were produced.

WHAT IS DIFFERENT FROM DIGER, AND WHY THERE IS NO ``finalize_unlearning`` HERE
------------------------------------------------------------------------------
DIGER's tokenizer is differentiable and trains jointly with the recommender, so
an unlearning update can MOVE an item's semantic id; ``DigerUnlearningModule``
therefore has to re-commit the ids and rewrite the SID tensor afterwards or every
downstream metric scores stale codes.

LETTER's tokenizer is learnable but not joint: it is trained in its own stage and
its output -- the ``(L, N)`` semantic-ID tensor -- is FROZEN before the
recommender ever runs, exactly as for TIGER. So a LETTER unlearning update moves
theta only, the ids on disk stay correct by construction, and there is nothing to
re-commit. That is a property worth stating rather than inferring from the
absence of a method.
"""

from __future__ import annotations

from typing import Any

from src.models.modules.semantic_id.letter_generation_model import (
    LetterEncoderDecoder,
)
from src.models.modules.semantic_id.tiger_unlearning_module import (
    TigerUnlearningModule,
)


class LetterUnlearningModule(LetterEncoderDecoder, TigerUnlearningModule):
    """LETTER + the unlearning algorithms.

    MRO is ``LetterUnlearningModule -> LetterEncoderDecoder ->
    TigerUnlearningModule -> SemanticIDEncoderDecoder``, so LETTER's tempered
    loss wins while every unlearning method is inherited. Both parents take
    ``**kwargs`` and cooperate through ``super().__init__``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
