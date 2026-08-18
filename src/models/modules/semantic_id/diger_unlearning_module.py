"""Unlearning entrypoint for DIGER checkpoints.

``DigerUnlearningModule`` is to :class:`DigerEncoderDecoder` what
``TigerUnlearningModule`` is to ``SemanticIDEncoderDecoder``: the same model,
plus the unlearning algorithms. Every algorithm (scif, seif, kookmin, fanchuan,
unified, finetune, neg_train, filter, tracer) works unchanged, because they
operate on parameters and batches, not on how the identifiers were produced.

WHAT IS DIFFERENT, AND WHY IT MATTERS HERE
------------------------------------------
With frozen identifiers an unlearning update can only move theta. DIGER's codes
are trainable, so ``tokenizer.codebooks`` and the tokenizer encoder are in the
update set and a deletion can move an item's identifier as well as the
recommender's weights. That is the natural home for the measured
levels-0/1-vs-2/3 asymmetry: a level-0 code is shared by ~47 items at width 256
(7.8% singletons) while a level-2 prefix is 90.4% singleton, so moving a coarse
code is a genuinely non-local edit and moving a fine one is nearly a no-op.

THE HAZARD THIS CLASS EXISTS TO CLOSE
-------------------------------------
If the codes move during unlearning but the SID tensor on disk does not, every
downstream metric silently scores the codes the run STARTED with -- SH/ASI/TPM
map their target items through ``semantic_id_path``. So
:meth:`finalize_unlearning` re-commits the ids and writes the tensor next to the
unlearned checkpoint; the caller threads that path into the post-unlearn eval.
This is the same failure that made TRACER's committed reassignment invisible.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

from src.models.modules.semantic_id.diger_generation_model import DigerEncoderDecoder
from src.models.modules.semantic_id.tiger_unlearning_module import (
    TigerUnlearningModule,
)

log = logging.getLogger(__name__)


class DigerUnlearningModule(DigerEncoderDecoder, TigerUnlearningModule):
    """DIGER + the unlearning algorithms.

    MRO is ``DigerUnlearningModule -> DigerEncoderDecoder ->
    TigerUnlearningModule -> SemanticIDEncoderDecoder``, so DIGER's
    ``_inject_soft_sid_embeddings`` / ``model_step`` overrides win while every
    unlearning method is inherited from the Tiger module. Both parents take
    ``**kwargs`` and cooperate through ``super().__init__``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Unlearning starts from a TRAINED checkpoint whose codebook is already
        # seeded, and the flag rides along in the state dict. Re-seeding here
        # would discard the identifier space that was actually trained, so it is
        # forced off rather than left to the config.
        self.diger_init_codebooks = False

    @torch.no_grad()
    def finalize_unlearning(self, ckpt_path: Optional[str] = None) -> Optional[str]:
        """Re-commit the semantic ids and persist them next to the checkpoint.

        Returns the written tensor path, or None when there is nothing to write.
        Call after the unlearning update and BEFORE evaluation -- the point of
        the exercise is that the identifiers may have moved.
        """
        if getattr(self, "tokenizer", None) is None:
            return None
        changed = self.commit_semantic_ids()
        log.info(
            "[diger-unlearn] re-committed semantic ids: %d items reassigned", changed
        )
        if not ckpt_path:
            return None
        out = os.path.join(
            os.path.dirname(os.path.abspath(ckpt_path)),
            "merged_predictions_tensor_diger_unlearned.pt",
        )
        self.write_semantic_id_tensor(out)
        if changed:
            log.warning(
                "[diger-unlearn] %d semantic ids MOVED. Post-unlearn eval must be "
                "pointed at %s via semantic_id_path, or SH/ASI/TPM will score the "
                "codes the run started with.",
                changed,
                out,
            )
        return out
