"""LETTER's generative recommender: TIGER's backbone under a tempered CE loss.

LETTER (arXiv:2405.07314) has two halves. The half that carries the method is
the TOKENIZER -- see :mod:`src.modules.clustering.letter_quantization`. The half
here is what the paper calls the *ranking-guided generation loss*, and it is
worth being precise about how little it is in the authors' own release:
``LETTER-TIGER/modeling_letter.py`` subclasses ``T5ForConditionalGeneration`` and
replaces the loss with

    CrossEntropy(lm_logits / temperature, labels)

with ``temperature`` defaulting to 1.0 in ``run_train.sh``. There is no negative
sampling, no ranking margin, and no reweighting in the released code; the module
even carries an unused ``sigmoid`` helper where such a term would have gone. So
at ``temperature = 1.0`` LETTER's recommender IS TIGER's, exactly, and every
difference in results comes from the identifiers it is trained on.

That is the honest thing for this class to implement, and it is implemented
honestly: the temperature is a real knob, its default reproduces the released
configuration, and nothing else about the backbone changes. Anyone who later
wants the paper's fuller ranking term should add it here rather than assume it
is already present.

**How the temperature is applied.** TIGER computes one cross-entropy per RQ
position (``model_step`` and ``per_hierarchy_losses`` both call
``self.loss_function``), so the temperature is folded into the loss function
itself rather than into either call site. Every consumer of ``loss_function`` --
including the unlearning algorithms, which recompute the training loss to build
their gradients -- then sees the same tempered objective the model was trained
under, which is what makes an unlearning update on a LETTER checkpoint
well posed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)

log = logging.getLogger(__name__)


class TemperatureScaledLoss(nn.Module):
    """Wrap a logit-consuming loss so its logits are divided by ``temperature``.

    Kept as a module (not a lambda or a partial) so it survives ``deepcopy`` --
    several unlearning algorithms deep-copy the model -- and so it shows up by
    name in a model summary instead of masquerading as the plain loss.
    """

    def __init__(self, loss_function: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.loss_function = loss_function
        self.temperature = float(temperature)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return self.loss_function(input=input / self.temperature, target=target, **kwargs)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


class LetterEncoderDecoder(SemanticIDEncoderDecoder):
    """TIGER backbone + LETTER's tempered generation loss.

    Args:
        letter_temperature: the paper's ``tau``. 1.0 (the released default) makes
            this numerically identical to TIGER, so a LETTER-vs-TIGER comparison
            at that setting is a comparison of IDENTIFIER SPACES with the
            recommender held fixed -- which is the comparison LETTER is actually
            making.
    """

    def __init__(
        self,
        *args: Any,
        letter_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.letter_temperature = float(letter_temperature)
        if self.loss_function is not None and self.letter_temperature != 1.0:
            self.loss_function = TemperatureScaledLoss(
                self.loss_function, self.letter_temperature
            )
            log.info(
                "[letter] generation loss tempered by tau=%.4g",
                self.letter_temperature,
            )
        elif self.loss_function is None and self.letter_temperature != 1.0:
            # Inference configs pass loss_function: null. A temperature there is
            # inert, and silently ignoring it would hide a mis-set eval config.
            log.warning(
                "[letter] letter_temperature=%.4g set but this model has no loss "
                "function (inference config); the temperature is inert.",
                self.letter_temperature,
            )
