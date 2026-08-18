"""Registry of recommender architectures available as an experiment axis.

The architecture is a first-class knob alongside the dataset and the identifier
space: ``MODEL=tiger`` / ``MODEL=diger`` on the shell runners, ``resolve_model``
in Python. Adding a model means adding ONE entry here, the matching case block
in ``scripts/resolve_model.sh``, and the three experiment configs named below.

Why a registry rather than passing ``model._target_`` directly: a model is not
just a class. It carries its own train / inference / unlearn experiment configs
(different loss wiring, different callbacks), and the unlearning entrypoints have
to know which config to compose. One name has to resolve all of it, in both
Python and bash, or the two drift.

``run_tag_prefix`` is deliberately EMPTY for tiger. Every recorded run dir,
extractor regex and results table was written before this axis existed and keys
off tags with no model token; giving tiger a prefix now would orphan all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelSpec:
    """Everything a runner needs to know to train / evaluate / unlearn a model."""

    name: str
    target: str
    train_experiment: str
    inference_experiment: str
    unlearn_experiment: str
    run_tag_prefix: str
    description: str
    # Does the architecture update the semantic IDs as part of its own training?
    # Unlearning has to know: for a differentiable tokenizer the codes are part
    # of what gets unlearned, and the SID tensor must be re-committed and written
    # back afterwards or SH/ASI/TPM silently score stale codes.
    differentiable_ids: bool = False


_MODELS: Dict[str, ModelSpec] = {
    "tiger": ModelSpec(
        name="tiger",
        target=(
            "src.models.modules.semantic_id.tiger_generation_model"
            ".SemanticIDEncoderDecoder"
        ),
        train_experiment="tiger_train_flat",
        inference_experiment="tiger_inference_flat",
        unlearn_experiment="tiger_unlearn_scif_flat",
        run_tag_prefix="",
        description=(
            "TIGER (arXiv:2305.05065): T5 encoder-decoder over FROZEN RQ "
            "semantic IDs produced by a separately trained quantizer."
        ),
        differentiable_ids=False,
    ),
    "diger": ModelSpec(
        name="diger",
        target=(
            "src.models.modules.semantic_id.diger_generation_model"
            ".DigerEncoderDecoder"
        ),
        train_experiment="diger_train_flat",
        inference_experiment="diger_inference_flat",
        unlearn_experiment="diger_unlearn_scif_flat",
        run_tag_prefix="diger_",
        description=(
            "DIGER (arXiv:2601.19711): TIGER backbone plus a DIFFERENTIABLE "
            "semantic-ID tokenizer trained jointly with the recommender via "
            "straight-through Gumbel-Softmax over the RQ codebooks."
        ),
        differentiable_ids=True,
    ),
}


def available_models() -> List[str]:
    return sorted(_MODELS)


def resolve_model(name: str) -> ModelSpec:
    """Short model name -> :class:`ModelSpec`. Raises on an unknown name."""
    key = (name or "tiger").strip().lower()
    if key not in _MODELS:
        raise KeyError(
            f"unknown model {name!r}; known models: {', '.join(available_models())}"
        )
    return _MODELS[key]
