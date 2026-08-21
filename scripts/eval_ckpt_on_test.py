"""Run ``trainer.test(model, datamodule, ckpt_path=...)`` against a pre-trained
TIGER checkpoint and dump the resulting NDCG@K / Recall@K to the experiment's
CSVLogger.

This is a small companion to ``scripts/compute_relative_utility.py`` because
``src/train.py``'s test branch ignores ``cfg.ckpt_path`` (it only honours the
best-model path produced by the in-run ``ModelCheckpoint`` callback). For the
unlearning workflow we want to evaluate arbitrary ckpts (clean / poisoned /
unlearned) without retraining first, so we expose a thin Hydra entry that just
calls ``trainer.test(ckpt_path=cfg.ckpt_path)``.

Usage
-----

::

    python -m scripts.eval_ckpt_on_test experiment=tiger_train_flat \\
        data_dir=src/data/amazon_data/beauty \\
        semantic_id_path=.../merged_predictions_tensor.pt \\
        ckpt_path=<the_ckpt_to_evaluate> \\
        num_hierarchies=4 \\
        train=False test=True

The resulting metrics land in
``${paths.output_dir}/csv/version_0/metrics.csv`` and can be passed to
``scripts/compute_relative_utility.py`` directly.

Decode-time filter
------------------

The ``filter`` unlearning baseline performs no weight update: it installs a mask
on the module that refuses forbidden semantic ids at the last decode level. That
mask is module state, NOT checkpoint state, so evaluating ``unlearned.ckpt`` in a
fresh process silently measures the UNFILTERED model. Pass the mask written by
the unlearn run to reinstall it::

    python -m scripts.eval_ckpt_on_test ... \\
        decode_filter_mask=<unlearn_run_dir>/filter_mask.json

The path is required to exist when set; a missing or unusable mask raises rather
than falling through to an unfiltered eval, because the two are indistinguishable
in the output metrics.
"""

from __future__ import annotations

import os

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.components.unlearning.filter_utils import (  # noqa: E402
    forbidden_sids_from_codebook,
    load_filter_mask,
    user_forbidden_sids_from_codebook,
)
from src.components.unlearning.neighborhood_sampler import load_codebook  # noqa: E402
from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402


command_line_logger = RankedLogger(__name__, rank_zero_only=True)


def install_decode_filter(cfg: DictConfig, model) -> None:
    """Reinstall the ``filter`` baseline's decode mask onto ``model``.

    No-op when ``decode_filter_mask`` is unset. Set but broken is an error: an
    eval that quietly skipped the mask would report the poisoned model's numbers
    under the filter baseline's name, which is precisely the failure this exists
    to close.

    Safe to call before ``trainer.test(ckpt_path=...)``: the mask is a plain
    Python attribute, so loading a state dict does not clear it.
    """
    mask_path = cfg.get("decode_filter_mask")
    if not mask_path:
        return
    mask_path = str(mask_path)
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"decode_filter_mask={mask_path!r} does not exist. The filter "
            "baseline's mask is written by the unlearn run (run-root "
            "filter_mask.json); without it this eval would measure the "
            "unfiltered model."
        )
    if not cfg.get("semantic_id_path"):
        raise ValueError(
            "decode_filter_mask needs semantic_id_path to map item ids to "
            "semantic ids."
        )
    if not hasattr(model, "set_decode_filter"):
        raise TypeError(
            f"model {type(model).__name__} has no set_decode_filter; the decode "
            "filter is only defined for SemanticIDEncoderDecoder models."
        )

    mask = load_filter_mask(mask_path)
    filter_mode = str(mask.get("filter_mode", "global"))
    codebook = load_codebook(str(cfg.semantic_id_path))
    forbidden_sids = forbidden_sids_from_codebook(
        codebook, mask.get("forbidden_item_ids") or []
    )
    user_forbidden_sids = user_forbidden_sids_from_codebook(
        codebook, mask.get("user_forget_items")
    )
    if not forbidden_sids and not user_forbidden_sids:
        raise ValueError(
            f"{mask_path} resolves to an EMPTY decode filter (no item mapped "
            "into the codebook). Check that semantic_id_path matches the "
            "identifier space the mask was built against."
        )
    model.set_decode_filter(
        forbidden_sids=forbidden_sids,
        filter_mode=filter_mode,
        user_forbidden_sids=user_forbidden_sids,
    )
    command_line_logger.info(
        f"Decode filter INSTALLED from {mask_path}: mode={filter_mode}, "
        f"{len(forbidden_sids)} forbidden semantic ids, "
        f"{len(user_forbidden_sids)} users with a per-user mask."
    )


def evaluate(cfg: DictConfig) -> None:
    """Load ``cfg.ckpt_path`` into the model and run ``trainer.test``."""
    if not cfg.get("ckpt_path"):
        raise ValueError(
            "ckpt_path is required; pass the checkpoint to evaluate via "
            "ckpt_path=<path>."
        )

    with pipeline_launcher(cfg) as pipeline_modules:
        install_decode_filter(cfg, pipeline_modules.model)
        command_line_logger.info(
            f"Running trainer.test(ckpt_path={cfg.ckpt_path}) ..."
        )
        pipeline_modules.trainer.test(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=cfg.ckpt_path,
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
