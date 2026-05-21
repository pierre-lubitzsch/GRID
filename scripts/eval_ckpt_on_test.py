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
"""

from __future__ import annotations

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402


command_line_logger = RankedLogger(__name__, rank_zero_only=True)


def evaluate(cfg: DictConfig) -> None:
    """Load ``cfg.ckpt_path`` into the model and run ``trainer.test``."""
    if not cfg.get("ckpt_path"):
        raise ValueError(
            "ckpt_path is required; pass the checkpoint to evaluate via "
            "ckpt_path=<path>."
        )

    with pipeline_launcher(cfg) as pipeline_modules:
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
