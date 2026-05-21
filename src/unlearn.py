"""Hydra entry point for TIGER unlearning.

Mirrors the structure of ``src/train.py`` and ``src/inference.py``:

1. Instantiate model / datamodule / trainer via ``pipeline_launcher`` so the
   exact same config plumbing the user is used to keeps working.
2. Load the pre-trained TIGER checkpoint into the (subclassed)
   ``TigerUnlearningModule``.
3. Hand control to ``model.run_scif_unlearning(...)`` -- which builds the
   forget / retain finite dataloaders, optionally runs the neighborhood
   sampler, and applies one SCIF parameter step in-place.
4. Save the modified ``state_dict`` as a Lightning-compatible checkpoint at
   ``${paths.output_dir}/checkpoints/unlearned.ckpt`` so downstream
   ``python -m src.inference experiment=tiger_inference_flat
   ckpt_path=...`` can consume it verbatim.

Run with::

    python -m src.unlearn experiment=tiger_unlearn_scif_flat \
        data_dir=src/data/amazon_data/beauty_spam_seed42_pct1_n10 \
        semantic_id_path=.../merged_predictions_tensor.pt \
        ckpt_path=.../checkpoints/checkpoint_best.ckpt \
        num_hierarchies=4 \
        unlearning.neighborhood_aware=true
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import hydra
import rootutils
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402
from src.models.modules.semantic_id.tiger_unlearning_module import (  # noqa: E402
    TigerUnlearningModule,
    save_unlearned_checkpoint,
)
from src.utils.unlearn_paths import (  # noqa: E402
    guard_fresh_unlearn_output_dir,
    run_metadata_from_cfg,
)


command_line_logger = RankedLogger(__name__, rank_zero_only=True)
torch.set_float32_matmul_precision("medium")


def _resolve_train_dl_cfg(
    pipeline_modules: Any, cfg: DictConfig
) -> Any:
    """Return a fully-instantiated ``SequenceDataloaderConfig`` for the train
    split. Uses the datamodule's already-built ``stage_to_config`` if
    available, otherwise re-instantiates from the raw cfg.
    """
    dm = pipeline_modules.datamodule
    train_dl_cfg = None
    if hasattr(dm, "stage_to_config"):
        train_dl_cfg = dm.stage_to_config.get(TrainerFn.FITTING)
    if train_dl_cfg is None:
        train_dl_cfg = hydra.utils.instantiate(
            cfg.data_loading.train_dataloader_config.dataloader
        )
    return train_dl_cfg


def unlearn(cfg: DictConfig) -> Dict[str, Any]:
    """Run a single SCIF unlearning step driven by ``cfg``.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: SCIF info dict (CG residuals, sizes, paths, ...).
    """
    guard_fresh_unlearn_output_dir(cfg.paths.output_dir)
    run_meta = run_metadata_from_cfg(cfg)
    command_line_logger.info(
        f"Unlearning output_dir={run_meta['hydra_output_dir']} "
        f"(tag={run_meta.get('unlearning_run_tag')}, job={run_meta.get('slurm_job_id')})"
    )

    with pipeline_launcher(cfg) as pipeline_modules:
        model = pipeline_modules.model
        if not isinstance(model, TigerUnlearningModule):
            raise TypeError(
                f"Expected `model._target_` to instantiate TigerUnlearningModule, "
                f"got {type(model).__name__}. Set "
                f"`model._target_: src.models.modules.semantic_id.tiger_unlearning_module."
                f"TigerUnlearningModule` in your experiment yaml."
            )

        ckpt_path = cfg.get("ckpt_path", None)
        if not ckpt_path:
            raise ValueError(
                "ckpt_path is required for unlearning -- pass the pre-trained "
                "(poisoned) TIGER checkpoint."
            )
        command_line_logger.info(f"Loading TIGER checkpoint from {ckpt_path}")
        source_ckpt = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
        if "state_dict" not in source_ckpt:
            raise KeyError(
                f"Checkpoint at {ckpt_path} has no 'state_dict' key; got "
                f"{sorted(source_ckpt.keys())}"
            )
        load_result = model.load_state_dict(source_ckpt["state_dict"], strict=False)
        if load_result.missing_keys:
            command_line_logger.warning(
                f"load_state_dict missing keys ({len(load_result.missing_keys)}): "
                f"{load_result.missing_keys[:5]}"
                f"{'...' if len(load_result.missing_keys) > 5 else ''}"
            )
        if load_result.unexpected_keys:
            command_line_logger.warning(
                f"load_state_dict unexpected keys ({len(load_result.unexpected_keys)}): "
                f"{load_result.unexpected_keys[:5]}"
                f"{'...' if len(load_result.unexpected_keys) > 5 else ''}"
            )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        command_line_logger.info(f"Moving model to {device}")
        model = model.to(device)

        train_dl_cfg = _resolve_train_dl_cfg(pipeline_modules, cfg)

        unlearning_cfg = OmegaConf.to_container(cfg.unlearning, resolve=True)
        data_dir = cfg.paths.data_dir
        forget_subdir = cfg.get("forget_subdir", "training_forget")
        retain_subdir = cfg.get("retain_subdir", "training_retain")
        retain_subset_dir = os.path.join(cfg.paths.output_dir, "retain_subset")
        semantic_id_path = cfg.get("semantic_id_path", None)
        forget_size_hint: Optional[int] = unlearning_cfg.get("forget_size", None)

        algorithm = str(unlearning_cfg.get("algorithm", "scif"))
        command_line_logger.info(
            f"Running unlearning algorithm={algorithm}: data_dir={data_dir} "
            f"forget={forget_subdir} retain={retain_subdir} "
            f"neighborhood_aware={unlearning_cfg.get('neighborhood_aware')} "
            f"deletion_spec={unlearning_cfg.get('deletion_spec', 'session')}"
        )
        info = model.run_unlearning(
            unlearning_cfg=unlearning_cfg,
            train_dataloader_config=train_dl_cfg,
            data_dir=data_dir,
            forget_subdir=forget_subdir,
            retain_subdir=retain_subdir,
            retain_subset_dir=retain_subset_dir,
            semantic_id_path=semantic_id_path,
            forget_size_hint=forget_size_hint,
            seed=int(cfg.get("seed", 2)),
            num_hierarchies=int(cfg.get("num_hierarchies", 4)),
            device=device,
            output_dir=cfg.paths.output_dir,
        )

        out_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
        out_path = os.path.join(out_dir, "unlearned.ckpt")
        save_unlearned_checkpoint(
            model=model,
            out_path=out_path,
            source_ckpt=source_ckpt,
            extra_metadata={
                "unlearning_info": info,
                "scif_info": info,
                "source_ckpt_path": os.path.abspath(ckpt_path),
                "data_dir": os.path.abspath(data_dir),
                "forget_subdir": forget_subdir,
                "retain_subdir": retain_subdir,
                "unlearning_cfg": unlearning_cfg,
                "algorithm": algorithm,
                "semantic_id_path": (
                    os.path.abspath(semantic_id_path)
                    if semantic_id_path
                    else None
                ),
                "num_hierarchies": cfg.get("num_hierarchies", None),
                **run_meta,
            },
        )

        info_path = os.path.join(cfg.paths.output_dir, "unlearn_info.json")
        info_out = {**info, **run_meta}
        with open(info_path, "w") as f:
            json.dump(_json_safe(info_out), f, indent=2)
        # Backward-compatible alias
        scif_info_path = os.path.join(cfg.paths.output_dir, "scif_info.json")
        with open(scif_info_path, "w") as f:
            json.dump(_json_safe(info_out), f, indent=2)
        command_line_logger.info(
            f"Unlearning done. Unlearned ckpt -> {out_path}; info -> {info_path}"
        )
        return info


def _json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


@hydra.main(version_base="1.3", config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    unlearn(cfg)


if __name__ == "__main__":
    main()
