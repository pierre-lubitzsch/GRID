"""Hydra entry point for TIGER RQ semantic-ID (RQ-ID) diagnosis.

Read-only counterpart to ``src/unlearn.py``: loads a trained / poisoned TIGER
checkpoint and runs the RQ-ID diagnostics WITHOUT modifying any weights.

Two analyses, selected by ``unlearning.diagnostics_mode`` (``both`` |
``positions`` | ``code_sharing``):

1. Position-wise signal analysis — per RQ code position (c1..cH): forget vs
   retain gradient strength and forget/retain gradient conflict (cosine). Tells
   you whether the spam influence and the forget/retain tension sit in the early
   (coarse) or late (fine) codes.
2. Code sharing — for each prefix length, how many retained (non-target) catalog
   items share a target item's code / code-prefix (the structural exposure for
   collateral forgetting).

Run with::

    python -m src.diagnose_rq_ids experiment=tiger_unlearn_scif_flat \
        data_dir=src/data/amazon_data/beauty_spam_clone_inject_seed2_pct1_n3 \
        semantic_id_path=embeddings/beauty/merged_predictions_tensor.pt \
        ckpt_path=.../checkpoints/checkpoint_best.ckpt \
        num_hierarchies=4 \
        unlearning.diagnostics_mode=both

Writes ``${paths.output_dir}/rq_id_diagnostics.json``.
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
)


command_line_logger = RankedLogger(__name__, rank_zero_only=True)
torch.set_float32_matmul_precision("medium")


def _resolve_train_dl_cfg(pipeline_modules: Any, cfg: DictConfig) -> Any:
    """Instantiate the train-split ``SequenceDataloaderConfig`` (see src/unlearn.py)."""
    dm = pipeline_modules.datamodule
    train_dl_cfg = None
    if hasattr(dm, "stage_to_config"):
        train_dl_cfg = dm.stage_to_config.get(TrainerFn.FITTING)
    if train_dl_cfg is None:
        train_dl_cfg = hydra.utils.instantiate(
            cfg.data_loading.train_dataloader_config.dataloader
        )
    return train_dl_cfg


def _json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _print_summary(diag: Dict[str, Any]) -> None:
    # NOTE: pre-format each line into a single string. The project's
    # RankedLogger prepends a rank tag and does not pass through %-style logging
    # args cleanly, so we must not rely on lazy `%` substitution here.
    pg = diag.get("position_gradients")
    if pg:
        command_line_logger.info("=== Position-wise gradient signal ===")
        command_line_logger.info(
            f"{'code':<5}{'forget_grad':>16}{'retain_grad':>16}{'fr_cosine':>16}"
        )
        for p in pg["positions"]:
            command_line_logger.info(
                f"{p['code']:<5}{p['forget_grad_norm']:>16.6e}"
                f"{p['retain_grad_norm']:>16.6e}{p['forget_retain_cosine']:>16.4f}"
            )
        command_line_logger.info(
            f"strongest forget={pg.get('strongest_forget_code')}  "
            f"strongest conflict={pg.get('strongest_conflict_code')}"
        )
    cs = diag.get("code_sharing")
    if cs:
        command_line_logger.info(
            "=== Code sharing (retained items sharing target prefix) ==="
        )
        command_line_logger.info(
            f"{'prefix':<8}{'unique':>10}{'mean/target':>14}{'max/target':>12}"
        )
        for row in cs["by_prefix_length"]:
            tag = f"p={row['prefix_length']}" + ("(full)" if row["is_full_code"] else "")
            command_line_logger.info(
                f"{tag:<8}{row['shared_retained_total_unique']:>10d}"
                f"{row['per_target_mean']:>14.2f}{row['per_target_max']:>12d}"
            )


def diagnose(cfg: DictConfig) -> Dict[str, Any]:
    """Run the RQ-ID diagnostics driven by ``cfg`` (no weight update)."""
    with pipeline_launcher(cfg) as pipeline_modules:
        model = pipeline_modules.model
        if not isinstance(model, TigerUnlearningModule):
            raise TypeError(
                f"Expected `model._target_` to instantiate TigerUnlearningModule, "
                f"got {type(model).__name__}. Set it in your experiment yaml."
            )

        ckpt_path = cfg.get("ckpt_path", None)
        if not ckpt_path:
            raise ValueError(
                "ckpt_path is required -- pass the trained / poisoned TIGER checkpoint."
            )
        command_line_logger.info(f"Loading TIGER checkpoint from {ckpt_path}")
        source_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # spam_forget_manifest lets a clean-baseline ckpt be diagnosed against a
        # poison dataset's targets (mirrors the eval-metric convention).
        forget_manifest_path = cfg.get("spam_forget_manifest", None)

        command_line_logger.info(
            f"RQ-ID diagnosis mode={unlearning_cfg.get('diagnostics_mode', 'both')}: "
            f"data_dir={data_dir} semantic_id_path={semantic_id_path}"
        )
        diag = model.diagnose_rq_ids(
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
            forget_manifest_path=forget_manifest_path,
        )
        diag["ckpt_path"] = os.path.abspath(ckpt_path)

        out_path = os.path.join(cfg.paths.output_dir, "rq_id_diagnostics.json")
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(_json_safe(diag), f, indent=2)
        _print_summary(diag)
        command_line_logger.info(f"RQ-ID diagnostics -> {out_path}")
        return diag


@hydra.main(version_base="1.3", config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    diagnose(cfg)


if __name__ == "__main__":
    main()
