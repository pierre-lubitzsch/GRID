"""Hydra entry point for PKM memory-slot access diagnosis (read-only).

Answers the question that gates top-t memory-slot selection: do FORGET and
RETAIN interactions route to DISJOINT product-key-memory slots? If they overlap
almost completely, no selection criterion can separate forget from retain and
sparse-memory unlearning cannot be targeted.

Reports, per PKM module:

* coverage       -- how many of the ``n_keys ** 2`` slots each split reads
* touched_jaccard -- overlap of the read slot SETS (forget vs retain)
* forget_exclusive_slots -- forget-read slots the retain split NEVER reads.
  This is the decisive number: it bounds how much of the memory can be edited
  without touching anything retain depends on.
* two candidate selection scores, for a range of top-t cutoffs:
    AF      raw access frequency on the forget set (the access-count baseline,
            i.e. Sparse Memory Finetuning's TF term)
    AF-IHF  access frequency x inverse history frequency,
            AF(s) * log((T_r + 1) / (HF(s) + 1)) -- the recommender-side
            analogue of TF-IDF, scoring slots the forget data hits often and
            the retain ("history") data rarely does.

NO WEIGHTS ARE MODIFIED. The model is only run forward, with the PKM access
counters enabled (persistent=False buffers, so checkpoint schemas are untouched).

Run with::

    python -m src.diagnose_pkm_slots experiment=tiger_unlearn_scif_flat \
        data_dir=src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1 \
        semantic_id_path=embeddings/beauty/merged_predictions_tensor.pt \
        ckpt_path=.../checkpoints/checkpoint_best.ckpt \
        num_hierarchies=4 \
        'model.pkm_layers={encoder:[2,3],decoder:[0,1]}' model.pkm_mode=replace

Writes ``${paths.output_dir}/pkm_slot_diagnostics.json``.
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
    # Pre-format each line: the project's RankedLogger prepends a rank tag and
    # does not pass %-style args through cleanly.
    command_line_logger.info(
        f"=== PKM slot access: {diag['n_memories']} memories, "
        f"{diag['n_forget_batches']} forget / {diag['n_retain_batches']} retain batches ==="
    )
    command_line_logger.info(
        f"{'memory':<40}{'slots':>9}{'f_cov':>8}{'r_cov':>8}"
        f"{'jaccard':>9}{'f_excl':>9}{'f_excl%':>9}"
    )
    for name, e in diag["per_memory"].items():
        command_line_logger.info(
            f"{name[-40:]:<40}{e['n_slots']:>9d}{e['forget_coverage']:>8.3f}"
            f"{e['retain_coverage']:>8.3f}{e['touched_jaccard']:>9.3f}"
            f"{e['forget_exclusive_slots']:>9d}"
            f"{e['forget_exclusive_frac_of_forget']:>9.3f}"
        )
    command_line_logger.info("=== top-t selection: AF vs AF-IHF ===")
    command_line_logger.info(
        f"{'memory':<28}{'t':>6}{'AF~IHF ovl':>12}"
        f"{'AF r-unused':>13}{'AFIHF r-unused':>15}"
    )
    for name, e in diag["per_memory"].items():
        for t, r in e["top_t"].items():
            command_line_logger.info(
                f"{name[-28:]:<28}{t:>6}{r['af_vs_afihf_overlap']:>12.3f}"
                f"{r['af_top_retain_unused']:>13.3f}"
                f"{r['afihf_top_retain_unused']:>15.3f}"
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
            f"PKM slot diagnosis: data_dir={data_dir} "
            f"semantic_id_path={semantic_id_path}"
        )
        diag = model.diagnose_pkm_slots(
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

        out_path = os.path.join(cfg.paths.output_dir, "pkm_slot_diagnostics.json")
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(_json_safe(diag), f, indent=2)
        _print_summary(diag)
        command_line_logger.info(f"PKM slot diagnostics -> {out_path}")
        return diag


@hydra.main(version_base="1.3", config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    diagnose(cfg)


if __name__ == "__main__":
    main()
