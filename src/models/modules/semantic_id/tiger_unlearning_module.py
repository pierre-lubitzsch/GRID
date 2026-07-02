"""TIGER unlearning Lightning module with multi-algorithm dispatch."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader

from src.components.unlearning.filter_utils import (
    build_filter_mask,
    forbidden_sids_from_codebook,
    save_filter_mask,
    scan_user_forget_items,
)
from src.components.unlearning.fanchuan import fanchuan_unlearn
from src.components.unlearning.finetune import finetune_unlearn
from src.components.unlearning.hvp import batch_size as tiger_batch_size
from src.components.unlearning.kookmin import kookmin_unlearn
from src.components.unlearning.neighborhood_sampler import (
    build_retain_subset,
    collect_items_in_shards,
    load_codebook,
)
from src.components.unlearning.neg_train import neg_train_unlearn
from src.components.unlearning.scif import scif_unlearn
from src.components.unlearning.seif import seif_unlearn
from src.components.unlearning.target_params import (
    select_code_position_params,
    select_target_params,
)
from src.components.unlearning.unified import unified_unlearn
from src.data.loading.utils import assign_files_to_workers
from src.data.unlearning.deletion_spec import (
    load_forget_manifest,
    load_target_items,
    manifest_deletion_spec,
    resolve_neighborhood_centers,
    resolve_forget_manifest_path,
)
from src.data.unlearning.forget_target_filter import (
    default_item_mode_forget_subdir,
    default_item_pairs_forget_subdir,
    materialize_item_mode_forget_dir,
    materialize_item_pairs_forget_dir,
)
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)
from src.utils.file_utils import list_files

if TYPE_CHECKING:
    from src.data.loading.components.interfaces import SequenceDataloaderConfig


log = logging.getLogger(__name__)


class TigerUnlearningModule(SemanticIDEncoderDecoder):
    """Drop-in TIGER subclass exposing multiple unlearning algorithms."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def run_unlearning(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        train_dataloader_config: "SequenceDataloaderConfig",
        data_dir: str,
        forget_subdir: str,
        retain_subdir: str,
        retain_subset_dir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        seed: int = 2,
        num_hierarchies: Optional[int] = None,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        algorithm = str(unlearning_cfg.get("algorithm", "scif")).strip().lower()
        if algorithm == "retrain":
            raise ValueError(
                "algorithm='retrain' is an external baseline; use run_tiger_train.sh "
                "on cleaned/retain data."
            )
        if algorithm == "scif":
            return self.run_scif_unlearning(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm == "finetune":
            return self._run_finetune(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm == "neg_train":
            return self._run_neg_train(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm == "filter":
            return self._run_filter(
                unlearning_cfg=unlearning_cfg,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                output_dir=output_dir,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm == "unified":
            return self._run_unified(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        if algorithm in ("kookmin", "fanchuan", "seif"):
            runner = {
                "kookmin": self._run_kookmin,
                "fanchuan": self._run_fanchuan,
                "seif": self._run_seif,
            }[algorithm]
            return runner(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
        raise ValueError(f"Unknown unlearning algorithm={algorithm!r}")

    def run_scif_unlearning(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        train_dataloader_config: "SequenceDataloaderConfig",
        data_dir: str,
        forget_subdir: str,
        retain_subdir: str,
        retain_subset_dir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        seed: int = 2,
        num_hierarchies: Optional[int] = None,
        device: Optional[torch.device] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        device = device or next(self.parameters()).device
        ctx = self._prepare_unlearning_context(
            unlearning_cfg=unlearning_cfg,
            train_dataloader_config=train_dataloader_config,
            data_dir=data_dir,
            forget_subdir=forget_subdir,
            retain_subdir=retain_subdir,
            retain_subset_dir=retain_subset_dir,
            semantic_id_path=semantic_id_path,
            forget_size_hint=forget_size_hint,
            seed=seed,
            num_hierarchies=num_hierarchies,
            device=device,
            forget_manifest_path=forget_manifest_path,
        )
        forget_batches = ctx["forget_batches"]
        retain_batches = ctx["retain_batches"]
        t0 = time.time()
        cg_solution_max_norm = unlearning_cfg.get("cg_solution_max_norm")
        if cg_solution_max_norm is None:
            cg_solution_max_norm = unlearning_cfg.get("max_norm")
        update_max_norm = unlearning_cfg.get("update_max_norm", 1.0)

        # Position-wise intervention: optionally confine the update to selected
        # RQ-code positions' parameters (decoder heads + SID embedding rows).
        positions = _resolve_update_positions(
            unlearning_cfg.get("update_positions"),
            int(num_hierarchies or self.num_hierarchies),
        )
        scif_params = None
        scif_grad_masks = None
        if positions is not None:
            scif_params, scif_grad_masks = select_code_position_params(
                self,
                positions=positions,
                update_backbone=bool(
                    unlearning_cfg.get("update_positions_backbone", False)
                ),
            )
            log.info(
                "[scif] position-wise intervention: update_positions=%s "
                "(backbone=%s) -> %d param tensors",
                positions,
                bool(unlearning_cfg.get("update_positions_backbone", False)),
                len(scif_params),
            )

        info = scif_unlearn(
            model=self,
            forget_batches=forget_batches,
            retain_batches=retain_batches,
            forget_size=ctx["forget_size_for_scif"],
            retain_size=ctx["retain_size_full"],
            retain_samples_used_for_update=ctx["retain_samples_used_for_update"],
            cg_max_iter=int(unlearning_cfg.get("cg_max_iter", 200)),
            cg_tol=float(unlearning_cfg.get("cg_tol", 1e-5)),
            cg_damping=float(unlearning_cfg.get("damping", 0.01)),
            target_params_policy=str(unlearning_cfg.get("target_params", "all")),
            params=scif_params,
            grad_masks=scif_grad_masks,
            cg_solution_max_norm=cg_solution_max_norm,
            update_max_norm=update_max_norm,
            eval_mode=bool(unlearning_cfg.get("eval_mode", True)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        info["algorithm"] = "scif"
        info["update_positions"] = positions
        info["update_positions_backbone"] = (
            bool(unlearning_cfg.get("update_positions_backbone", False))
            if positions is not None
            else None
        )
        return info

    def diagnose_rq_ids(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        train_dataloader_config: "SequenceDataloaderConfig",
        data_dir: str,
        forget_subdir: str,
        retain_subdir: str,
        retain_subset_dir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        seed: int = 2,
        num_hierarchies: Optional[int] = None,
        device: Optional[torch.device] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """RQ-ID diagnosis (read-only; no weight update).

        Runs the position-wise gradient signal analysis and/or the static
        code-sharing analysis selected by ``unlearning_cfg['diagnostics_mode']``
        (``both`` | ``positions`` | ``code_sharing``) and returns a combined
        report dict. ``positions`` reuses ``_prepare_unlearning_context`` to
        build the same forget/retain batches the unlearning algorithms use.
        """
        device = device or next(self.parameters()).device
        H = int(num_hierarchies or self.num_hierarchies)
        mode = str(unlearning_cfg.get("diagnostics_mode", "both")).strip().lower()
        if mode not in ("both", "positions", "code_sharing"):
            raise ValueError(
                f"diagnostics_mode={mode!r} must be both|positions|code_sharing"
            )

        manifest_path = forget_manifest_path or resolve_forget_manifest_path(data_dir)
        diag: Dict[str, Any] = {
            "num_hierarchies": H,
            "diagnostics_mode": mode,
            "data_dir": os.path.abspath(data_dir),
            "forget_manifest_path": manifest_path,
            "semantic_id_path": (
                os.path.abspath(semantic_id_path) if semantic_id_path else None
            ),
        }

        if mode in ("positions", "both"):
            from src.components.unlearning.position_diagnostics import (
                per_position_gradient_report,
            )

            ctx = self._prepare_unlearning_context(
                unlearning_cfg=unlearning_cfg,
                train_dataloader_config=train_dataloader_config,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                retain_subdir=retain_subdir,
                retain_subset_dir=retain_subset_dir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                seed=seed,
                num_hierarchies=num_hierarchies,
                device=device,
                forget_manifest_path=forget_manifest_path,
            )
            params = select_target_params(
                self, policy=str(unlearning_cfg.get("target_params", "tiger"))
            )
            diag["position_gradients"] = per_position_gradient_report(
                self,
                ctx["forget_batches"],
                ctx["retain_batches"],
                params,
                num_hierarchies=H,
                eval_mode=bool(unlearning_cfg.get("eval_mode", True)),
            )
            diag["target_items"] = ctx["meta"]["target_items"]
            diag["forget_size"] = ctx["forget_size_for_scif"]

        if mode in ("code_sharing", "both"):
            from src.components.unlearning.code_sharing import code_sharing_report

            if not semantic_id_path:
                raise ValueError(
                    "code-sharing analysis requires semantic_id_path (the RQ-ID "
                    "tensor) to be set"
                )
            targets = load_target_items(load_forget_manifest(manifest_path))
            diag["code_sharing"] = code_sharing_report(
                semantic_id_path, sorted(targets), num_hierarchies=H
            )

        return diag

    def _run_finetune(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        info = finetune_unlearn(
            self,
            ctx["retain_batches"],
            steps=int(cfg.get("finetune_steps", 500)),
            lr=float(cfg.get("finetune_lr", 1e-3)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_neg_train(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        info = neg_train_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            steps=int(cfg.get("neg_train_steps", 200)),
            lr=float(cfg.get("neg_train_lr", 1e-3)),
            neg_retain_every=int(cfg.get("neg_retain_every", 5)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_unified(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        local_repair = cfg.get("local_repair") or {}
        n_epochs_cfg = cfg.get("n_epochs")
        info = unified_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            steps=int(cfg.get("unified_steps", 500)),
            n_epochs=(
                int(n_epochs_cfg) if n_epochs_cfg is not None else None
            ),
            lr=float(cfg.get("unified_lr", 1e-4)),
            lambda_forget=float(cfg.get("lambda_forget", 1.0)),
            lambda_sep=float(cfg.get("lambda_sep", 0.1)),
            forget_loss_level=str(cfg.get("forget_loss_level", "token")),
            sep_temperature=float(cfg.get("sep_temperature", 0.07)),
            deletion_spec=ctx["deletion_spec"],
            forget_item_ids=ctx["visible_forget_items"],
            neighbor_item_ids=ctx["neighborhood_centers"],
            sep_negative_item_ids=ctx["sep_negative_items"],
            sep_negatives_mode=str(cfg.get("sep_negatives", "forget")),
            local_repair_cfg=local_repair,
            restrict_adaptive_codes=bool(cfg.get("adaptive_codes", False)),
            stable_codes=int(cfg.get("stable_codes", 2)),
            adaptive_update_backbone=bool(cfg.get("adaptive_update_backbone", False)),
            adaptive_adapter=bool(cfg.get("adaptive_adapter", False)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_kookmin(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        info = kookmin_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            init_rate=float(cfg.get("kookmin_init_rate", 0.01)),
            neg_grad_sample_size=int(cfg.get("kookmin_neg_grad_sample_size", 128)),
            retain_epochs=int(cfg.get("kookmin_retain_epochs", 1)),
            retain_lr=float(cfg.get("kookmin_retain_lr", 1e-3)),
            scale_for_reinit_params=float(cfg.get("kookmin_scale_for_reinit", 10.0)),
            target_params_policy=str(cfg.get("target_params", "all")),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_fanchuan(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        info = fanchuan_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            lr=float(cfg.get("fanchuan_lr", 1e-3)),
            uniform_epochs=int(cfg.get("fanchuan_uniform_epochs", 1)),
            contrastive_iters=int(cfg.get("fanchuan_contrastive_iters", 8)),
            contrastive_temperature=float(
                cfg.get("fanchuan_contrastive_temperature", 1.15)
            ),
            retain_epochs_per_iter=int(cfg.get("fanchuan_retain_epochs_per_iter", 1)),
            seed=int(kwargs.get("seed", 2)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_seif(self, **kwargs: Any) -> Dict[str, Any]:
        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()
        keywords = cfg.get("seif_noise_param_keywords")
        # `unlearning.n_epochs`, when set, is the shared "number of passes" knob
        # and overrides the per-algorithm `seif_repair_epochs` default.
        n_epochs_cfg = cfg.get("n_epochs")
        repair_epochs = (
            int(n_epochs_cfg)
            if n_epochs_cfg is not None
            else int(cfg.get("seif_repair_epochs", 4))
        )
        info = seif_unlearn(
            self,
            ctx["retain_batches"],
            ctx["forget_batches"],
            erase_std=float(cfg.get("seif_erase_std", 0.6)),
            erase_std_final=float(cfg.get("seif_erase_std_final", 0.005)),
            repair_epochs=repair_epochs,
            repair_lr=float(cfg.get("seif_repair_lr", 7e-4)),
            weight_decay=float(cfg.get("seif_weight_decay", 5e-4)),
            noise_param_keywords=list(keywords) if keywords else None,
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        return info

    def _run_filter(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        data_dir: str,
        forget_subdir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        output_dir: Optional[str] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        manifest_path = forget_manifest_path or resolve_forget_manifest_path(data_dir)
        manifest = load_forget_manifest(manifest_path)
        deletion_spec = manifest_deletion_spec(
            manifest, unlearning_cfg.get("deletion_spec")
        )
        target_items = load_target_items(manifest)
        forget_dir = os.path.join(data_dir, forget_subdir)
        forget_shard_items = collect_items_in_shards(_list_shards_safe(forget_dir))
        visible_forget = resolve_neighborhood_centers(
            deletion_spec=deletion_spec,
            forget_shard_items=forget_shard_items,
            target_items=target_items,
        )
        filter_mode = str(unlearning_cfg.get("filter_mode", "global"))
        user_map = (
            scan_user_forget_items(forget_dir)
            if filter_mode == "user_dependent"
            else None
        )
        mask = build_filter_mask(
            deletion_spec=deletion_spec,
            target_items=target_items,
            forget_shard_items=forget_shard_items,
            filter_mode=filter_mode,
            user_forget_items=user_map,
        )
        if semantic_id_path:
            codebook = load_codebook(semantic_id_path)
            forbidden_sids = forbidden_sids_from_codebook(
                codebook, mask["forbidden_item_ids"]
            )
            self.set_decode_filter(
                forbidden_sids=forbidden_sids,
                filter_mode=filter_mode,
                user_forbidden_items=user_map,
            )
        mask_path = os.path.join(output_dir or ".", "filter_mask.json")
        save_filter_mask(mask, mask_path)
        return {
            "algorithm": "filter",
            "deletion_spec": deletion_spec,
            "filter_mode": filter_mode,
            "filter_mask_path": os.path.abspath(mask_path),
            "n_forbidden_items": len(mask["forbidden_item_ids"]),
            "forget_size_input": forget_size_hint,
            "visible_forget_items": sorted(visible_forget),
        }

    def _prepare_unlearning_context(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        train_dataloader_config: "SequenceDataloaderConfig",
        data_dir: str,
        forget_subdir: str,
        retain_subdir: str,
        retain_subset_dir: str,
        semantic_id_path: Optional[str],
        forget_size_hint: Optional[int] = None,
        seed: int = 2,
        num_hierarchies: Optional[int] = None,
        device: Optional[torch.device] = None,
        forget_manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        device = device or next(self.parameters()).device
        manifest_path = forget_manifest_path or resolve_forget_manifest_path(data_dir)
        manifest = load_forget_manifest(manifest_path)
        deletion_spec = manifest_deletion_spec(
            manifest, unlearning_cfg.get("deletion_spec")
        )
        target_items = load_target_items(manifest)

        forget_dir = os.path.join(data_dir, forget_subdir)
        retain_dir = os.path.join(data_dir, retain_subdir)

        if deletion_spec == "item" and target_items:
            filtered_subdir = default_item_mode_forget_subdir(forget_subdir)
            filtered_dir = os.path.join(data_dir, filtered_subdir)
            if not _list_shards_safe(filtered_dir):
                materialize_item_mode_forget_dir(
                    forget_dir=forget_dir,
                    out_dir=filtered_dir,
                    target_items=target_items,
                    rows_per_shard=int(unlearning_cfg.get("rows_per_shard", 4096)),
                )
            forget_dir = filtered_dir

        if deletion_spec == "item_pairs" and target_items:
            item_pairs_subdir = default_item_pairs_forget_subdir(forget_subdir)
            item_pairs_dir = os.path.join(data_dir, item_pairs_subdir)
            unlearn_whole_items = bool(unlearning_cfg.get("unlearn_whole_items", False))
            extra_dirs: Optional[List[str]] = [retain_dir] if unlearn_whole_items else None
            if not _list_shards_safe(item_pairs_dir):
                log.info(
                    "[item_pairs] materialising (prefix→target) pairs "
                    "from %s (unlearn_whole_items=%s)",
                    forget_dir,
                    unlearn_whole_items,
                )
                materialize_item_pairs_forget_dir(
                    forget_dir=forget_dir,
                    out_dir=item_pairs_dir,
                    target_items=target_items,
                    extra_source_dirs=extra_dirs,
                    rows_per_shard=int(unlearning_cfg.get("rows_per_shard", 4096)),
                )
            forget_dir = item_pairs_dir

        if forget_size_hint is None:
            forget_size_hint = _count_rows_in_tfrecord_dir(forget_dir)
        if forget_size_hint <= 0:
            raise ValueError(f"Could not infer |D_f| from {forget_dir}")

        forget_shard_items = collect_items_in_shards(_list_shards_safe(forget_dir))
        visible_forget = resolve_neighborhood_centers(
            deletion_spec=deletion_spec,
            forget_shard_items=forget_shard_items,
            target_items=target_items,
        )

        neighborhood_aware = bool(unlearning_cfg.get("neighborhood_aware", False))
        subset_info = build_retain_subset(
            forget_dir=os.path.join(data_dir, forget_subdir),
            retain_dir=retain_dir,
            out_dir=retain_subset_dir,
            neighborhood_aware=neighborhood_aware,
            semantic_id_path=semantic_id_path,
            sid_prefix_length=int(unlearning_cfg.get("sid_prefix_length", 2)),
            forget_size=forget_size_hint,
            neighbor_aware_factor=float(unlearning_cfg.get("neighbor_aware_factor", 8.0)),
            retain_samples_used_for_update=int(
                unlearning_cfg.get("retain_samples_used_for_update") or 16
            ),
            retain_sample_size=unlearning_cfg.get("retain_sample_size"),
            repair_sample_bound=unlearning_cfg.get("repair_sample_bound"),
            retain_max_rows=unlearning_cfg.get("retain_max_rows"),
            progressive_sid_prefix=bool(unlearning_cfg.get("progressive_sid_prefix", True)),
            neighborhood_aware_sample_rate=float(
                unlearning_cfg.get("neighborhood_aware_sample_rate", 1.0)
            ),
            neighborhood_method=str(unlearning_cfg.get("neighborhood_method", "prefix")),
            embedding_path=unlearning_cfg.get("embedding_path"),
            embedding_epsilon=unlearning_cfg.get("embedding_epsilon"),
            embedding_max_neighbors=int(
                unlearning_cfg.get("embedding_max_neighbors", 100)
            ),
            deletion_spec=deletion_spec,
            target_items=target_items if deletion_spec in ("item", "item_pairs") else None,
            num_hierarchies=num_hierarchies,
            rows_per_shard=int(unlearning_cfg.get("rows_per_shard", 4096)),
            seed=int(seed),
            overwrite=True,
        )

        unlearn_batch_size = unlearning_cfg.get("batch_size_per_device")
        forget_loader = _build_finite_loader(
            base_train_cfg=train_dataloader_config,
            data_folder=forget_dir,
            batch_size_per_device_override=unlearn_batch_size,
        )
        retain_loader = _build_finite_loader(
            base_train_cfg=train_dataloader_config,
            data_folder=retain_subset_dir,
            batch_size_per_device_override=unlearn_batch_size,
        )
        forget_batches = _drain_loader(forget_loader, device=device)
        retain_batches = _drain_loader(retain_loader, device=device)
        if not forget_batches:
            raise RuntimeError(f"No forget batches from {forget_dir}")
        if not retain_batches:
            raise RuntimeError(f"No retain batches from {retain_subset_dir}")

        retain_size_full = _count_rows_in_tfrecord_dir(retain_dir)
        retain_samples_used = int(unlearning_cfg.get("retain_samples_used_for_update") or 16)

        sep_negative_items = self._sample_sep_random_negatives(
            unlearning_cfg=unlearning_cfg,
            retain_dir=retain_dir,
            exclude_items=visible_forget | target_items,
            default_count=len(visible_forget),
            target_items=target_items,
            seed=int(seed),
        )

        return {
            "forget_batches": forget_batches,
            "retain_batches": retain_batches,
            "forget_size_for_scif": int(forget_size_hint),
            "retain_size_full": int(retain_size_full),
            "retain_samples_used_for_update": retain_samples_used,
            "deletion_spec": deletion_spec,
            "visible_forget_items": visible_forget,
            "neighborhood_centers": visible_forget,
            "sep_negative_items": sep_negative_items,
            "meta": {
                "forget_size_input": int(forget_size_hint),
                "forget_size_augmented": sum(tiger_batch_size(b) for b in forget_batches),
                "retain_size_augmented": sum(tiger_batch_size(b) for b in retain_batches),
                "retain_size_full": int(retain_size_full),
                "retain_subset": subset_info,
                "neighborhood_aware": neighborhood_aware,
                "deletion_spec": deletion_spec,
                "target_items": sorted(target_items),
            },
        }

    def _sample_sep_random_negatives(
        self,
        *,
        unlearning_cfg: Dict[str, Any],
        retain_dir: str,
        exclude_items: Set[int],
        default_count: int,
        target_items: Optional[Set[int]] = None,
        seed: int,
    ) -> Optional[Set[int]]:
        """Resolve the sep-loss negative item ids from ``sep_negatives``.

        Modes:

        * ``forget`` (default) / ``neighbors`` (legacy alias) → returns ``None``;
          the caller then uses all visible forget items ``I_f`` as negatives
          (every distinct item in the forget shards under ``deletion_spec=session``).
        * ``forget_target_only`` → returns exactly the manifest ``target_items``
          (the ``n_target`` spam targets), independent of ``deletion_spec``.
        * ``random_retain`` → random retain-set item ids (ablation): every item
          id in the retain shards minus forget/target items, sampled to
          ``default_count`` (``sep_num_random_negatives`` overrides). The item
          pool is cached per resolved retain dir so symlinked sequential request
          dirs scan once.
        """
        mode = str(unlearning_cfg.get("sep_negatives", "forget")).strip().lower()
        # 'forget' (slide default, I_f only); 'neighbors' kept as a legacy alias
        # for the same forget-only behavior (neighbors are no longer negatives).
        if mode in ("", "forget", "neighbors"):
            return None
        if mode == "forget_target_only":
            targets = set(target_items or [])
            if not targets:
                raise ValueError(
                    "sep_negatives='forget_target_only' requires target_items in "
                    "the forget manifest (none found)"
                )
            log.info(
                "[sep_negatives] forget_target_only: using %d target item(s) as "
                "sep-loss negatives",
                len(targets),
            )
            return targets
        if mode != "random_retain":
            raise ValueError(
                "sep_negatives must be 'forget', 'forget_target_only', or "
                f"'random_retain', got {mode!r}"
            )
        pool_key = os.path.realpath(retain_dir)
        cache = getattr(self, "_retain_item_pool_cache", None)
        if cache is None or cache[0] != pool_key:
            cache = (pool_key, collect_items_in_shards(_list_shards_safe(retain_dir)))
            self._retain_item_pool_cache = cache
        pool = sorted(cache[1] - exclude_items)
        if not pool:
            log.warning(
                "[sep_negatives] empty retain item pool after excluding "
                "forget/target items; falling back to neighbor negatives"
            )
            return None
        n_cfg = unlearning_cfg.get("sep_num_random_negatives")
        n = int(n_cfg) if n_cfg is not None else int(default_count)
        n = max(1, min(n, len(pool)))
        sampled = set(random.Random(seed).sample(pool, n))
        log.info(
            "[sep_negatives] random_retain: sampled %d of %d retain items "
            "as sep-loss negatives (default_count=%d)",
            len(sampled),
            len(pool),
            default_count,
        )
        return sampled


def _resolve_update_positions(
    value: Any, num_hierarchies: int
) -> Optional[List[int]]:
    """Parse the ``unlearning.update_positions`` knob into 0-based hierarchy ids.

    Accepts:
      * ``None`` / ``"all"`` / ``"null"`` / ``"none"`` / ``""`` -> ``None`` (no
        restriction).
      * a list of 0-based indices, e.g. ``[0, 1]`` (== c1, c2).
      * a list / comma string of code names, e.g. ``["c1", "c2"]`` or
        ``"c1,c2"`` (1-based names mapped to 0-based indices).
      * a comma/space string of indices, e.g. ``"2,3"``.

    Returns a sorted list of distinct indices, or ``None`` when the selection is
    empty or equals the full set ``{0..H-1}`` (both mean "update everything").
    Do not mix bare numbers and ``cN`` names in one list (numbers are 0-based,
    ``cN`` is 1-based).
    """
    if value is None:
        return None
    items: Any = value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "all", "null", "none"):
            return None
        items = [tok for tok in re.split(r"[,\s]+", s.strip("[]() ")) if tok]
    idxs: List[int] = []
    for it in items:
        if isinstance(it, str):
            tok = it.strip().lower()
            if tok.startswith("c"):
                idxs.append(int(tok[1:]) - 1)  # c1 -> 0
            else:
                idxs.append(int(tok))
        else:
            idxs.append(int(it))
    idxs = sorted(set(idxs))
    if not idxs:
        return None
    for h in idxs:
        if not 0 <= h < num_hierarchies:
            raise ValueError(
                f"update_positions index {h} out of range [0, {num_hierarchies})"
            )
    if len(idxs) == num_hierarchies:
        return None  # full set == no restriction
    return idxs


def _list_shards_safe(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".tfrecord.gz")
    ]


def _count_rows_in_tfrecord_dir(directory: str) -> int:
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
    except Exception as ex:
        raise RuntimeError(f"TensorFlow is required to count TFRecord rows: {ex}")

    shards = _list_shards_safe(directory)
    n = 0
    for path in shards:
        for _ in tf.data.TFRecordDataset([path], compression_type="GZIP"):
            n += 1
    return n


def _build_finite_loader(
    base_train_cfg: "SequenceDataloaderConfig",
    data_folder: str,
    batch_size_per_device_override: Optional[int] = None,
) -> DataLoader:
    cfg = deepcopy(base_train_cfg)
    cfg.data_folder = data_folder
    if batch_size_per_device_override is not None:
        cfg.batch_size_per_device = int(batch_size_per_device_override)

    suffix_provider = cfg.dataset_config.data_iterator
    file_suffix = (
        getattr(cfg.dataset_config, "file_format", None)
        or suffix_provider.get_file_suffix()
    )
    files = list_files(folder_path=data_folder, suffix=f"*{file_suffix}")
    file_map, _ = assign_files_to_workers(
        list_of_files=files,
        total_workers=1,
        assign_by_size=False,
        should_shuffle_rows=False,
        assign_all_files_per_worker=False,
    )

    dataset = cfg.dataset_class(
        dataset_config=cfg.dataset_config,
        data_folder=data_folder,
        should_shuffle_rows=False,
        batch_size=cfg.batch_size_per_device,
        is_for_training=False,
        assign_all_files_per_worker=False,
    )
    dataset.set_list_of_files(list_of_files=file_map.get(0, []))
    dataset.set_distributed_params(total_workers=1, global_worker_id=0)

    collate_fn_partial = partial(
        cfg.collate_fn,
        labels=cfg.labels,
        sequence_length=cfg.sequence_length,
        masking_token=cfg.masking_token,
        padding_token=cfg.padding_token,
        oov_token=cfg.get("oov_token", None) if hasattr(cfg, "get") else None,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=(
            cfg.batch_size_per_device if cfg.dataset_config.iterate_per_row else None
        ),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn_partial,
        timeout=0,
    )


def _drain_loader(loader: DataLoader, device: torch.device) -> List[Any]:
    from src.components.unlearning.hvp import batch_to_device

    out: List[Any] = []
    for batch in loader:
        out.append(batch_to_device(batch, device))
    return out


def save_unlearned_checkpoint(
    *,
    model: TigerUnlearningModule,
    out_path: str,
    source_ckpt: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": _safe_lightning_version(),
        "callbacks": {},
        "optimizer_states": [],
        "lr_schedulers": [],
        "hparams_name": "kwargs",
        "hyper_parameters": {},
    }
    if source_ckpt is not None:
        for key in (
            "epoch",
            "global_step",
            "pytorch-lightning_version",
            "callbacks",
            "optimizer_states",
            "lr_schedulers",
            "hparams_name",
            "hyper_parameters",
        ):
            if key in source_ckpt:
                payload[key] = source_ckpt[key]
    if extra_metadata:
        payload["unlearning_metadata"] = _json_safe(extra_metadata)
        payload["scif_metadata"] = _json_safe(extra_metadata)
    torch.save(payload, out_path)
    log.info("[unlearn] saved unlearned checkpoint -> %s", out_path)


def _safe_lightning_version() -> str:
    try:
        import lightning

        return getattr(lightning, "__version__", "unknown")
    except Exception:
        return "unknown"


def _json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)
