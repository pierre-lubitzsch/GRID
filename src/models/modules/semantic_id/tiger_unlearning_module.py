"""TIGER unlearning Lightning module with multi-algorithm dispatch."""

from __future__ import annotations

import json
import logging
import os
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
from src.components.unlearning.finetune import finetune_unlearn
from src.components.unlearning.hvp import batch_size as tiger_batch_size
from src.components.unlearning.neighborhood_sampler import (
    build_retain_subset,
    collect_items_in_shards,
    load_codebook,
)
from src.components.unlearning.neg_train import neg_train_unlearn
from src.components.unlearning.scif import scif_unlearn
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
            )
        if algorithm == "filter":
            return self._run_filter(
                unlearning_cfg=unlearning_cfg,
                data_dir=data_dir,
                forget_subdir=forget_subdir,
                semantic_id_path=semantic_id_path,
                forget_size_hint=forget_size_hint,
                output_dir=output_dir,
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
        )
        forget_batches = ctx["forget_batches"]
        retain_batches = ctx["retain_batches"]
        t0 = time.time()
        cg_solution_max_norm = unlearning_cfg.get("cg_solution_max_norm")
        if cg_solution_max_norm is None:
            cg_solution_max_norm = unlearning_cfg.get("max_norm")
        update_max_norm = unlearning_cfg.get("update_max_norm", 1.0)

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
            cg_solution_max_norm=cg_solution_max_norm,
            update_max_norm=update_max_norm,
            eval_mode=bool(unlearning_cfg.get("eval_mode", True)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info.update(ctx["meta"])
        info["algorithm"] = "scif"
        return info

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
        info = unified_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            steps=int(cfg.get("unified_steps", 500)),
            lr=float(cfg.get("unified_lr", 1e-4)),
            lambda_forget=float(cfg.get("lambda_forget", 1.0)),
            lambda_sep=float(cfg.get("lambda_sep", 0.1)),
            forget_loss_level=str(cfg.get("forget_loss_level", "token")),
            sep_temperature=float(cfg.get("sep_temperature", 0.07)),
            deletion_spec=ctx["deletion_spec"],
            forget_item_ids=ctx["visible_forget_items"],
            neighbor_item_ids=ctx["neighborhood_centers"],
            local_repair_cfg=local_repair,
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
    ) -> Dict[str, Any]:
        manifest_path = resolve_forget_manifest_path(data_dir)
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
    ) -> Dict[str, Any]:
        device = device or next(self.parameters()).device
        manifest_path = resolve_forget_manifest_path(data_dir)
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

        return {
            "forget_batches": forget_batches,
            "retain_batches": retain_batches,
            "forget_size_for_scif": int(forget_size_hint),
            "retain_size_full": int(retain_size_full),
            "retain_samples_used_for_update": retain_samples_used,
            "deletion_spec": deletion_spec,
            "visible_forget_items": visible_forget,
            "neighborhood_centers": visible_forget,
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
