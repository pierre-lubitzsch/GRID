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
    build_sorted_sid_index,
    closest_prefix_neighbors,
    collect_items_in_shards,
    load_codebook,
    load_dense_embeddings,
    topk_embedding_neighbors,
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

# Algorithms that read unlearning.update_scope. 'scif' is excluded (no second
# derivative through PKM's EmbeddingBag) and 'filter' performs no weight update.
_PKM_SCOPE_ALGOS = frozenset(
    {"unified", "finetune", "neg_train", "kookmin", "fanchuan", "seif"}
)


def _resolve_optimizer(cfg, algo: str, default: str = "adam") -> str:
    """Optimizer name for `algo`, honouring a global fallback.

    Precedence: unlearning.<algo>_optimizer  >  unlearning.optimizer  > default.
    The per-algorithm keys predate the global one and are kept so recorded
    commands keep their meaning; the global key is what makes "optimizer" a real
    experiment axis instead of something only three algorithms respect.
    """
    specific = cfg.get(f"{algo}_optimizer")
    if specific:
        return str(specific)
    shared = cfg.get("optimizer")
    return str(shared) if shared else default


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
        # update_scope=pkm_only is only honoured by the algorithms wired for it.
        # Fail LOUDLY rather than silently performing a full-model update that
        # would be recorded as a memory-only result.
        _scope = str(unlearning_cfg.get("update_scope", "all") or "all").strip().lower()
        if _scope not in ("all", "pkm_only", "ffn_only"):
            raise ValueError(
                f"unlearning.update_scope must be 'all', 'pkm_only' or "
                f"'ffn_only', got {_scope!r}"
            )
        if _scope in ("pkm_only", "ffn_only") and algorithm not in _PKM_SCOPE_ALGOS:
            raise ValueError(
                f"unlearning.update_scope={_scope!r} is not supported for "
                f"algorithm={algorithm!r}. Supported: {sorted(_PKM_SCOPE_ALGOS)}. "
                + ("'scif' cannot work on PKM at all: its HVP needs a second "
                   "derivative and PKM's EmbeddingBag has none. "
                   if algorithm == "scif" else "")
                + ("'filter' performs no weight update (it masks forbidden SIDs "
                   "at decode time), so a parameter scope is meaningless. "
                   if algorithm == "filter" else "")
            )
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
        if algorithm in ("kookmin", "fanchuan", "seif", "tracer"):
            runner = {
                "kookmin": self._run_kookmin,
                "fanchuan": self._run_fanchuan,
                "seif": self._run_seif,
                "tracer": self._run_tracer,
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
            # Same 'modular stabilizer' scope as unified. With a PKM installed
            # over a checkpoint that was trained WITHOUT it (replace mode), this
            # is the ablate-then-repair setup; finetune_steps=0 measures the
            # pure ablation with no repair at all.
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "finetune"),
            patience=int(cfg.get("finetune_patience", 0) or 0),
            min_delta=float(cfg.get("finetune_min_delta", 0.0) or 0.0),
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
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "neg_train"),
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

        # Coherence loss L_n (TRACER Eq. 9): precompute, per forget batch, the
        # prefix-neighbour semantic ids of each sample's target item.
        lambda_neighborhood = float(cfg.get("lambda_n", 0.0))
        coherence_neighbors = None
        if lambda_neighborhood != 0.0:
            coherence_neighbors = self._build_coherence_neighbors(
                forget_batches=ctx["forget_batches"],
                semantic_id_path=kwargs.get("semantic_id_path"),
                num_hierarchies=kwargs.get("num_hierarchies"),
                neighborhood_count=int(cfg.get("neighborhood_count", 4)),
                neighborhood_prefix_length=int(
                    cfg.get("neighborhood_prefix_length", 2)
                ),
                exclude_items=ctx["visible_forget_items"],
                # 'target_only' (default) restricts L_n to forget rows whose
                # label IS a deletion target. 'all' reproduces the pre-2026-08-05
                # behaviour, where >=92% of the term boosted neighbours of
                # popular filler items instead (see _build_coherence_neighbors).
                coherence_rows=str(cfg.get("coherence_rows", "target_only")),
                target_items=set(ctx["meta"].get("target_items") or []),
                # 'embedding' defines P(i_T) as a fixed-size top-k in the
                # pre-quantization space: never empty, independent of codebook
                # width, and the ground truth that a prefix bucket only
                # approximates (0.24-0.68 overlap, sid_fidelity.json).
                neighbor_method=str(
                    cfg.get("coherence_neighbor_method", "prefix")
                ),
                embedding_path=cfg.get("embedding_path"),
                embedding_metric=str(cfg.get("coherence_embedding_metric", "cosine")),
            )

        info = unified_unlearn(
            self,
            ctx["forget_batches"],
            ctx["retain_batches"],
            steps=int(cfg.get("unified_steps", 500)),
            n_epochs=(
                int(n_epochs_cfg) if n_epochs_cfg is not None else None
            ),
            lr=float(cfg.get("unified_lr", 1e-4)),
            lambda_forget=float(cfg.get("lambda_f", 1.0)),
            lambda_sep=float(cfg.get("lambda_s", 0.1)),
            lambda_neighborhood=lambda_neighborhood,
            coherence_neighbors=coherence_neighbors,
            # 'mass' (logsumexp over the neighbourhood) is the bounded, feasible
            # form; 'nll' is the original per-neighbour TRACER Eq. 9 whose optimum
            # needs every neighbour at probability 1 (see compute_coherence_loss).
            coherence_loss_type=str(cfg.get("coherence_loss_type", "nll")),
            coherence_mass_cap=float(cfg.get("coherence_mass_cap", 0.999)),
            forget_loss_level=str(cfg.get("forget_loss_level", "token")),
            sep_temperature=float(cfg.get("sep_temperature", 0.07)),
            deletion_spec=ctx["deletion_spec"],
            forget_item_ids=ctx["visible_forget_items"],
            neighbor_item_ids=ctx["neighborhood_centers"],
            sep_negative_item_ids=ctx["sep_negative_items"],
            sep_negatives_mode=str(cfg.get("sep_negatives", "forget_target_only")),
            # history (default, back-compatible but tautological: r_u IS the mean
            # of the history items) | label (the true next item; a real positive).
            sep_positives=str(cfg.get("sep_positives", "history")),
            # cosine (default) = pooled-encoder similarity; generative = the
            # model's own sequence log-prob, i.e. on the generation path.
            sep_loss_type=str(cfg.get("sep_loss_type", "cosine")),
            sep_gen_temperature=float(cfg.get("sep_gen_temperature", 1.0)),
            local_repair_cfg=local_repair,
            restrict_adaptive_codes=bool(cfg.get("adaptive_codes", False)),
            stable_codes=int(cfg.get("stable_codes", 2)),
            adaptive_update_backbone=bool(cfg.get("adaptive_update_backbone", False)),
            adaptive_adapter=bool(cfg.get("adaptive_adapter", False)),
            # Position-wise intervention (same unlearning.update_positions knob
            # as SCIF): confine the unified update to an arbitrary subset of RQ
            # code positions, e.g. [0] = only c1 moves. Overrides adaptive_codes.
            update_positions=_resolve_update_positions(
                cfg.get("update_positions"),
                num_hierarchies=int(self.num_hierarchies),
            ),
            update_positions_backbone=bool(
                cfg.get("update_positions_backbone", False)
            ),
            # "Modular stabilizer" scope: update_scope='pkm_only' keeps ONLY the
            # Product-Key Memory in the optimizer (backbone/SID/heads frozen), so
            # forgetting must be expressible as a sparse-memory edit. Requires a
            # PKM-bearing model (pass model.pkm_layers / model.pkm_mode).
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            # top-t memory-slot restriction (requires update_scope=pkm_only)
            slot_selection=str(cfg.get("slot_selection", "none")),
            slot_top_t=int(cfg.get("slot_top_t", 32)),
            slot_lambda=float(cfg.get("slot_lambda", 1.0)),
            slot_mu=float(cfg.get("slot_mu", 5.0)),
            slot_dot_abs=bool(cfg.get("slot_dot_abs", False)),
            optimizer=_resolve_optimizer(cfg, "unified"),
            # 1.0 = one lr for every parameter, the value every recorded run
            # used. Lower it (0.1 / 0.01) to slow the identifier space down.
            code_lr_scale=float(cfg.get("code_lr_scale", 1.0)),
            # Extra multiplier on the ADAPTIVE tail [stable_codes, H) only, on
            # top of code_lr_scale. 1.0 = one code group (previous behaviour).
            adaptive_code_lr_scale=float(cfg.get("adaptive_code_lr_scale", 1.0)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info["update_scope"] = str(cfg.get("update_scope", "all"))
        info["optimizer"] = str(cfg.get("unified_optimizer", "adam"))
        info.update(ctx["meta"])
        return info

    def _run_tracer(self, **kwargs: Any) -> Dict[str, Any]:
        """TRACER (arXiv:2606.07688) -- token reassignment, as a baseline.

        Needs the RQ-KMeans codebook the semantic ids were built from
        (``unlearning.tracer_rqkmeans_ckpt``) plus the pre-quantization item
        embeddings (``unlearning.embedding_path``); ``phi=0`` is asserted to
        reproduce the stored codes before anything is trained, because a codebook
        that does not match the SID tensor would silently reassign items before
        unlearning even starts.
        """
        import torch as _torch

        from src.components.unlearning.tracer import tracer_unlearn
        from src.components.unlearning.tracer_tokenizer import (
            assert_reproduces_sids,
            compute_residuals,
            load_rq_centroids,
        )
        from src.components.unlearning.neighborhood_sampler import load_dense_embeddings

        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device
        cfg = kwargs["unlearning_cfg"]
        t0 = time.time()

        ckpt = cfg.get("tracer_rqkmeans_ckpt")
        if not ckpt:
            raise ValueError(
                "algorithm=tracer requires unlearning.tracer_rqkmeans_ckpt (the "
                "RQ-KMeans checkpoint holding the codeword centroids). The "
                "original width-256/L4 beauty codebook no longer exists, so use "
                "an identifier space whose checkpoint survives (w16, w8l6, L8)."
            )
        emb_path = cfg.get("embedding_path")
        if not emb_path:
            raise ValueError("algorithm=tracer requires unlearning.embedding_path")

        num_hierarchies = int(kwargs.get("num_hierarchies") or self.num_hierarchies)
        n_levels = int(cfg.get("tracer_levels") or (num_hierarchies - 1))
        centroids = load_rq_centroids(str(ckpt), n_levels=n_levels)
        centroids = _torch.stack([c for c in centroids])              # [L, K, D]

        codes = self.codebooks.t().to(_torch.long).cpu()               # [H, N]
        n_items = int(codes.shape[1])

        # load_dense_embeddings returns a DenseEmbeddings whose rows are indexed by
        # RAW item id (rsc15's reach into the hundreds of millions), while `codes`
        # is indexed by dense id 0..N-1. Align explicitly rather than assuming the
        # two coincide -- they happen to for Amazon, but silently mismatched rows
        # would make every distance in Eq. 6 refer to the wrong item.
        z_obj = load_dense_embeddings(str(emb_path))
        if hasattr(z_obj, "tensor"):
            id_to_idx = z_obj.item_id_to_idx
            missing = [i for i in range(n_items) if i not in id_to_idx]
            if missing:
                raise ValueError(
                    f"{len(missing)} of {n_items} dense item ids are absent from "
                    f"{emb_path} (first few: {missing[:5]}). The embeddings must "
                    "cover every item in the SID tensor."
                )
            row_idx = _torch.tensor(
                [id_to_idx[i] for i in range(n_items)], dtype=_torch.long
            )
            z = z_obj.tensor[row_idx]
        else:
            z = z_obj
        if int(z.shape[0]) != n_items:
            raise ValueError(
                f"embeddings have {z.shape[0]} rows but the SID tensor has "
                f"{n_items} items"
            )

        # Correctness anchor: refuses unless phi=0 reproduces every stored code.
        assert_reproduces_sids(z, [centroids[i] for i in range(n_levels)], codes)

        # Target items live under ctx["meta"], not ctx itself -- reading the
        # wrong key made this look like "no targets" and tripped the
        # coherence_rows=target_only guard on a dataset that has them.
        target_items = sorted(int(i) for i in (ctx["meta"].get("target_items") or []))
        if not target_items:
            target_items = sorted(int(i) for i in (ctx.get("visible_forget_items") or []))
        if not target_items:
            raise ValueError("tracer found no concept items to reassign")
        concept = _torch.tensor(target_items, dtype=_torch.long)
        all_items = _torch.arange(codes.shape[1])
        keep = _torch.ones(codes.shape[1], dtype=_torch.bool)
        keep[concept] = False
        retain_items = all_items[keep]

        res_levels = compute_residuals(z, [centroids[i] for i in range(n_levels)], codes)
        residuals = _torch.stack([r[concept] for r in res_levels], dim=1)  # [M, L, D]

        # L_Coh's P(i_T) -- BUILT HERE, BY TRACER, ON PURPOSE.
        #
        # The paper's P(i_T) is the K nearest items to the CONCEPT item i_T in
        # the frozen embedding space. That is computed directly below from `z`
        # (the same dense embeddings the RQ-KMeans quantizer was fitted on, in
        # dense-item-id row order), with the concept set excluded from its own
        # neighbourhood.
        #
        # It deliberately does NOT call self._build_coherence_neighbors and does
        # not read any coherence_* / neighborhood_* config key: our prefix and
        # L_n neighbourhood construction is a contribution of ours, so routing a
        # baseline through it would stop it being a baseline. The only knob is
        # unlearning.tracer_neighborhood_count (the paper's K ~ 5).
        k_coh = int(cfg.get("tracer_neighborhood_count", 5))
        neighbor_items_log: Optional[List[List[int]]] = None
        zc = _torch.nn.functional.normalize(z[concept].double(), dim=-1)   # [M, D]
        za = _torch.nn.functional.normalize(z.double(), dim=-1)            # [N, D]
        sim = zc @ za.t()                                                  # [M, N]
        sim[:, concept] = float("-inf")   # never a neighbour of itself/the concept
        k_eff = int(min(k_coh, max(0, sim.shape[1] - int(concept.numel()))))
        if k_eff <= 0:
            concept_neighbor_sids = None
        else:
            nbr_items = sim.topk(k_eff, dim=-1).indices                    # [M, k]
            # codes is [H, N]; transpose to [N, H] and gather the neighbours'
            # full semantic ids (including the trailing dedup digit).
            concept_neighbor_sids = codes.t()[nbr_items].contiguous()      # [M, k, H]
            neighbor_items_log = nbr_items.tolist()
            log.info(
                "[tracer] P(i_T): cosine top-%d over %s for %d concept item(s), "
                "concept set excluded (built inside the TRACER path, not via "
                "_build_coherence_neighbors). First concept item %d -> %s",
                k_eff,
                emb_path,
                int(concept.numel()),
                int(concept[0]),
                nbr_items[0].tolist(),
            )
        del sim, zc, za

        info = tracer_unlearn(
            self,
            forget_batches=ctx["forget_batches"],
            retain_batches=ctx["retain_batches"],
            concept_item_ids=concept,
            residuals=residuals,
            centroids=centroids,
            codes=codes,
            retain_item_ids=retain_items,
            concept_neighbor_sids=concept_neighbor_sids,
            steps=cfg.get("tracer_steps", 500),
            n_epochs=cfg.get("n_epochs"),
            lr=float(cfg.get("tracer_lr", 1e-4)),
            phi_lr=float(cfg.get("tracer_phi_lr", 1e-2)),
            tau=float(cfg.get("tracer_temperature", 0.005)),
            lambda_forget=float(cfg.get("tracer_lambda_forget", 1.0)),
            lambda_coherence=float(cfg.get("tracer_lambda_coherence", 1.0)),
            lambda_reg=float(cfg.get("tracer_lambda_reg", 1e-3)),
            selective_update=bool(cfg.get("tracer_selective_update", True)),
            optimizer=_resolve_optimizer(cfg, "tracer", default="sgd"),
            commit=bool(cfg.get("tracer_commit", True)),
            device=device,
        )
        info["wall_seconds"] = time.time() - t0
        info["tracer_rqkmeans_ckpt"] = str(ckpt)
        # Audit trail for P(i_T): the exact neighbour item ids TRACER used.
        info["tracer_coherence_neighbor_items"] = neighbor_items_log
        info["tracer_coherence_metric"] = "cosine_topk_on_concept_items"
        info.update(ctx["meta"])
        return info

    def _build_coherence_neighbors(
        self,
        *,
        forget_batches: List[Any],
        semantic_id_path: Optional[str],
        num_hierarchies: Optional[int],
        neighborhood_count: int,
        neighborhood_prefix_length: int,
        exclude_items: Set[int],
        coherence_rows: str = "target_only",
        target_items: Optional[Set[int]] = None,
        neighbor_method: str = "prefix",
        embedding_path: Optional[str] = None,
        embedding_metric: str = "cosine",
    ) -> List[Optional[Any]]:
        """Per-forget-batch neighbour semantic ids for the coherence loss.

        For every eligible forget sample ``(H_f, i_T)`` we resolve the label item
        ``i_T`` from its label semantic id, then look up the
        ``neighborhood_count`` closest catalog items by shared SID prefix (length
        ``>= neighborhood_prefix_length``), excluding forget items. Returns a
        list aligned to ``forget_batches``; each element is
        ``(neighbor_sids[B, C, H], neighbor_mask[B, C])`` or ``None`` when a
        batch has no eligible neighbours anywhere.

        ``coherence_rows`` decides which forget rows are eligible:

        ``target_only`` (default)
            Only rows whose label item is in ``target_items`` (the deletion
            targets, i.e. the spam items). This is the documented TRACER
            semantics — ``P(i_T)`` is the neighbourhood *of the removed item*.

        ``all``
            Every forget row, keyed on whatever its label happens to be. This
            was the behaviour up to 2026-08-05 and is kept only to reproduce the
            432-run λ_n grid and its 3-seed replication. It is **not** the
            intended objective: TIGER's collate expands each session into all
            contiguous sub-sequences and supervises the last item of each, so in
            a bandwagon session the label is a popular *filler* item on all but
            one row. Measured on beauty bandwagon pct1/n1 (226 spam users, one
            target click each, 4110 forget rows over 10 chunks): only 226 rows
            (8.3% of the 2739 rows that had a neighbour) could possibly be the
            target, so >=91.7% of the λ_n gradient budget was spent boosting the
            prefix-neighbours of popular filler items from spam contexts —
            which is why λ_n never helped and hurt the `mid` stratum most (that
            target has zero prefix-2 neighbours, so its share was exactly 0%).
        """
        rows_mode = str(coherence_rows).lower()
        if rows_mode not in ("target_only", "all"):
            raise ValueError(
                f"coherence_rows must be 'target_only'|'all', got {coherence_rows!r}"
            )
        nbr_method = str(neighbor_method).lower()
        if nbr_method not in ("prefix", "embedding"):
            raise ValueError(
                "coherence_neighbor_method must be 'prefix'|'embedding', got "
                f"{neighbor_method!r}"
            )
        if nbr_method == "embedding" and not embedding_path:
            raise ValueError(
                "coherence_neighbor_method='embedding' requires "
                "unlearning.embedding_path (pre-quantization item embeddings, "
                "e.g. embeddings/beauty_merged_predictions_tensor_latest.pt)."
            )
        eligible = {int(x) for x in (target_items or set())}
        if rows_mode == "target_only" and not eligible:
            raise ValueError(
                "coherence_rows='target_only' needs a non-empty target item set "
                "(manifest target_items / visible forget items). Pass "
                "coherence_rows='all' to score every forget row instead."
            )
        if not semantic_id_path:
            raise ValueError(
                "lambda_n > 0 (coherence loss) requires semantic_id_path "
                "(merged_predictions_tensor.pt) to define SID neighbours."
            )
        codebook = load_codebook(semantic_id_path, num_hierarchies=num_hierarchies)
        num_items, H = int(codebook.shape[0]), int(codebook.shape[1])
        count = int(neighborhood_count)

        # SID tuple -> item id (bijective; the dedup digit makes codes unique).
        sid_to_item: Dict[tuple, int] = {
            tuple(int(x) for x in codebook[i].tolist()): i for i in range(num_items)
        }
        sorted_ids = build_sorted_sid_index(codebook)
        sorted_sids = codebook.numpy()[sorted_ids]
        exclude = {int(x) for x in (exclude_items or set())}

        # Embedding mode: the neighbourhood is a fixed-size top-k in the
        # pre-quantization space, so every target gets exactly `count` neighbours
        # regardless of codebook width. Loaded once; the k-NN itself is cached
        # per item below (with coherence_rows='target_only' and n_target=1 that
        # is a single query for the whole run).
        embeddings = None
        if nbr_method == "embedding":
            embeddings = load_dense_embeddings(embedding_path)
            # We index the embedding matrix BY ROW, because the codebook is
            # indexed 0..N-1 by row and raw item IDs are not valid codebook
            # indices for datasets with non-sequential IDs (rsc15's reach ~1e9).
            # That is only sound if row i means the same item in both, which
            # holds when the SID tensor came from RQ-KMeans over these very
            # embeddings, in order. Assert the shapes agree rather than silently
            # mismatching items.
            if len(embeddings) != num_items:
                raise ValueError(
                    f"embedding_path has {len(embeddings)} items but the "
                    f"codebook has {num_items}: they must be row-aligned "
                    f"(same catalog, same order) for coherence_neighbor_method="
                    f"'embedding'. Check that {embedding_path!r} is the "
                    f"pre-quantization tensor the SID codebook was built from."
                )
            log.info(
                "[unified] coherence L_n neighbours: embedding top-k "
                "(metric=%s, k=%d) from %s [%d items, dim %d]",
                embedding_metric,
                count,
                embedding_path,
                len(embeddings),
                int(embeddings.shape[1]),
            )

        # item id -> neighbour SID rows [k, H] (cached; forget targets repeat).
        neighbor_cache: Dict[int, List[List[int]]] = {}

        def _neighbor_rows(item_id: int) -> List[List[int]]:
            if item_id not in neighbor_cache:
                if nbr_method == "embedding":
                    # by_row: item_id and the returned neighbours are codebook
                    # row indices, directly usable as codebook[n] below.
                    nbr_ids = topk_embedding_neighbors(
                        item_id,
                        embeddings,
                        count,
                        metric=embedding_metric,
                        exclude_ids=exclude,
                        by_row=True,
                    )
                else:
                    nbr_ids = closest_prefix_neighbors(
                        codebook,
                        item_id,
                        count,
                        neighborhood_prefix_length,
                        sorted_ids=sorted_ids,
                        sorted_sids=sorted_sids,
                        exclude_ids=exclude,
                    )
                neighbor_cache[item_id] = [
                    [int(x) for x in codebook[n].tolist()] for n in nbr_ids
                ]
            return neighbor_cache[item_id]

        out: List[Optional[Any]] = []
        n_samples = 0
        n_eligible = 0
        n_with_neighbors = 0
        n_targets_missing = 0
        for model_input, label_data in forget_batches:
            bsz = int(model_input.mask.size(0))
            fut_ids = None
            for label in label_data.labels:
                fut_ids = label_data.labels[label].reshape(bsz, -1)
            neighbor_sids = torch.zeros((bsz, count, H), dtype=torch.long)
            neighbor_mask = torch.zeros((bsz, count), dtype=torch.float32)
            for row in range(bsz):
                n_samples += 1
                sid_tuple = tuple(int(x) for x in fut_ids[row, :H].tolist())
                item_id = sid_to_item.get(sid_tuple)
                if item_id is None:
                    n_targets_missing += 1
                    continue
                # Skip rows whose label is not a deletion target: their
                # neighbourhood is irrelevant to the removal (see docstring).
                if rows_mode == "target_only" and item_id not in eligible:
                    continue
                n_eligible += 1
                rows = _neighbor_rows(item_id)
                if rows:
                    n_with_neighbors += 1
                for c, sid_row in enumerate(rows[:count]):
                    neighbor_sids[row, c] = torch.tensor(sid_row, dtype=torch.long)
                    neighbor_mask[row, c] = 1.0
            out.append(
                (neighbor_sids, neighbor_mask)
                if float(neighbor_mask.sum()) > 0
                else None
            )

        log.info(
            "[unified] coherence L_n: method=%s rows=%s, %d/%d forget rows "
            "eligible, %d of those have >=1 neighbour (count=%d, "
            "min_prefix_length=%s, %d labels not found in codebook)",
            nbr_method,
            rows_mode,
            n_eligible,
            n_samples,
            n_with_neighbors,
            count,
            int(neighborhood_prefix_length) if nbr_method == "prefix" else "n/a",
            n_targets_missing,
        )
        if n_with_neighbors == 0:
            if nbr_method == "prefix":
                log.warning(
                    "[unified] coherence L_n is identically ZERO: no eligible "
                    "forget row has a catalog neighbour sharing a prefix of "
                    "length >=%d. lambda_n cannot have any effect. Lower "
                    "unlearning.neighborhood_prefix_length, switch to "
                    "coherence_neighbor_method=embedding (fixed-size top-k, "
                    "never empty), or use a narrower codebook (at beauty width "
                    "256 the mean prefix-2 neighbourhood is ~3 items and ~31%% "
                    "of items have none).",
                    int(neighborhood_prefix_length),
                )
            else:
                log.warning(
                    "[unified] coherence L_n is identically ZERO despite "
                    "embedding top-k neighbours — no eligible forget row was "
                    "found at all (check coherence_rows/target_items)."
                )
        return out

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
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "kookmin"),
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
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "fanchuan"),
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
            update_scope=str(cfg.get("update_scope", "all")),
            pkm_update_keys=bool(cfg.get("pkm_update_keys", True)),
            pkm_update_query=bool(cfg.get("pkm_update_query", True)),
            optimizer=_resolve_optimizer(cfg, "seif"),
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

    def diagnose_pkm_slots(
        self,
        *,
        top_t: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Per-slot access statistics for every PKM, forget vs retain (read-only).

        Answers the question that gates top-t memory-slot selection: do forget
        and retain interactions route to DISJOINT memory slots? If they overlap
        almost completely, no selection criterion can separate them.

        Produces two candidate selection scores per slot:

        * ``AF``      -- raw access frequency on the forget set. The
          access-count baseline (what Sparse Memory Finetuning's TF term does).
        * ``AF-IHF``  -- access frequency x inverse history frequency,
          ``AF(s) * log((T_r + 1) / (HF(s) + 1))``, where ``HF`` is the retain
          access count and ``T_r`` the total retain reads. This is the
          recommender-side analogue of TF-IDF: a slot scores highly when the
          forget data hits it often AND the retain ("history") data rarely does.

        Nothing is updated; the model is only run forward.
        """
        from src.models.components.network_blocks.product_key_memory import (
            HashingMemory,
        )

        tops = [int(t) for t in (top_t or [25, 50, 100, 200, 500, 1000])]
        mems = [
            (n, m) for n, m in self.named_modules() if isinstance(m, HashingMemory)
        ]
        if not mems:
            raise ValueError(
                "diagnose_pkm_slots requires a PKM-bearing model; pass the same "
                "model.pkm_layers / model.pkm_mode the checkpoint was trained with."
            )

        ctx = self._prepare_unlearning_context(**kwargs)
        device = kwargs.get("device") or next(self.parameters()).device

        def _sweep(batches: List[Any]) -> Dict[str, Any]:
            for _, m in mems:
                m.enable_access_counting()
                m.reset_access_counts()
            was_training = self.training
            self.eval()
            with torch.no_grad():
                for b in batches:
                    self.model_step(*b)
            out = {n: m.get_access_counts() for n, m in mems}
            for _, m in mems:
                m.disable_access_counting()
            if was_training:
                self.train()
            return out

        log.info(
            "[pkm-slots] sweeping %d forget / %d retain batches over %d memories",
            len(ctx["forget_batches"]), len(ctx["retain_batches"]), len(mems),
        )
        af_all = _sweep(ctx["forget_batches"])
        hf_all = _sweep(ctx["retain_batches"])

        # ---- per-slot GRADIENT signal -------------------------------------
        # Access counts cannot separate forget from retain when both read the
        # same slots. Gradient MAGNITUDE on those slots still can, so this is
        # strictly more informative than AF-IHF on a collapsed memory.
        # Accumulates the gradient of the summed loss w.r.t. values.weight
        # (shape (size, v_dim)), then takes the per-slot row norm — i.e. the
        # gradient of the objective, not a sum of per-batch norms.
        def _grad_sweep(batches: List[Any]) -> Dict[str, torch.Tensor]:
            acc: Dict[str, torch.Tensor] = {
                n: torch.zeros_like(m.values.weight) for n, m in mems
            }
            was_training = self.training
            self.eval()  # no dropout noise; grads still flow
            for b in batches:
                self.zero_grad(set_to_none=True)
                _, loss = self.model_step(*b)
                loss.backward()
                for n, m in mems:
                    g = m.values.weight.grad
                    if g is not None:
                        acc[n] += g.detach()
            self.zero_grad(set_to_none=True)
            if was_training:
                self.train()
            return acc

        gf_vec = _grad_sweep(ctx["forget_batches"])
        gr_vec = _grad_sweep(ctx["retain_batches"])

        per_mem: Dict[str, Any] = {}
        for name, mem in mems:
            af = af_all[name][0].double()
            hf = hf_all[name][0].double()
            n_slots = int(af.numel())
            t_f = float(af.sum().item())
            t_r = float(hf.sum().item())
            f_touch = af > 0
            r_touch = hf > 0
            inter = int((f_touch & r_touch).sum().item())
            union = int((f_touch | r_touch).sum().item())

            # AF-IHF: high forget access, low retain ("history") access.
            ihf = torch.log((t_r + 1.0) / (hf + 1.0))
            af_ihf = af * ihf

            # gradient-based scores: g_f, g_r, and g_f - lambda*g_r
            gf = gf_vec[name]
            gr = gr_vec[name]
            gf_n = gf.norm(dim=1).double()
            gr_n = gr.norm(dim=1).double()
            denom = (gf.norm(dim=1) * gr.norm(dim=1)).clamp_min(1e-12)
            cos_fr = ((gf * gr).sum(dim=1) / denom).double()
            # ---- selection scores -------------------------------------
            # Magnitude-only criterion (g_f - lambda*g_r) compares NORMS and so
            # is blind to DIRECTION: it cannot tell a slot whose forget-ascent
            # direction also happens to help retain from one that wrecks it.
            #
            # The unlearning update moves along +g_f (ascend the forget loss),
            # so the first-order change in the RETAIN loss from editing slot i is
            # <g_f,i , g_r,i>. That inner product — not ||g_r,i|| — is the actual
            # collateral-damage term, and it is signed: negative means editing
            # for forgetting also IMPROVES retain.
            #
            # Combined score (all three terms max-normalised so lambda/mu are
            # scale-free and comparable across memories):
            #     s_i = gf_i - lambda * gr_i - mu * dot_i
            # 'dotabs' variant penalises |dot| instead (pure orthogonality: we
            # only care that the gradients are UNRELATED, either sign).
            # Both are additively separable over slots, so exact top-t is just
            # topk — no greedy approximation needed.
            def _nrm(v: torch.Tensor) -> torch.Tensor:
                m = v.abs().max()
                return v / m if float(m) > 0 else v

            dot_fr = (gf * gr).sum(dim=1).double()
            gf_hat, gr_hat = _nrm(gf_n), _nrm(gr_n)
            dot_hat = _nrm(dot_fr)
            grad_scores = {}
            for lam in (0.0, 1.0):
                grad_scores[f"lam{lam}"] = gf_n - float(lam) * gr_n  # magnitude-only
            for lam in (0.0, 1.0):
                for mu in (1.0, 5.0):
                    grad_scores[f"lam{lam}_mu{mu}"] = (
                        gf_hat - float(lam) * gr_hat - float(mu) * dot_hat
                    )
                    grad_scores[f"lam{lam}_mu{mu}_dotabs"] = (
                        gf_hat - float(lam) * gr_hat - float(mu) * dot_hat.abs()
                    )

            entry: Dict[str, Any] = {
                "n_slots": n_slots,
                "grad": {
                    "forget_grad_norm_total": float(gf_n.sum().item()),
                    "retain_grad_norm_total": float(gr_n.sum().item()),
                    "slots_with_forget_grad": int((gf_n > 0).sum().item()),
                    "slots_with_retain_grad": int((gr_n > 0).sum().item()),
                    # mean cosine over slots that BOTH objectives touch: >0 means
                    # the forget and retain updates pull the same way (conflict).
                    "mean_fr_cosine_on_shared": float(
                        cos_fr[(gf_n > 0) & (gr_n > 0)].mean().item()
                    ) if int(((gf_n > 0) & (gr_n > 0)).sum().item()) else None,
                    # how far apart are the two objectives' per-slot rankings?
                    "mean_dot_fr": float(dot_fr.mean().item()),
                    "frac_slots_dot_negative": float(
                        (dot_fr < 0).double().mean().item()
                    ),
                },
                "forget_reads": t_f,
                "retain_reads": t_r,
                "forget_slots_touched": int(f_touch.sum().item()),
                "retain_slots_touched": int(r_touch.sum().item()),
                "forget_coverage": float(f_touch.sum().item()) / n_slots,
                "retain_coverage": float(r_touch.sum().item()) / n_slots,
                "touched_jaccard": (inter / union) if union else 0.0,
                # the decisive statistic: forget-hit slots the retain set never reads
                "forget_exclusive_slots": int((f_touch & ~r_touch).sum().item()),
                "forget_exclusive_frac_of_forget": (
                    float((f_touch & ~r_touch).sum().item())
                    / max(1, int(f_touch.sum().item()))
                ),
                "top_t": {},
            }
            for t in tops:
                k = min(t, n_slots)
                af_top = torch.topk(af, k).indices
                ihf_top = torch.topk(af_ihf, k).indices
                hf_top = torch.topk(hf, k).indices
                af_set, ihf_set, hf_set = set(af_top.tolist()), set(ihf_top.tolist()), set(hf_top.tolist())
                entry["top_t"][str(t)] = {
                    # how different is AF-IHF from the plain access-count baseline?
                    "af_vs_afihf_overlap": len(af_set & ihf_set) / max(1, k),
                    # how much do the selected slots collide with retain's own top-t?
                    "af_top_in_retain_top": len(af_set & hf_set) / max(1, k),
                    "afihf_top_in_retain_top": len(ihf_set & hf_set) / max(1, k),
                    # fraction of selected slots the retain set NEVER touches
                    "af_top_retain_unused": float(
                        (hf[af_top] == 0).sum().item()
                    ) / max(1, k),
                    "afihf_top_retain_unused": float(
                        (hf[ihf_top] == 0).sum().item()
                    ) / max(1, k),
                    "afihf_top_slot_ids": ihf_top[: min(k, 32)].tolist(),
                }
                # gradient-criterion selections at the same cutoff
                for sname, sc in grad_scores.items():
                    g_top = torch.topk(sc, k).indices
                    g_set = set(g_top.tolist())
                    entry["top_t"][str(t)][f"grad_{sname}"] = {
                        # the collateral-damage term on the SELECTED slots:
                        # negative is good (forgetting also helps retain)
                        "mean_dot_selected": float(dot_fr[g_top].mean().item()),
                        # does the gradient criterion pick different slots than
                        # the access-count criteria?
                        "overlap_with_af": len(g_set & af_set) / max(1, k),
                        "overlap_with_afihf": len(g_set & ihf_set) / max(1, k),
                        "retain_unused": float(
                            (hf[g_top] == 0).sum().item()
                        ) / max(1, k),
                        "mean_gf_selected": float(gf_n[g_top].mean().item()),
                        "mean_gr_selected": float(gr_n[g_top].mean().item()),
                    }
            per_mem[name] = entry

        return {
            "diagnostic": "pkm_slots",
            "n_memories": len(mems),
            "top_t": tops,
            "n_forget_batches": len(ctx["forget_batches"]),
            "n_retain_batches": len(ctx["retain_batches"]),
            "retain_source": ctx["meta"].get("retain_source"),
            "per_memory": per_mem,
            "meta": ctx["meta"],
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
                    include_context_rows=bool(
                        unlearning_cfg.get("include_context_rows", False)
                    ),
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
        # Where the retain batches come from.
        #   subset (DEFAULT) -> retain_subset_dir, the sampled subset of size
        #       retain_samples_used_for_update * |D_f|. This is what every
        #       influence-function / unlearning-scale algorithm expects and is
        #       what all existing results were produced with.
        #   full -> the ENTIRE retain split. Needed when the update has to REBUILD
        #       capacity rather than nudge it (post-hoc PKM repair): fine-tuning
        #       thousands of steps over a 2-batch subset just memorises it, which
        #       collapsed utility 0.832 -> 0.360 in jobs 10280081-84.
        retain_source = str(
            unlearning_cfg.get("retain_source", "subset") or "subset"
        ).strip().lower()
        if retain_source not in ("subset", "full"):
            raise ValueError(
                f"unlearning.retain_source must be 'subset' or 'full', got {retain_source!r}"
            )
        retain_loader_dir = retain_dir if retain_source == "full" else retain_subset_dir
        if retain_source == "full":
            algo_name = str(unlearning_cfg.get("algorithm", "scif")).strip().lower()
            log.warning(
                "[retain_source=full] retain batches come from the FULL retain "
                "split (%s), NOT the %d-per-|D_f| subset. Intended for capacity "
                "REBUILD (post-hoc PKM repair).",
                retain_dir,
                int(unlearning_cfg.get("retain_samples_used_for_update") or 16),
            )
            if algo_name in ("scif", "seif"):
                log.warning(
                    "[retain_source=full] algorithm=%s derives its influence "
                    "scaling from the SAMPLED retain subset "
                    "(retain_count = retain_samples_used_for_update * |D_f|); "
                    "using the full split changes that estimator's assumptions. "
                    "Results will NOT be comparable to the recorded %s numbers.",
                    algo_name, algo_name,
                )
        retain_loader = _build_finite_loader(
            base_train_cfg=train_dataloader_config,
            data_folder=retain_loader_dir,
            batch_size_per_device_override=unlearn_batch_size,
        )
        forget_batches = _drain_loader(forget_loader, device=device)
        retain_batches = _drain_loader(retain_loader, device=device)
        if not forget_batches:
            raise RuntimeError(f"No forget batches from {forget_dir}")
        if not retain_batches:
            raise RuntimeError(f"No retain batches from {retain_loader_dir}")

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
                "retain_source": retain_source,
                "retain_loader_dir": retain_loader_dir,
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
