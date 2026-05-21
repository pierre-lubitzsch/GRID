"""Compare clean-reference vs poisoned TIGER checkpoints on the test split.

For each test example, record whether the top-k semantic-ID prediction hits
the ground-truth label. Save items that are **hit by the reference model** but
**missed by the poisoned model**, and report how many of those target items
appear in poisoned training sessions (bandwagon spam shards).

Usage::

    python -m scripts.compare_ref_poison_test_items \\
        experiment=tiger_train_flat \\
        data_dir=src/data/amazon_data/beauty \\
        poison_data_dir=src/data/amazon_data/beauty_spam_seed2_pct1_n10 \\
        semantic_id_path=embeddings/beauty/merged_predictions_tensor.pt \\
        reference_ckpt_path='logs/train/.../clean.ckpt' \\
        poisoned_ckpt_path='logs/train/.../poisoned.ckpt' \\
        num_hierarchies=4 \\
        train=False test=True

Outputs (under ``hydra.run.dir``):

* ``ref_right_poison_wrong.json`` — per-item semantic IDs and metadata
* ``summary.json`` — counts and overlap with poisoned sessions
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.components.unlearning.hvp import batch_to_device  # noqa: E402
from src.components.unlearning.neighborhood_sampler import (  # noqa: E402
    _list_shards,
    collect_items_in_shards,
    load_codebook,
)
from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def _hit_at_k(
    generated_ids: torch.Tensor, labels: torch.Tensor, num_hierarchies: int
) -> torch.Tensor:
    """Return a boolean vector of shape ``(batch_size,)`` for SID hit@k."""
    batch_size = generated_ids.shape[0]
    labels = labels.reshape(batch_size, 1, num_hierarchies)
    match = (generated_ids == labels.to(generated_ids.device)).all(dim=2)
    return match.any(dim=1).cpu()


def _semantic_id_tuple(label_row: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(x) for x in label_row.tolist())


def _build_sparse_id_lookup(
    codebook: torch.Tensor,
) -> Dict[Tuple[int, ...], int]:
    """Map full semantic-id tuple -> sparse catalog item id."""
    rows = codebook.numpy()
    lookup: Dict[Tuple[int, ...], int] = {}
    for item_id in range(rows.shape[0]):
        key = tuple(int(x) for x in rows[item_id].tolist())
        lookup.setdefault(key, item_id)
    return lookup


def _collect_poisoned_session_items(
    poison_data_dir: str,
    *,
    spam_shards_only: bool = True,
) -> Set[int]:
    """Items appearing in poisoned training sessions (spam TFRecord shards)."""
    training_dir = os.path.join(poison_data_dir, "training")
    if spam_shards_only:
        shard_paths = sorted(glob(os.path.join(training_dir, "data_spam_*.tfrecord.gz")))
    else:
        shard_paths = _list_shards(training_dir)
    if not shard_paths:
        raise FileNotFoundError(
            f"No poisoned training shards under {training_dir} "
            f"(spam_shards_only={spam_shards_only})"
        )
    return collect_items_in_shards(shard_paths)


@torch.inference_mode()
def _run_test_pass(
    model: torch.nn.Module,
    test_loader: Any,
    device: torch.device,
    num_hierarchies: int,
) -> List[Dict[str, Any]]:
    """Run generation on the test loader; return per-row hit flags + labels."""
    model.eval()
    model.to(device)
    records: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(test_loader):
        batch_to_device(batch, device)
        model_input, label_data = batch

        generated_ids, _ = model.generate(
            attention_mask=model_input.mask,
            **{
                model.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        labels = list(label_data.labels.values())[0]
        hits = _hit_at_k(generated_ids, labels, num_hierarchies)

        user_ids = model_input.user_id_list
        batch_size = hits.shape[0]
        labels_view = labels.reshape(batch_size, num_hierarchies)

        for i in range(batch_size):
            uid = user_ids[i]
            if isinstance(uid, torch.Tensor):
                uid = int(uid.item())
            records.append(
                {
                    "batch_idx": batch_idx,
                    "row_in_batch": i,
                    "user_id": uid,
                    "semantic_id": list(_semantic_id_tuple(labels_view[i])),
                    "hit": bool(hits[i].item()),
                }
            )

        if (batch_idx + 1) % 50 == 0:
            log.info("Processed %d test batches (%d rows)", batch_idx + 1, len(records))

    return records


def _load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)


def compare(cfg: DictConfig) -> Dict[str, Any]:
    ref_ckpt = cfg.get("reference_ckpt_path")
    poison_ckpt = cfg.get("poisoned_ckpt_path")
    if not ref_ckpt or not poison_ckpt:
        raise ValueError(
            "reference_ckpt_path and poisoned_ckpt_path are required Hydra overrides."
        )
    poison_data_dir = cfg.get("poison_data_dir") or cfg.paths.data_dir
    eval_top_k = int(cfg.get("eval_top_k") or 10)
    num_hierarchies = int(cfg.get("num_hierarchies", 4))
    semantic_id_path = cfg.get("semantic_id_path")

    if not semantic_id_path:
        raise ValueError("semantic_id_path is required.")

    codebook = load_codebook(semantic_id_path, num_hierarchies=num_hierarchies)
    sid_to_sparse = _build_sparse_id_lookup(codebook)

    log.info("Collecting items from poisoned spam sessions under %s", poison_data_dir)
    poison_session_items = _collect_poisoned_session_items(
        poison_data_dir, spam_shards_only=bool(cfg.get("spam_shards_only", True))
    )
    log.info("Poisoned session pool: %d distinct item ids", len(poison_session_items))

    with pipeline_launcher(cfg) as pipeline_modules:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        test_loader = pipeline_modules.datamodule.test_dataloader()
        model = pipeline_modules.model

        log.info("Reference checkpoint: %s", ref_ckpt)
        _load_checkpoint_into_model(model, ref_ckpt)
        ref_records = _run_test_pass(model, test_loader, device, num_hierarchies)

        log.info("Poisoned checkpoint: %s", poison_ckpt)
        _load_checkpoint_into_model(model, poison_ckpt)
        poison_records = _run_test_pass(model, test_loader, device, num_hierarchies)

    if len(ref_records) != len(poison_records):
        raise RuntimeError(
            f"Reference and poisoned passes produced different row counts: "
            f"{len(ref_records)} vs {len(poison_records)}"
        )

    ref_right_poison_wrong: List[Dict[str, Any]] = []
    n_ref_hit = 0
    n_poison_hit = 0

    for ref_row, poi_row in zip(ref_records, poison_records):
        if ref_row["semantic_id"] != poi_row["semantic_id"]:
            raise RuntimeError(
                "Test loader order mismatch between checkpoint passes."
            )
        if ref_row["hit"]:
            n_ref_hit += 1
        if poi_row["hit"]:
            n_poison_hit += 1
        if ref_row["hit"] and not poi_row["hit"]:
            sid_tuple = tuple(ref_row["semantic_id"])
            sparse_id = sid_to_sparse.get(sid_tuple)
            in_poison_session = (
                sparse_id is not None and sparse_id in poison_session_items
            )
            ref_right_poison_wrong.append(
                {
                    "semantic_id": ref_row["semantic_id"],
                    "sparse_item_id": sparse_id,
                    "user_id": ref_row["user_id"],
                    "in_poisoned_session": in_poison_session,
                    "batch_idx": ref_row["batch_idx"],
                    "row_in_batch": ref_row["row_in_batch"],
                }
            )

    n_special = len(ref_right_poison_wrong)
    n_in_poison = sum(1 for r in ref_right_poison_wrong if r["in_poisoned_session"])

    summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_dir": os.path.abspath(cfg.paths.data_dir),
        "poison_data_dir": os.path.abspath(poison_data_dir),
        "semantic_id_path": os.path.abspath(semantic_id_path),
        "reference_ckpt_path": os.path.abspath(ref_ckpt),
        "poisoned_ckpt_path": os.path.abspath(poison_ckpt),
        "num_hierarchies": num_hierarchies,
        "eval_top_k": eval_top_k,
        "n_test_rows": len(ref_records),
        "n_ref_hit": n_ref_hit,
        "n_poison_hit": n_poison_hit,
        "n_ref_right_poison_wrong": n_special,
        "n_ref_right_poison_wrong_in_poisoned_session": n_in_poison,
        "frac_ref_right_poison_wrong_in_poisoned_session": (
            float(n_in_poison) / n_special if n_special else 0.0
        ),
        "n_poison_session_items": len(poison_session_items),
        "spam_shards_only": bool(cfg.get("spam_shards_only", True)),
    }

    out_dir = cfg.paths.output_dir
    os.makedirs(out_dir, exist_ok=True)
    items_path = os.path.join(out_dir, "ref_right_poison_wrong.json")
    summary_path = os.path.join(out_dir, "summary.json")

    with open(items_path, "w") as f:
        json.dump(
            {"summary": summary, "items": ref_right_poison_wrong},
            f,
            indent=2,
        )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "Done: %d / %d test rows hit by ref only; %d of those items appear in "
        "poisoned spam sessions (%.1f%%)",
        n_special,
        len(ref_records),
        n_in_poison,
        100.0 * summary["frac_ref_right_poison_wrong_in_poisoned_session"],
    )
    log.info("Wrote %s and %s", items_path, summary_path)
    return summary


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    compare(cfg)


if __name__ == "__main__":
    main()
