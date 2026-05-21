"""Decode-time filtering utilities for the filter unlearning baseline."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch


def build_filter_mask(
    *,
    deletion_spec: str,
    target_items: Set[int],
    forget_shard_items: Set[int],
    filter_mode: str = "global",
    user_forget_items: Optional[Dict[int, Set[int]]] = None,
) -> Dict[str, Any]:
    """Build a serialisable decode mask specification."""
    mode = str(filter_mode).strip().lower()
    if mode == "global":
        if deletion_spec == "item" and target_items:
            forbidden = sorted(int(x) for x in target_items)
        else:
            forbidden = sorted(int(x) for x in forget_shard_items)
        return {
            "filter_mode": "global",
            "deletion_spec": deletion_spec,
            "forbidden_item_ids": forbidden,
            "user_forget_items": None,
        }
    if mode == "user_dependent":
        serial_user: Dict[str, List[int]] = {}
        if user_forget_items:
            for uid, items in user_forget_items.items():
                serial_user[str(int(uid))] = sorted(int(x) for x in items)
        return {
            "filter_mode": "user_dependent",
            "deletion_spec": deletion_spec,
            "forbidden_item_ids": sorted(int(x) for x in target_items or forget_shard_items),
            "user_forget_items": serial_user,
        }
    raise ValueError(f"Unknown filter_mode={filter_mode!r}")


def save_filter_mask(mask: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(mask, fh, indent=2)
    return path


def load_filter_mask(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def forbidden_sids_from_codebook(
    codebook: torch.Tensor,
    forbidden_item_ids: Iterable[int],
) -> Set[Tuple[int, ...]]:
    """Map sparse item ids to full SID tuples for decode masking."""
    forbidden: Set[Tuple[int, ...]] = set()
    cb = codebook.long()
    if cb.shape[0] < cb.shape[1]:
        cb = cb.t()
    for iid in forbidden_item_ids:
        i = int(iid)
        if 0 <= i < cb.shape[0]:
            forbidden.add(tuple(int(x) for x in cb[i].tolist()))
    return forbidden


def scan_user_forget_items(forget_dir: str) -> Dict[int, Set[int]]:
    """Scan forget shards and return ``user_id -> items in sequence``."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from src.components.unlearning.neighborhood_sampler import (
        SEQUENCE_FIELD,
        USER_ID_FIELD,
        _infer_feature_description,
        _list_shards,
    )

    shards = _list_shards(forget_dir)
    if not shards:
        return {}
    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample = next(iter(raw))
    feat_desc = _infer_feature_description(sample)
    parsed = raw.map(lambda x: tf.io.parse_single_example(x, feat_desc))
    out: Dict[int, Set[int]] = {}
    for ex in parsed:
        uid = int(tf.sparse.to_dense(ex[USER_ID_FIELD]).numpy().flatten()[0])
        seq = tf.sparse.to_dense(ex[SEQUENCE_FIELD]).numpy()
        out.setdefault(uid, set()).update(int(x) for x in seq.tolist())
    return out
