"""Item-mode forget preprocessing: remove ``I_f`` from histories and drop invalid rows.

When ``deletion_spec='item'`` the unlearning algorithms know the target item
set ``I_f`` but not which spam sessions contain them.  This module materialises
a sibling TFRecord directory whose rows have ``I_f`` stripped from
``sequence_data`` and rows that fall below ``min_sequence_length`` removed.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

from src.data.unlearning.split_forget_retain import (
    FORGET_SUBDIR,
    SEQUENCE_FIELD,
    USER_ID_FIELD,
    _infer_feature_description,
    _list_shards,
    _ShardWriter,
)

tf.config.set_visible_devices([], "GPU")


def _filter_sequence(seq: List[int], forbidden: Set[int]) -> List[int]:
    return [int(x) for x in seq if int(x) not in forbidden]


def materialize_item_mode_forget_dir(
    *,
    forget_dir: str,
    out_dir: str,
    target_items: Set[int],
    min_sequence_length: int = 1,
    rows_per_shard: int = 4096,
    overwrite: bool = True,
) -> Dict[str, object]:
    """Write filtered forget shards with ``I_f`` removed from each sequence."""
    if not target_items:
        raise ValueError("target_items must be non-empty for item-mode filtering.")

    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass overwrite=True")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    shards = _list_shards(forget_dir)
    if not shards:
        raise FileNotFoundError(f"No shards under {forget_dir}")

    raw = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample = next(iter(raw))
    feat_desc = _infer_feature_description(sample)
    if SEQUENCE_FIELD not in feat_desc:
        raise ValueError(f"{forget_dir}: missing {SEQUENCE_FIELD!r}")

    writer = _ShardWriter(out_dir, "data", rows_per_shard)
    parsed = raw.map(lambda x: (x, tf.io.parse_single_example(x, feat_desc)))

    rows_in = 0
    rows_out = 0
    rows_dropped = 0
    for raw_bytes, example in parsed:
        rows_in += 1
        seq = tf.sparse.to_dense(example[SEQUENCE_FIELD]).numpy()
        if seq.size == 0:
            rows_dropped += 1
            continue
        filtered = _filter_sequence(seq.tolist(), target_items)
        if len(filtered) < min_sequence_length:
            rows_dropped += 1
            continue

        ex = tf.train.Example()
        ex.ParseFromString(raw_bytes.numpy())  # type: ignore[arg-type]
        feat = ex.features.feature
        del feat[SEQUENCE_FIELD]
        feat[SEQUENCE_FIELD].int64_list.value.extend(filtered)
        writer.write(ex.SerializeToString())
        rows_out += 1

    writer.close()
    return {
        "source_forget_dir": os.path.abspath(forget_dir),
        "out_dir": os.path.abspath(out_dir),
        "n_target_items": len(target_items),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped": rows_dropped,
        "min_sequence_length": min_sequence_length,
        "shard_paths": writer.shard_paths,
    }


def default_item_mode_forget_subdir(forget_subdir: str = FORGET_SUBDIR) -> str:
    return f"{forget_subdir}_item_filtered"
