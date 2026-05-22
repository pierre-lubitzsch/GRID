"""Bandwagon spam-injection preprocessing for TIGER on GRID.

Adapted from the user's RecBole-flavoured ``FraudSessionGenerator`` (per-click
``.inter`` TSV) and retargeted to TIGER's per-user TFRecord pipeline. Produces
a sibling poisoned dataset directory that downstream Hydra configs can consume
unchanged via ``data_dir=...``.

Output layout
-------------
``data/amazon_data/<dataset>_spam_seed<S>_pct<P>_n<C>/``

* ``training/data_*.tfrecord.gz``        clean shards copied verbatim
* ``training/data_spam_*.tfrecord.gz``   new spam-user shards
* ``evaluation/``, ``testing/``, ``items/``  copied verbatim from source
* ``forget_manifest.json``               drives ``split_forget_retain.py``

Each spam example is a single row mirroring the source schema:
``user_id`` is a fresh int64 ID past ``max(clean user_id)``, ``sequence_data``
is the bandwagon attack sequence; any other declared features
(``embedding``, ``text``, ...) are emitted as empty defaults so per-shard
schema inference in ``TFRecordIterator`` stays consistent.

Usage
-----
``python -m src.data.poisoning.bandwagon \\
    --data_dir src/data/amazon_data/beauty \\
    --attack bandwagon --target_strategy unpopular \\
    --poisoning_ratio 0.01 --n_target_items 10 \\
    --placement alternating --seed 42``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


SEQUENCE_FIELD = "sequence_data"
USER_ID_FIELD = "user_id"
TRAINING_SUBDIR = "training"
SIBLING_SUBDIRS = ("evaluation", "testing", "items")


# ---------------------------------------------------------------------------
# Schema inference and shard reading
# ---------------------------------------------------------------------------


def _list_training_shards(training_dir: str) -> List[str]:
    paths = [
        os.path.join(training_dir, f)
        for f in sorted(os.listdir(training_dir))
        if f.endswith(".tfrecord.gz")
    ]
    if not paths:
        raise FileNotFoundError(f"No .tfrecord.gz shards found under {training_dir}")
    return paths


def _infer_feature_description(sample_record: tf.Tensor) -> Dict[str, tf.io.VarLenFeature]:
    """Mirror :class:`TFRecordIterator.infer_feature_type` (VarLen for all fields)."""
    example = tf.train.Example()
    example.ParseFromString(sample_record.numpy())  # type: ignore[arg-type]
    feature_description: Dict[str, tf.io.VarLenFeature] = {}
    for key, value in example.features.feature.items():
        if value.HasField("bytes_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.string)
        elif value.HasField("float_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.float32)
        elif value.HasField("int64_list"):
            feature_description[key] = tf.io.VarLenFeature(tf.int64)
        else:
            raise ValueError(f"Unknown feature type for key {key!r}")
    return feature_description


def _feature_kinds(feature_description: Dict[str, tf.io.VarLenFeature]) -> Dict[str, str]:
    """Return ``{name: 'int64'|'float'|'bytes'}`` for default-filling spam rows."""
    kinds: Dict[str, str] = {}
    for name, spec in feature_description.items():
        dtype = spec.dtype
        if dtype == tf.int64:
            kinds[name] = "int64"
        elif dtype == tf.float32:
            kinds[name] = "float"
        elif dtype == tf.string:
            kinds[name] = "bytes"
        else:  # pragma: no cover  defensive
            raise ValueError(f"Unsupported dtype {dtype} for feature {name!r}")
    return kinds


def _scan_clean_training(
    training_dir: str,
) -> Tuple[List[str], Dict[str, str], Counter, int, int, np.ndarray]:
    """Single-pass scan that collects everything we need for the attack.

    Returns
    -------
    shards
        List of source shard paths (sorted).
    feature_kinds
        ``{name: 'int64'|'float'|'bytes'}`` for every declared feature.
    item_counts
        Counter of ``item_id -> #occurrences in sequence_data`` across training.
    max_user_id
        Highest observed integer ``user_id``.
    n_users
        Total number of training rows (one row per user).
    seq_lengths
        ``np.ndarray`` of per-user sequence lengths (used for length sampling).
    """
    shards = _list_training_shards(training_dir)

    raw_dataset = tf.data.TFRecordDataset(shards, compression_type="GZIP")
    sample_record = next(iter(raw_dataset))
    feature_description = _infer_feature_description(sample_record)
    kinds = _feature_kinds(feature_description)

    if SEQUENCE_FIELD not in feature_description:
        raise ValueError(
            f"Source training shards do not contain a {SEQUENCE_FIELD!r} feature; "
            f"got {sorted(feature_description)}"
        )
    if USER_ID_FIELD not in feature_description:
        raise ValueError(
            f"Source training shards do not contain a {USER_ID_FIELD!r} feature; "
            f"got {sorted(feature_description)}"
        )

    item_counts: Counter = Counter()
    max_user_id = -1
    n_users = 0
    seq_lengths: List[int] = []

    parsed = raw_dataset.map(
        lambda x: tf.io.parse_single_example(x, feature_description)
    )
    for example in parsed:
        seq_sparse = example[SEQUENCE_FIELD]
        seq = tf.sparse.to_dense(seq_sparse).numpy()
        if seq.size == 0:
            continue
        item_counts.update(int(x) for x in seq.tolist())
        seq_lengths.append(int(seq.size))

        uid_sparse = example[USER_ID_FIELD]
        uid_dense = tf.sparse.to_dense(uid_sparse).numpy()
        if uid_dense.size == 0:
            continue
        uid = int(uid_dense.flatten()[0])
        if uid > max_user_id:
            max_user_id = uid
        n_users += 1

    if n_users == 0:
        raise ValueError(f"Training scan found 0 users under {training_dir}")

    return (
        shards,
        kinds,
        item_counts,
        max_user_id,
        n_users,
        np.asarray(seq_lengths, dtype=np.int64),
    )


def _feature_kinds_from_one_shard(training_dir: str) -> Dict[str, str]:
    """Infer TFRecord feature dtypes from a single training shard."""
    shards = _list_training_shards(training_dir)
    raw = tf.data.TFRecordDataset([shards[0]], compression_type="GZIP")
    sample_record = next(iter(raw))
    feature_description = _infer_feature_description(sample_record)
    return _feature_kinds(feature_description)


def _scan_stats_from_inter(
    inter_path: str,
    n_clean_users: Optional[int] = None,
    chunksize: int = 2_000_000,
) -> Tuple[Counter, int, int, np.ndarray]:
    """One-pass pandas scan of a RecBole ``.inter`` file (ERASE-style).

    Mirrors ``FraudSessionGenerator`` statistics: item popularity and per-session
    click counts. Much faster than iterating millions of TFRecord rows when the
    source ``.inter`` is already on disk.

    Parameters
    ----------
    n_clean_users
        Training-session count for the poisoning-ratio denominator. Required when
        the ``.inter`` file contains more than training sessions (e.g. merged
        ``rsc15.inter``); pass ``7990324`` from ``dataset_meta.json`` for GRID
        rsc15 training.
    """
    import pandas as pd

    print(f"[bandwagon] Scanning .inter for stats (chunksize={chunksize}) ...")
    item_counts: Counter = Counter()
    session_clicks: Counter = Counter()
    max_session_id = -1

    compression = "gzip" if inter_path.endswith(".gz") else None
    sid_col: Optional[str] = None

    for chunk in pd.read_csv(
        inter_path, sep="\t", chunksize=chunksize, compression=compression
    ):
        chunk.columns = [str(c).split(":")[0] for c in chunk.columns]
        if sid_col is None:
            sid_col = "session_id" if "session_id" in chunk.columns else "user_id"
        item_counts.update(chunk["item_id"].value_counts().to_dict())
        session_clicks.update(chunk.groupby(sid_col).size().to_dict())
        chunk_max = int(chunk[sid_col].max())
        if chunk_max > max_session_id:
            max_session_id = chunk_max

    if n_clean_users is None:
        n_clean_users = len(session_clicks)

    seq_lengths = np.asarray(list(session_clicks.values()), dtype=np.int64)
    print(
        f"[bandwagon] .inter scan done: unique_items={len(item_counts)} "
        f"| sessions_in_inter={len(session_clicks)} | "
        f"n_clean_users(for ratio)={n_clean_users} | max_session_id={max_session_id}"
    )
    return item_counts, max_session_id, int(n_clean_users), seq_lengths


# ---------------------------------------------------------------------------
# Item-bin selection (popularity bins + targets) — matches the user's script
# ---------------------------------------------------------------------------


def _popularity_bins(
    item_counts: Counter,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Return ``(popular, average, unpopular, all_items)`` lists ordered by popularity.

    Mirrors ``FraudSessionGenerator._analyze_normal_sessions``: top 20 % popular,
    middle 40 % (after skipping the next 30 %), bottom 20 % unpopular.
    """
    items_by_pop = [item for item, _ in item_counts.most_common()]
    n = len(items_by_pop)
    n_popular = max(1, int(n * 0.2))
    n_skip = int(n * 0.3)
    n_average = max(1, int(n * 0.4))
    popular = items_by_pop[:n_popular]
    average = items_by_pop[n_skip : n_skip + n_average]
    n_unpopular = max(1, int(n * 0.2))
    unpopular = items_by_pop[-n_unpopular:]
    return popular, average, unpopular, items_by_pop


def _select_target_items(
    items_by_pop: List[int],
    strategy: str,
    n_target_items: int,
    rng: np.random.Generator,
) -> List[int]:
    n = len(items_by_pop)
    if strategy == "unpopular":
        bottom = items_by_pop[-max(1, int(n * 0.2)) :]
        pool = bottom
    elif strategy == "popular":
        pool = items_by_pop[: max(1, int(n * 0.05))]
    elif strategy == "random":
        pool = items_by_pop
    else:
        raise ValueError(f"Unknown target_strategy={strategy!r}")
    size = min(n_target_items, len(pool))
    selected = rng.choice(pool, size=size, replace=False)
    return [int(x) for x in selected.tolist()]


def _filler_pool(
    attack: str,
    popular: List[int],
    average: List[int],
    all_items: List[int],
) -> List[int]:
    if attack == "bandwagon":
        return popular
    if attack == "average":
        return average
    if attack in ("random", "push"):
        return all_items
    raise ValueError(f"Unknown attack={attack!r}")


# ---------------------------------------------------------------------------
# Spam-sequence construction
# ---------------------------------------------------------------------------


def _sample_session_length(
    seq_lengths: np.ndarray,
    rng: np.random.Generator,
    bot_speed_factor: float = 0.8,
    min_len: int = 4,
) -> int:
    """Lognormal-Poisson sample, clipped at observed [min, max]. Matches user's logic."""
    mean = max(min_len, float(seq_lengths.mean()) * bot_speed_factor)
    std = max(1.0, float(seq_lengths.std()))
    sigma_squared = math.log(1.0 + (std**2 / mean**2))
    mu = math.log(mean) - sigma_squared / 2
    lambda_param = float(rng.lognormal(mu, math.sqrt(sigma_squared)))
    length = int(max(min_len, rng.poisson(lambda_param)))
    return int(np.clip(length, min_len, int(seq_lengths.max())))


def _build_alternating_sequence(
    length: int,
    targets: List[int],
    fillers: List[int],
    rng: np.random.Generator,
) -> List[int]:
    """``[popular, target, popular, target, ...]`` of total ``length`` items."""
    seq: List[int] = []
    for i in range(length):
        if i % 2 == 0:
            seq.append(int(rng.choice(fillers)))
        else:
            seq.append(int(rng.choice(targets)))
    return seq


def _build_sprinkled_sequence(
    length: int,
    targets: List[int],
    fillers: List[int],
    rng: np.random.Generator,
    p_two_targets: float = 0.119,
) -> List[int]:
    """1 target item per spam session, occasionally 2 -- matches the
    rsc15_fraud_sessions_* distribution (mean per session ~1.119, max 2).

    * Number of target clicks per session is sampled as 1 with probability
      ``1 - p_two_targets`` and 2 with probability ``p_two_targets`` (capped
      to ``len(targets)`` and to a max of 2).
    * Target positions are sampled **without replacement** from the window
      ``[0.2*L, 0.9*L]`` so the two targets cannot collide and the empirical
      mean is exactly ``1 + p_two_targets``.
    * Each placed target is drawn uniformly at random (with replacement)
      from ``targets`` so the per-target click distribution stays
      near-uniform at scale (max/min ratio ~1.02-1.10 for tens of thousands
      of sessions, matching your rsc15 measurements).
    * Every other slot is filled by a uniform draw from ``fillers``
      (popular / average / random pool, depending on the attack type).
    """
    n_targets_max = min(2, len(targets))
    if length < 6:
        # Sequences this short cannot accommodate two well-separated targets.
        n_targets = min(1, n_targets_max)
    else:
        n_targets = 2 if rng.random() < p_two_targets else 1
        n_targets = min(n_targets, n_targets_max)

    lo = max(1, int(length * 0.2))
    hi = max(lo + 1, int(length * 0.9))
    candidate_positions = list(range(lo, hi))
    if n_targets > len(candidate_positions):
        n_targets = len(candidate_positions)
    if n_targets > 0:
        chosen = rng.choice(
            len(candidate_positions), size=n_targets, replace=False
        )
        target_positions = {int(candidate_positions[i]) for i in chosen.tolist()}
    else:
        target_positions = set()

    seq: List[int] = []
    for i in range(length):
        if i in target_positions:
            seq.append(int(rng.choice(targets)))
        else:
            seq.append(int(rng.choice(fillers)))
    return seq


def _build_spam_sequence(
    placement: str,
    length: int,
    targets: List[int],
    fillers: List[int],
    rng: np.random.Generator,
    p_two_targets: float = 0.119,
) -> List[int]:
    if placement == "alternating":
        return _build_alternating_sequence(length, targets, fillers, rng)
    if placement == "sprinkled":
        return _build_sprinkled_sequence(
            length, targets, fillers, rng, p_two_targets=p_two_targets
        )
    raise ValueError(f"Unknown placement={placement!r}")


# ---------------------------------------------------------------------------
# TFRecord writing
# ---------------------------------------------------------------------------


def _make_example(
    user_id: int,
    sequence: List[int],
    feature_kinds: Dict[str, str],
) -> tf.train.Example:
    """Assemble a ``tf.train.Example`` matching the source schema.

    Fields other than ``user_id`` / ``sequence_data`` are emitted as empty
    lists of the correct dtype so per-shard schema inference stays consistent
    with clean shards.
    """
    feature: Dict[str, tf.train.Feature] = {}
    for name, kind in feature_kinds.items():
        if name == USER_ID_FIELD:
            feature[name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(user_id)])
            )
        elif name == SEQUENCE_FIELD:
            feature[name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(x) for x in sequence])
            )
        elif kind == "int64":
            feature[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=[]))
        elif kind == "float":
            feature[name] = tf.train.Feature(float_list=tf.train.FloatList(value=[]))
        elif kind == "bytes":
            feature[name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))
        else:  # pragma: no cover — defensive
            raise ValueError(f"Unsupported kind {kind!r} for feature {name!r}")
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _write_spam_shards(
    out_training_dir: str,
    spam_user_ids: List[int],
    spam_sequences: List[List[int]],
    feature_kinds: Dict[str, str],
    rows_per_shard: int,
) -> List[str]:
    """Write spam rows split across ``data_spam_<i>.tfrecord.gz`` shards."""
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    written: List[str] = []
    n_total = len(spam_user_ids)
    if n_total == 0:
        return written
    n_shards = max(1, math.ceil(n_total / rows_per_shard))
    for shard_idx in range(n_shards):
        start = shard_idx * rows_per_shard
        end = min(n_total, start + rows_per_shard)
        path = os.path.join(out_training_dir, f"data_spam_{shard_idx}.tfrecord.gz")
        with tf.io.TFRecordWriter(path, options=options) as writer:
            for i in range(start, end):
                example = _make_example(
                    spam_user_ids[i], spam_sequences[i], feature_kinds
                )
                writer.write(example.SerializeToString())
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _default_out_dir(data_dir: str, seed: int, ratio: float, n_targets: int) -> str:
    parent = os.path.dirname(os.path.abspath(data_dir.rstrip("/"))) or "."
    base = os.path.basename(os.path.abspath(data_dir.rstrip("/")))
    pct = int(round(ratio * 100))
    return os.path.join(parent, f"{base}_spam_seed{seed}_pct{pct}_n{n_targets}")


def _copy_clean_shards(src_training: str, dst_training: str, src_shards: List[str]) -> None:
    os.makedirs(dst_training, exist_ok=True)
    for shard in src_shards:
        dst = os.path.join(dst_training, os.path.basename(shard))
        shutil.copy2(shard, dst)


def _copy_sibling_subdirs(src_dir: str, dst_dir: str) -> None:
    for sub in SIBLING_SUBDIRS:
        s = os.path.join(src_dir, sub)
        if not os.path.isdir(s):
            continue
        d = os.path.join(dst_dir, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_dir: str,
    out_dir: Optional[str],
    attack: str,
    target_strategy: str,
    poisoning_ratio: float,
    n_target_items: int,
    placement: str,
    seed: int,
    rows_per_shard: int,
    overwrite: bool,
    p_two_targets: float = 0.119,
    stats_inter: Optional[str] = None,
    n_clean_users: Optional[int] = None,
    deletion_spec: str = "session",
) -> str:
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    training_dir = os.path.join(data_dir, TRAINING_SUBDIR)
    if not os.path.isdir(training_dir):
        raise FileNotFoundError(f"Expected training dir at {training_dir}")

    if stats_inter:
        if not os.path.isfile(stats_inter):
            raise FileNotFoundError(f"--stats-inter not found: {stats_inter}")
        clean_shards = _list_training_shards(training_dir)
        feature_kinds = _feature_kinds_from_one_shard(training_dir)
        item_counts, max_user_id, n_clean_users, seq_lengths = _scan_stats_from_inter(
            stats_inter, n_clean_users=n_clean_users
        )
        # _scan_stats_from_inter returns item_counts keyed by raw item IDs (e.g.
        # 214844368).  The TFRecords store sequential IDs (0..N-1), so we must
        # remap before selecting targets/fillers, otherwise item IDs written to
        # spam shards will be out-of-bounds for the codebook tensor at training.
        id_map_path = os.path.join(data_dir, "item_id_map.json")
        if os.path.isfile(id_map_path):
            with open(id_map_path, encoding="utf-8") as _f:
                _id_map_data = json.load(_f)
            raw_to_seq = {
                raw: seq
                for seq, raw in enumerate(_id_map_data["seq_to_raw"])
            }
            item_counts = Counter(
                {raw_to_seq[k]: v for k, v in item_counts.items() if k in raw_to_seq}
            )
            print(
                f"[bandwagon] Remapped item_counts from raw IDs to sequential IDs "
                f"using {id_map_path} ({len(item_counts)} items retained)."
            )
        else:
            print(
                f"[bandwagon] WARNING: {id_map_path} not found. "
                "item_counts keys are raw item IDs which will cause an "
                "out-of-bounds crash at training time. "
                "Re-generate the dataset with convert_rsc15_inter.py to create "
                "item_id_map.json, or use the slow path (omit --stats-inter)."
            )
    else:
        print(f"[bandwagon] Scanning clean training shards under {training_dir} ...")
        (
            clean_shards,
            feature_kinds,
            item_counts,
            max_user_id,
            n_clean_users,
            seq_lengths,
        ) = _scan_clean_training(training_dir)
    print(
        f"[bandwagon] features={list(feature_kinds.keys())} | "
        f"users={n_clean_users} | unique_items={len(item_counts)} | "
        f"max_user_id={max_user_id} | seq_len mean={seq_lengths.mean():.2f} "
        f"std={seq_lengths.std():.2f} min={seq_lengths.min()} max={seq_lengths.max()}"
    )

    sessions_to_add = math.ceil(
        poisoning_ratio * n_clean_users / max(1.0 - poisoning_ratio, 1e-9)
    )
    if sessions_to_add <= 0:
        raise ValueError(
            f"poisoning_ratio={poisoning_ratio} produces 0 spam sessions "
            f"for n_clean_users={n_clean_users}; choose a larger ratio."
        )
    print(
        f"[bandwagon] Will add {sessions_to_add} spam users to hit "
        f"poisoning_ratio={poisoning_ratio:.4f}"
    )

    popular, average, unpopular, items_by_pop = _popularity_bins(item_counts)
    target_items = _select_target_items(
        items_by_pop, target_strategy, n_target_items, rng
    )
    fillers = _filler_pool(attack, popular, average, items_by_pop)
    print(
        f"[bandwagon] popular={len(popular)} avg={len(average)} unpopular={len(unpopular)} "
        f"| targets={target_items[:5]}{'...' if len(target_items) > 5 else ''} "
        f"| filler_pool={len(fillers)}"
    )

    spam_user_ids: List[int] = []
    spam_sequences: List[List[int]] = []
    target_set = set(target_items)
    n_target_clicks_per_session: List[int] = []
    for i in range(sessions_to_add):
        length = _sample_session_length(seq_lengths, rng)
        seq = _build_spam_sequence(
            placement,
            length,
            target_items,
            fillers,
            rng,
            p_two_targets=p_two_targets,
        )
        spam_user_ids.append(int(max_user_id + 1 + i))
        spam_sequences.append(seq)
        n_target_clicks_per_session.append(sum(1 for x in seq if x in target_set))
    n_clicks = sum(len(s) for s in spam_sequences)
    if n_target_clicks_per_session:
        tgt_arr = np.asarray(n_target_clicks_per_session)
        target_stats = (
            f"target clicks/session: mean={tgt_arr.mean():.3f} "
            f"min={tgt_arr.min()} max={tgt_arr.max()}"
        )
    else:
        target_stats = "target clicks/session: n/a"
    print(
        f"[bandwagon] Generated {len(spam_user_ids)} spam users, "
        f"{n_clicks} total spam clicks "
        f"({n_clicks / max(1, len(spam_user_ids)):.2f} clicks/user) | "
        f"{target_stats}"
    )

    if out_dir is None:
        out_dir = _default_out_dir(
            data_dir, seed=seed, ratio=poisoning_ratio, n_targets=n_target_items
        )
    out_training_dir = os.path.join(out_dir, TRAINING_SUBDIR)
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {out_dir} already exists; pass --overwrite to replace."
            )
        shutil.rmtree(out_dir)
    os.makedirs(out_training_dir, exist_ok=True)

    print(f"[bandwagon] Copying {len(clean_shards)} clean training shards -> {out_training_dir}")
    _copy_clean_shards(training_dir, out_training_dir, clean_shards)

    print(f"[bandwagon] Writing spam shards (rows_per_shard={rows_per_shard}) ...")
    spam_shard_paths = _write_spam_shards(
        out_training_dir,
        spam_user_ids=spam_user_ids,
        spam_sequences=spam_sequences,
        feature_kinds=feature_kinds,
        rows_per_shard=rows_per_shard,
    )
    print(f"[bandwagon] Wrote {len(spam_shard_paths)} spam shards.")

    print(f"[bandwagon] Copying sibling dirs ({', '.join(SIBLING_SUBDIRS)}) ...")
    _copy_sibling_subdirs(data_dir, out_dir)

    manifest = {
        "spam_user_ids": spam_user_ids,
        "target_items": target_items,
        "attack_type": attack,
        "target_strategy": target_strategy,
        "placement": placement,
        "p_two_targets": float(p_two_targets) if placement == "sprinkled" else None,
        "poisoning_ratio": poisoning_ratio,
        "n_target_items": n_target_items,
        "n_clean_users": int(n_clean_users),
        "n_spam_users": int(len(spam_user_ids)),
        "max_clean_user_id": int(max_user_id),
        "first_spam_user_id": int(max_user_id + 1),
        "last_spam_user_id": int(max_user_id + len(spam_user_ids)),
        "seed": int(seed),
        "rows_per_spam_shard": int(rows_per_shard),
        "spam_shard_paths": [os.path.relpath(p, out_dir) for p in spam_shard_paths],
        "source_dataset": os.path.abspath(data_dir),
        "schema_features": sorted(feature_kinds.keys()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "target_clicks_mean": (
            float(np.mean(n_target_clicks_per_session))
            if n_target_clicks_per_session else None
        ),
        "target_clicks_min": (
            int(min(n_target_clicks_per_session))
            if n_target_clicks_per_session else None
        ),
        "target_clicks_max": (
            int(max(n_target_clicks_per_session))
            if n_target_clicks_per_session else None
        ),
        "deletion_spec": str(deletion_spec).strip().lower(),
    }
    manifest_path = os.path.join(out_dir, "forget_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[bandwagon] Wrote manifest -> {manifest_path}")
    print(f"[bandwagon] Done. Poisoned dataset at {out_dir}")
    return out_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a bandwagon-poisoned TIGER dataset (TFRecord shards)."
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Source clean dataset dir (e.g. src/data/amazon_data/beauty)",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output dir; defaults to <data_dir>_spam_seed<S>_pct<P>_n<C>",
    )
    p.add_argument(
        "--attack",
        choices=("bandwagon", "random", "average", "push"),
        default="bandwagon",
    )
    p.add_argument(
        "--target_strategy",
        choices=("unpopular", "popular", "random"),
        default="unpopular",
    )
    p.add_argument("--poisoning_ratio", type=float, default=0.01)
    p.add_argument("--n_target_items", type=int, default=10)
    p.add_argument(
        "--placement",
        choices=("sprinkled", "alternating"),
        default="sprinkled",
        help=(
            "Spam-sequence pattern. 'sprinkled' (default) matches the "
            "rsc15_fraud_sessions_* distribution: 1 target per session "
            "(occasionally 2) at random positions in [0.2*L, 0.9*L]. "
            "'alternating' interleaves [filler, target, filler, target, ...] "
            "for the entire sequence -- much stronger attack."
        ),
    )
    p.add_argument(
        "--p_two_targets",
        type=float,
        default=0.119,
        help=(
            "Probability that a sprinkled spam session contains 2 targets "
            "instead of 1. Default 0.119 reproduces the rsc15_fraud_sessions_* "
            "empirical mean of ~1.119 target clicks per session "
            "(min 1, max 2). Ignored when --placement=alternating."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rows_per_shard",
        type=int,
        default=1024,
        help="Rows per spam tfrecord shard.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--stats-inter",
        default=None,
        help=(
            "RecBole .inter for ERASE-fast stats (one pandas pass). "
            "Skips scanning all training TFRecords. For merged rsc15.inter also pass "
            "--n-clean-users from dataset_meta.json training split size."
        ),
    )
    p.add_argument(
        "--n-clean-users",
        type=int,
        default=None,
        help="Training-session count for poisoning_ratio (required with merged .inter).",
    )
    p.add_argument(
        "--deletion_spec",
        default="session",
        choices=["session", "item"],
        help="Default deletion specification stored in forget_manifest.json.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        attack=args.attack,
        target_strategy=args.target_strategy,
        poisoning_ratio=args.poisoning_ratio,
        n_target_items=args.n_target_items,
        placement=args.placement,
        seed=args.seed,
        rows_per_shard=args.rows_per_shard,
        overwrite=args.overwrite,
        p_two_targets=args.p_two_targets,
        stats_inter=args.stats_inter,
        n_clean_users=args.n_clean_users,
        deletion_spec=args.deletion_spec,
    )
