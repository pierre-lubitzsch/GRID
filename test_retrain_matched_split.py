"""CPU tests for unlearning.split_mode=retrain_matched (sensitive item deletion).

retrain_matched: forget set = only the training rows whose LABEL is a deleted
interaction (original histories); retain set = every user incl. the requesting
users with exactly the deleted items removed, i.e. the retrained reference's
training data (scripts/build_retrain_dataset.py).

  python -m pytest -q test_retrain_matched_split.py          # synthetic + one real cell
  RMSPLIT_CELLS=all python -m pytest -q test_retrain_matched_split.py   # all 15 cells

The real-data part checks, per cell (default Beauty seed 3):
  (a) the retain pool (retrain dir training/) holds, for every requesting user,
      exactly their original training sequence with the manifest's target items
      removed (users left with < 2 items are absent, as in the retrained model),
      and for every other user the byte-identical row of training_retain/;
  (b) the forget rows produced by the real pipeline pieces (item-pairs
      materialisation with include_context_rows=false + last_label_only collate)
      are exactly the requesting users' training rows (all contiguous
      subsequences, as the training collate expands them) whose label is a
      target item, with their original histories.
"""
from __future__ import annotations

import itertools
import json
import os
import tempfile
from collections import Counter
from typing import Dict, List

import pytest
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
AD = os.path.join(ROOT, "src/data/amazon_data")
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}


# ---------------------------------------------------------------------------
# collate: last_label_only
# ---------------------------------------------------------------------------


def _expand(seqs: List[List[int]], H: int, **kw) -> List[tuple]:
    """Run the real collate with collate_fn_train stubbed to return its batch."""
    import src.utils  # noqa: F401  (import order: avoids the interfaces <-> utils cycle)
    import src.data.loading.components.collate_functions as cf

    orig = cf.collate_fn_train
    cf.collate_fn_train = lambda batch, **_: batch
    try:
        out = cf.collate_with_sid_causal_duplicate(
            batch={"seq": [torch.tensor(s) for s in seqs], "uid": [torch.tensor(i) for i in range(len(seqs))]},
            sequence_field_name="seq", sid_hierarchy=H, labels={}, max_batch_size=10 ** 9, **kw)
    finally:
        cf.collate_fn_train = orig
    return [tuple(int(x) for x in t) for t in out["seq"]]


def _all_subseqs_items(seq: List[int]) -> List[tuple]:
    n = len(seq)
    return [tuple(seq[s:e + 1]) for e in range(1, n) for s in range(0, e)]


def test_collate_default_unchanged_and_last_label_only():
    seqs = [[1, 2, 3, 4, 5], [7, 8], [9]]
    # H=1: default = every contiguous subsequence with >= 2 items
    full = _expand(seqs, 1)
    assert Counter(full) == Counter(itertools.chain.from_iterable(_all_subseqs_items(s) for s in seqs))
    last = _expand(seqs, 1, last_label_only=True)
    assert Counter(last) == Counter([(1, 2, 3, 4, 5), (2, 3, 4, 5), (3, 4, 5), (4, 5), (7, 8)])
    # H=2 (two tokens per item): same structure on item boundaries
    tok = [[10, 11, 20, 21, 30, 31]]
    assert Counter(_expand(tok, 2, last_label_only=True)) == Counter(
        [(10, 11, 20, 21, 30, 31), (20, 21, 30, 31)])
    assert len(_expand(tok, 2)) == 3


# ---------------------------------------------------------------------------
# helpers on TFRecord dirs
# ---------------------------------------------------------------------------


def _read_dir(d: str) -> Dict[int, bytes]:
    import tensorflow as tf
    out: Dict[int, bytes] = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".tfrecord.gz"):
            continue
        for rec in tf.data.TFRecordDataset(os.path.join(d, f), compression_type="GZIP"):
            ex = tf.train.Example.FromString(rec.numpy())
            uid = int(ex.features.feature["user_id"].int64_list.value[0])
            assert uid not in out, (d, uid)
            out[uid] = rec.numpy()
    return out


def _ex(raw: bytes):
    import tensorflow as tf
    return tf.train.Example.FromString(raw)


def _seq(raw: bytes) -> List[int]:
    import tensorflow as tf
    return [int(x) for x in tf.train.Example.FromString(raw).features.feature["sequence_data"].int64_list.value]


def test_materialize_request_dirs_keeps_forget_uids_when_asked():
    from src.data.unlearning.sequential_split import (index_tfrecord_dir_by_user_id,
                                                      materialize_request_dirs)
    from src.data.unlearning.split_forget_retain import _ShardWriter
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp:
        def write(d, rows):
            w = _ShardWriter(d, "data", 4096)
            for uid, seq in rows:
                ex = tf.train.Example()
                ex.features.feature["user_id"].int64_list.value.append(uid)
                ex.features.feature["sequence_data"].int64_list.value.extend(seq)
                w.write(ex.SerializeToString())
            w.close()
        os.makedirs(f"{tmp}/f"); os.makedirs(f"{tmp}/r")
        write(f"{tmp}/f", [(1, [5, 99, 6])])
        write(f"{tmp}/r", [(1, [5, 6]), (2, [7, 8])])
        fi, ri = index_tfrecord_dir_by_user_id(f"{tmp}/f"), index_tfrecord_dir_by_user_id(f"{tmp}/r")
        a = materialize_request_dirs(request_dir=f"{tmp}/a", forget_index=fi, retain_index=ri, forget_uids=[1])
        b = materialize_request_dirs(request_dir=f"{tmp}/b", forget_index=fi, retain_index=ri, forget_uids=[1],
                                     exclude_forget_from_retain=False)
        assert a["n_retain_rows_written"] == 1 and not a["retain_reused_source"]
        assert b["n_retain_rows_written"] == 2 and b["retain_reused_source"]
        assert set(_read_dir(f"{tmp}/b/training_retain")) == {1, 2}


# ---------------------------------------------------------------------------
# real cell(s)
# ---------------------------------------------------------------------------


def _cells():
    if os.environ.get("RMSPLIT_CELLS") == "all":
        return [(ds, sd) for ds in CAT for sd in (2, 3, 5, 7, 11)]
    return [("beauty", 3)]


@pytest.mark.parametrize("ds,sd", _cells())
def test_real_cell_retain_and_forget_rows(ds, sd):
    cat = CAT[ds]
    sens = f"{AD}/{ds}_sens_{cat}_1e-4_seed{sd}"
    retr = f"{AD}/{ds}_retrain_{cat}_1e-4_seed{sd}"
    man_p = f"{AD}/{ds}/forget_manifest_{cat}_1e-4_seed{sd}.json"
    if not (os.path.isdir(sens) and os.path.isdir(retr) and os.path.isfile(man_p)):
        pytest.skip("cell data not present")
    man = json.load(open(man_p))
    targets = {int(i) for i in man["target_items"]}
    users = {int(u) for u in man["spam_user_ids"]}
    meta = json.load(open(f"{retr}/retrain_meta.json"))
    assert os.path.realpath(meta["manifest"]) == os.path.realpath(man_p)

    full = _read_dir(f"{sens}/training")          # clean training (symlink)
    forget = _read_dir(f"{sens}/training_forget")
    retain = _read_dir(f"{sens}/training_retain")
    pool = _read_dir(f"{retr}/training")          # retrain_matched retain pool
    assert set(forget) == users and set(forget) | set(retain) == set(full)

    # (a) retain pool == retrained reference's training rows
    n_req = 0
    for uid, raw in full.items():
        if uid in users:
            kept = [i for i in _seq(raw) if i not in targets]
            assert _seq(forget[uid]) == _seq(raw)
            if len(kept) < 2:
                assert uid not in pool
            else:
                assert _seq(pool[uid]) == kept
                n_req += 1
        else:
            assert _ex(pool[uid]) == _ex(retain[uid]), uid     # identical Example (all features)
    assert set(pool) <= set(full) and len(pool) == len(retain) + n_req

    # (b) forget rows == all training rows of requesting users with a deleted label
    from src.data.unlearning.forget_target_filter import materialize_item_pairs_forget_dir
    with tempfile.TemporaryDirectory() as tmp:
        materialize_item_pairs_forget_dir(forget_dir=f"{sens}/training_forget", out_dir=f"{tmp}/ip",
                                          target_items=targets, include_context_rows=False)
        rows = [_seq(r) for r in _read_dir(f"{tmp}/ip").values()]
    got = set(_expand(rows, 1, last_label_only=True))
    want = set()
    for uid in users:
        want |= {t for t in _all_subseqs_items(_seq(forget[uid])) if t[-1] in targets}
    assert got == want
    assert all(t[-1] in targets for t in got)
    # every deleted training interaction with a history (position >= 1) is a label somewhere
    n_del = sum(1 for uid in users for j, i in enumerate(_seq(forget[uid])) if j >= 1 and i in targets)
    assert n_del > 0 and len({t[-1] for t in got}) <= n_del
