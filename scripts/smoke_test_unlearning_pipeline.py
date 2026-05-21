"""End-to-end smoke test for the TIGER unlearning data pipeline.

Builds a tiny synthetic TFRecord dataset under ``/tmp`` that mirrors TIGER's
schema (``user_id`` + ``sequence_data`` + empty ``embedding`` / ``text``),
then walks the full pipeline:

1. ``src.data.poisoning.bandwagon`` produces a poisoned dataset directory and
   ``forget_manifest.json``.
2. ``src.data.unlearning.split_forget_retain`` partitions training shards
   into ``training_forget`` / ``training_retain``.
3. ``src.components.unlearning.neighborhood_sampler.build_retain_subset``
   produces a (capped + optionally SID-prefix-filtered) retain subset and a
   bookkeeping JSON.
4. ``src.components.unlearning.{target_params, hvp, scif}`` are exercised
   against a *toy nn.Module* on a single dummy batch, verifying that a SCIF
   parameter update runs without raising.

Skip the actual TIGER ckpt + Hydra entry -- those need a multi-GB dataset
and a real GPU run. The data pipeline + algorithm bits are what we want to
validate quickly.

Run::

    python -m scripts.smoke_test_unlearning_pipeline
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from src.components.unlearning.hvp import (
    batch_grad,
    cg_inv_hvp,
    hvp_single,
)
from src.components.unlearning.neighborhood_sampler import build_retain_subset
from src.components.unlearning.scif import scif_unlearn
from src.components.unlearning.target_params import select_target_params
from src.data.unlearning.deletion_spec import (
    normalize_deletion_spec,
    resolve_neighborhood_centers,
)
from src.data.poisoning import bandwagon as bw
from src.data.unlearning import split_forget_retain as sfr


SEED = 1234
NUM_USERS = 60
NUM_ITEMS = 200
SEQ_LEN_RANGE = (5, 30)
NUM_HIERARCHIES = 3
NUM_EMB_PER_HIERARCHY = 16
SHARDS_PER_DIR = 3


# ---------------------------------------------------------------------------
# Synthetic dataset construction
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(root: str) -> Dict[str, str]:
    """Create a tiny TIGER-style dataset under ``root``."""
    rng = np.random.default_rng(SEED)
    users: List[Tuple[int, List[int]]] = []
    for uid in range(NUM_USERS):
        seq_len = int(rng.integers(*SEQ_LEN_RANGE))
        users.append((uid, [int(x) for x in rng.integers(0, NUM_ITEMS, size=seq_len).tolist()]))

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    for sub in ("training", "evaluation", "testing", "items"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    def _write(rows: List[Tuple[int, List[int]]], out_dir: str) -> None:
        per_shard = max(1, len(rows) // SHARDS_PER_DIR)
        for shard_idx in range(SHARDS_PER_DIR):
            start = shard_idx * per_shard
            end = (shard_idx + 1) * per_shard if shard_idx < SHARDS_PER_DIR - 1 else len(rows)
            path = os.path.join(out_dir, f"partition_{shard_idx}.tfrecord.gz")
            with tf.io.TFRecordWriter(path, options=options) as w:
                for uid, seq in rows[start:end]:
                    feat = {
                        "user_id": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[uid])
                        ),
                        "sequence_data": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=seq)
                        ),
                        "embedding": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[])
                        ),
                        "text": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[])
                        ),
                    }
                    ex = tf.train.Example(features=tf.train.Features(feature=feat))
                    w.write(ex.SerializeToString())

    _write(users, os.path.join(root, "training"))
    _write(users[: NUM_USERS // 4], os.path.join(root, "evaluation"))
    _write(users[: NUM_USERS // 4], os.path.join(root, "testing"))
    # items/ schema: a single int item_id field is enough for our smoke test.
    items_path = os.path.join(root, "items", "partition_0.tfrecord.gz")
    with tf.io.TFRecordWriter(items_path, options=options) as w:
        for i in range(NUM_ITEMS):
            ex = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "item_id": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[i])
                        ),
                    }
                )
            )
            w.write(ex.SerializeToString())

    # Synthetic codebook: each item gets a random hierarchical SID.
    codebook = torch.tensor(
        rng.integers(0, NUM_EMB_PER_HIERARCHY, size=(NUM_ITEMS, NUM_HIERARCHIES)),
        dtype=torch.long,
    )
    codebook_path = os.path.join(root, "merged_predictions_tensor.pt")
    torch.save(codebook, codebook_path)

    return {"data_dir": root, "codebook": codebook_path}


# ---------------------------------------------------------------------------
# Toy SCIF target (a tiny linear model with the same model_step contract)
# ---------------------------------------------------------------------------


class _ToyModel(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=False)

    def model_step(self, model_input, label_data):
        # `model_input` is a synthetic dummy with `.transformed_sequences['x']`
        x = model_input["x"]
        y = label_data["y"]
        out = self.linear(x)
        loss = ((out - y) ** 2).mean()
        return out, loss


def _make_dummy_batch(n: int, dim: int, device: torch.device):
    rng = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn(n, dim, device=device, generator=rng)
    y = torch.randn(n, dim, device=device, generator=rng)
    return ({"x": x}, {"y": y})


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _step_bandwagon(data_dir: str) -> str:
    out_dir = data_dir + "_spam"
    bw.main(
        data_dir=data_dir,
        out_dir=out_dir,
        attack="bandwagon",
        target_strategy="unpopular",
        poisoning_ratio=0.1,
        n_target_items=5,
        placement="sprinkled",
        seed=SEED,
        rows_per_shard=8,
        overwrite=True,
    )
    manifest_path = os.path.join(out_dir, "forget_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["n_spam_users"] > 0, "bandwagon: no spam users produced"
    assert len(manifest["spam_user_ids"]) == manifest["n_spam_users"]
    assert os.path.isdir(os.path.join(out_dir, "training"))
    assert os.path.isdir(os.path.join(out_dir, "evaluation"))
    assert os.path.isdir(os.path.join(out_dir, "testing"))
    assert os.path.isdir(os.path.join(out_dir, "items"))
    print(
        f"  [step1] bandwagon OK: "
        f"{manifest['n_spam_users']} spam users, "
        f"{len(manifest['target_items'])} target items, "
        f"first_uid={manifest['first_spam_user_id']}, "
        f"last_uid={manifest['last_spam_user_id']}"
    )
    return out_dir


def _step_split(data_dir: str) -> Dict[str, str]:
    sfr.main(
        data_dir=data_dir,
        forget_manifest=os.path.join(data_dir, "forget_manifest.json"),
        forget_user_ids=None,
        out_subdir_forget="training_forget",
        out_subdir_retain="training_retain",
        rows_per_shard=8,
        overwrite=True,
    )
    forget_dir = os.path.join(data_dir, "training_forget")
    retain_dir = os.path.join(data_dir, "training_retain")
    bookkeeping = os.path.join(data_dir, "forget_retain_split.json")
    assert os.path.isfile(bookkeeping)
    with open(bookkeeping) as f:
        info = json.load(f)
    assert info["n_forget_rows"] > 0, "split: no forget rows produced"
    assert info["n_retain_rows"] > 0, "split: no retain rows produced"
    print(
        f"  [step2] split OK: forget={info['n_forget_rows']} rows, "
        f"retain={info['n_retain_rows']} rows"
    )
    return {"forget_dir": forget_dir, "retain_dir": retain_dir}


def _step_neighborhood(
    data_dir: str,
    forget_dir: str,
    retain_dir: str,
    codebook_path: str,
) -> None:
    for mode, label in ((True, "neighborhood"), (False, "baseline")):
        out_dir = os.path.join(data_dir, f"training_retain_subset_{label}")
        info = build_retain_subset(
            forget_dir=forget_dir,
            retain_dir=retain_dir,
            out_dir=out_dir,
            neighborhood_aware=mode,
            semantic_id_path=codebook_path if mode else None,
            sid_prefix_length=2,
            forget_size=None,
            neighbor_aware_factor=8.0,
            retain_samples_used_for_update=16,
            rows_per_shard=8,
            seed=SEED,
            overwrite=True,
        )
        assert info["n_retain_rows_kept"] >= 0
        if mode:
            assert info["n_neighbor_items"] is not None
        print(
            f"  [step3:{label}] subset OK: kept={info['n_retain_rows_kept']} / "
            f"seen={info['n_retain_rows_seen']} cap={info['max_rows']}"
        )


def _step_scif_components() -> None:
    """Drive ``select_target_params`` / ``batch_grad`` / ``hvp_single`` /
    ``cg_inv_hvp`` / ``scif_unlearn`` against the toy model + a fixed batch.
    """
    device = torch.device("cpu")
    dim = 4
    model = _ToyModel(dim=dim).to(device)

    forget_batch = _make_dummy_batch(n=4, dim=dim, device=device)
    retain_batch = _make_dummy_batch(n=8, dim=dim, device=device)

    params = select_target_params(model, policy="all")
    assert len(params) >= 1

    g = batch_grad(model, forget_batch, params, average_scale=4.0)
    assert all(t.shape == p.shape for t, p in zip(g, params))

    h = hvp_single(model, retain_batch, g, params)
    assert all(t.shape == p.shape for t, p in zip(h, params))

    x, info = cg_inv_hvp(
        model=model,
        hvp_batches=[retain_batch],
        v_list=g,
        params=params,
        damping=0.1,
        max_iter=10,
        tol=1e-8,
    )
    assert info["iters"] > 0

    before = [p.detach().clone() for p in params]
    scif_info = scif_unlearn(
        model=model,
        forget_batches=[forget_batch],
        retain_batches=[retain_batch, retain_batch],
        forget_size=4,
        retain_size=16,
        retain_samples_used_for_update=4,
        cg_max_iter=10,
        cg_tol=1e-8,
        cg_damping=0.1,
        target_params_policy="all",
        eval_mode=False,
    )
    after = [p.detach().clone() for p in params]
    moved = any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert moved, "SCIF update did not move any parameter"
    print(
        f"  [step4] SCIF OK on toy model: "
        f"updated={scif_info['n_param_tensors_updated']} / "
        f"skipped={scif_info['n_param_tensors_skipped_nan']} | "
        f"CG iters={scif_info['cg']['iters']} converged={scif_info['cg']['converged']}"
    )


def _step_baseline_dispatch() -> None:
    """Verify baseline modules and deletion_spec helpers."""
    assert normalize_deletion_spec("session") == "session"
    assert normalize_deletion_spec("item") == "item"
    centers = resolve_neighborhood_centers(
        deletion_spec="item",
        forget_shard_items={1, 2, 3},
        target_items={10, 11},
    )
    assert centers == {10, 11}
    from src.components.unlearning.filter_utils import build_filter_mask

    mask = build_filter_mask(
        deletion_spec="item",
        target_items={10, 11},
        forget_shard_items={1, 2, 3},
        filter_mode="global",
    )
    assert mask["forbidden_item_ids"] == [10, 11]
    print("  [step5] baseline dispatch OK (deletion_spec, filter mask)")


def main() -> None:
    print("[smoke] Running TIGER unlearning pipeline smoke test ...")
    tmpdir = tempfile.mkdtemp(prefix="tiger_unlearn_smoke_")
    try:
        clean_root = os.path.join(tmpdir, "synthetic_beauty")
        os.makedirs(clean_root, exist_ok=True)

        print("[smoke] Building synthetic dataset ...")
        artifacts = _make_synthetic_dataset(clean_root)
        print(f"  data_dir = {artifacts['data_dir']}")

        spam_dir = _step_bandwagon(artifacts["data_dir"])
        split = _step_split(spam_dir)
        _step_neighborhood(
            data_dir=spam_dir,
            forget_dir=split["forget_dir"],
            retain_dir=split["retain_dir"],
            codebook_path=artifacts["codebook"],
        )
        _step_scif_components()
        _step_baseline_dispatch()
        print("[smoke] All pipeline steps passed.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
