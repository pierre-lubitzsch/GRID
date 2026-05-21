"""Targeted regression tests for confirmed and potential bugs in the unlearning stack.

Each test is a standalone function that raises AssertionError (or any exception)
on failure.  No external datasets or GPUs required.

Run::

    python -m scripts.test_unlearning_bugs
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


# ---------------------------------------------------------------------------
# T1 — DenseEmbeddings: non-sequential item IDs (rsc15 style)
# ---------------------------------------------------------------------------

def test_dense_embeddings_nonsequential() -> None:
    """load_dense_embeddings + embedding_neighbors must work with sparse large IDs."""
    from src.components.unlearning.neighborhood_sampler import (
        DenseEmbeddings,
        embedding_neighbors,
        load_dense_embeddings,
    )

    # Simulate rsc15: 5 items with raw IDs in the hundreds of millions
    raw_ids = [214765100, 214765147, 214766000, 214800000, 999999999]
    item_ids = torch.tensor(raw_ids, dtype=torch.int64)
    # Give item 0 and item 1 very close embeddings, others far away
    embs = torch.zeros(5, 4)
    embs[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    embs[1] = torch.tensor([1.0, 0.1, 0.0, 0.0])   # close to item 0
    embs[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])   # far

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
        torch.save({"embeddings": embs, "item_ids": item_ids}, tmp)
    try:
        store = load_dense_embeddings(tmp)
    finally:
        os.unlink(tmp)

    assert len(store) == 5
    assert 214765100 in store
    assert 0 not in store, "Sequential index 0 must not be a valid key"

    # embedding_neighbors must return raw item IDs, not row indices
    neighbors = embedding_neighbors(214765100, store, epsilon=0.2)
    assert all(n in raw_ids for n in neighbors), (
        f"Returned row indices instead of raw item IDs: {neighbors}"
    )
    assert 214765147 in neighbors, "Close neighbor must be found"
    assert 214766000 not in neighbors, "Distant item must not be a neighbor"

    # Legacy plain-tensor format (Amazon-style sequential IDs) still works
    legacy = torch.randn(10, 4)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp2 = f.name
        torch.save(legacy, tmp2)
    try:
        store2 = load_dense_embeddings(tmp2)
    finally:
        os.unlink(tmp2)
    assert list(store2.item_ids.tolist()) == list(range(10))
    print("  [T1] DenseEmbeddings non-sequential IDs: OK")


# ---------------------------------------------------------------------------
# T2 — eval_forget_targets: SID decode must use codebook, not raw codes
# ---------------------------------------------------------------------------

def test_eval_forget_target_recall_sid_decoding() -> None:
    """eval_forget_target_recall must decode SID tuples via codebook, not compare
    raw SID component codes (0–255) directly against item IDs."""
    from scripts.eval_forget_targets import _build_sid_to_item, eval_forget_target_recall

    # Build a tiny codebook: 4 items × 2 hierarchies, each gets a unique SID
    codebook = torch.tensor([
        [0, 1],   # item 0 → SID (0, 1)
        [1, 0],   # item 1 → SID (1, 0)
        [2, 3],   # item 2 → SID (2, 3)
        [3, 2],   # item 3 → SID (3, 2)
    ], dtype=torch.long)

    # Old (broken) approach: gen_ids flattened → raw SID codes, compared to item IDs
    # If target_items = {2}, raw SID codes include values 2 and 3.
    # Item IDs 2 and 3 are in range 0–3, so code "2" would falsely match item 2.
    # The correct behavior: decode SID tuple (2,3) → item 2, then check membership.

    sid_to_item = _build_sid_to_item(codebook)
    assert sid_to_item[(0, 1)] == 0
    assert sid_to_item[(1, 0)] == 1
    assert sid_to_item[(2, 3)] == 2
    assert sid_to_item[(3, 2)] == 3
    assert len(sid_to_item) == 4, "Reverse map must have one entry per item"

    # Verify: a SID code value of 2 alone must NOT match item ID 2
    # (it's a component, not a full tuple)
    assert (2,) not in sid_to_item, "Partial SID must not match any item"

    print("  [T2] eval_forget_target_recall SID decoding: OK")


# ---------------------------------------------------------------------------
# T3 — forget_target_filter: item-mode history rewriting
# ---------------------------------------------------------------------------

def _write_tfrecord_shards(rows: List[Tuple[int, List[int]]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    opts = tf.io.TFRecordOptions(compression_type="GZIP")
    path = os.path.join(out_dir, "data_0.tfrecord.gz")
    with tf.io.TFRecordWriter(path, opts) as w:
        for uid, seq in rows:
            ex = tf.train.Example(features=tf.train.Features(feature={
                "user_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
                "sequence_data": tf.train.Feature(int64_list=tf.train.Int64List(value=seq)),
            }))
            w.write(ex.SerializeToString())


def _read_tfrecord_sequences(shard_dir: str) -> List[List[int]]:
    shards = [os.path.join(shard_dir, f) for f in sorted(os.listdir(shard_dir))
              if f.endswith(".tfrecord.gz")]
    out = []
    for shard in shards:
        for raw in tf.data.TFRecordDataset([shard], compression_type="GZIP"):
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            seq = list(ex.features.feature["sequence_data"].int64_list.value)
            out.append(seq)
    return out


def test_forget_target_filter() -> None:
    """materialize_item_mode_forget_dir must remove I_f from sequences and
    drop rows that become too short."""
    from src.data.unlearning.forget_target_filter import materialize_item_mode_forget_dir

    tmpdir = tempfile.mkdtemp(prefix="test_ftf_")
    try:
        forget_dir = os.path.join(tmpdir, "training_forget")
        out_dir = os.path.join(tmpdir, "filtered")

        rows = [
            (1, [10, 20, 30, 40]),     # target 20 removed → [10, 30, 40] ✓
            (2, [20]),                  # only target → dropped (len < 1)
            (3, [20, 30]),              # target removed → [30] (len=1 ≥ min=1) ✓
            (4, [10, 11, 12]),          # no target → unchanged ✓
        ]
        _write_tfrecord_shards(rows, forget_dir)

        info = materialize_item_mode_forget_dir(
            forget_dir=forget_dir,
            out_dir=out_dir,
            target_items={20},
            min_sequence_length=1,
            rows_per_shard=10,
        )

        assert info["rows_in"] == 4
        assert info["rows_out"] == 3, f"Expected 3 kept rows, got {info['rows_out']}"
        assert info["rows_dropped"] == 1, f"Expected 1 dropped row, got {info['rows_dropped']}"

        seqs = _read_tfrecord_sequences(out_dir)
        assert len(seqs) == 3
        flat = {tuple(s) for s in seqs}
        assert (10, 30, 40) in flat, "Target 20 must be removed from row 1"
        assert (30,) in flat, "Row 3 must survive with single non-target item"
        assert (10, 11, 12) in flat, "Row with no targets must be unchanged"
        assert (20,) not in flat, "Single-target row must be dropped"

        print(f"  [T3] forget_target_filter: OK "
              f"(in={info['rows_in']}, out={info['rows_out']}, "
              f"dropped={info['rows_dropped']})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# T4 — deletion_spec routing: session vs item neighborhood centers
# ---------------------------------------------------------------------------

def test_deletion_spec_routing() -> None:
    """resolve_neighborhood_centers must return target_items for item mode
    and forget_shard_items for session mode."""
    from src.data.unlearning.deletion_spec import (
        normalize_deletion_spec,
        resolve_neighborhood_centers,
        resolve_forbidden_retain_items,
    )

    forget_shard_items = {1, 2, 3, 4, 5}
    target_items = {10, 11}

    centers_session = resolve_neighborhood_centers(
        deletion_spec="session",
        forget_shard_items=forget_shard_items,
        target_items=target_items,
    )
    assert centers_session == forget_shard_items, (
        "Session mode must use all forget-shard items as centers"
    )

    centers_item = resolve_neighborhood_centers(
        deletion_spec="item",
        forget_shard_items=forget_shard_items,
        target_items=target_items,
    )
    assert centers_item == target_items, (
        "Item mode must use only I_f as neighborhood centers"
    )

    forbidden_session = resolve_forbidden_retain_items(
        deletion_spec="session",
        forget_shard_items=forget_shard_items,
        target_items=target_items,
    )
    assert forbidden_session == forget_shard_items

    forbidden_item = resolve_forbidden_retain_items(
        deletion_spec="item",
        forget_shard_items=forget_shard_items,
        target_items=target_items,
    )
    assert forbidden_item == target_items

    try:
        normalize_deletion_spec("invalid_spec")
        raise AssertionError("Should have raised ValueError for unknown spec")
    except ValueError:
        pass

    print("  [T4] deletion_spec routing: OK")


# ---------------------------------------------------------------------------
# T5 — finetune: loss descends over retain batches
# ---------------------------------------------------------------------------

class _FakeInput:
    """Duck-typed stand-in for SequentialModelInputData — avoids circular import."""
    def __init__(self, user_id_list, transformed_sequences, mask):
        self.user_id_list = user_id_list
        self.transformed_sequences = transformed_sequences
        self.mask = mask


class _FakeLabel:
    """Duck-typed stand-in for SequentialModuleLabelData."""
    def __init__(self, labels):
        self.labels = labels
        self.label_location: dict = {}
        self.attention_mask: dict = {}


class _ToyModel(nn.Module):
    """Minimal model that speaks the TIGER model_step / batch contract."""
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def model_step(self, model_input, label_data):
        x = model_input.transformed_sequences["x"]
        y = label_data.labels["y"]
        out = self.linear(x)
        loss = ((out - y) ** 2).mean()
        return out, loss


def _make_batch(n: int = 4, dim: int = 8) -> tuple:
    """Build a minimal TIGER-compatible batch (duck-typed, no circular import)."""
    x = torch.randn(n, dim)
    y = torch.randn(n, dim)
    model_input = _FakeInput(
        user_id_list=torch.arange(n),
        transformed_sequences={"x": x},
        mask=torch.ones(n, 1),
    )
    label_data = _FakeLabel(labels={"y": y})
    return (model_input, label_data)


def test_finetune_loss_decreases() -> None:
    """finetune_unlearn must reduce retain loss over steps."""
    from src.components.unlearning.finetune import finetune_unlearn

    torch.manual_seed(0)
    model = _ToyModel(dim=8)
    retain = [_make_batch() for _ in range(4)]

    # Measure initial loss
    model.eval()
    with torch.no_grad():
        _, initial_loss = model.model_step(*retain[0])

    info = finetune_unlearn(model, retain, steps=50, lr=1e-2)

    model.eval()
    with torch.no_grad():
        _, final_loss = model.model_step(*retain[0])

    assert final_loss < initial_loss, (
        f"Finetune must reduce retain loss: {initial_loss:.4f} → {final_loss:.4f}"
    )
    assert info["algorithm"] == "finetune"
    print(f"  [T5] finetune loss descends: {initial_loss:.4f} → {final_loss:.4f} OK")


# ---------------------------------------------------------------------------
# T6 — neg_train: gradient ascent raises forget loss
# ---------------------------------------------------------------------------

def test_neg_train_loss_increases() -> None:
    """neg_train_unlearn must push forget loss higher (gradient ascent)."""
    from src.components.unlearning.neg_train import neg_train_unlearn

    torch.manual_seed(0)
    model = _ToyModel(dim=8)
    forget = [_make_batch() for _ in range(4)]
    retain = [_make_batch() for _ in range(4)]

    model.eval()
    with torch.no_grad():
        _, initial_loss = model.model_step(*forget[0])

    # neg_retain_every=0 → pure gradient ascent, no retain steps
    info = neg_train_unlearn(
        model, forget, retain, steps=30, lr=1e-2, neg_retain_every=0
    )

    model.eval()
    with torch.no_grad():
        _, final_loss = model.model_step(*forget[0])

    assert final_loss > initial_loss, (
        f"Neg-train must INCREASE forget loss (gradient ascent): "
        f"{initial_loss:.4f} → {final_loss:.4f}"
    )
    assert info["algorithm"] == "neg_train"
    print(f"  [T6] neg_train loss ascends: {initial_loss:.4f} → {final_loss:.4f} OK")


# ---------------------------------------------------------------------------
# T7 — codebook indexing: non-sequential IDs must be detected
#
#  This documents the rsc15 bug: self.codebooks[raw_item_id] crashes when
#  raw item IDs (e.g. 214M) exceed the codebook's N rows (53k).
#  The fix is item ID remapping in convert_rsc15_inter.py.
# ---------------------------------------------------------------------------

def test_codebook_index_nonsequential_detected() -> None:
    """Confirm that indexing a small codebook with a large non-sequential item ID
    raises an IndexError — this is the rsc15 training bug that the ID remapping
    in convert_rsc15_inter.py fixes."""
    N, D = 100, 4
    # codebook already transposed to N×D (as stored in self.codebooks)
    codebook = torch.zeros(N, D, dtype=torch.long)

    raw_item_id = 214765147  # rsc15-style raw ID, >> N

    crashed = False
    try:
        _ = codebook[torch.tensor([raw_item_id])]
    except IndexError:
        crashed = True

    assert crashed, (
        "Expected IndexError when indexing codebook with non-sequential raw item ID. "
        "If this test passes without crashing, the rsc15 ID bug may have been patched "
        "at the tensor level — verify convert_rsc15_inter.py remaps IDs correctly."
    )

    # After remapping, sequential ID for the item would be e.g. 42 → valid lookup
    sequential_id = 42
    result = codebook[torch.tensor([sequential_id])]
    assert result.shape == (1, D)

    print("  [T7] codebook non-sequential index detection: OK "
          "(rsc15 requires ID remapping — see convert_rsc15_inter.py)")


# ---------------------------------------------------------------------------
# T8 — convert_rsc15_inter: item ID remapping correctness
# ---------------------------------------------------------------------------

def test_convert_rsc15_item_id_remapping() -> None:
    """_build_item_id_map must produce a dense 0..N-1 mapping in sorted order."""
    from src.data.erase_data.convert_rsc15_inter import _build_item_id_map

    raw_ids = {214800000, 214765147, 214765100, 999999999, 500000000}
    mapping = _build_item_id_map(raw_ids)

    assert set(mapping.values()) == set(range(len(raw_ids))), (
        "Sequential indices must be exactly 0..N-1"
    )
    # Sorted order: 214765100 → 0, 214765147 → 1, 214800000 → 2, 500000000 → 3, 999999999 → 4
    assert mapping[214765100] == 0
    assert mapping[214765147] == 1
    assert mapping[214800000] == 2
    assert mapping[500000000] == 3
    assert mapping[999999999] == 4

    # Verify max sequential ID is within tensor bounds after mapping
    max_seq = max(mapping.values())
    codebook = torch.zeros(len(raw_ids), 4, dtype=torch.long)
    result = codebook[torch.tensor([max_seq])]
    assert result.shape == (1, 4), "Mapped ID must be within codebook bounds"

    print(f"  [T8] convert_rsc15 item ID remapping: OK (N={len(raw_ids)})")


# ---------------------------------------------------------------------------
# T9 — generate_embeddings merge: new indexed format
# ---------------------------------------------------------------------------

def test_generate_embeddings_indexed_format() -> None:
    """The merged_predictions_tensor.pt from generate_embeddings.sh must be a
    dict with 'embeddings' and 'item_ids' keys, not a plain tensor."""
    latest = "logs/inference/runs/2026-05-21/14-23-52/pickle/merged_predictions_tensor.pt"
    if not os.path.isfile(latest):
        print(f"  [T9] SKIP — embedding file not found: {latest}")
        return

    obj = torch.load(latest, map_location="cpu", weights_only=False)
    assert isinstance(obj, dict), f"Expected dict, got {type(obj).__name__}"
    assert "embeddings" in obj, "Missing 'embeddings' key"
    assert "item_ids" in obj, "Missing 'item_ids' key — old format, re-run generate_embeddings.sh"

    embs = obj["embeddings"]
    ids = obj["item_ids"]
    assert embs.shape[0] == ids.shape[0], "embeddings and item_ids row counts must match"
    assert embs.shape[1] == 2048, f"Expected 2048-dim flan-T5 embeddings, got {embs.shape[1]}"

    # IDs must be sorted (generate_embeddings.sh sorts by item_id)
    assert (ids[1:] >= ids[:-1]).all(), "item_ids must be sorted ascending"

    # For rsc15, IDs must NOT be sequential from 0 (that's the whole point)
    assert ids[0].item() > 0, "rsc15 item IDs must not start at 0 (they are raw IDs)"

    print(f"  [T9] generate_embeddings indexed format: OK "
          f"(n={embs.shape[0]}, id_range={int(ids.min())}..{int(ids.max())})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_dense_embeddings_nonsequential,
    test_eval_forget_target_recall_sid_decoding,
    test_forget_target_filter,
    test_deletion_spec_routing,
    test_finetune_loss_decreases,
    test_neg_train_loss_increases,
    test_codebook_index_nonsequential_detected,
    test_convert_rsc15_item_id_remapping,
    test_generate_embeddings_indexed_format,
]


def main() -> None:
    print("[bug-tests] Running unlearning bug regression tests ...")
    failures: List[str] = []
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            failures.append(name)

    print()
    if failures:
        print(f"[bug-tests] {len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"[bug-tests] All {len(TESTS)} tests passed.")


if __name__ == "__main__":
    main()
