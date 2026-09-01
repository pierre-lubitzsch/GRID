"""Tests for the refined ID space (latent refiner) and its L_n neighbourhoods.

Two halves:

TESTS 1-4  ``src/components/unlearning/latent_refiner.py`` in isolation — the
           SID-table indexing (which silently produces garbage if the ``h*K``
           offset convention is wrong), the cosine top-k, the prefix sibling
           pool, and an end-to-end train/export.

TESTS 5-8  ``_build_coherence_neighbors`` with ``coherence_neighbor_method`` set
           to ``latent`` and ``embedding+latent``. The decisive one is TEST 7:
           the union must DEDUPE, because a neighbour ranked by both sources
           would otherwise be teacher-forced twice and get double weight in
           L_n — an easy mistake that would silently bias the union arm.

Run:  python test_latent_refiner.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.components.unlearning.latent_refiner import (  # noqa: E402
    LatentRefiner,
    LatentRefinerConfig,
    build_sid_embedding_matrix,
    export_latents,
    prefix_positive_pool,
    topk_cosine_neighbors,
    train_latent_refiner,
)

log = logging.getLogger("test_latent_refiner")

N_ITEMS, H, K, D_MODEL, X_DIM = 60, 4, 8, 16, 32
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


def _fixtures(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    codebook = torch.randint(0, K, (N_ITEMS, H), generator=g)
    sid_table = torch.randn(H * K, D_MODEL, generator=g)
    x = torch.randn(N_ITEMS, X_DIM, generator=g)
    return codebook, sid_table, x


# --------------------------------------------------------------------------- #
print("\nTEST 1: build_sid_embedding_matrix reproduces TIGER's h*K offsetting")
codebook, sid_table, x = _fixtures()
sid_emb = build_sid_embedding_matrix(codebook, sid_table)
check(tuple(sid_emb.shape) == (N_ITEMS, H * D_MODEL),
      f"shape is [N, H*d] = {tuple(sid_emb.shape)}")
# Independent re-derivation of the SAME indexing the encoder uses:
# _add_repeating_offset_to_rows maps code c at hierarchy h to row c + h*K.
manual = torch.stack([
    torch.cat([sid_table[int(codebook[i, h]) + h * K] for h in range(H)])
    for i in range(N_ITEMS)
])
check(torch.allclose(sid_emb, manual), "matches per-item manual row gather")
# A wrong convention (no offset) must NOT coincide, or the test proves nothing.
no_offset = torch.stack([
    torch.cat([sid_table[int(codebook[i, h])] for h in range(H)])
    for i in range(N_ITEMS)
])
check(not torch.allclose(sid_emb, no_offset),
      "differs from the un-offset gather (test is discriminating)")
try:
    build_sid_embedding_matrix(torch.full((N_ITEMS, H), K + 1), sid_table)
    check(False, "out-of-range code rejected")
except ValueError:
    check(True, "out-of-range code rejected")

# --------------------------------------------------------------------------- #
print("\nTEST 2: topk_cosine_neighbors matches brute force and excludes self")
nbr = topk_cosine_neighbors(x, 5)
check(tuple(nbr.shape) == (N_ITEMS, 5), f"shape {tuple(nbr.shape)}")
check(all(i not in nbr[i].tolist() for i in range(N_ITEMS)), "self never returned")
xn = torch.nn.functional.normalize(x, dim=1)
brute = (xn @ xn.t())
brute.fill_diagonal_(float("-inf"))
check(torch.equal(nbr, brute.topk(5, dim=1).indices), "equals brute-force top-k")
# Chunking must not change the answer.
check(torch.equal(topk_cosine_neighbors(x, 5, chunk=7), nbr),
      "chunked pass is identical (chunk=7)")
check(tuple(topk_cosine_neighbors(x, N_ITEMS + 100).shape) == (N_ITEMS, N_ITEMS - 1),
      "k > N-1 clamps to N-1")

# --------------------------------------------------------------------------- #
print("\nTEST 3: prefix_positive_pool only pairs true prefix siblings")
pos, counts = prefix_positive_pool(codebook, 2, max_per_item=16)
check(tuple(pos.shape) == (N_ITEMS, 16), f"shape {tuple(pos.shape)}")
bad_prefix = bad_self = 0
for i in range(N_ITEMS):
    for j in pos[i, : int(counts[i])].tolist():
        if not torch.equal(codebook[i, :2], codebook[j, :2]):
            bad_prefix += 1
        if j == i:
            bad_self += 1
check(bad_prefix == 0, "every sampled positive shares the 2-code prefix")
check(bad_self == 0, "an item is never its own positive")
# Items in a singleton bucket must report zero, not a fabricated sibling.
_, c_small = prefix_positive_pool(torch.arange(K).unsqueeze(1).repeat(1, H), 4, max_per_item=4)
check(int(c_small.sum()) == 0, "singleton prefix buckets report count 0")

# --------------------------------------------------------------------------- #
print("\nTEST 4: refiner trains, z is finite, and z is a DIFFERENT geometry")
cfg = LatentRefinerConfig(latent_dim=8, hidden_dim=32, epochs=6, batch_size=16,
                          neighbor_k=4, sid_prefix_length=2, seed=1)
model, stats = train_latent_refiner(sid_emb=sid_emb, item_emb=x, codebook=codebook,
                                    config=cfg, device=torch.device("cpu"))
z = export_latents(model, sid_emb=sid_emb, item_emb=x, device=torch.device("cpu"))
check(tuple(z.shape) == (N_ITEMS, 8), f"z shape {tuple(z.shape)}")
check(bool(torch.isfinite(z).all()), "z is all-finite")
check(len(stats.epochs) == 6 and stats.final["epoch"] == 6, "6 epochs recorded")
check(stats.epochs[-1]["total"] < stats.epochs[0]["total"], "total loss decreased")
z_nbr = topk_cosine_neighbors(z, 4)
x_nbr = topk_cosine_neighbors(x, 4)
same = sum(int(torch.equal(z_nbr[i], x_nbr[i])) for i in range(N_ITEMS))
check(same < N_ITEMS, f"N_z is not identical to N_emb ({same}/{N_ITEMS} rows equal)")
# Determinism: same seed, same z.
m2, _ = train_latent_refiner(sid_emb=sid_emb, item_emb=x, codebook=codebook,
                             config=cfg, device=torch.device("cpu"))
z2 = export_latents(m2, sid_emb=sid_emb, item_emb=x, device=torch.device("cpu"))
check(torch.allclose(z, z2, atol=1e-5), "training is seed-deterministic")

# --------------------------------------------------------------------------- #
# Second half: the L_n neighbourhood dispatch.
# --------------------------------------------------------------------------- #
print("\nTEST 5-8: _build_coherence_neighbors with latent / embedding+latent")
from src.models.modules.semantic_id.tiger_unlearning_module import (  # noqa: E402
    TigerUnlearningModule,
)

BUILD = TigerUnlearningModule._build_coherence_neighbors  # does not touch self

# Unique codebook (the dedup digit makes real SIDs unique) so sid_to_item is
# bijective, as the production codebook is.
uniq = torch.stack([
    torch.tensor([i // (K * K), (i // K) % K, i % K, 0]) for i in range(N_ITEMS)
])
TARGET = 7
tmp = tempfile.mkdtemp(prefix="latent_refiner_test_")
sid_path = os.path.join(tmp, "sid.pt")
emb_path = os.path.join(tmp, "x.pt")
lat_path = os.path.join(tmp, "z.pt")
torch.save(uniq, sid_path)
torch.save(x, emb_path)
torch.save(z, lat_path)


def _batch(label_item: int):
    """One forget batch of a single row whose LABEL is `label_item`'s SID."""
    mi = types.SimpleNamespace(mask=torch.ones(1, H))
    ld = types.SimpleNamespace(labels={"sid": uniq[label_item].reshape(1, -1).clone()})
    return (mi, ld)


def build(method: str, count: int, **kw):
    return BUILD(
        None,
        forget_batches=[_batch(TARGET)],
        semantic_id_path=sid_path,
        num_hierarchies=H,
        neighborhood_count=count,
        neighborhood_prefix_length=2,
        exclude_items={TARGET},
        coherence_rows="target_only",
        target_items={TARGET},
        neighbor_method=method,
        embedding_path=emb_path,
        embedding_metric="cosine",
        latent_path=lat_path,
        **kw,
    )


def ids_of(out, count_expected=None):
    """Recover neighbour ITEM ids from the returned SID rows."""
    sids, mask = out[0]
    sid_to_item = {tuple(uniq[i].tolist()): i for i in range(N_ITEMS)}
    got = [sid_to_item[tuple(sids[0, c].tolist())]
           for c in range(sids.shape[1]) if float(mask[0, c]) > 0]
    return got, tuple(sids.shape), tuple(mask.shape)

print("\nTEST 5: method=latent returns the cosine top-k of z, target excluded")
got, sshape, _ = ids_of(build("latent", 4))
expect = topk_cosine_neighbors(z, 5)[TARGET].tolist()
expect = [i for i in expect if i != TARGET][:4]
check(got == expect, f"latent neighbours match top-k on z: {got} vs {expect}")
check(TARGET not in got, "the forget target itself is excluded")
check(sshape == (1, 4, H), f"allocated exactly count rows {sshape}")

print("\nTEST 6: latent and embedding arms select genuinely different sets")
got_emb, _, _ = ids_of(build("embedding", 4))
got_lat, _, _ = ids_of(build("latent", 4))
check(got_emb != got_lat or set(got_emb) != set(got_lat),
      f"N_emb={got_emb} differs from N_z={got_lat}")

print("\nTEST 7: embedding+latent is a DEDUPED union")
got_u, sshape_u, mshape_u = ids_of(build("embedding+latent", 4))
check(len(got_u) == len(set(got_u)), f"no duplicate neighbour in the union: {got_u}")
check(set(got_u) == set(got_emb) | set(got_lat),
      f"union equals N_emb ∪ N_z ({sorted(got_u)} vs "
      f"{sorted(set(got_emb) | set(got_lat))})")
check(sshape_u == (1, 8, H), f"full union allocates 2*count rows {sshape_u}")
check(len(got_u) <= 8, "union never exceeds its allocation")
check(TARGET not in got_u, "target excluded from the union too")

print("\nTEST 8: union_size=matched keeps the neighbourhood at count")
got_m, sshape_m, _ = ids_of(build("embedding+latent", 4, union_size="matched"))
check(sshape_m == (1, 4, H), f"matched allocates exactly count rows {sshape_m}")
check(len(got_m) <= 4, f"matched union holds <= count neighbours ({len(got_m)})")
check(set(got_m) <= set(got_emb) | set(got_lat),
      "matched union is a subset of the full union")

print("\nTEST 9: config validation")
for method, kw, why in [
    ("latent", {"latent_path": None}, "latent without coherence_latent_path"),
    ("embedding+latent", {"latent_path": None}, "union without coherence_latent_path"),
    ("embedding+latent", {"embedding_path": None}, "union without embedding_path"),
    ("bogus", {}, "unknown method"),
    ("latent", {"union_size": "half"}, "unknown union_size"),
]:
    args = dict(
        forget_batches=[_batch(TARGET)], semantic_id_path=sid_path,
        num_hierarchies=H, neighborhood_count=4, neighborhood_prefix_length=2,
        exclude_items={TARGET}, coherence_rows="target_only",
        target_items={TARGET}, neighbor_method=method,
        embedding_path=emb_path, embedding_metric="cosine", latent_path=lat_path,
    )
    args.update(kw)
    try:
        BUILD(None, **args)
        check(False, f"rejects {why}")
    except ValueError:
        check(True, f"rejects {why}")

# Row-misalignment must be caught, not silently mismatch items.
bad_lat = os.path.join(tmp, "z_bad.pt")
torch.save(z[: N_ITEMS - 3], bad_lat)
try:
    BUILD(None, forget_batches=[_batch(TARGET)], semantic_id_path=sid_path,
          num_hierarchies=H, neighborhood_count=4, neighborhood_prefix_length=2,
          exclude_items={TARGET}, coherence_rows="target_only",
          target_items={TARGET}, neighbor_method="latent",
          embedding_path=emb_path, embedding_metric="cosine", latent_path=bad_lat)
    check(False, "rejects a latent tensor with the wrong row count")
except ValueError:
    check(True, "rejects a latent tensor with the wrong row count")

print("\n" + "=" * 70)
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s)")
    for f in FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
print("All checks passed.")
