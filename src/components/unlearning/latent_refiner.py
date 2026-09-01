"""Latent refiner: a refined ID space for fine-grained unlearning.

Motivation
----------
A 4-token semantic ID is efficient to generate but coarse: at beauty width 256
roughly 47 items share a coarse code on average, and the mean prefix-2
neighbourhood is only ~3.16 items with 30.6% of items having none at all (see
``topk_embedding_neighbors`` in ``neighborhood_sampler.py``). Forget and retain
signals can therefore share the same token path, and a neighbourhood defined on
the SID prefix is both width-dependent and frequently empty.

This module keeps the SID pipeline untouched and learns an AUXILIARY continuous
space on top of it::

    z_i = MLP([SIDEmb(i), x_i])

``SIDEmb(i)`` is the concatenation of the ``H`` rows of the trained recommender's
semantic-ID embedding table that item ``i``'s codes select, so the refined space
inherits the geometry the decoder actually reads. ``x_i`` is the pre-quantization
item embedding, which carries the fine-grained item identity that quantization
threw away. The refined space is used for NEIGHBOUR SELECTION only -- generation
still runs entirely through the unchanged SID decoder.

Training objectives (all three from the plan, individually weighted)
--------------------------------------------------------------------
``reconstruction``
    A decoder head maps ``z_i`` back to ``x_i``. This is the "align z_i with the
    item representation" term: it forces the latent to retain item-level detail
    rather than collapsing onto the coarse SID partition.

``neighborhood``
    InfoNCE that pulls each anchor's top-k neighbours in the ORIGINAL embedding
    space together in latent space, against in-batch negatives. This is what
    makes ``N_z`` a meaningful refinement of ``N_emb`` instead of noise: the
    latent ranking stays anchored to the pre-quantization geometry.

``sid_consistency``
    InfoNCE with positives drawn from items sharing an SID prefix. Keeps the
    refined space compatible with the generation-side partition, so a latent
    neighbour is still reachable by the decoder.

The three terms pull in different directions on purpose. Reconstruction alone
would reproduce ``N_emb`` exactly (an MLP that inverts to ``x`` preserves its
ranking), and SID consistency alone would reproduce the prefix neighbourhood.
The refined space is useful only in the middle, which is why all three weights
are exposed rather than fixed.

Outputs
-------
``export_latents`` writes a plain ``[N, latent_dim]`` float tensor, ROW-ALIGNED
with the SID codebook. That is exactly the layout ``load_dense_embeddings``
accepts, so the resulting file drops into ``unlearning.coherence_latent_path``
and is consumed by the same ``topk_embedding_neighbors`` helper the ``embedding``
neighbour method already uses -- no new k-NN code path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = [
    "LatentRefiner",
    "LatentRefinerConfig",
    "build_sid_embedding_matrix",
    "load_sid_table_from_checkpoint",
    "topk_cosine_neighbors",
    "prefix_positive_pool",
    "train_latent_refiner",
    "export_latents",
]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class LatentRefinerConfig:
    """Hyperparameters for the latent refiner and its training loop."""

    latent_dim: int = 128
    hidden_dim: int = 512
    dropout: float = 0.0
    # Loss weights. reconstruction is the anchor term; the other two shape the
    # neighbourhood. All three are active by default -- see module docstring for
    # why none of them is safe to drop.
    w_reconstruction: float = 1.0
    w_neighborhood: float = 1.0
    w_sid_consistency: float = 0.1
    # Neighbourhood-alignment term
    neighbor_k: int = 8  # x-space neighbours treated as positives
    temperature: float = 0.07  # matches sep_temperature, the repo-wide default
    # SID-consistency term
    sid_prefix_length: int = 2
    # Optimisation
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    seed: int = 2
    # Chunk size for the O(N^2) cosine top-k passes (memory, not semantics).
    knn_chunk: int = 1024


# --------------------------------------------------------------------------- #
# inputs: SID embeddings and item representations
# --------------------------------------------------------------------------- #
def load_sid_table_from_checkpoint(
    ckpt_path: str,
    *,
    table_key_suffix: str = "item_sid_embedding_table_encoder.weight",
) -> torch.Tensor:
    """Return the trained semantic-ID embedding table as ``[H*K, d]``.

    Reads the Lightning checkpoint's ``state_dict`` and picks the single key
    ending in ``table_key_suffix``. The table is the recommender's own SID
    geometry, which is the point: the refined space must be built on what the
    decoder reads, not on a fresh random table.
    """
    # These checkpoints pickle references to repo classes, so unpickling imports
    # src.data.loading.components.interfaces -- which is circular with
    # src.utils.utils unless src.utils is initialised FIRST. Importing it here
    # (not at module scope) keeps this file usable without the hydra/lightning
    # stack for callers that only need the pure-torch helpers. weights_only=True
    # is not an option: Lightning stores non-tensor hyper_parameters.
    import src.utils  # noqa: F401

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(
            f"{ckpt_path!r} does not contain a state_dict (got {type(state)})"
        )
    matches = [k for k in state if k.endswith(table_key_suffix)]
    if not matches:
        raise KeyError(
            f"No key ending in {table_key_suffix!r} in {ckpt_path!r}. "
            f"Sample keys: {sorted(state)[:5]}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"Ambiguous SID table in {ckpt_path!r}: {matches}. "
            "Pass an explicit table_key_suffix."
        )
    table = state[matches[0]]
    if not isinstance(table, torch.Tensor) or table.dim() != 2:
        raise TypeError(
            f"{matches[0]!r} is not a 2-D tensor (got {type(table)} "
            f"{getattr(table, 'shape', None)})"
        )
    log.info(
        "[latent-refiner] SID table %s: %s from %s",
        matches[0],
        tuple(table.shape),
        ckpt_path,
    )
    return table.float()


def build_sid_embedding_matrix(
    codebook: torch.Tensor,
    sid_table: torch.Tensor,
    *,
    num_embeddings_per_hierarchy: Optional[int] = None,
) -> torch.Tensor:
    """Concatenate each item's ``H`` SID embedding rows into ``[N, H*d]``.

    TIGER stores all hierarchies in ONE table and offsets hierarchy ``h`` by
    ``h * K`` (``_add_repeating_offset_to_rows``: ``[0,1,2] -> [0,301,602]`` for
    ``K=301``). We reproduce that indexing exactly, so ``SIDEmb(i)`` is the same
    set of vectors the encoder embeds for item ``i``.

    Concatenation rather than mean pooling: the hierarchies are ordered
    coarse-to-fine and mean pooling would discard which level a code came from,
    the very distinction the refined space is meant to sharpen.
    """
    if codebook.dim() != 2:
        raise ValueError(f"codebook must be [N, H], got {tuple(codebook.shape)}")
    num_items, H = int(codebook.shape[0]), int(codebook.shape[1])
    total_rows, dim = int(sid_table.shape[0]), int(sid_table.shape[1])

    K = int(num_embeddings_per_hierarchy or 0)
    if K <= 0:
        if total_rows % H != 0:
            raise ValueError(
                f"SID table has {total_rows} rows, not divisible by H={H}; "
                "pass num_embeddings_per_hierarchy explicitly."
            )
        K = total_rows // H
    if K * H > total_rows:
        raise ValueError(
            f"num_embeddings_per_hierarchy={K} x H={H} exceeds the "
            f"{total_rows}-row SID table."
        )

    codes = codebook.long()
    if int(codes.max()) >= K:
        raise ValueError(
            f"codebook holds code {int(codes.max())} but each hierarchy only "
            f"has K={K} rows -- codebook and checkpoint disagree."
        )
    offsets = torch.arange(H, dtype=torch.long).unsqueeze(0) * K  # [1, H]
    token_ids = codes + offsets  # [N, H]
    out = sid_table[token_ids.reshape(-1)].reshape(num_items, H * dim)
    log.info(
        "[latent-refiner] SIDEmb: %d items x (H=%d * d=%d) = %d dims (K=%d)",
        num_items,
        H,
        dim,
        out.shape[1],
        K,
    )
    return out.contiguous()


# --------------------------------------------------------------------------- #
# neighbour targets for the training objectives
# --------------------------------------------------------------------------- #
def topk_cosine_neighbors(
    matrix: torch.Tensor,
    k: int,
    *,
    chunk: int = 1024,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Row-wise cosine top-``k`` over ``matrix`` ``[N, D]``, self excluded.

    Returns ``[N, k]`` row indices. Chunked so the ``N x N`` similarity matrix is
    never materialised (beauty is only 12k items, but rsc15 is 53k and the same
    code path serves both).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = int(matrix.shape[0])
    k_eff = min(int(k), max(n - 1, 0))
    if k_eff == 0:
        return torch.zeros((n, 0), dtype=torch.long)
    dev = device or matrix.device
    normed = F.normalize(matrix.to(dev).float(), dim=1)
    out = torch.empty((n, k_eff), dtype=torch.long)
    for start in range(0, n, int(chunk)):
        stop = min(start + int(chunk), n)
        sims = normed[start:stop] @ normed.t()  # [c, N]
        rows = torch.arange(start, stop, device=dev)
        sims[torch.arange(stop - start, device=dev), rows] = float("-inf")
        out[start:stop] = sims.topk(k_eff, dim=1).indices.cpu()
    return out


def prefix_positive_pool(
    codebook: torch.Tensor,
    prefix_length: int,
    *,
    max_per_item: int = 32,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample SID-prefix siblings per item for the SID-consistency term.

    Returns ``(positives[N, max_per_item], counts[N])``; row ``i`` holds up to
    ``max_per_item`` other items sharing item ``i``'s first ``prefix_length``
    codes, and ``counts[i]`` says how many are valid. Items with no sibling get
    ``counts[i] == 0`` and are skipped by the loss -- which is precisely the
    coverage gap the refined space exists to fill, so it must not be faked.
    """
    codes = codebook.long()
    p = max(1, min(int(prefix_length), int(codes.shape[1])))
    buckets: Dict[tuple, List[int]] = {}
    for i, row in enumerate(codes[:, :p].tolist()):
        buckets.setdefault(tuple(row), []).append(i)

    n = int(codes.shape[0])
    positives = torch.zeros((n, int(max_per_item)), dtype=torch.long)
    counts = torch.zeros(n, dtype=torch.long)
    for members in buckets.values():
        if len(members) < 2:
            continue
        member_t = torch.tensor(members, dtype=torch.long)
        for i in members:
            siblings = member_t[member_t != i]
            if siblings.numel() > int(max_per_item):
                pick = torch.randperm(siblings.numel(), generator=generator)[
                    : int(max_per_item)
                ]
                siblings = siblings[pick]
            positives[i, : siblings.numel()] = siblings
            counts[i] = siblings.numel()
    n_covered = int((counts > 0).sum())
    log.info(
        "[latent-refiner] SID-prefix siblings (p=%d): %d/%d items covered "
        "(%.1f%%), mean pool %.2f",
        p,
        n_covered,
        n,
        100.0 * n_covered / max(n, 1),
        float(counts.float().mean()),
    )
    return positives, counts


# --------------------------------------------------------------------------- #
# the refiner
# --------------------------------------------------------------------------- #
class LatentRefiner(nn.Module):
    """``z_i = MLP([SIDEmb(i), x_i])`` with a reconstruction head back to ``x``.

    Deliberately small: one hidden layer each way. The point of the minimal
    validation is to test whether a refined neighbourhood helps at all, so the
    refiner must not be capable enough for its own capacity to be the
    explanation.
    """

    def __init__(
        self,
        sid_dim: int,
        item_dim: int,
        *,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sid_dim = int(sid_dim)
        self.item_dim = int(item_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.sid_dim + self.item_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.item_dim),
        )

    def forward(self, sid_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([sid_emb, item_emb], dim=-1))

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def _infonce(
    anchor_z: torch.Tensor,
    positive_z: torch.Tensor,
    bank_z: torch.Tensor,
    temperature: float,
    *,
    exclude_col: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE of ``anchor`` against one positive and a shared negative bank.

    Cosine similarities on L2-normalised vectors, so logits are bounded to
    ``+-1/temperature`` -- the same regime as ``L_sep`` at ``sep_temperature``.

    ``exclude_col[i]`` is the column of ``bank_z`` holding anchor ``i`` itself
    (or a negative value if it is absent). The bank here is the minibatch, so
    every anchor IS in it, and ``cos(a, a) == 1`` would enter the denominator as
    a constant ``1/temperature`` logit -- 14.3 at the default temperature. Its
    gradient w.r.t. ``a`` is identically zero, so it does not push in a wrong
    direction, but it floors the loss near ``log 2`` and halves the effective
    signal, which makes the reported curve unreadable. Masked out.

    Note the remaining, standard approximation: a genuine neighbour of anchor
    ``i`` that happens to be in the same minibatch acts as a false negative
    (~4% of pairs at batch 512 over a 12k catalog). That is ordinary in-batch
    InfoNCE and is left as-is.
    """
    a = F.normalize(anchor_z, dim=-1)
    p = F.normalize(positive_z, dim=-1)
    bank = F.normalize(bank_z, dim=-1)
    pos = (a * p).sum(-1, keepdim=True) / temperature  # [B, 1]
    neg = (a @ bank.t()) / temperature  # [B, M]
    if exclude_col is not None:
        rows = torch.arange(neg.shape[0], device=neg.device)
        valid = exclude_col >= 0
        if bool(valid.any()):
            neg[rows[valid], exclude_col[valid]] = float("-inf")
    logits = torch.cat([pos, neg], dim=1)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


@dataclass
class LatentRefinerStats:
    """Per-epoch loss history plus the geometry diagnostics that matter."""

    epochs: List[Dict[str, float]] = field(default_factory=list)
    final: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"epochs": self.epochs, "final": self.final}


def train_latent_refiner(
    *,
    sid_emb: torch.Tensor,
    item_emb: torch.Tensor,
    codebook: torch.Tensor,
    config: LatentRefinerConfig,
    device: Optional[torch.device] = None,
) -> Tuple[LatentRefiner, LatentRefinerStats]:
    """Train the refiner on the full catalog and return it with its stats.

    All three objectives run every step. Anchors are a shuffled minibatch of
    items; the negative bank is the batch itself, which is why ``batch_size``
    doubles as the number of InfoNCE negatives.
    """
    if sid_emb.shape[0] != item_emb.shape[0]:
        raise ValueError(
            f"SIDEmb has {sid_emb.shape[0]} rows but item embeddings have "
            f"{item_emb.shape[0]}: they must be row-aligned (same catalog, same "
            "order)."
        )
    if codebook.shape[0] != sid_emb.shape[0]:
        raise ValueError(
            f"codebook has {codebook.shape[0]} rows but SIDEmb has "
            f"{sid_emb.shape[0]}."
        )
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config.seed))
    gen = torch.Generator().manual_seed(int(config.seed))

    n_items = int(sid_emb.shape[0])
    # Standardise the item embeddings: flan-t5 features are not zero-centred and
    # an unnormalised MSE target would let the reconstruction term dominate by
    # scale alone rather than by the weight we set.
    item_mu = item_emb.mean(dim=0, keepdim=True)
    item_sigma = item_emb.std(dim=0, keepdim=True).clamp_min(1e-6)
    item_std = (item_emb - item_mu) / item_sigma

    sid_dev = sid_emb.to(dev)
    item_dev = item_std.to(dev)

    log.info(
        "[latent-refiner] neighbourhood targets: cosine top-%d in the "
        "pre-quantization space",
        config.neighbor_k,
    )
    nbr_idx = topk_cosine_neighbors(
        item_emb, config.neighbor_k, chunk=config.knn_chunk, device=dev
    ).to(dev)
    sid_pos, sid_counts = prefix_positive_pool(
        codebook, config.sid_prefix_length, generator=gen
    )
    sid_pos, sid_counts = sid_pos.to(dev), sid_counts.to(dev)

    model = LatentRefiner(
        sid_dim=int(sid_emb.shape[1]),
        item_dim=int(item_emb.shape[1]),
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay)
    )

    stats = LatentRefinerStats()
    n_batches = max(1, (n_items + config.batch_size - 1) // config.batch_size)
    for epoch in range(int(config.epochs)):
        model.train()
        perm = torch.randperm(n_items, generator=gen).to(dev)
        acc = {"total": 0.0, "rec": 0.0, "nbr": 0.0, "sid": 0.0}
        for b in range(n_batches):
            idx = perm[b * config.batch_size : (b + 1) * config.batch_size]
            if idx.numel() < 2:
                continue
            z = model(sid_dev[idx], item_dev[idx])

            # (1) item-representation alignment / reconstruction
            l_rec = F.mse_loss(model.reconstruct(z), item_dev[idx])

            # (2) neighbourhood alignment: one x-space neighbour per anchor,
            #     resampled each step so all k contribute over the epoch.
            pick = torch.randint(
                0, max(1, nbr_idx.shape[1]), (idx.numel(),), device=dev
            )
            pos_ids = nbr_idx[idx, pick]
            z_pos = model(sid_dev[pos_ids], item_dev[pos_ids])
            # Anchor j occupies column j of the bank (the bank IS this batch).
            batch_cols = torch.arange(idx.numel(), device=dev)
            l_nbr = _infonce(
                z, z_pos, z, float(config.temperature), exclude_col=batch_cols
            )

            # (3) SID consistency, over the anchors that actually have a sibling.
            has_sib = sid_counts[idx] > 0
            if bool(has_sib.any()):
                sel = has_sib.nonzero(as_tuple=True)[0]  # columns into the bank
                a_idx = idx[sel]
                pool_n = sid_counts[a_idx]
                pick_s = (
                    torch.rand(a_idx.numel(), device=dev) * pool_n.float()
                ).long().clamp_max_(sid_pos.shape[1] - 1)
                sib_ids = sid_pos[a_idx, pick_s]
                z_a = model(sid_dev[a_idx], item_dev[a_idx])
                z_s = model(sid_dev[sib_ids], item_dev[sib_ids])
                l_sid = _infonce(
                    z_a, z_s, z, float(config.temperature), exclude_col=sel
                )
            else:
                l_sid = z.new_zeros(())

            loss = (
                float(config.w_reconstruction) * l_rec
                + float(config.w_neighborhood) * l_nbr
                + float(config.w_sid_consistency) * l_sid
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            acc["total"] += float(loss)
            acc["rec"] += float(l_rec)
            acc["nbr"] += float(l_nbr)
            acc["sid"] += float(l_sid)

        row = {k: v / n_batches for k, v in acc.items()}
        row["epoch"] = epoch + 1
        stats.epochs.append(row)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(config.epochs):
            log.info(
                "[latent-refiner] epoch %d/%d total=%.4f rec=%.4f nbr=%.4f sid=%.4f",
                epoch + 1,
                int(config.epochs),
                row["total"],
                row["rec"],
                row["nbr"],
                row["sid"],
            )

    stats.final = dict(stats.epochs[-1]) if stats.epochs else {}
    return model, stats


@torch.no_grad()
def export_latents(
    model: LatentRefiner,
    *,
    sid_emb: torch.Tensor,
    item_emb: torch.Tensor,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode the whole catalog to ``z`` ``[N, latent_dim]``, row-aligned.

    Re-standardises ``item_emb`` with its own statistics, matching training. The
    result is a plain tensor so ``load_dense_embeddings`` reads it directly.
    """
    dev = device or next(model.parameters()).device
    model.eval()
    mu = item_emb.mean(dim=0, keepdim=True)
    sigma = item_emb.std(dim=0, keepdim=True).clamp_min(1e-6)
    item_std = (item_emb - mu) / sigma
    out = torch.empty((int(sid_emb.shape[0]), model.latent_dim), dtype=torch.float32)
    for start in range(0, int(sid_emb.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(sid_emb.shape[0]))
        out[start:stop] = model(
            sid_emb[start:stop].to(dev), item_std[start:stop].to(dev)
        ).float().cpu()
    return out
