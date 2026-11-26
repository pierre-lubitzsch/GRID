"""Item-keyed exemplar memory with neighbour substitution.

Adapted from MUNKEY ("Models Designed to Forget: Machine Unlearning via Key
Deletion", Laguna et al. 2026, arXiv:2603.15033). MUNKEY keeps one learnable
exemplar token per training INSTANCE, keyed by a frozen encoder, and trains with
(i) stochastic pathway dropout and (ii) retrieval regularisation that replaces an
instance's own token by a similarity-weighted average of its neighbours' tokens.
Forgetting = deleting the key.

Here the memory is keyed by ITEM, because items are the persistent unit a
recommender keeps serving: a forgotten item must stay in the catalogue, so its
entry is not deleted but UPDATED by an unlearning algorithm, which can put most
of the update on the forget items' entries and little on the shared backbone
(see unified's item_memory_lr_scale / backbone_lr_scale / item_memory_rows).
Training with neighbour substitution makes an entry a meaningful, swappable
carrier of item-specific information rather than an arbitrary bias.

Each item in the encoder input gets ``adapter(v)`` added to all of its SID token
embeddings, where per occurrence during training

    v = null                                   with prob p_drop  (pathway dropout)
      = sum_j w_j * stopgrad(v_j), j in N_K'(i) with prob p_nbr   (neighbours)
      = v_i                                    otherwise

with N_K'(i) the K' ~ U{1..max_neighbors} nearest items by cosine similarity of
a FROZEN item embedding (self excluded) and w = softmax(sim / tau). At eval
v = v_i.

recipe="munkey" follows the authors' code (MUNKEY/experiments/mul.py) instead:
training draws ONCE PER BATCH, independently, (a) w.p. p_nbr every entry is its
neighbour mix with one K' ~ U{2..max_neighbors} for the batch (detached), and
(b) w.p. p_drop the whole batch gets the null token; eval ALWAYS retrieves:
softmax(sim / tau) over the top-max_neighbors of [the item itself (sim 1) +
its neighbours], because items, unlike MUNKEY's test images, are instances.
Deviation: MUNKEY excludes every instance of the batch from the retrieval
candidates; a batch holds thousands of items, so only the item itself is.
"""

import torch
from torch import nn
from torch.nn import functional as F


def nearest_items(
    embeddings: torch.Tensor, k: int, chunk: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` cosine neighbours of every row, excluding the row itself.

    Returns ``(idx, sim)``, both ``(n, k)``.
    """
    e = F.normalize(embeddings.float(), dim=-1)
    n = e.size(0)
    idx_out, sim_out = [], []
    for s in range(0, n, chunk):
        sims = e[s : s + chunk] @ e.T  # (c, n)
        rows = torch.arange(s, min(s + chunk, n))
        sims[torch.arange(len(rows)), rows] = -float("inf")  # exclude self
        sv, si = sims.topk(k, dim=-1)
        idx_out.append(si)
        sim_out.append(sv)
    return torch.cat(idx_out), torch.cat(sim_out)


class ItemExemplarMemory(nn.Module):
    def __init__(
        self,
        num_items: int,
        dim: int,
        neighbor_idx: torch.Tensor,
        neighbor_sim: torch.Tensor,
        p_drop: float = 0.1,
        p_nbr: float = 0.2,
        tau: float = 0.1,
        init_scale: float = 0.02,
        recipe: str = "ours",
        row_dim: int | None = None,
        head_ids: torch.Tensor | None = None,
        tail_nbr_idx: torch.Tensor | None = None,
        tail_nbr_sim: torch.Tensor | None = None,
        eval_quant: str = "none",
    ) -> None:
        """Memory-size options (2026-11-25, the memory-reduction study; all default off):

        ``row_dim`` r < dim: rows are r-dimensional, the adapter (and the score head's
        out_proj) map r -> dim. ``head_ids``: only these items (e.g. the most frequent
        in training) get a row; every other item's entry is the softmax(sim/tau) mix of
        its ``tail_nbr_idx`` / ``tail_nbr_sim`` nearest HEAD items (content similarity),
        not detached, so head rows also learn to serve their tail neighbours.
        ``eval_quant`` int8 | fp16: rows are quantized (per-row symmetric int8, or
        half precision) and dequantized at evaluation only (post-hoc compression).
        """
        super().__init__()
        self.recipe = str(recipe).strip().lower()
        if self.recipe not in ("ours", "munkey"):
            raise ValueError(f"recipe must be ours|munkey, got {recipe!r}")
        if self.recipe == "munkey" and not (0 <= p_drop < 1 and 0 <= p_nbr <= 1):
            raise ValueError(f"recipe=munkey needs p_drop in [0, 1), p_nbr in [0, 1], got {p_drop}, {p_nbr}")
        if self.recipe == "ours" and not (0 <= p_drop and 0 <= p_nbr and p_drop + p_nbr < 1):
            raise ValueError(f"need p_drop, p_nbr >= 0 and p_drop + p_nbr < 1, got {p_drop}, {p_nbr}")
        if neighbor_idx.shape != (num_items, neighbor_idx.size(1)):
            raise ValueError(
                f"neighbor_idx must be (num_items={num_items}, K), got {tuple(neighbor_idx.shape)}"
            )
        self.num_items = int(num_items)
        self.max_neighbors = int(neighbor_idx.size(1))
        self.p_drop, self.p_nbr, self.tau = float(p_drop), float(p_nbr), float(tau)
        self.dim = int(dim)
        r = int(row_dim) if row_dim else int(dim)
        self.eval_quant = str(eval_quant).lower()
        if self.eval_quant not in ("none", "int8", "fp16"):
            raise ValueError(f"eval_quant must be none|int8|fp16, got {eval_quant!r}")
        if head_ids is not None:
            head_ids = torch.as_tensor(head_ids, dtype=torch.long)
            if tail_nbr_idx is None or tail_nbr_sim is None:
                raise ValueError("head_ids needs tail_nbr_idx / tail_nbr_sim")
            row_of = torch.full((num_items,), -1, dtype=torch.long)
            row_of[head_ids] = torch.arange(head_ids.numel())
            # tail_nbr_idx are ROW indices (positions in head_ids) for every item
            self.register_buffer("row_of", row_of, persistent=False)
            self.register_buffer("tail_nbr_idx", tail_nbr_idx.long(), persistent=False)
            self.register_buffer("tail_nbr_sim", tail_nbr_sim.float(), persistent=False)
            num_rows = int(head_ids.numel())
        else:
            self.row_of = None
            num_rows = int(num_items)
        self.values = nn.Embedding(num_rows, r)
        nn.init.normal_(self.values.weight, std=float(init_scale))
        self.null_value = nn.Parameter(torch.zeros(r))
        self.adapter = nn.Linear(r, dim)  # h_psi in MUNKEY
        # Output side (optional, see enable_item_score): s_i(h) = <h, W_out v_i>,
        # so an item's own entry also shapes how strongly IT is recommended.
        self.out_proj: nn.Linear | None = None
        # derived from a frozen embedding file at construction -> not checkpointed
        self.register_buffer("neighbor_idx", neighbor_idx.long(), persistent=False)
        self.register_buffer("neighbor_sim", neighbor_sim.float(), persistent=False)
        self._count_access = False
        # "neighbor": every entry is replaced by its deterministic full-K
        # neighbour average (detached) -- used by the consistency loss.
        self.force_mode: str | None = None
        # Zero-shot item deletion (MUNKEY-style key deletion, the item is KEPT in the
        # catalogue): a deleted item's entry is replaced everywhere (input term and
        # output score) by the softmax(sim/tau) mix of its non-deleted neighbours;
        # under recipe=munkey deleted keys are simply never retrieved. State, so it
        # travels with an unlearned checkpoint; older checkpoints load with none.
        self.register_buffer("deleted", torch.zeros(num_items, dtype=torch.bool))

    def _table(self) -> torch.Tensor:
        w = self.values.weight
        if self.training or self.eval_quant == "none":
            return w
        if self.eval_quant == "fp16":
            return w.half().to(w.dtype)
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0  # per-row int8
        return (w / scale).round().clamp(-127, 127) * scale

    def _rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Entries of item ids (..., ) -> (..., r): own row, or for items without a row
        (head-only memory) the mix of their nearest head rows."""
        w = self._table()
        if self.row_of is None:
            return F.embedding(ids, w)
        row = self.row_of[ids]
        own = F.embedding(row.clamp(min=0), w)
        nb = F.embedding(self.tail_nbr_idx[ids], w)  # (..., K, r)
        a = torch.softmax(self.tail_nbr_sim[ids] / self.tau, dim=-1)
        mix = (nb * a.unsqueeze(-1)).sum(-2)
        return torch.where((row >= 0).unsqueeze(-1), own, mix)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "deleted" not in state_dict:  # checkpoints from before 2026-10-21
            state_dict[prefix + "deleted"] = torch.zeros_like(self.deleted)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def delete(self, item_ids) -> int:
        """Delete items' keys; returns the number newly deleted."""
        ids = torch.as_tensor(item_ids, dtype=torch.long, device=self.deleted.device).flatten()
        ids = ids[(ids >= 0) & (ids < self.num_items)].unique()
        new = int((~self.deleted[ids]).sum().item())
        self.deleted[ids] = True
        return new

    def _substitute_deleted(self, ids: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Entries of deleted items -> mix of their non-deleted neighbours."""
        if not bool(self.deleted.any()):
            return v
        gone = self.deleted[ids]
        if not bool(gone.any()):
            return v
        nid = self.neighbor_idx[ids]
        logits = (self.neighbor_sim[ids] / self.tau).masked_fill(self.deleted[nid], -float("inf"))
        w = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        sub = (self._rows(nid).detach() * w.unsqueeze(-1)).sum(-2)
        return torch.where(gone.unsqueeze(-1), sub.to(v.dtype), v)

    # ---- per-item read counts (for AF-IHF row selection in unlearning) ------
    def enable_access_counting(self) -> None:
        if getattr(self, "_access_count", None) is None:
            self.register_buffer(
                "_access_count",
                torch.zeros(self.num_items, device=self.values.weight.device),
                persistent=False,
            )
        self._access_count.zero_()
        self._count_access = True

    def disable_access_counting(self) -> None:
        self._count_access = False

    def get_access_counts(self) -> torch.Tensor:
        return self._access_count.detach().clone()

    def enable_item_score(self) -> None:
        if self.out_proj is None:
            d = self.values.weight.size(1)
            self.out_proj = nn.Linear(d, self.dim, bias=False).to(self.values.weight.device)

    def item_score(self, h: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """``h`` (B, d) user state, ``item_ids`` (B, ...) -> scores (B, ...).

        Uses each item's OWN entry (no substitution): the score is what the
        unlearning update acts on. Invalid ids (-1) score 0.
        """
        valid = item_ids >= 0
        ids = item_ids.clamp(min=0)
        v = self.out_proj(self._substitute_deleted(ids, self._rows(ids)))  # (B, ..., d)
        hh = h.view(h.size(0), *([1] * (item_ids.dim() - 1)), h.size(-1))
        return (v * hh).sum(-1) * valid.to(v.dtype)

    def _neighbor_average(self, ids: torch.Tensor) -> torch.Tensor:
        nid = self.neighbor_idx[ids]  # (..., K)
        sim = self.neighbor_sim[ids]
        k = torch.randint(1, self.max_neighbors + 1, ids.shape, device=ids.device)
        keep = torch.arange(self.max_neighbors, device=ids.device) < k.unsqueeze(-1)
        w = torch.softmax((sim / self.tau).masked_fill(~keep, -float("inf")), dim=-1)
        return (self._rows(nid).detach() * w.unsqueeze(-1)).sum(-2)

    def _munkey_values(self, ids: torch.Tensor) -> torch.Tensor:
        """recipe=munkey entries (see the module docstring)."""
        K = self.max_neighbors
        if self.training:
            # DDP needs every parameter in the graph on every step: the own
            # entries and the null value are always used, an inactive branch
            # contributing exactly zero
            v = self._substitute_deleted(ids, self._rows(ids))
            if torch.rand(()).item() < self.p_nbr:
                k = int(torch.randint(min(2, K), K + 1, (1,)).item())
                w = torch.softmax(self.neighbor_sim[ids][..., :k] / self.tau, dim=-1)
                v = (self._rows(self.neighbor_idx[ids][..., :k]).detach() * w.unsqueeze(-1)).sum(-2) + 0.0 * v
            drop = torch.tensor(torch.rand(()).item() < self.p_drop, device=v.device)
            return torch.where(drop, self.null_value.to(v.dtype).expand_as(v), v)
        # eval: top-K of [self (sim 1)] + neighbours (lists are sorted by sim)
        sim = torch.cat([torch.ones_like(self.neighbor_sim[ids][..., :1]),
                         self.neighbor_sim[ids][..., : K - 1]], dim=-1)
        nid = torch.cat([ids.unsqueeze(-1), self.neighbor_idx[ids][..., : K - 1]], dim=-1)
        logits = (sim / self.tau).masked_fill(self.deleted[nid], -float("inf"))  # key deletion
        w = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        return (self._rows(nid) * w.unsqueeze(-1)).sum(-2)

    def forward(self, item_ids: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``item_ids`` (..., ) with -1 for padding -> (..., dim) additive term.

        ``drop_mask`` (same shape as ``item_ids``, bool): if given, it decides
        which entries are replaced by the null token (instead of sampling with
        p_drop), so a caller can make token dropout mutually exclusive with
        dropping the other pathway, as in MUNKEY's categorical mask.
        """
        valid = item_ids >= 0
        ids = item_ids.clamp(min=0)
        if self._count_access:
            with torch.no_grad():
                self._access_count.index_add_(
                    0, ids[valid], torch.ones_like(ids[valid], dtype=self._access_count.dtype)
                )
        if self.recipe == "munkey" and drop_mask is None and self.force_mode is None:
            return self.adapter(self._munkey_values(ids)) * valid.unsqueeze(-1).to(self.values.weight.dtype)
        if self.force_mode == "neighbor":
            logits = (self.neighbor_sim[ids] / self.tau).masked_fill(
                self.deleted[self.neighbor_idx[ids]], -float("inf"))
            w = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
            v = (self._rows(self.neighbor_idx[ids]).detach() * w.unsqueeze(-1)).sum(-2)
            return self.adapter(v) * valid.unsqueeze(-1).to(v.dtype)
        v = self._substitute_deleted(ids, self._rows(ids))
        if drop_mask is not None:
            # explicit mask: applied in eval too (pathway-sensitivity evaluation);
            # neighbour substitution stays training-only
            if self.training and self.p_nbr > 0:
                u = torch.rand(ids.shape, device=ids.device).unsqueeze(-1)
                v = torch.where(u < self.p_nbr, self._neighbor_average(ids), v)
            v = torch.where(drop_mask.unsqueeze(-1), self.null_value.to(v.dtype).expand_as(v), v)
        elif self.training and (self.p_drop > 0 or self.p_nbr > 0):
            u = torch.rand(ids.shape, device=ids.device).unsqueeze(-1)
            if self.p_nbr > 0:
                v = torch.where(u < self.p_drop + self.p_nbr, self._neighbor_average(ids), v)
            if self.p_drop > 0:
                v = torch.where(u < self.p_drop, self.null_value.to(v.dtype).expand_as(v), v)
        return self.adapter(v) * valid.unsqueeze(-1).to(v.dtype)


def build_item_memory(
    num_items: int,
    dim: int,
    embedding_path: str,
    max_neighbors: int = 4,
    p_drop: float = 0.1,
    p_nbr: float = 0.2,
    tau: float = 0.1,
    init_scale: float = 0.02,
    recipe: str = "ours",
    row_dim: int | None = None,
    head_frac: float = 1.0,
    counts_path: str | None = None,
    eval_quant: str = "none",
) -> ItemExemplarMemory:
    """Load the frozen item embeddings, find neighbours, build the memory.

    ``head_frac`` < 1: only the top head_frac of items by training frequency
    (``counts_path``: a (num_items,) tensor of training counts, from
    scripts/item_train_counts.py) get their own row; see ItemExemplarMemory."""
    emb = torch.load(embedding_path, map_location="cpu", weights_only=False)
    if emb.size(0) != num_items:
        raise ValueError(
            f"item embedding rows ({emb.size(0)}) != num_items ({num_items}); "
            f"{embedding_path} must be indexed by item id"
        )
    idx, sim = nearest_items(emb, int(max_neighbors))
    head = tidx = tsim = None
    if float(head_frac) < 1.0:
        if not counts_path:
            raise ValueError("head_frac < 1 needs counts_path (item training counts)")
        counts = torch.load(counts_path, map_location="cpu", weights_only=False).float()
        if counts.numel() != num_items:
            raise ValueError(f"counts ({counts.numel()}) != num_items ({num_items})")
        n_head = max(1, int(round(float(head_frac) * num_items)))
        head = counts.argsort(descending=True, stable=True)[:n_head].sort().values
        e = F.normalize(emb.float(), dim=-1)
        eh = e[head]
        K = int(max_neighbors)
        tidx = torch.empty(num_items, K, dtype=torch.long)
        tsim = torch.empty(num_items, K)
        for s0 in range(0, num_items, 4096):
            sv, si = (e[s0:s0 + 4096] @ eh.T).topk(K, dim=-1)
            tidx[s0:s0 + 4096], tsim[s0:s0 + 4096] = si, sv
    return ItemExemplarMemory(
        num_items, dim, idx, sim, p_drop=p_drop, p_nbr=p_nbr, tau=tau, init_scale=init_scale,
        recipe=recipe, row_dim=row_dim, head_ids=head, tail_nbr_idx=tidx, tail_nbr_sim=tsim,
        eval_quant=eval_quant,
    )


def build_user_memory(
    neighbor_file: str,
    dim: int,
    p_drop: float = 0.1,
    p_nbr: float = 0.2,
    tau: float = 0.1,
    init_scale: float = 0.02,
) -> ItemExemplarMemory:
    """User-keyed exemplar memory: one learnable token per TRAINING user.

    ``neighbor_file`` (scripts/build_user_neighbors.py) fixes the rows (sorted
    user ids) and the neighbours (by frozen history embedding). The same module
    class as the item memory, with users as the instances; the user-id -> row
    map is stored as the non-persistent buffer ``uid_to_row`` (-1 = unknown
    user, e.g. a user absent from training: that user gets no token).

    At unlearning time a DIFFERENT neighbour file with the same rows may be
    passed (built from the retained histories), so a reset towards the
    neighbour average uses neighbours that no longer see the deleted pairs.
    """
    d = torch.load(neighbor_file, map_location="cpu", weights_only=False)
    uids = d["user_ids"].long()
    mem = ItemExemplarMemory(
        int(uids.numel()), dim, d["neighbor_idx"], d["neighbor_sim"],
        p_drop=p_drop, p_nbr=p_nbr, tau=tau, init_scale=init_scale,
    )
    lut = torch.full((int(uids.max()) + 1,), -1, dtype=torch.long)
    lut[uids] = torch.arange(uids.numel())
    mem.register_buffer("uid_to_row", lut, persistent=False)
    return mem


class GroupMemory(nn.Module):
    """Category memory keyed by the FIRST semantic-ID level (between the item memory and
    a PKM): every group (level-0 code) owns ``slots`` key/value pairs; each history item
    reads the top-``knn`` slots of ITS group by content (query = the item's mean SID-token
    embedding), softmax-weighted, and the result is added to all of the item's tokens.
    Structure is explicit per category; capacity per category is ``slots`` vectors chosen
    dynamically, rather than one fixed vector per item.

    Training (optional): w.p. ``p_drop`` per sequence the read is replaced by a learned null
    vector (pathway dropout). The null is always in the graph (DDP).
    """

    def __init__(self, num_groups: int, dim: int, slots: int = 32, knn: int = 4,
                 key_dim: int = 64, p_drop: float = 0.0, init_scale: float = 0.02) -> None:
        super().__init__()
        if knn > slots:
            raise ValueError(f"knn ({knn}) must be <= slots ({slots})")
        if not 0.0 <= p_drop < 1.0:
            raise ValueError(f"p_drop must be in [0, 1), got {p_drop}")
        self.num_groups, self.slots, self.knn = int(num_groups), int(slots), int(knn)
        self.p_drop = float(p_drop)
        self.query = nn.Linear(dim, key_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(num_groups, slots, key_dim) / key_dim ** 0.5)
        self.values = nn.Parameter(torch.randn(num_groups, slots, dim) * init_scale)
        self.null_value = nn.Parameter(torch.zeros(dim))
        self.adapter = nn.Linear(dim, dim)

    def forward(self, groups: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """``groups`` (B, n) level-0 codes (-1 = padding), ``item_emb`` (B, n, dim) ->
        (B, n, dim) additive term (0 for padding)."""
        valid = (groups >= 0) & (groups < self.num_groups)
        g = groups.clamp(0, self.num_groups - 1)
        q = self.query(item_emb)                                  # (B, n, k)
        scores = torch.einsum("bnk,bnmk->bnm", q, self.keys[g])   # (B, n, slots)
        top, idx = scores.topk(self.knn, dim=-1)                  # (B, n, knn)
        w = torch.softmax(top, dim=-1)
        vals = self.values[g.unsqueeze(-1).expand_as(idx), idx]   # (B, n, knn, dim)
        out = (w.unsqueeze(-1) * vals).sum(-2)
        if self.training and self.p_drop > 0:
            drop = torch.rand(out.size(0), 1, 1, device=out.device) < self.p_drop
            out = torch.where(drop, self.null_value.to(out.dtype).expand_as(out), out)
        elif self.training:
            out = out + 0.0 * self.null_value.sum()               # keep it in the graph (DDP)
        return self.adapter(out) * valid.unsqueeze(-1).to(out.dtype)
