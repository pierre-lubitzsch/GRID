"""MUNKEY instance memory, ported from the authors' code (MUNKEY/ in this repo:
models/models.py TokenMemory, experiments/mul.py MUL.knn_mix_token / training_step /
global_step), "Models Designed to Forget: Machine Unlearning via Key Deletion".

One learnable ``token_dim`` token per training INSTANCE, mapped to the model
width by a linear adapter, plus a frozen, L2-normalised KEY per instance from a
frozen pretrained encoder (theirs: ViT-Base CLS; here: the mean of the frozen
item text embeddings over the instance's history). The key is only used for
retrieval; the token is not derived from it.

Training (MUL.training_step), decided ONCE PER BATCH:
  * w.p. ``p_retrieval`` (their knn_train_prob) every row's token is the
    softmax(sim / tau)-weighted mix of its K' nearest instances' tokens, with
    K' ~ U{2..topk} (var_k), every instance of the batch excluded from the
    candidates (they mask the batch ids, simulating unlearning) and the
    neighbour tokens DETACHED (freeze_memory);
  * otherwise the instance's own token.
Inference (MUL.global_step): ALWAYS retrieval with the query's key, top-``topk``,
no var_k, nothing detached, deleted keys masked. Unlearning = key deletion
(``delete``): a deleted instance is never retrieved again; no weight changes.

Pathway dropout (also once per batch, applied by the model) lives in the model,
because the "image" pathway is the model's own input.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class MunkeyMemory(nn.Module):
    def __init__(
        self,
        keys: torch.Tensor,
        model_dim: int,
        token_dim: int = 128,
        topk: int = 4,
        tau: float = 0.07,
        p_retrieval: float = 0.0,
        var_k: bool = True,
        freeze_memory: bool = True,
    ) -> None:
        super().__init__()
        if keys.dim() != 2:
            raise ValueError(f"keys must be (num_instances, key_dim), got {tuple(keys.shape)}")
        if not 0.0 <= p_retrieval <= 1.0:
            raise ValueError(f"p_retrieval must be in [0, 1], got {p_retrieval}")
        n = int(keys.size(0))
        self.num_instances = n
        self.topk = int(topk)
        self.tau = float(tau)
        self.p_retrieval = float(p_retrieval)
        self.var_k = bool(var_k)
        self.freeze_memory = bool(freeze_memory)
        # TokenMemory: table ~ N(0, 0.02), adapter to the model width, null token
        self.table = nn.Embedding(n, int(token_dim))
        nn.init.normal_(self.table.weight, std=0.02)
        self.adapter = nn.Linear(int(token_dim), int(model_dim))
        self.null_token = nn.Parameter(torch.randn(1, int(token_dim)) * 0.02)
        # frozen keys are derived data (rebuilt from the key file), the deletion
        # set is state: it must travel with an unlearned checkpoint
        self.register_buffer("keys", F.normalize(keys.float(), dim=-1), persistent=False)
        self.register_buffer("deleted", torch.zeros(n, dtype=torch.bool))

    def tokens(self, rows: torch.Tensor) -> torch.Tensor:
        """Own tokens of ``rows`` (B,), at the model width."""
        return self.adapter(self.table(rows))

    def null_tokens(self, batch: int) -> torch.Tensor:
        return self.adapter(self.null_token).expand(batch, -1)

    def sample_k(self) -> int:
        """K' ~ U{2..topk} (MUNKEY var_k); topk itself when topk < 2."""
        if self.topk < 2:
            return self.topk
        return int(torch.randint(2, self.topk + 1, (1,)).item())

    def retrieve(
        self,
        query: torch.Tensor,
        k: Optional[int] = None,
        exclude: Optional[torch.Tensor] = None,
        exclude_self: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """softmax(sim / tau)-weighted mix of the top-k instances' tokens, at the
        model width.

        query: (B, key_dim) keys (normalised here). exclude: instance rows masked
        for every query (the batch, in training). exclude_self: (B,) row of each
        query's own instance, -1 = none. Deleted instances are always masked.
        Neighbour tokens are detached in training when freeze_memory is set.
        """
        k = self.topk if k is None else int(k)
        with torch.no_grad():
            q = F.normalize(query.float(), dim=-1)
            sims = q @ self.keys.T  # (B, N)
            sims[:, self.deleted] = -float("inf")
            if exclude is not None and exclude.numel() > 0:
                sims[:, exclude.long()] = -float("inf")
            if exclude_self is not None:
                ok = exclude_self >= 0
                if ok.any():
                    sims[ok.nonzero().flatten(), exclude_self[ok].long()] = -float("inf")
            topv, topi = sims.topk(min(k, sims.size(1)), dim=-1)  # (B, K)
        neigh = self.table.weight[topi]  # (B, K, token_dim)
        if self.training and self.freeze_memory:
            neigh = neigh.detach()
        w = F.softmax(topv / self.tau, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)  # a query with no candidate at all gets 0
        return self.adapter((w.unsqueeze(-1) * neigh).sum(dim=1))

    @torch.no_grad()
    def delete(self, rows: torch.Tensor) -> int:
        """Key deletion; returns the number of newly deleted instances."""
        rows = rows.long().to(self.deleted.device)
        rows = rows[(rows >= 0) & (rows < self.num_instances)].unique()
        new = int((~self.deleted[rows]).sum().item())
        self.deleted[rows] = True
        return new


def build_munkey_user_memory(key_file: str, model_dim: int, **kwargs) -> MunkeyMemory:
    """User-instance MUNKEY memory from scripts/build_user_neighbors.py --save_keys.

    Rows follow the file's sorted ``user_ids``; the user-id -> row map is the
    non-persistent buffer ``uid_to_row`` (-1 = a user without an instance).
    """
    d = torch.load(key_file, map_location="cpu", weights_only=False)
    if "keys" not in d:
        raise ValueError(f"{key_file} has no 'keys': rebuild it with build_user_neighbors.py --save_keys")
    uids = d["user_ids"].long()
    mem = MunkeyMemory(d["keys"], model_dim, **kwargs)
    lut = torch.full((int(uids.max()) + 1,), -1, dtype=torch.long)
    lut[uids] = torch.arange(uids.numel())
    mem.register_buffer("uid_to_row", lut, persistent=False)
    return mem
