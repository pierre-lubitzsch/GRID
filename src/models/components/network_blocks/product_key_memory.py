"""Product-Key Memory (PKM) layer.

Ported from the minimalist reference implementation shipped with Facebook's
XLM repo (``XLM/PKM-layer.ipynb``), described in

    Lample, Sablayrolles, Ranzato, Denoyer, Jegou.
    "Large Memory Layers with Product Keys", NeurIPS 2019.
    https://arxiv.org/abs/1907.05242

The reference notebook already implements the vectorized ``pq_fast`` style
retrieval used in ``XLM/xlm/model/memory/memory.py`` (``HashingMemoryProductFast``):
each head's query is split in half, the top-``knn`` sub-keys are found for each
half with a plain ``topk`` (no FAISS dependency), and the two candidate lists are
combined with a cartesian product before a final top-``knn`` selection.

A PKM maps ``R^input_dim -> R^output_dim`` through a large, sparse key-value
store holding ``n_keys ** 2`` value vectors, of which only ``heads * knn`` are
read (and gradient-updated) per token. It is a drop-in, high-capacity
replacement for a transformer feed-forward sub-layer.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_uniform_keys(n_keys: int, dim: int, seed: int) -> np.ndarray:
    """Generate random uniform keys (same initialization as ``nn.Linear``)."""
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)


class HashingMemory(nn.Module):
    """Minimalist Product-Key Memory layer.

    Args:
        input_dim: dimension of the input vectors (the model dimension).
        output_dim: dimension of the output / value vectors.
        k_dim: dimension of the query/key space. Must be even (it is split in
            half for product quantization).
        heads: number of independent reading heads (their outputs are summed).
        knn: number of memory slots read per head (k nearest sub-keys per half).
        n_keys: number of sub-keys per half. The memory holds ``n_keys ** 2``
            values, so this is the dominant capacity / parameter-count knob.
        query_batchnorm: apply BatchNorm1d to the queries. Improves usage of the
            memory, but corrupts the running stats when batches contain padding
            tokens of varying counts -- keep ``False`` for TIGER unless training
            on padding-free fixed-size batches.
        input_dropout / query_dropout / value_dropout: dropout rates.
        sparse: use sparse gradients for the value ``EmbeddingBag`` (requires a
            sparse-capable optimizer, e.g. SparseAdam).
        value_init: ``normal`` (default, ``N(0, v_dim**-0.5)``) or ``zeros``.
            ``zeros`` makes the layer an exact no-op at step 0. For a POST-HOC
            ``add`` adapter this is the right choice -- the model starts exactly
            at its trained behaviour and the memory learns a pure correction
            (the LoRA convention). For post-hoc ``replace`` it means the FFN is
            *deleted* rather than *randomised*, which is a cleaner starting point
            for a rebuild than injecting noise.
            CAVEAT: with all-zero values the layer output is 0 AND
            ``d(output)/d(scores) = 0``, so ``query_proj`` and ``keys`` receive
            NO gradient at step 0 -- routing only starts learning once the values
            become non-zero. Since PKM collapse is a ROUTING failure (see
            WORKFLOW.md section H), prefer a small non-zero scale over exact
            zeros when the memory must also learn to route (i.e. trained-in).
        value_init_scale: multiplier on the ``normal`` std. ``1.0`` reproduces
            the reference init; e.g. ``0.01`` gives a near-no-op start that still
            keeps the routing gradients alive.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k_dim: int = 128,
        heads: int = 4,
        knn: int = 32,
        n_keys: int = 512,
        query_batchnorm: bool = False,
        input_dropout: float = 0.0,
        query_dropout: float = 0.0,
        value_dropout: float = 0.0,
        sparse: bool = False,
        value_init: str = "normal",
        value_init_scale: float = 1.0,
    ) -> None:
        super().__init__()

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_dim = k_dim
        self.v_dim = output_dim
        self.n_keys = n_keys
        self.size = n_keys ** 2
        self.heads = heads
        self.knn = knn
        assert self.k_dim >= 2 and self.k_dim % 2 == 0, "k_dim must be even and >= 2"
        assert self.knn <= self.n_keys, "knn must be <= n_keys"

        # dropout
        self.input_dropout = input_dropout
        self.query_dropout = query_dropout
        self.value_dropout = value_dropout

        # initialize keys / values
        self.initialize_keys()
        self.values = nn.EmbeddingBag(self.size, self.v_dim, mode="sum", sparse=sparse)
        vi = str(value_init).strip().lower()
        if vi == "zeros":
            nn.init.zeros_(self.values.weight)
        elif vi == "normal":
            nn.init.normal_(
                self.values.weight,
                mean=0,
                std=(self.v_dim ** -0.5) * float(value_init_scale),
            )
        else:
            raise ValueError(
                f"value_init must be 'normal' or 'zeros', got {value_init!r}"
            )
        self.value_init = vi
        self.value_init_scale = float(value_init_scale)

        # slot-access instrumentation: off by default, zero overhead when off.
        # The counter BUFFERS are created lazily by enable_access_counting();
        # assigning them here as plain attributes would make register_buffer raise.
        self._count_access = False

        # query network (linear projection, optionally followed by batchnorm)
        self.query_proj = nn.Sequential(
            *filter(
                None,
                [
                    nn.Linear(self.input_dim, self.heads * self.k_dim, bias=True),
                    nn.BatchNorm1d(self.heads * self.k_dim)
                    if query_batchnorm
                    else None,
                ],
            )
        )

    # ---- slot-access instrumentation (opt-in, off by default) --------------
    # Records which of the ``size`` value slots each forward pass reads. Used to
    # build the access statistics that drive top-t memory-slot selection
    # (access-frequency / inverse-history-frequency, the AF-IHF analogue of the
    # TF-IDF criterion in Sparse Memory Finetuning).
    #
    # The counters are registered with persistent=False so they NEVER enter the
    # state_dict: adding them must not change checkpoint schemas or break the
    # strict=False loads used throughout the unlearning pipeline.
    def enable_access_counting(self) -> None:
        """Start accumulating per-slot read counts (and softmax mass)."""
        dev = self.keys.device
        if getattr(self, "_access_count", None) is None:
            self.register_buffer(
                "_access_count", torch.zeros(self.size, device=dev), persistent=False
            )
            self.register_buffer(
                "_access_mass", torch.zeros(self.size, device=dev), persistent=False
            )
        self._count_access = True

    def disable_access_counting(self) -> None:
        self._count_access = False

    def reset_access_counts(self) -> None:
        if getattr(self, "_access_count", None) is not None:
            self._access_count.zero_()
            self._access_mass.zero_()

    def get_access_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(read_count, softmax_mass)`` per slot, both length ``size``."""
        if getattr(self, "_access_count", None) is None:
            raise RuntimeError(
                "access counting was never enabled; call enable_access_counting()"
            )
        return self._access_count.detach().clone(), self._access_mass.detach().clone()

    def initialize_keys(self) -> None:
        """Create two sub-key sets per head.

        ``self.keys`` has shape ``(heads, 2, n_keys, k_dim // 2)``.
        """
        half = self.k_dim // 2
        keys = torch.from_numpy(
            np.array(
                [
                    get_uniform_keys(self.n_keys, half, seed=(2 * i + j))
                    for i in range(self.heads)
                    for j in range(2)
                ]
            )
        ).view(self.heads, 2, self.n_keys, half)
        self.keys = nn.Parameter(keys)

    def _get_indices(
        self, query: torch.Tensor, subkeys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate scores and indices for a single head (vectorized pq_fast)."""
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = query.size(0)
        knn = self.knn
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]  # (bs, half)
        q2 = query[:, half:]  # (bs, half)

        # compute indices with associated scores
        scores1 = F.linear(q1, subkeys[0], bias=None)  # (bs, n_keys)
        scores2 = F.linear(q2, subkeys[1], bias=None)  # (bs, n_keys)
        scores1, indices1 = scores1.topk(knn, dim=1)  # (bs, knn)
        scores2, indices2 = scores2.topk(knn, dim=1)  # (bs, knn)

        # cartesian product on the best candidate keys
        all_scores = (
            scores1.view(bs, knn, 1).expand(bs, knn, knn)
            + scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)  # (bs, knn ** 2)
        all_indices = (
            indices1.view(bs, knn, 1).expand(bs, knn, knn) * n_keys
            + indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)  # (bs, knn ** 2)

        # select the overall best scores with associated indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1)  # (bs, knn)
        indices = all_indices.gather(1, best_indices)  # (bs, knn)

        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices

    def get_indices(
        self, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate scores and indices for all heads."""
        assert query.dim() == 2 and query.size(1) == self.k_dim
        query = query.view(-1, self.heads, self.k_dim)
        bs = len(query)
        outputs = [self._get_indices(query[:, i], self.keys[i]) for i in range(self.heads)]
        s = torch.cat([s.view(bs, 1, self.knn) for s, _ in outputs], 1)  # (bs, heads, knn)
        i = torch.cat([i.view(bs, 1, self.knn) for _, i in outputs], 1)  # (bs, heads, knn)
        return s.view(-1, self.knn), i.view(-1, self.knn)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Read from the memory.

        Accepts an input of shape ``(..., input_dim)`` and returns ``(..., v_dim)``.
        """
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        bs = int(np.prod(prefix_shape)) if len(prefix_shape) > 0 else 1

        # compute query
        input = F.dropout(input, p=self.input_dropout, training=self.training)  # (..., i_dim)
        query = self.query_proj(input.contiguous().view(-1, self.input_dim))  # (bs, heads*k_dim)
        query = query.view(bs * self.heads, self.k_dim)  # (bs*heads, k_dim)
        query = F.dropout(query, p=self.query_dropout, training=self.training)  # (bs*heads, k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        # retrieve indices and scores
        scores, indices = self.get_indices(query)  # (bs*heads, knn)
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs*heads, knn)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * self.knn)  # (bs, heads*knn)
        scores = scores.view(bs, self.heads * self.knn)  # (bs, heads*knn)

        if self._count_access:
            # Read-only bookkeeping: detached, no autograd, no effect on output.
            with torch.no_grad():
                flat_idx = indices.reshape(-1)
                self._access_count.index_add_(
                    0, flat_idx, torch.ones_like(flat_idx, dtype=self._access_count.dtype)
                )
                self._access_mass.index_add_(
                    0, flat_idx, scores.reshape(-1).detach().to(self._access_mass.dtype)
                )

        # weighted sum of values
        output = self.values(indices, per_sample_weights=scores)  # (bs, v_dim)
        output = F.dropout(output, p=self.value_dropout, training=self.training)  # (bs, v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))  # (..., v_dim)
        elif len(prefix_shape) == 0:
            output = output.view(self.v_dim)
        return output
