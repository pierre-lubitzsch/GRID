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
import torch.utils.checkpoint
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
        query_norm: anti-collapse normalisation of the query, applied BEFORE the
            product-key lookup. ``none`` (default) reproduces prior behaviour.

            ``batchnorm`` normalises each query dimension across the batch with
            ``track_running_stats=False``, so batch statistics are used at train
            AND eval time. This is the mechanism Lample et al. use to spread
            memory usage, and it is the ONLY one that removes a component shared
            by every token -- exactly the failure we measured (every token and
            head selecting the identical 32 slots). ``query_batchnorm=True``
            (the stock BatchNorm) was avoided here because its RUNNING stats are
            corrupted by TIGER's variable padding; dropping the running stats
            removes that objection at the cost of making eval batch-composition
            dependent (deterministic for our fixed eval batches).

            ``layernorm`` normalises per token across dims. Cheaper and
            padding-safe, but it CANNOT remove a token-shared constant
            direction, so it is the weaker option for this failure mode.
        warmup_knn / warmup_steps: for the first ``warmup_steps`` forward passes
            in training mode, read ``warmup_knn`` slots per head instead of
            ``knn``. Rationale: at init the values are random, so which slots the
            router picks barely changes the loss -> almost no gradient pressure on
            the query projection -> it collapses to a near-constant, after which
            only the surviving slots ever train and the collapse self-reinforces.
            Touching many more slots early makes slot CONTENTS meaningful, so
            routing choices start to matter while the router can still move.
            ``warmup_knn=0`` (default) disables this.

    Exploration (explore / exploit training): the owning model may switch the
    layer into an *explore* pass via :meth:`set_explore`. On such a pass the
    read covers ALL ``n_keys ** 2`` slots instead of the top-``knn`` ones, so
    every value row, every sub-key and the query net receive gradient. This is
    exact and cheap thanks to the product structure: the softmax over all
    pairs ``s1_i + s2_j`` factorises into ``softmax(s1) (x) softmax(s2)``, so
    the read is ``p1^T V p2`` per head (see :meth:`_dense_read`). Two variants:

        ``dense``             the forward output IS the dense read (literal
                              "update all slots" step; the next layers see a
                              smoothed memory output on that step).
        ``straight_through``  the forward output stays the usual sparse read,
                              but gradient additionally flows through the
                              dense read (``sparse + dense - dense.detach()``),
                              so all slots are updated without changing the
                              forward pass.

    Exploration is OFF unless the model turns it on (it is driven per optimizer
    step by ``pkm_explore`` on the model), so inference / unlearning are
    unaffected. Adds nothing to the state_dict.
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
        query_norm: str = "none",
        warmup_knn: int = 0,
        warmup_noise: float = 0.0,
        warmup_steps: int = 0,
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

        # anti-collapse query normalisation (see the class docstring)
        self.query_norm = str(query_norm).strip().lower()
        if self.query_norm not in ("none", "batchnorm", "layernorm"):
            raise ValueError(
                f"query_norm must be none|batchnorm|layernorm, got {query_norm!r}"
            )
        if self.query_norm == "batchnorm":
            # track_running_stats=False -> batch stats at train AND eval, so the
            # padding-corrupted running stats that motivated disabling BatchNorm
            # never exist.
            self.q_norm = nn.BatchNorm1d(self.k_dim, track_running_stats=False)
        elif self.query_norm == "layernorm":
            self.q_norm = nn.LayerNorm(self.k_dim)
        else:
            self.q_norm = None

        # dense-ish warm-up: read more slots per head for the first N steps
        self.warmup_knn = int(warmup_knn or 0)
        self.warmup_steps = int(warmup_steps or 0)
        # Cost of the lookup scales as knn**2 (the cartesian product over the two
        # sub-key halves), so a large warmup_knn is not just slow: at knn=256 the
        # candidate tensor is (rows, 65536), which overflowed int32 indexing in
        # topk and aborted with 'CUDA error: illegal memory access' (job
        # 10449833). Cap it well below that and prefer warmup_noise instead.
        _MAX_WARMUP_KNN = 64
        if self.warmup_knn and self.warmup_knn > min(self.n_keys, _MAX_WARMUP_KNN):
            raise ValueError(
                f"warmup_knn={self.warmup_knn} is too large: the lookup builds a "
                f"knn**2 candidate tensor, so values above {_MAX_WARMUP_KNN} "
                f"overflow topk. Use warmup_noise for a cheap warm-up instead."
            )
        self.warmup_noise = float(warmup_noise or 0.0)
        # persistent=False: a step counter must not change the checkpoint schema.
        self.register_buffer(
            "_pkm_step", torch.zeros((), dtype=torch.long), persistent=False
        )

        # explore/exploit training (set by the owning model, see set_explore)
        self._explore = False
        self._explore_forward = "dense"
        self._explore_chunk = 1024
        # slot-utilisation tracking over exploit passes (see enable_util_tracking)
        self._track_util = False
        # routing balance loss (see set_balance); the model reads self.balance_loss
        self._balance = False
        self._balance_token_entropy = 1.0
        self._balance_kind = "mi"
        self.balance_loss = None
        self.balance_stats = {}

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

    # ---- explore / exploit training ------------------------------------------
    def set_explore(
        self, active: bool, forward: str = "dense", chunk: int = 1024
    ) -> None:
        """Make the following TRAINING forward passes read all slots (``active``)
        or the usual top-``knn`` slots. ``forward`` is ``dense`` or
        ``straight_through`` (see the class docstring); ``chunk`` bounds the
        number of queries materialised at once by the dense read."""
        forward = str(forward).strip().lower()
        if forward not in ("dense", "straight_through"):
            raise ValueError(
                f"explore forward must be dense|straight_through, got {forward!r}"
            )
        self._explore = bool(active)
        self._explore_forward = forward
        self._explore_chunk = max(1, int(chunk))

    def enable_util_tracking(self) -> None:
        """Record which slots are read on exploit (sparse) training passes."""
        if getattr(self, "_util_seen", None) is None:
            self.register_buffer(
                "_util_seen",
                torch.zeros(self.size, dtype=torch.bool, device=self.keys.device),
                persistent=False,
            )
        self._track_util = True

    def pop_util_seen(self) -> torch.Tensor:
        """Return the bool mask of slots read since the last call, and reset it."""
        seen = self._util_seen.clone()
        self._util_seen.zero_()
        return seen

    # ---- routing balance (anti-collapse) -------------------------------------
    def set_balance(
        self, active: bool, token_entropy_weight: float = 1.0, kind: str = "mi"
    ) -> None:
        """Compute a routing balance loss on the following TRAINING passes.

        ``kind="switch"`` (Switch-Transformer load balancing on the HARD routing):
        per head and sub-key half, ``L = n_keys * sum_i f_i * P_i`` where ``f_i``
        is the fraction of top-``knn`` sub-key assignments that go to sub-key
        ``i`` (counted, no gradient) and ``P_i`` the batch-mean softmax
        probability of ``i``. ``L = 1`` under perfectly uniform routing and grows
        as tokens pile onto the same sub-keys; its gradient lowers the scores of
        over-used sub-keys. ``P`` is computed on PER-TOKEN STANDARDISED scores:
        on raw scores the router could reach the minimum by shrinking every
        score (flat ``P``) while the top-k assignment stays collapsed.

        ``kind="mi"``: per head and sub-key half, with ``p_b = softmax(s_b)`` over the
        ``n_keys`` sub-keys for token ``b`` and ``m = mean_b p_b``::

            L = log(n_keys) - H(m) + w * mean_b H(p_b)

        With ``w = 1`` this is ``log(n_keys) - I(token; sub-key)``: the batch
        must spread over all sub-keys (high ``H(m)``) while each token routes
        sharply (low ``H(p_b)``), i.e. routing has to DEPEND on the token. The
        marginal term alone (``w = 0``) is satisfiable by making every token's
        routing flat and identical, which leaves top-k picking the same slots,
        the collapse itself. Stored as ``self.balance_loss`` (mean over heads
        and halves) for the owning model to add to the training loss.
        """
        kind = str(kind).strip().lower()
        if kind not in ("mi", "switch"):
            raise ValueError(f"balance kind must be mi|switch, got {kind!r}")
        self._balance = bool(active)
        self._balance_token_entropy = float(token_entropy_weight)
        self._balance_kind = kind

    def _balance_term(self, scores: torch.Tensor) -> torch.Tensor:
        if self._balance_kind == "switch":
            return self._switch_balance_term(scores)
        logp = F.log_softmax(scores.float(), dim=-1)  # (bs, n_keys)
        p = logp.exp()
        m = p.mean(0)
        h_marg = -(m * torch.log(m.clamp_min(1e-12))).sum()
        h_tok = -(p * logp).sum(-1).mean()
        self._bal_h_marg.append(h_marg.detach())
        self._bal_h_tok.append(h_tok.detach())
        return math.log(scores.size(-1)) - h_marg + self._balance_token_entropy * h_tok

    def _switch_balance_term(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        bs, n = scores.shape
        with torch.no_grad():
            top = scores.topk(min(self._effective_knn(), n), dim=-1).indices
            f = torch.zeros(n, device=scores.device).scatter_add_(
                0, top.reshape(-1), torch.ones(top.numel(), device=scores.device)
            )
            f = f / top.numel()
        z = (scores - scores.mean(-1, keepdim=True)) / scores.std(-1, keepdim=True).clamp_min(1e-6)
        P = F.softmax(z, dim=-1).mean(0)
        with torch.no_grad():
            self._bal_h_marg.append(-(P * torch.log(P.clamp_min(1e-12))).sum())
            # "token" slot reused for the HARD assignment entropy H(f)
            self._bal_h_tok.append(-(f * torch.log(f.clamp_min(1e-12))).sum())
        return n * (f * P).sum()

    def _dense_read_chunk(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Dense read for one chunk: ``query`` is ``(n, heads, k_dim)``."""
        half = self.k_dim // 2
        n_keys, v_dim = self.n_keys, self.v_dim
        # V[i, j] is slot i * n_keys + j (same layout as _get_indices); arrange it
        # as (j, i * v_dim) so that sum_j p2_j V[i, j] is a single GEMM.
        v_t = values.view(n_keys, n_keys, v_dim).transpose(0, 1).reshape(n_keys, n_keys * v_dim)
        out = None
        for h in range(self.heads):
            p1 = F.softmax(F.linear(query[:, h, :half], self.keys[h, 0]).float(), dim=-1)
            p2 = F.softmax(F.linear(query[:, h, half:], self.keys[h, 1]).float(), dim=-1)
            t = (p2.type_as(v_t) @ v_t).view(-1, n_keys, v_dim)  # (n, i, v_dim)
            o = torch.bmm(p1.type_as(t).unsqueeze(1), t).squeeze(1)  # (n, v_dim)
            out = o if out is None else out + o
        return out

    def _dense_read(self, query: torch.Tensor) -> torch.Tensor:
        """Exact read over ALL ``n_keys ** 2`` slots (softmax over every pair).

        Because the product-key score is ``s1_i + s2_j``, the full softmax is
        ``softmax(s1)_i * softmax(s2)_j``, so the read is ``p1^T V p2`` per head
        and costs ``O(n_keys**2 * v_dim)`` per query -- no ``n_keys**2`` score
        tensor is built. Queries are processed in chunks under activation
        checkpointing, so the ``(chunk, n_keys, v_dim)`` intermediate is never
        stored for backward.
        """
        query = query.view(-1, self.heads, self.k_dim)
        values = self.values.weight
        outs = []
        for start in range(0, query.size(0), self._explore_chunk):
            q = query[start : start + self._explore_chunk]
            outs.append(
                torch.utils.checkpoint.checkpoint(
                    self._dense_read_chunk, q, values, use_reentrant=False
                )
            )
        return torch.cat(outs, 0)

    def _in_warmup(self) -> bool:
        return bool(
            self.training
            and self.warmup_steps
            and int(self._pkm_step.item()) < self.warmup_steps
        )

    def _effective_knn(self) -> int:
        """``warmup_knn`` while training inside the warm-up window, else ``knn``."""
        if self.warmup_knn and self._in_warmup():
            return self.warmup_knn
        return self.knn

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
        knn = self._effective_knn()
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]  # (bs, half)
        q2 = query[:, half:]  # (bs, half)

        # compute indices with associated scores
        scores1 = F.linear(q1, subkeys[0], bias=None)  # (bs, n_keys)
        scores2 = F.linear(q2, subkeys[1], bias=None)  # (bs, n_keys)

        if self._balance and self.training:
            # before any warm-up noise: the loss acts on the router's own scores
            self._bal_terms.append(self._balance_term(scores1))
            self._bal_terms.append(self._balance_term(scores2))

        if self.warmup_noise and self._in_warmup():
            # Noisy top-k gating (Shazeer et al.): perturb the sub-key scores so
            # selection VARIES across steps. Over the warm-up window this spreads
            # gradient over many slots, making slot CONTENTS meaningful while the
            # router can still move -- without the knn**2 blow-up of raising knn.
            # Scaled by each half's own score std so it is dimensionless.
            for _s in (scores1, scores2):
                _s.add_(
                    torch.randn_like(_s)
                    * (self.warmup_noise * _s.detach().std().clamp_min(1e-6))
                )
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
        knn = self._effective_knn()
        outputs = [self._get_indices(query[:, i], self.keys[i]) for i in range(self.heads)]
        s = torch.cat([s.view(bs, 1, knn) for s, _ in outputs], 1)  # (bs, heads, knn)
        i = torch.cat([i.view(bs, 1, knn) for _, i in outputs], 1)  # (bs, heads, knn)
        return s.view(-1, knn), i.view(-1, knn)

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
        if self.q_norm is not None:
            query = self.q_norm(query)
        query = F.dropout(query, p=self.query_dropout, training=self.training)  # (bs*heads, k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        explore = self._explore and self.training
        if explore and self._explore_forward == "dense":
            self.balance_loss = None  # no top-k routing on this pass
            output = self._dense_read(query)  # (bs, v_dim)
            output = F.dropout(output, p=self.value_dropout, training=self.training)
            return self._reshape_output(output, prefix_shape)

        balance = self._balance and self.training
        if balance:
            self._bal_terms, self._bal_h_marg, self._bal_h_tok = [], [], []

        # retrieve indices and scores
        scores, indices = self.get_indices(query)  # (bs*heads, knn)

        if balance:
            self.balance_loss = torch.stack(self._bal_terms).mean()
            self.balance_stats = {
                # exp(H): effective number of sub-keys used by the batch / per token
                "eff_subkeys_batch": torch.stack(self._bal_h_marg).mean().exp(),
                ("eff_subkeys_assigned" if self._balance_kind == "switch" else "eff_subkeys_token"):
                    torch.stack(self._bal_h_tok).mean().exp(),
            }
            self._bal_terms = self._bal_h_marg = self._bal_h_tok = None
        else:
            self.balance_loss = None
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs*heads, knn)

        # merge heads / knn (since we sum heads)
        _knn = self._effective_knn()
        indices = indices.view(bs, self.heads * _knn)  # (bs, heads*knn)
        scores = scores.view(bs, self.heads * _knn)  # (bs, heads*knn)

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

        if self._track_util and self.training and not explore:
            with torch.no_grad():
                self._util_seen[indices.reshape(-1)] = True

        # weighted sum of values
        output = self.values(indices, per_sample_weights=scores)  # (bs, v_dim)
        if explore:  # straight_through: sparse forward, dense gradient
            dense = self._dense_read(query)
            output = output + (dense - dense.detach())
        output = F.dropout(output, p=self.value_dropout, training=self.training)  # (bs, v_dim)

        if (self.warmup_knn or self.warmup_noise) and self.training:
            self._pkm_step += 1

        return self._reshape_output(output, prefix_shape)

    def _reshape_output(self, output: torch.Tensor, prefix_shape: torch.Size) -> torch.Tensor:
        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))  # (..., v_dim)
        elif len(prefix_shape) == 0:
            output = output.view(self.v_dim)
        return output


class PKMExploreController:
    """Per-optimizer-step explore / exploit draw for all PKM layers of a model.

    With probability ``p(step)`` a training step is an *explore* step: every
    :class:`HashingMemory` reads (and so updates) ALL slots; otherwise it is a
    normal sparse top-``knn`` *exploit* step. The draw is seeded by
    ``(seed, global_step)``, so it is identical across DDP ranks, across the
    PKM layers, and across gradient-accumulation micro-batches.

    Config (``model.pkm_explore``)::

        p: 0.1                  # explore probability (start value)
        schedule: constant      # constant | linear | cosine | adaptive
        p_final: 0.0            # linear/cosine: end value; adaptive: floor
        start_step: 0           # linear/cosine: decay starts here
        decay_steps: 10000      # linear/cosine: ... and ends this many steps later
        util_target: 0.5        # adaptive: p reaches p_final once the fraction of
                                #   slots read in a window of exploit steps hits this
        util_window: 200        # steps per utilisation window (also logging)
        forward: dense          # dense | straight_through (see HashingMemory)
        chunk: 1024             # dense-read chunk size (queries)
        seed: 0

    ``p: 0`` with a ``pkm_explore`` block still tracks and logs utilisation, which
    makes a matched no-exploration baseline arm.

    Adaptive rule: ``p = p_final + (p - p_final) * max(0, 1 - util / util_target)``
    where ``util`` is the mean (over PKM layers) fraction of slots read during the
    last window of exploit steps. Until the first window completes, ``p`` is used.
    The adaptive state is not checkpointed; after a resume it re-estimates
    ``util`` within one window.
    """

    _SCHEDULES = ("constant", "linear", "cosine", "adaptive")

    def __init__(
        self,
        p: float = 0.1,
        schedule: str = "constant",
        p_final: float = 0.0,
        start_step: int = 0,
        decay_steps: int = 10000,
        util_target: float = 0.5,
        util_window: int = 200,
        forward: str = "dense",
        chunk: int = 1024,
        seed: int = 0,
    ) -> None:
        self.p0 = float(p)
        self.schedule = str(schedule).strip().lower()
        if self.schedule not in self._SCHEDULES:
            raise ValueError(
                f"pkm_explore.schedule must be one of {self._SCHEDULES}, got {schedule!r}"
            )
        if not (0.0 <= self.p0 <= 1.0 and 0.0 <= float(p_final) <= 1.0):
            raise ValueError("pkm_explore.p and p_final must lie in [0, 1]")
        self.p_final = float(p_final)
        self.start_step = int(start_step)
        self.decay_steps = max(1, int(decay_steps))
        self.util_target = float(util_target)
        self.util_window = max(1, int(util_window))
        self.forward = str(forward).strip().lower()
        self.chunk = int(chunk)
        self.seed = int(seed)
        self.util: float | None = None  # last measured mean utilisation

    def prob(self, step: int) -> float:
        if self.schedule == "constant":
            return self.p0
        if self.schedule == "adaptive":
            if self.util is None:
                return self.p0
            frac = max(0.0, 1.0 - self.util / self.util_target)
            return self.p_final + (self.p0 - self.p_final) * frac
        t = min(1.0, max(0.0, (step - self.start_step) / self.decay_steps))
        if self.schedule == "cosine":
            t = 0.5 * (1.0 - math.cos(math.pi * t))
        return self.p0 + (self.p_final - self.p0) * t

    def draw(self, step: int) -> tuple[bool, float]:
        """Return ``(explore, p)`` for optimizer step ``step``."""
        p = self.prob(step)
        if p <= 0.0:
            return False, p
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(step))
        return bool(torch.rand((), generator=g).item() < p), p
