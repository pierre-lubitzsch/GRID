"""Input-side aggregation of an item's semantic-ID *token* embeddings into a
compact item representation.

Motivation ("Longer IDs for finer-grained neighborhoods", Jul 3): with longer RQ
IDs (L in {8, 16}) the TIGER encoder would otherwise see L token positions per
history item, making the encoder input sequence L times longer. These mergers
compress each item's L per-hierarchy token embeddings into fewer encoder input
vectors, keeping the encoder sequence short.

Two strategies are provided (both operate purely on the *input* side; the
decoder still generates the full L-token semantic ID autoregressively):

* :class:`MeanTokenMerger` -- Option 1: mean pooling over the token embeddings
  -> ONE vector per item.
* :class:`AttentiveTokenMerger` -- Option 2: the Attentive Token Merger of
  ACERec (https://arxiv.org/abs/2602.13573):

      Ẽ_i = E_i + P                            (learnable positional embeddings)
      s_i = f_s(E_i)                           (item-level summary; nonlinear proj)
      Q_i = Q  [+ f_q(s_i)]                     (k learnable latents; optionally
                                                 content-adaptive from s_i)
      Z_i = f_out( f_attn(Q_i, Ẽ_i, Ẽ_i) )     (f_out = MLP)
      Z̃_i = [Z_i ; h_i],  h_i = s_i            (optional per-item Intent Token)

  i.e. a compression from ``L`` tokens to ``k`` latents (default k=4), plus an
  optional Intent Token (off by default here -> ``k`` output tokens per item;
  ``k+1`` when the Intent Token is enabled).

All mergers share the contract::

    forward(item_tokens: [B, N, L, d]) -> [B, N, k_out, d]

where ``B`` is batch size, ``N`` the number of items per sequence, ``L`` the
number of ID tokens per item (``num_hierarchies``), ``d`` the embedding dim, and
``k_out`` the number of output tokens per item (1 for mean pooling; ``k`` or
``k+1`` for the attentive merger, depending on the Intent Token).
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class ItemTokenMerger(nn.Module):
    """Base class: merge per-item ID-token embeddings ``[B, N, L, d] -> [B, N, k_out, d]``."""

    def forward(self, item_tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MeanTokenMerger(ItemTokenMerger):
    """Option 1 -- mean pooling over an item's token embeddings.

    Parameter-free: the compact item representation is the arithmetic mean of the
    item's ``L`` per-hierarchy token embeddings (``k_out = 1``). All ``L`` tokens
    of a (non-padded) item are always valid -- padding is applied whole-item -- so
    no intra-item masking is needed; padded items are dropped downstream via the
    encoder attention mask.
    """

    def forward(self, item_tokens: torch.Tensor) -> torch.Tensor:
        # item_tokens: (B, N, L, d) -> (B, N, 1, d)
        return item_tokens.mean(dim=2, keepdim=True)


class SumTokenMerger(ItemTokenMerger):
    """Option 1b -- sum pooling over an item's token embeddings (``k_out = 1``).

    Since padding is applied whole-item, every non-padded item has exactly
    ``L = num_hierarchies`` tokens, so ``sum = L * mean`` EXACTLY -- the two
    differ only by a fixed scalar. That is not a no-op in this architecture:
    T5's ``T5LayerNorm`` is RMS-norm and therefore scale-invariant, but T5 is
    **pre-norm**, so the residual stream carries the un-scaled value. Multiplying
    the encoder inputs by ``L`` thus changes the ratio of raw-embedding signal to
    processed signal along the residual path -- effectively an embedding-scale
    change, which is worth measuring rather than assuming away.

    Shares :class:`MeanTokenMerger`'s weakness: pooling treats the RQ levels as
    exchangeable and discards which level a code came from. Only
    :class:`AttentiveTokenMerger` preserves per-hierarchy identity (via its
    positional embeddings).
    """

    def forward(self, item_tokens: torch.Tensor) -> torch.Tensor:
        # item_tokens: (B, N, L, d) -> (B, N, 1, d)
        return item_tokens.sum(dim=2, keepdim=True)


class AttentiveTokenMerger(ItemTokenMerger):
    """Option 2 -- ACERec Attentive Token Merger (learnable-query cross-attention).

    Compresses each item's ``L`` ID-token embeddings into ``k`` latent tokens
    (``num_query_tokens``) plus an optional per-item Intent Token, following
    https://arxiv.org/abs/2602.13573:

    1. add learnable positional embeddings ``P`` to the token embeddings
       (``Ẽ_i = E_i + P``) to preserve the per-hierarchy subspace identity of the
       RQ digits;
    2. compute an item-level summary ``s_i = f_s(E_i)`` -- a nonlinear projection
       aggregating the item's token embeddings (used to init the Intent Token and,
       optionally, to make the queries content-adaptive);
    3. ``k`` learnable latent query vectors ``Q_i`` (default 4); when
       ``content_adaptive_queries`` is set, they are offset by ``f_q(s_i)``;
    4. multi-head cross-attention of the queries over the item's ``L`` token
       embeddings (keys/values), with a residual on the latents, then an MLP
       ``f_out`` (with residual) producing the ``k`` compact latent tokens
       ``Z_i in R^{k x d}``;
    5. optionally append a per-item Intent Token ``h_i = s_i`` giving
       ``Z̃_i = [Z_i ; h_i] in R^{(k+1) x d}`` (ACERec Sec 2.3.1). In the
       encoder-based TIGER, the Intent Token then "evolves" through the encoder's
       self-attention like any other position.

    Output shape is ``[B, N, k_out, d]`` with ``k_out = k (+1 if intent token)``
    -- each item becomes ``k_out`` encoder positions (a compression from ``L``).

    Parameters
    ----------
    embedding_dim: int
        Token / output embedding dimension ``d``.
    num_tokens: int
        Number of ID tokens per item ``L`` (== ``num_hierarchies``); sizes the
        positional-embedding table ``P in R^{L x d}``.
    num_query_tokens: int
        Number ``k`` of learnable latent queries / latent output tokens per item.
    num_heads: int
        Number of attention heads; must divide ``embedding_dim``.
    dropout: float
        Dropout used in attention and the MLPs.
    mlp_ratio: float
        Hidden width of ``f_out`` / ``f_s`` as a multiple of ``embedding_dim``.
    use_positional_embedding: bool
        Whether to add the learnable positional embeddings ``P`` (paper: yes).
    use_intent_token: bool
        Whether to append the per-item Intent Token ``h_i = s_i`` (ACERec: yes).
        Defaults OFF here: in ACERec the intent token is the prediction anchor
        (its evolved state ``h_pred`` drives the output), but TIGER predicts with a
        separate autoregressive decoder that cross-attends over all encoder
        positions, so it is redundant with the ``k`` latents. Turn on for strict
        ACERec parity.
    content_adaptive_queries: bool
        Whether to make the queries content-adaptive via ``Q_i = Q + f_q(s_i)``.
        The paper does this; its exact form is underspecified, so it defaults off.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_tokens: int,
        num_query_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        use_positional_embedding: bool = True,
        use_intent_token: bool = False,
        content_adaptive_queries: bool = False,
        allow_no_compression: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by num_heads="
                f"{num_heads} for AttentiveTokenMerger"
            )
        if num_query_tokens < 1:
            raise ValueError(f"num_query_tokens must be >= 1, got {num_query_tokens}")
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")

        # The whole point of this module is to emit FEWER vectors per item than the
        # item has ID tokens. k_out >= num_tokens silently turns it into a
        # same-width transform, which reads as "aggregation enabled" in the config
        # and the run label while saving nothing — the failure mode that made
        # `num_query_tokens=4` (the ACERec default, chosen for L=8/16) a no-op at
        # L=4. Refuse it unless the caller is deliberately running that control.
        k_out_check = int(num_query_tokens) + (1 if use_intent_token else 0)
        if k_out_check >= int(num_tokens) and not allow_no_compression:
            _suggest = max(1, int(num_tokens) // 2)
            raise ValueError(
                f"AttentiveTokenMerger would not compress: it emits "
                f"k_out={k_out_check} vector(s) per item "
                f"(num_query_tokens={num_query_tokens}"
                f"{' + 1 intent token' if use_intent_token else ''}) for an item "
                f"of num_tokens={num_tokens} ID tokens, so the encoder sequence "
                f"is not shortened at all"
                f"{' (it grows)' if k_out_check > int(num_tokens) else ''}. "
                f"Lower num_query_tokens (e.g. {_suggest}, giving "
                f"{int(num_tokens) / _suggest:.1f}x compression), or pass "
                f"allow_no_compression=true to run it as an explicit "
                f"transform-only control."
            )

        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.num_query_tokens = num_query_tokens
        self.use_intent_token = use_intent_token
        self.content_adaptive_queries = content_adaptive_queries
        # number of output tokens per item (k latents + optional intent token)
        self.k_out = num_query_tokens + (1 if use_intent_token else 0)

        hidden_dim = int(embedding_dim * mlp_ratio)

        # (2) learnable latent query vectors Q_i in R^{k x d}.
        # SMALL init (std 0.02, BERT-style) is load-bearing: the queries are
        # shared across ALL items and sit on the residual path
        # (latents = queries + attended). With the original randn(0,1) init the
        # shared query term dominated the item-dependent attended term, so
        # early in training every item produced near-identical latents, the
        # decoder learned unconditional code priors while ignoring the encoder,
        # and val recall never left ~0 -> early stopping killed the run
        # (observed on all clean + all L=16 runs, 2026-07-14). At std 0.02 the
        # residual is negligible and latents are item-discriminative from
        # step 0.
        self.query = nn.Parameter(
            0.02 * torch.randn(num_query_tokens, embedding_dim)
        )
        # (1) learnable positional embeddings P in R^{L x d}
        if use_positional_embedding:
            self.pos_embedding: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(num_tokens, embedding_dim)
            )
        else:
            self.register_parameter("pos_embedding", None)

        # item-level summary f_s (nonlinear projection of the token embeddings),
        # needed by the Intent Token and/or the content-adaptive queries.
        self._needs_summary = use_intent_token or content_adaptive_queries
        if self._needs_summary:
            self.summary_norm = nn.LayerNorm(embedding_dim)
            self.summary = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )
        if content_adaptive_queries:
            # f_q: summary -> per-item query offsets (k x d)
            self.query_gen = nn.Linear(embedding_dim, num_query_tokens * embedding_dim)

        # (4) cross-attention (pre-norm, residual on the latents)
        self.q_norm = nn.LayerNorm(embedding_dim)
        self.kv_norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # f_out: an MLP with a residual
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, item_tokens: torch.Tensor) -> torch.Tensor:
        # item_tokens: (B, N, L, d)
        batch, n_items, num_tokens, dim = item_tokens.shape
        if num_tokens != self.num_tokens:
            raise ValueError(
                f"AttentiveTokenMerger built for L={self.num_tokens} tokens per item "
                f"but got L={num_tokens}"
            )
        # fold (B, N) into a single "item" batch dimension for attention
        raw_tokens = item_tokens.reshape(batch * n_items, num_tokens, dim)

        # (2) item-level summary s_i = f_s(E_i) from the raw token embeddings
        summary = None
        if self._needs_summary:
            summary = self.summary(self.summary_norm(raw_tokens.mean(dim=1)))  # (B*N, d)

        # (1) Ẽ_i = E_i + P
        tokens = raw_tokens
        if self.pos_embedding is not None:
            tokens = tokens + self.pos_embedding.unsqueeze(0)

        # (3) queries, optionally content-adaptive via f_q(s_i)
        queries = self.query.unsqueeze(0).expand(batch * n_items, -1, -1)
        if self.content_adaptive_queries:
            queries = queries + self.query_gen(summary).view(
                batch * n_items, self.num_query_tokens, dim
            )

        # (4) cross-attention -> k latents, with residual + MLP f_out (residual)
        keys_values = self.kv_norm(tokens)
        attended, _ = self.attn(
            self.q_norm(queries), keys_values, keys_values, need_weights=False
        )
        latents = queries + attended
        latents = latents + self.ffn(self.ffn_norm(latents))  # (B*N, k, d)

        # (5) append per-item Intent Token h_i = s_i -> (B*N, k+1, d)
        if self.use_intent_token:
            latents = torch.cat([latents, summary.unsqueeze(1)], dim=1)

        return latents.view(batch, n_items, self.k_out, dim)


def _default_num_heads(embedding_dim: int) -> int:
    """Largest of {8,4,2,1} that divides ``embedding_dim`` (a safe head default).

    ``d_model`` is not always divisible by the transformer's own ``num_heads``
    (e.g. 128 is not divisible by 6), so the attentive merger picks its own.
    """
    for candidate in (8, 4, 2, 1):
        if embedding_dim % candidate == 0:
            return candidate
    return 1


def build_item_token_merger(
    spec: Optional[Union[str, Dict[str, Any]]],
    embedding_dim: int,
    num_tokens: int,
) -> Optional[ItemTokenMerger]:
    """Construct an :class:`ItemTokenMerger` from a config ``spec``.

    ``num_tokens`` is the number of ID tokens per item (``num_hierarchies``); it
    sizes the attentive merger's positional-embedding table.

    ``spec`` forms:
      * ``None`` / ``"none"`` / ``"off"``  -> ``None`` (feature disabled; the
        model keeps the default per-token + separator-token encoder input).
      * ``"mean"``                          -> :class:`MeanTokenMerger` (1 vec/item).
      * ``"sum"``                           -> :class:`SumTokenMerger` (1 vec/item;
        ``= L * mean``, which differs only via T5's pre-norm residual path).
      * ``"attentive"``                     -> :class:`AttentiveTokenMerger` with
        positional embeddings on and the intent token off by default (set
        ``intent_token: true`` for strict ACERec parity). ``num_query_tokens``
        defaults to ``min(4, L//2)``, i.e. the historical ACERec k=4 at L=8/16 but
        k=2 at L=4, where a literal 4 would compress nothing. A configuration
        whose ``k_out >= L`` is rejected unless ``allow_no_compression: true``.
      * mapping ``{type: mean|sum|attentive, ...}`` -> the named merger, with the
        remaining keys forwarded to the attentive merger (``num_query_tokens``,
        ``num_heads``, ``dropout``, ``mlp_ratio``, ``positional_embedding``,
        ``intent_token``, ``content_adaptive_queries``).
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        spec_dict: Dict[str, Any] = {"type": spec}
    else:
        # dict / OmegaConf DictConfig
        spec_dict = dict(spec)

    agg_type = str(spec_dict.get("type", "mean")).lower()
    if agg_type in ("none", "off", "null"):
        return None
    if agg_type == "mean":
        return MeanTokenMerger()
    if agg_type == "sum":
        return SumTokenMerger()
    if agg_type in ("attentive", "attention", "merger"):
        # Default k adapts to L instead of being pinned to the ACERec value of 4,
        # which only compresses when L > 4. min(4, L//2) keeps the historical
        # k=4 at L=8 and L=16 (so existing runs/labels are unchanged) while
        # giving L=4 a k=2 that actually halves the encoder sequence.
        default_k = min(4, max(1, int(num_tokens) // 2))
        return AttentiveTokenMerger(
            embedding_dim=embedding_dim,
            num_tokens=int(num_tokens),
            num_query_tokens=int(spec_dict.get("num_query_tokens", default_k)),
            num_heads=int(spec_dict.get("num_heads", _default_num_heads(embedding_dim))),
            dropout=float(spec_dict.get("dropout", 0.0)),
            mlp_ratio=float(spec_dict.get("mlp_ratio", 4.0)),
            use_positional_embedding=bool(
                spec_dict.get("positional_embedding", True)
            ),
            use_intent_token=bool(spec_dict.get("intent_token", False)),
            content_adaptive_queries=bool(
                spec_dict.get("content_adaptive_queries", False)
            ),
            allow_no_compression=bool(
                spec_dict.get("allow_no_compression", False)
            ),
        )
    raise ValueError(
        f"Unknown item_token_aggregation type: {agg_type!r} "
        f"(expected 'mean', 'attentive', or 'none')"
    )
