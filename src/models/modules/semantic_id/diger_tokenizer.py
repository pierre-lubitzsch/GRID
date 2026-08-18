"""DIGER's differentiable semantic-ID tokenizer (arXiv:2601.19711).

TIGER freezes the semantic IDs after the quantizer is trained for content
reconstruction, so no recommendation gradient ever reaches them. DIGER makes the
indexing step differentiable and trains the tokenizer jointly with the
recommender. Three pieces:

**DRIL** -- differentiable residual indexing with exploratory learning. Per item
``v`` and level ``j`` the residual ``r_{v,j}`` is scored against the codebook,

    l_{v,j,i} = sim(r_{v,j}, e_i)

Gumbel noise is added, the FORWARD pass takes a hard code

    c_{v,j} = argmax_i (l_{v,j,i} + g_{v,j,i})

and the BACKWARD pass flows through the Gumbel-Softmax mixture

    y~_{v,j,i} = softmax_i( (l_{v,j,i} + g_{v,j,i}) / tau )
    e-_{v,j}   = sum_i y~_{v,j,i} e_i

i.e. a straight-through estimator: discrete codes downstream, dense gradients
upstream.

**FrqUD** -- frequency-based uncertainty decay. Exploration is only useful where
the codebook is congested, so Gumbel noise is applied ONLY to *hot* codes,

    f_i^(e) = beta f_i^(e-1) + (1 - beta) f^_i^(e)
    I_high^(e) = { i : f_i^(e) > r / K }

and low-frequency codes get a deterministic (noise-free) assignment. Without
this, noise on a rarely used code just destabilises an assignment that was
already fine.

**SDUD** -- standard-deviation uncertainty decay, an uncertainty weighting on the
generation loss (see :func:`sdud_loss`).

This module is deliberately free of any T5 / Lightning import so it can be unit
tested on its own; :mod:`diger_generation_model` wires it into the recommender.

A NOTE ON WHICH EMBEDDINGS ARE MIXED. ``forward`` returns the assignment
probabilities, not a token embedding. The soft mixture the RECOMMENDER consumes
is over its own SID token-embedding rows (``item_sid_embedding_table_*``, width
``d_model``), not over the quantizer codebook (width ``latent_dim``) -- those are
different spaces. Mixing the recommender's rows with these probabilities is what
puts the codebook and the encoder on the generation loss's gradient path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn

SIMILARITIES = ("neg_l2", "cosine")


@dataclass
class TokenizerOutput:
    """One forward pass of the tokenizer."""

    codes: torch.Tensor          # [B, L] long, the HARD assignment
    probs: torch.Tensor          # [B, L, K] Gumbel-Softmax probabilities
    latent: torch.Tensor         # [B, D] encoder output z_v
    quantized: torch.Tensor      # [B, D] sum of selected codewords (straight-through)
    vq_loss: torch.Tensor        # scalar
    recon_loss: torch.Tensor     # scalar


def sdud_loss(
    l_gen: torch.Tensor,
    lam: float = 1.4,
    sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard-deviation uncertainty decay.

        L_sigma = L_gen / (2 (sigma + lam)^2) + log(sigma + lam)

    The paper gives the minimiser in closed form,
    ``sigma* = max(0, sqrt(L_gen) - lam)``, so by default sigma is set to it
    rather than learned as a free parameter -- one fewer thing to tune, and it is
    exactly optimal at every step.

    ``sigma*`` is computed under ``no_grad`` on purpose. It is the *solution* of
    the inner minimisation, so differentiating through it would add a spurious
    term that the envelope theorem says contributes nothing at the optimum.

    Note the effect is a self-annealing weight on L_gen: while the model fits
    badly (large L_gen) sigma* grows and the generation term is DOWN-weighted,
    which is what keeps early, high-variance recommendation gradients from
    wrecking the codebook before it has stabilised.
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0 (it is a floor on sigma+lam), got {lam}")
    if sigma is None:
        with torch.no_grad():
            sigma = torch.clamp(l_gen.detach().sqrt() - lam, min=0.0)
    denom = sigma + lam
    return l_gen / (2.0 * denom.pow(2)) + torch.log(denom)


class DigerTokenizer(nn.Module):
    """Residual-quantization tokenizer with a differentiable index.

    Parameters
    ----------
    content_dim
        Width of the frozen item content embeddings (2048 for the flan-T5-XL
        features this repo uses).
    n_levels
        Number of RQ levels the tokenizer produces. This is the SID length BEFORE
        the trailing dedup digit, i.e. ``num_hierarchies - 1`` -- the same
        convention as ``RKMEANS_HIER``.
    codebook_size
        Codewords per level (K=256 here, matching the model's ``vocab_size``).
    latent_dim
        Width of the quantizer's latent space.
    tau
        Gumbel-Softmax temperature. 2.0 in the paper.
    frq_decay_ratio
        ``r`` in ``I_high = {i : f_i > r/K}``. 1.0 makes "hot" mean "used more
        than uniformly"; larger values restrict exploration to the worst
        offenders.
    ema_beta
        ``beta`` in the usage-frequency EMA (0.25 in the paper).
    commitment_weight
        ``beta`` of the standard VQ commitment term.
    similarity
        ``neg_l2`` (default, the RQ-VAE convention) or ``cosine``.
    """

    def __init__(
        self,
        *,
        content_dim: int,
        n_levels: int,
        codebook_size: int,
        latent_dim: int = 64,
        encoder_hidden: Sequence[int] = (768, 256, 128),
        tau: float = 2.0,
        frq_decay_ratio: float = 2.0,
        ema_beta: float = 0.25,
        commitment_weight: float = 0.25,
        similarity: str = "neg_l2",
        use_gumbel_noise: bool = True,
        normalize_inputs: bool = True,
        normalize_residuals: bool = False,
        input_norm: str = "batchnorm",
    ) -> None:
        super().__init__()
        if similarity not in SIMILARITIES:
            raise ValueError(
                f"similarity must be one of {SIMILARITIES}, got {similarity!r}"
            )
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.content_dim = int(content_dim)
        self.n_levels = int(n_levels)
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
        self.tau = float(tau)
        self.frq_decay_ratio = float(frq_decay_ratio)
        self.ema_beta = float(ema_beta)
        self.commitment_weight = float(commitment_weight)
        self.similarity = similarity
        # The paper's sharpest ablation: with NO exploration noise B-Shop R@10
        # falls 0.0683 -> 0.0283. Kept as an explicit flag so that ablation is
        # runnable rather than encoded as a magic frq_decay_ratio.
        self.use_gumbel_noise = bool(use_gumbel_noise)
        # Match THIS REPO's RQ-KMeans, which sets normalize_inputs /
        # normalize_residuals True by default and whose recursion
        #   r^1 = normalize(z);  r^(l+1) = normalize(r^l - c_{s^l})
        # is the one verified to reproduce the stored semantic ids exactly.
        #
        # Not a stylistic choice. The flan-T5 content features carry a large
        # common component, so the unnormalized latents sit in a ball of radius
        # ~2.1e-3. Adam's step size is ~lr regardless of gradient scale, so at
        # the paper's tokenizer lr of 1e-4 FIVE steps move the codebook 4.3e-4 --
        # a fifth of the entire space -- and the assignment collapses (measured:
        # 317 items needing a dedup digit >= 256 after 5 steps). On the unit
        # sphere the same lr is a sane step.
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_residuals = bool(normalize_residuals)

        # Standardize the content features BEFORE the encoder -- this is the
        # repo's own RQ-VAE front (BatchNorm1d + NormalizeLayer,
        # rqvae_train_flat.yaml `normalization_layer`) and it is load-bearing.
        #
        # Measured on beauty: the raw flan-T5 features have per-dim std 1.0e-02
        # against a large shared mean, and pushing them through ANY deep MLP
        # collapses the catalog to mean pairwise cosine 0.99995 -- my encoder and
        # the repo's encoder both. After BatchNorm the same encoders give 0.820 /
        # 0.871, in line with the raw features' own 0.863. Without this the
        # codebook is fitted on noise and dies within ~25 joint-training steps.
        if input_norm not in ("standardize", "batchnorm", "none"):
            raise ValueError(
                "input_norm must be 'standardize', 'batchnorm' or 'none', got "
                f"{input_norm!r}"
            )
        self.input_norm_kind = input_norm
        # `standardize` (default) is a FIXED per-dimension (x - mean) / std fitted
        # once from the frozen catalog by fit_input_norm(). Preferred over the
        # repo RQ-VAE's BatchNorm1d because the content features never change, so
        # the statistics are a constant -- and BatchNorm would introduce a
        # train/eval skew that breaks committing: `assign()` runs under eval(),
        # where BatchNorm uses RUNNING stats, which start at mean 0 / var 1 and
        # so act as an identity until enough training-mode batches have gone
        # through. The ids committed early would then be fitted on collapsed
        # latents, silently.
        self.register_buffer("_in_mean", torch.zeros(self.content_dim))
        self.register_buffer("_in_std", torch.ones(self.content_dim))
        self.register_buffer("_in_fitted", torch.zeros((), dtype=torch.long))
        self.input_norm = (
            nn.BatchNorm1d(self.content_dim) if input_norm == "batchnorm"
            else nn.Identity()
        )
        self.encoder = _mlp(self.content_dim, encoder_hidden, self.latent_dim)
        self.decoder = _mlp(self.latent_dim, tuple(reversed(tuple(encoder_hidden))),
                            self.content_dim)
        # [L, K, D]. Initialised small and spread out; on_train_start in the
        # model may re-seed these from k-means, which is what the RQ-VAE
        # literature does and what stops a large fraction of codes going dead in
        # the first epoch.
        self.codebooks = nn.Parameter(
            torch.randn(self.n_levels, self.codebook_size, self.latent_dim) * 0.02
        )
        # EMA usage frequency per level, used by FrqUD. A BUFFER, not a
        # parameter: it is optimizer state of the exploration schedule, and it
        # must ride along with the checkpoint or a resumed run explores the wrong
        # codes. Initialised uniform so nothing is "hot" before any data is seen.
        self.register_buffer(
            "code_freq",
            torch.full((self.n_levels, self.codebook_size), 1.0 / self.codebook_size),
        )
        # Per-epoch accumulator of raw counts; folded into code_freq by
        # :meth:`step_frequency_ema`.
        self.register_buffer(
            "_code_counts", torch.zeros(self.n_levels, self.codebook_size)
        )

    @torch.no_grad()
    def fit_input_norm(self, content: torch.Tensor, eps: float = 1e-6) -> None:
        """Fit the fixed standardization from the frozen catalog. Idempotent."""
        if self.input_norm_kind != "standardize":
            return
        self._in_mean.copy_(content.mean(dim=0))
        self._in_std.copy_(content.std(dim=0).clamp_min(eps))
        self._in_fitted.fill_(1)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm_kind == "standardize":
            if not bool(self._in_fitted):
                raise RuntimeError(
                    "input_norm='standardize' but fit_input_norm() was never "
                    "called; the encoder would see raw features and collapse the "
                    "catalog (mean pairwise cosine 0.99995 measured on beauty)."
                )
            return (x - self._in_mean) / self._in_std
        return self.input_norm(x)

    # ---------------------------------------------------------------- scoring
    def assignment_logits(self, residual: torch.Tensor, level: int) -> torch.Tensor:
        """``l_{v,j,i} = sim(r_{v,j}, e_i)`` for one level. ``[B, D] -> [B, K]``."""
        codebook = self.codebooks[level]                      # [K, D]
        if self.similarity == "cosine":
            r = nn.functional.normalize(residual, dim=-1)
            e = nn.functional.normalize(codebook, dim=-1)
            return r @ e.t()
        # neg_l2: -||r - e||^2, expanded so it stays a single matmul.
        r2 = residual.pow(2).sum(-1, keepdim=True)            # [B, 1]
        e2 = codebook.pow(2).sum(-1).unsqueeze(0)             # [1, K]
        return -(r2 + e2 - 2.0 * residual @ codebook.t())     # [B, K]

    def _exploration_noise(self, logits: torch.Tensor, level: int) -> torch.Tensor:
        """FrqUD: Gumbel noise on HOT codes only, zeros elsewhere.

        Sampled as ``-log(-log(u))`` with u strictly inside (0, 1) -- clamping
        the uniform away from both ends, because ``u=0`` gives +inf and ``u=1``
        gives -inf, and a single inf here poisons the whole softmax row.
        """
        if not self.training or not self.use_gumbel_noise:
            return torch.zeros_like(logits)
        hot = self.code_freq[level] > (self.frq_decay_ratio / self.codebook_size)
        if not bool(hot.any()):
            return torch.zeros_like(logits)
        u = torch.rand_like(logits).clamp_(min=1e-9, max=1.0 - 1e-7)
        g = -torch.log(-torch.log(u))
        return g * hot.to(logits.dtype).unsqueeze(0)

    # ---------------------------------------------------------------- forward
    def forward(self, content: torch.Tensor) -> TokenizerOutput:
        """Tokenize a batch of item content embeddings. ``[B, content_dim]``."""
        if content.dim() != 2 or content.shape[-1] != self.content_dim:
            raise ValueError(
                f"content must be [B, {self.content_dim}], got {tuple(content.shape)}"
            )
        content_n = self._normalize_input(content)                   # [B, C]
        z = self.encoder(content_n)                            # [B, D]
        residual = z
        codes: List[torch.Tensor] = []
        probs: List[torch.Tensor] = []
        quantized = torch.zeros_like(z)
        vq = z.new_zeros(())
        for level in range(self.n_levels):
            logits = self.assignment_logits(residual, level)   # [B, K]
            noisy = logits + self._exploration_noise(logits, level)
            y = torch.softmax(noisy / self.tau, dim=-1)        # [B, K]
            hard = noisy.argmax(dim=-1)                        # [B]
            codebook = self.codebooks[level]                   # [K, D]
            e_hard = codebook[hard]                            # [B, D]
            e_soft = y @ codebook                              # [B, D]
            # Straight-through: forward value is the HARD codeword, gradient is
            # the soft mixture's. The ORDER matters -- `e_soft - e_soft.detach()`
            # is exactly 0.0 in floating point (x - x never rounds), so the
            # forward tensor is bit-identical to e_hard. The other arrangement,
            # `e_soft + (e_hard - e_soft).detach()`, is only approximately e_hard
            # and leaks quantization error into the residual recursion, where it
            # compounds level over level.
            # e_hard is detached here so the codebook's gradient for the SELECTED
            # codeword comes from the VQ term below (the RQ-VAE convention)
            # rather than from two paths at once.
            e_st = e_hard.detach() + (e_soft - e_soft.detach())
            # Standard VQ terms, on the hard selection (the ST path already
            # carries the recommender's gradient into the codebook).
            # MEAN over elements, matching the repo's RQ-VAE
            # (rqvae_train_flat.yaml uses MSELoss(reduction='mean') for both the
            # quantization and reconstruction terms). Summing over the feature
            # dimension instead would scale these by 2048 / latent_dim and let
            # them swamp L_gen ~50:1 at lambda=1.0 -- which would quietly reduce
            # DIGER to a plain RQ-VAE that ignores the recommender.
            vq = vq + (
                (residual.detach() - e_hard).pow(2).mean()
                + self.commitment_weight
                * (residual - e_hard.detach()).pow(2).mean()
            )
            quantized = quantized + e_st
            residual = residual - e_st
            if self.normalize_residuals:
                residual = nn.functional.normalize(residual, dim=-1)
            codes.append(hard)
            probs.append(y)
            if self.training:
                with torch.no_grad():
                    self._code_counts[level] += torch.bincount(
                        hard.detach().flatten(), minlength=self.codebook_size
                    ).to(self._code_counts.dtype)
        # Reconstruct the STANDARDIZED features: the decoder mirrors an encoder
        # that now sees BatchNorm'd input, so `content` is the wrong target and
        # would make L_recon dominated by the shared mean the front removed.
        recon = self.decoder(quantized)
        recon_loss = (recon - content_n).pow(2).mean()
        return TokenizerOutput(
            codes=torch.stack(codes, dim=1),
            probs=torch.stack(probs, dim=1),
            latent=z,
            quantized=quantized,
            vq_loss=vq,
            recon_loss=recon_loss,
        )

    # ------------------------------------------------------------------- init
    def pretrain_autoencoder(
        self,
        content: torch.Tensor,
        steps: int = 2000,
        lr: float = 1e-3,
        batch_size: int = 1024,
        seed: int = 0,
    ) -> float:
        """Fit encoder+decoder as a plain autoencoder before any quantization.

        THIS IS REQUIRED, and it is the step that makes DIGER trainable at all
        on these features. A randomly initialised deep MLP destroys the item
        structure: measured on beauty, the raw content embeddings have mean
        pairwise cosine 0.8617, but after a random 2048-512-256-128-32 encoder
        the latents sit at mean pairwise cosine 0.99995 (min 0.9995) -- every one
        of the 12,101 items collapsed onto a single point. k-means then partitions
        pure noise, which looks healthy (11,938 distinct prefixes) but is erased
        by the first gradient step, and the dedup digit overflows immediately.

        Conceptually this is the two-stage tokenizer's stage 1. DIGER's
        contribution is removing the FREEZE that normally follows it, not
        removing stage 1 -- the paper starts from a trained RQ-VAE too.

        Returns the final reconstruction loss.
        """
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=float(lr))
        n = content.size(0)
        loss = content.new_zeros(())
        for _ in range(int(steps)):
            idx = torch.randint(0, n, (min(batch_size, n),), generator=g)
            x = content[idx.to(content.device)]
            xn = self._normalize_input(x)
            z = self.encoder(xn)
            if self.normalize_inputs:
                z = nn.functional.normalize(z, dim=-1)
            # Reconstruct the NORMALISED target: the decoder mirrors the encoder,
            # which now sees standardized input.
            loss = (self.decoder(z) - xn).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach())

    @torch.no_grad()
    def latent_dispersion(self, content: torch.Tensor, sample: int = 2000) -> float:
        """Mean pairwise cosine of the (normalised) latents. Lower is healthier.

        A value above ~0.99 means the encoder has collapsed the catalog onto one
        point and any codebook fit on top of it is partitioning noise.
        """
        x = content[: int(sample)]
        z = nn.functional.normalize(self.encoder(self._normalize_input(x)), dim=-1)
        sim = z @ z.t()
        off = sim[~torch.eye(z.size(0), dtype=torch.bool, device=z.device)]
        return float(off.mean())

    @torch.no_grad()
    def init_codebooks_from_content(
        self,
        content: torch.Tensor,
        n_iter: int = 15,
        max_points: int = 50_000,
        seed: int = 0,
    ) -> None:
        """Seed each level's codebook by k-means on that level's residuals.

        NOT optional in practice. From a random codebook the assignment is
        degenerate -- measured on beauty, 11,845 of 12,101 items land in a single
        ``c_0..c_2`` bucket, which both starves the recommender of any signal
        from the identifier and makes the dedup digit overflow the vocabulary.
        This is the standard RQ-VAE / RQ-KMeans initialisation and it is what
        makes the first epoch trainable at all.

        Run AFTER the encoder exists but before training. The encoder is still
        random here, so this spreads codes over the actual latent distribution
        rather than over nothing; the codebook keeps training from there.
        """
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = content
        if x.size(0) > max_points:
            idx = torch.randperm(x.size(0), generator=g)[:max_points]
            x = x[idx.to(x.device)]
        residual = self.encoder(self._normalize_input(x))
        if self.normalize_inputs:
            residual = nn.functional.normalize(residual, dim=-1)
        for level in range(self.n_levels):
            centroids = _kmeans(residual, self.codebook_size, n_iter=n_iter, gen=g)
            self.codebooks[level].copy_(centroids)
            assign = torch.cdist(residual, centroids).argmin(dim=-1)
            residual = residual - centroids[assign]
            # The seeds must be fit in the SAME space the forward pass scores in,
            # so the recursion here has to match forward()/assign() exactly.
            if self.normalize_residuals:
                residual = nn.functional.normalize(residual, dim=-1)
        # A fresh codebook has no usage history; start everything cold so FrqUD
        # does not explore on the strength of a stale EMA.
        self.code_freq.fill_(1.0 / self.codebook_size)
        self._code_counts.zero_()

    # -------------------------------------------------------------- frequency
    @torch.no_grad()
    def step_frequency_ema(self) -> None:
        """Fold this epoch's raw counts into the EMA and reset the accumulator.

        Call once per epoch (``on_train_epoch_end``). The paper's ``f^_i^(e)`` is
        an epoch-level usage rate, so folding per step would make the EMA track
        batch noise and the hot set would flicker.
        """
        total = self._code_counts.sum(dim=-1, keepdim=True)
        # A level with no observations leaves its EMA untouched rather than
        # collapsing it to zero -- otherwise an eval-only epoch would mark every
        # code cold and silently switch exploration off.
        seen = (total > 0).squeeze(-1)
        if bool(seen.any()):
            rate = self._code_counts[seen] / total[seen]
            self.code_freq[seen] = (
                self.ema_beta * self.code_freq[seen] + (1.0 - self.ema_beta) * rate
            )
        self._code_counts.zero_()

    @torch.no_grad()
    def hot_code_fraction(self) -> torch.Tensor:
        """Per-level share of codes currently eligible for exploration. ``[L]``."""
        return (
            self.code_freq > (self.frq_decay_ratio / self.codebook_size)
        ).to(torch.float32).mean(dim=-1)

    @torch.no_grad()
    def assign(self, content: torch.Tensor) -> torch.Tensor:
        """Deterministic (noise-free) hard codes. ``[B, content_dim] -> [B, L]``.

        This is the assignment used to COMMIT semantic IDs -- for evaluation, for
        the SID tensor written back to disk, and for the dedup digit. Never uses
        Gumbel noise regardless of ``self.training``, so committing is
        reproducible.
        """
        was_training = self.training
        self.eval()
        try:
            z = self.encoder(self._normalize_input(content))
            if self.normalize_inputs:
                z = nn.functional.normalize(z, dim=-1)
            residual = z
            out: List[torch.Tensor] = []
            for level in range(self.n_levels):
                hard = self.assignment_logits(residual, level).argmax(dim=-1)
                out.append(hard)
                residual = residual - self.codebooks[level][hard]
                if self.normalize_residuals:
                    residual = nn.functional.normalize(residual, dim=-1)
            return torch.stack(out, dim=1)
        finally:
            if was_training:
                self.train()


@torch.no_grad()
def _kmeans(
    x: torch.Tensor, k: int, n_iter: int = 15, gen: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Lloyd's algorithm, centroids seeded from distinct data points. ``[N, D] -> [k, D]``.

    Empty clusters are re-seeded to the points furthest from their assigned
    centroid rather than left in place: a dead code stays dead forever otherwise,
    and dead codes are precisely what shrinks the usable identifier space.
    """
    n = x.size(0)
    if n == 0:
        raise ValueError("cannot run k-means on an empty tensor")
    if n <= k:
        # Fewer points than codes: use them all and pad by repetition, so every
        # row is at least a real point rather than an arbitrary vector.
        reps = (k + n - 1) // n
        return x.repeat(reps, 1)[:k].clone()
    perm = torch.randperm(n, generator=gen)[:k].to(x.device)
    centroids = x[perm].clone()
    for _ in range(int(n_iter)):
        d = torch.cdist(x, centroids)                     # [N, k]
        assign = d.argmin(dim=-1)
        for j in range(k):
            sel = assign == j
            if bool(sel.any()):
                centroids[j] = x[sel].mean(dim=0)
            else:
                centroids[j] = x[d.min(dim=-1).values.argmax()]
    return centroids


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = int(in_dim)
    for h in hidden:
        layers += [nn.Linear(prev, int(h)), nn.ReLU()]
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)
