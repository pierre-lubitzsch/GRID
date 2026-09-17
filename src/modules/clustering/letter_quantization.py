"""LETTER's learnable tokenizer, as a drop-in for GRID's RQ-VAE.

LETTER (arXiv:2405.07314, "Learnable Item Tokenization for Generative
Recommendation") is an RQ-VAE with two extra regularisers and one extra
assignment rule:

1. **Semantic regularisation** -- the plain RQ-VAE reconstruction + commitment
   terms. Unchanged from :class:`ResidualQuantization`; that is the whole point
   of subclassing rather than porting the authors' file.
2. **Collaborative regularisation** (``alpha``) -- an InfoNCE term pulling each
   item's quantised latent towards that item's vector in a sequential CF model
   (the paper uses a 32-d SASRec; here, whatever
   ``scripts/train_cf_embeddings.py`` produced). This is the term that makes
   LETTER's identifiers behaviour-aware rather than purely content-derived.
3. **Diversity regularisation** (``beta``) -- a per-level contrastive term over
   the codebook that counteracts code-assignment bias, plus optional Sinkhorn
   balancing on the final level (see
   :class:`~src.components.quantization_strategies.SinkhornQuantization`).

HOW IT PLUGS INTO GRID
----------------------
The extra terms are folded into the value :meth:`model_step` already returns as
``quantization_loss``, so the inherited ``training_step`` -- including the manual
``training_loop_function`` used for codebook initialisation -- is untouched. With
``quantization_loss_weight = 1.0`` the total is exactly LETTER's

    L = L_recon + L_quant + beta * L_diversity + alpha * L_CF

Validation, test and predict paths are untouched: the extra terms are added only
while fitting. In particular ID ASSIGNMENT is the inherited deterministic
``argmin``, matching LETTER's own ``use_sk=False`` at tokenization time.

ID LAYOUT
---------
This module trains ``num_hierarchies`` real codebooks; GRID's inference pipeline
then appends a dedup digit, exactly as for RQ-VAE / RQ-KMeans. So a LETTER
tensor has the same ``(L, N)`` shape and the same "last position is the dedup
digit" convention as every other GRID identifier space, and every prefix-based
diagnostic (code sharing, neighbourhood sampling, TRACER, the position analysis)
applies unchanged. LETTER's own release instead trains 4 semantic levels and
breaks collisions by resampling; that variant is NOT what this file implements,
because it would silently change what a depth-3 prefix means everywhere.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

import torch
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn

from src.data.loading.components.interfaces import ItemData
from src.modules.clustering.residual_quantization import ResidualQuantization
from src.modules.clustering.vector_quantization import VectorQuantization

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# balanced k-means over the codebook
# --------------------------------------------------------------------------- #
def balanced_kmeans(
    points: torch.Tensor,
    n_clusters: int,
    n_iters: int = 10,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Size-constrained k-means: every cluster gets at most ``ceil(n / k)`` points.

    LETTER calls ``k_means_constrained.KMeansConstrained`` here. That package is
    not installed in this environment (and pulls in ortools), so the constraint
    is solved directly instead: with a per-cluster capacity of ``cap``, replicate
    each cluster ``cap`` times and run a rectangular linear-sum assignment. That
    is the exact optimum of the balanced assignment step, not an approximation of
    it, and it is cheap at the size that matters here (256 codes, 10 clusters).

    Why the size constraint is load-bearing: the diversity loss samples a
    positive from the assigned code's cluster, so a singleton cluster has no
    positive to offer. Capacity ``ceil(n/k)`` forces every cluster to at least
    ``n - (k-1) * cap`` members (22 for the default 256/10).

    Args:
        points: ``[n, d]`` points to cluster (the codebook).
        n_clusters: number of clusters.
        n_iters: Lloyd iterations.
        seed: seeds the initial centre choice.

    Returns:
        ``(centers [k, d], labels [n])``, both on ``points``' device.
    """
    from scipy.optimize import linear_sum_assignment

    device = points.device
    x = points.detach().to(torch.float32)
    n = x.size(0)
    if n_clusters < 1 or n_clusters > n:
        raise ValueError(f"n_clusters={n_clusters} invalid for {n} points")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = x[torch.randperm(n, generator=generator)[:n_clusters].to(device)].clone()

    cap = int(math.ceil(n / n_clusters))
    # Column j of the expanded problem is cluster (j // cap): cap slots each.
    slot_to_cluster = (
        torch.arange(n_clusters * cap, device=device) // cap
    )
    labels = torch.zeros(n, dtype=torch.long, device=device)

    for _ in range(max(1, n_iters)):
        cost = torch.cdist(x, centers) ** 2                 # [n, k]
        expanded = cost[:, slot_to_cluster]                 # [n, k * cap]
        rows, cols = linear_sum_assignment(
            expanded.detach().cpu().numpy().astype("float64")
        )
        new_labels = torch.zeros(n, dtype=torch.long, device=device)
        new_labels[torch.as_tensor(rows, device=device)] = slot_to_cluster[
            torch.as_tensor(cols, device=device)
        ]
        converged = bool(torch.equal(new_labels, labels))
        labels = new_labels
        one_hot = torch.zeros(n, n_clusters, device=device, dtype=x.dtype)
        one_hot[torch.arange(n, device=device), labels] = 1.0
        counts = one_hot.sum(0).clamp(min=1.0).unsqueeze(-1)
        centers = (one_hot.t() @ x) / counts
        if converged:
            break

    return centers, labels


# --------------------------------------------------------------------------- #
# per-level quantizer with the diversity loss
# --------------------------------------------------------------------------- #
class LetterVectorQuantization(VectorQuantization):
    """A codebook level plus LETTER's diversity loss.

    The diversity loss is a cross-entropy over the codebook in which the
    "positive" is a DIFFERENT code drawn from the same balanced-k-means cluster
    as the code the item was actually assigned, and the item's own code is masked
    out. Pushing an assigned code towards its cluster-mates makes nearby codes
    interchangeable, which is what stops the assignment piling onto a few
    dominant codes.

    The cluster labels are recomputed from the live codebook by
    :meth:`refresh_code_clusters`; the parent
    :class:`LetterResidualQuantization` drives that on a step schedule.
    """

    def __init__(
        self,
        *args: Any,
        diversity_weight: float = 0.0,
        diversity_clusters: int = 10,
        diversity_temperature: float = 1.0,
        diversity_kmeans_iters: int = 10,
        diversity_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.diversity_weight = float(diversity_weight)
        self.diversity_clusters = int(diversity_clusters)
        self.diversity_temperature = float(diversity_temperature)
        self.diversity_kmeans_iters = int(diversity_kmeans_iters)
        self.diversity_seed = int(diversity_seed)
        # [n_clusters, n_positives]: for code k, the OTHER codes in its cluster.
        # A buffer so it follows .to(device), but NOT persistent: its width is
        # the largest cluster size and therefore varies between refreshes, so
        # checkpointing it would make resume fail on a shape mismatch. It is
        # rebuilt from the live codebook on the first fitting step instead.
        self.register_buffer(
            "code_cluster_positives",
            torch.zeros(self.n_clusters, 1, dtype=torch.long),
            persistent=False,
        )
        self._clusters_ready = False
        self.last_diversity_loss = torch.tensor(0.0)

    @torch.no_grad()
    def refresh_code_clusters(self) -> None:
        """Recluster the codebook and rebuild the positive-sampling table."""
        if self.diversity_weight <= 0 or not self.is_initialized:
            return
        codebook = self.get_centroids().data
        if not bool(torch.isfinite(codebook).all()):
            log.warning("[letter] codebook has non-finite entries; skipping recluster")
            return
        _, labels = balanced_kmeans(
            codebook,
            n_clusters=self.diversity_clusters,
            n_iters=self.diversity_kmeans_iters,
            seed=self.diversity_seed,
        )
        n = self.n_clusters
        # Per cluster, the member list; per code, that list minus the code
        # itself, right-padded by repeating members so every row is full.
        positives: List[torch.Tensor] = []
        width = 1
        members_by_cluster = [
            torch.nonzero(labels == c, as_tuple=False).flatten()
            for c in range(self.diversity_clusters)
        ]
        for code in range(n):
            members = members_by_cluster[int(labels[code])]
            others = members[members != code]
            if others.numel() == 0:
                # Degenerate singleton cluster: fall back to the whole codebook
                # minus self, so the loss stays defined instead of hanging (the
                # reference implementation loops forever here).
                others = torch.cat(
                    [
                        torch.arange(code, device=labels.device),
                        torch.arange(code + 1, n, device=labels.device),
                    ]
                )
            positives.append(others)
            width = max(width, int(others.numel()))
        table = torch.zeros(n, width, dtype=torch.long, device=labels.device)
        for code, others in enumerate(positives):
            reps = int(math.ceil(width / others.numel()))
            table[code] = others.repeat(reps)[:width]
        self.code_cluster_positives = table
        self._clusters_ready = True

    def diversity_loss(
        self, quantized: torch.Tensor, ids: torch.Tensor
    ) -> torch.Tensor:
        """LETTER's per-level diversity term.

        Args:
            quantized: ``[B, d]`` the assigned codebook vectors (gradient flows
                into the codebook, which is where this term is meant to act).
            ids: ``[B]`` assigned code per item.
        """
        if self.diversity_weight <= 0 or not self._clusters_ready:
            return quantized.new_zeros(())
        codebook = self.get_centroids()
        table = self.code_cluster_positives
        if table.size(0) != codebook.size(0):
            return quantized.new_zeros(())
        column = torch.randint(
            0, table.size(1), (ids.numel(),), device=table.device
        )
        targets = table[ids, column]                                # [B]
        similarity = quantized @ codebook.t()                       # [B, K]
        # Mask the item's own code, exactly as the reference does: without this
        # the argmin-assigned code trivially wins and the term has no gradient
        # towards its cluster-mates.
        similarity = similarity.scatter(
            1, ids.unsqueeze(-1), torch.finfo(similarity.dtype).min
        )
        return nn.functional.cross_entropy(
            similarity / self.diversity_temperature, targets
        )

    def model_step(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assignments, embeddings, loss = super().model_step(batch)
        self.last_diversity_loss = torch.zeros((), device=loss.device)
        if (
            self.diversity_weight > 0
            and self.is_initialized
            and self.training
            and assignments.numel() > 0
        ):
            quantized = self.get_centroids()[assignments]
            diversity = self.diversity_loss(quantized, assignments)
            self.last_diversity_loss = diversity.detach()
            loss = loss + self.diversity_weight * diversity
        return assignments, embeddings, loss


# --------------------------------------------------------------------------- #
# the tokenizer
# --------------------------------------------------------------------------- #
class LetterResidualQuantization(ResidualQuantization):
    """RQ-VAE + LETTER's collaborative alignment, over LETTER quantizer levels.

    Args:
        cf_embeddings: ``[N, d_cf]`` per-item collaborative vectors, row ``i`` =
            item ``i`` (the indexing of ``sequence_data`` item ids and of the
            semantic-ID tensor). Produced by
            ``scripts/train_cf_embeddings.py``. Required whenever
            ``cf_loss_weight > 0``.
        cf_loss_weight: LETTER's ``alpha``. 0 disables the term, which reduces
            the tokenizer to "RQ-VAE + diversity" -- useful as an ablation, not
            as LETTER.
        cf_projection: if the latent dim and ``d_cf`` differ, align them with a
            learned bias-free linear map. The paper has no such map (its latent
            and its SASRec are both 32-d); prefer matching the dims by training
            the CF model at the latent width, and treat this as the escape hatch.
        diversity_refresh_every_n_steps: how often to recluster the codebooks for
            the diversity term. The reference reclusters once per epoch, but
            GRID's item loader is unbounded and its "epochs" are a few steps
            long, so this is expressed in steps.
        sinkhorn_epsilon: entropic regularisation for balanced code assignment.
            LETTER applies it to the FINAL level only (``sk_epsilons =
            [0, 0, 0, 0.003]``), which is what ``sinkhorn_layers="last"``
            reproduces for any number of levels. Levels are configured from ONE
            deep-copied prototype, so per-level epsilons have to be set here
            rather than in the config.
        sinkhorn_layers: ``"last"`` (default), ``"all"``, ``"none"``, or an
            explicit list of level indices.
    """

    def __init__(
        self,
        *args: Any,
        cf_embeddings: Optional[torch.Tensor] = None,
        cf_loss_weight: float = 0.0,
        cf_projection: bool = False,
        cf_normalize: bool = False,
        diversity_refresh_every_n_steps: int = 200,
        sinkhorn_epsilon: float = 0.0,
        sinkhorn_layers: Any = "last",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._apply_sinkhorn_schedule(sinkhorn_epsilon, sinkhorn_layers)
        self.cf_loss_weight = float(cf_loss_weight)
        self.diversity_refresh_every_n_steps = int(diversity_refresh_every_n_steps)
        self.cf_normalize = bool(cf_normalize)
        self.cf_projection: Optional[nn.Module] = None
        self.last_cf_loss = torch.tensor(0.0)
        self._last_quantized: Optional[torch.Tensor] = None

        if self.cf_loss_weight > 0 and cf_embeddings is None:
            raise ValueError(
                "cf_loss_weight > 0 but no cf_embeddings were given. LETTER's "
                "collaborative alignment has nothing to align to; run "
                "scripts/train_cf_embeddings.py first, or set cf_loss_weight=0 "
                "to train the ablation explicitly."
            )

        if cf_embeddings is None:
            self.register_buffer("cf_embeddings", None, persistent=False)
            return

        matrix = self._as_cf_matrix(cf_embeddings)
        latent_dim = self._latent_dim()
        if matrix.size(-1) != latent_dim:
            if not cf_projection:
                raise ValueError(
                    f"cf_embeddings are {matrix.size(-1)}-d but the tokenizer's "
                    f"latent is {latent_dim}-d. LETTER's alignment term is a dot "
                    f"product between the two, so they must match. Retrain the CF "
                    f"model with --dim {latent_dim}, or set cf_projection=true to "
                    f"learn a map (a deviation from the paper)."
                )
            self.cf_projection = nn.Linear(latent_dim, matrix.size(-1), bias=False)
        # Not a Parameter: the CF model is frozen, it is an input to this one.
        self.register_buffer("cf_embeddings", matrix, persistent=False)
        log.info(
            "[letter] CF alignment enabled: alpha=%.4g over [%d, %d] embeddings%s",
            self.cf_loss_weight,
            matrix.size(0),
            matrix.size(1),
            " (projected)" if self.cf_projection is not None else "",
        )

    # -- construction helpers ------------------------------------------------ #
    def _apply_sinkhorn_schedule(self, epsilon: float, layers: Any) -> None:
        """Set each level's Sinkhorn epsilon, 0 for the levels not selected."""
        n = len(self.quantization_layer_list)
        if isinstance(layers, str):
            key = layers.strip().lower()
            if key == "last":
                selected = {n - 1}
            elif key == "all":
                selected = set(range(n))
            elif key in ("none", "null", ""):
                selected = set()
            else:
                raise ValueError(
                    f"sinkhorn_layers must be 'last', 'all', 'none' or a list of "
                    f"level indices; got {layers!r}"
                )
        elif layers is None:
            selected = set()
        else:
            selected = {int(i) % n for i in layers}

        for idx, layer in enumerate(self.quantization_layer_list):
            strategy = getattr(layer, "quantization_strategy", None)
            value = float(epsilon) if idx in selected else 0.0
            if not hasattr(strategy, "epsilon"):
                if value > 0:
                    raise TypeError(
                        f"sinkhorn_epsilon={epsilon} selects level {idx}, but its "
                        f"quantization_strategy is {type(strategy).__name__}, which "
                        f"has no epsilon. Use SinkhornQuantization for the levels "
                        f"balanced assignment is meant to act on."
                    )
                continue
            strategy.epsilon = value
        if float(epsilon) > 0:
            log.info(
                "[letter] Sinkhorn balancing: epsilon=%.4g on level(s) %s of %d",
                epsilon,
                sorted(selected),
                n,
            )

    def _latent_dim(self) -> int:
        return int(self.quantization_layer_list[0].n_features)

    @staticmethod
    def _as_cf_matrix(value: Any) -> torch.Tensor:
        """Accept a bare ``[N, d]`` tensor or a dict with an ``item_ids`` key.

        The dict form is reordered by ``item_ids``. Row ``i`` is only item ``i``
        when the ids happen to be ``arange``; a differently ordered artefact
        would otherwise pair every item's identifier with another item's
        behaviour, with no error anywhere.
        """
        if isinstance(value, dict):
            keys = [k for k in ("embeddings", "cf_embeddings", "weight") if k in value]
            if not keys:
                raise ValueError(
                    f"cf_embeddings dict has no embedding key; got {sorted(value)}"
                )
            matrix = torch.as_tensor(value[keys[0]]).float()
            if "item_ids" in value:
                order = torch.as_tensor(value["item_ids"]).long()
                if order.numel() != matrix.size(0):
                    raise ValueError(
                        f"item_ids has {order.numel()} entries for "
                        f"{matrix.size(0)} embedding rows"
                    )
                reordered = torch.empty_like(matrix)
                reordered[order] = matrix
                matrix = reordered
            return matrix
        matrix = torch.as_tensor(value).float()
        if matrix.dim() == 3 and matrix.size(0) == 1:
            matrix = matrix.squeeze(0)
        if matrix.dim() != 2:
            raise ValueError(f"cf_embeddings must be [N, d]; got {tuple(matrix.shape)}")
        return matrix

    # -- LETTER's collaborative alignment ------------------------------------ #
    def cf_loss(
        self, quantized: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE between quantised latents and CF vectors, in-batch negatives.

        Identical in form to the reference ``RQVAE.CF_loss``: the similarity
        matrix is ``x_q @ cf^T`` and the label of row ``i`` is ``i``, so the item
        must score its OWN behaviour above every other item's in the batch.
        """
        cf = self.cf_embeddings
        if cf is None or self.cf_loss_weight <= 0:
            return quantized.new_zeros(())
        if int(item_ids.max()) >= cf.size(0):
            raise IndexError(
                f"item id {int(item_ids.max())} is outside the CF table "
                f"({cf.size(0)} rows). The CF embeddings were built for a "
                f"different catalog than the items being tokenized."
            )
        target = cf[item_ids].to(quantized.dtype)
        source = quantized
        if self.cf_projection is not None:
            source = self.cf_projection(source)
        if self.cf_normalize:
            source = nn.functional.normalize(source, dim=-1)
            target = nn.functional.normalize(target, dim=-1)
        similarities = source @ target.t()
        labels = torch.arange(
            similarities.size(0), device=similarities.device, dtype=torch.long
        )
        return nn.functional.cross_entropy(similarities, labels)

    @staticmethod
    def _item_ids_tensor(model_input: ItemData, device: torch.device) -> Optional[torch.Tensor]:
        ids = getattr(model_input, "item_ids", None)
        if ids is None:
            return None
        if isinstance(ids, torch.Tensor):
            return ids.flatten().long().to(device)
        flat = [
            int(i.item()) if isinstance(i, torch.Tensor) else int(i) for i in ids
        ]
        return torch.tensor(flat, dtype=torch.long, device=device)

    # -- refresh schedule ---------------------------------------------------- #
    def maybe_refresh_code_clusters(self, force: bool = False) -> None:
        every = self.diversity_refresh_every_n_steps
        due = force or (every > 0 and int(self.global_step) % every == 0)
        for layer in self.quantization_layer_list:
            if not isinstance(layer, LetterVectorQuantization):
                continue
            # A level that has just finished initialising has no cluster table
            # yet. Waiting for the next scheduled refresh would leave the
            # diversity term silently at zero for up to `every` steps -- which
            # is most of a short run.
            if due or (layer.is_initialized and not layer._clusters_ready):
                layer.refresh_code_clusters()

    def on_train_start(self) -> None:
        super().on_train_start()
        self.maybe_refresh_code_clusters(force=True)

    def forward(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """As the parent, but keeps the summed quantised latent for the CF term.

        ``quantized_embeddings`` is the straight-through sum over levels, i.e.
        exactly the ``dense_out`` the reference implementation feeds to its CF
        loss. Recomputing it instead would run the residual chain twice and
        re-draw the Sinkhorn assignment.
        """
        cluster_ids, all_residuals, quantized, quantization_loss = super().forward(
            embeddings
        )
        self._last_quantized = quantized
        return cluster_ids, all_residuals, quantized, quantization_loss

    # -- the one override that carries the extra terms ----------------------- #
    def model_step(
        self, model_input: ItemData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """As the parent, but folds LETTER's extra terms into ``quantization_loss``.

        Folding rather than returning a fifth value keeps the inherited
        ``training_step`` -- and with it the manual ``training_loop_function``
        that drives codebook initialisation -- working unmodified. With
        ``quantization_loss_weight = 1.0`` the total loss is exactly LETTER's.
        """
        (
            cluster_ids,
            all_residuals,
            quantization_loss,
            reconstruction_loss,
        ) = super().model_step(model_input)

        trainer = getattr(self, "_trainer", None)
        is_fitting = (
            trainer is not None
            and trainer.state.fn == TrainerFn.FITTING
            and self.training
        )
        # The diversity terms were already added inside each level's model_step.
        diversity = sum(
            float(getattr(layer, "last_diversity_loss", 0.0))
            for layer in self.quantization_layer_list
        )
        cf_value = 0.0
        if is_fitting and self.cf_loss_weight > 0:
            item_ids = self._item_ids_tensor(model_input, quantization_loss.device)
            last_layer = self.quantization_layer_list[-1]
            if item_ids is not None and last_layer.is_initialized:
                quantized = self._last_quantized
                if quantized is not None and quantized.size(0) == item_ids.numel():
                    cf = self.cf_loss(quantized, item_ids)
                    quantization_loss = quantization_loss + self.cf_loss_weight * cf
                    cf_value = float(cf.detach())
        self.last_cf_loss = torch.tensor(cf_value)

        if is_fitting:
            self.maybe_refresh_code_clusters()
            self.log_dict(
                {
                    "train/letter_cf_loss": cf_value,
                    "train/letter_diversity_loss": diversity,
                },
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        return cluster_ids, all_residuals, quantization_loss, reconstruction_loss

__all__ = [
    "LetterResidualQuantization",
    "LetterVectorQuantization",
    "balanced_kmeans",
]
