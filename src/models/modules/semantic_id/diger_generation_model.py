"""DIGER: TIGER with a differentiable, jointly trained semantic-ID tokenizer.

arXiv:2601.19711. TIGER trains an RQ quantizer for content reconstruction, then
FREEZES the resulting semantic IDs and trains the recommender on top, so no
recommendation gradient ever reaches the identifiers. DIGER removes the freeze:
the indexing step becomes differentiable (straight-through Gumbel-Softmax over
the codebooks, see :mod:`diger_tokenizer`) and the tokenizer trains jointly with
the recommender under

    L = L_gen (SDUD-weighted) + lambda_vq * L_vq + lambda_recon * L_recon

**How it plugs in.** The recommender's token embeddings for an item's SID block
are replaced by the soft mixture

    e~_v^l = sum_k y~_{v,l,k} E_l[k]

where ``y~`` comes from the tokenizer and ``E_l`` is the l-th K-row block of the
model's own SID embedding table. That single substitution is what puts the
codebook and the tokenizer encoder on the generation loss's gradient path. It
reuses :meth:`_inject_soft_sid_embeddings`, the hook TRACER already added, so
DIGER gets both the encoder- and decoder-side call sites and their alignment
guards for free.

**Why this matters for unlearning.** With frozen IDs, an unlearning update can
only move theta; the identifier space is fixed. Here the codes are part of what
gets unlearned, which is exactly the regime where the measured
levels-0/1-vs-2/3 asymmetry becomes actionable rather than descriptive.

**COMMITTING.** The soft mixture is a training-time device. Evaluation, the SID
tensor on disk, and every spam metric read HARD codes, so
:meth:`commit_semantic_ids` re-derives them (deterministically, no Gumbel) plus
the trailing dedup digit and rewrites ``self.codebooks`` in place.

> Known limitation, stated rather than hidden: the dataloader materialises its
> own item->SID mapping, and with ``num_workers > 0`` each worker holds a copy.
> A mid-training commit therefore refreshes the model's view immediately but the
> LABELS only when the dataloader is rebuilt. ``commit_every_n_epochs=0`` (the
> default) sidesteps this entirely by committing once at train end, which is
> well defined; set it >0 only with ``persistent_workers=false``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.modules.semantic_id.diger_tokenizer import DigerTokenizer, sdud_loss
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)

log = logging.getLogger(__name__)

# How many training steps to observe before concluding that injection never
# fires. Large enough that an unlucky run of all-padding micro-batches cannot
# trip it, small enough to catch a genuinely broken run early.
_DIGER_NOAUX_GRACE_STEPS = 50


class DigerEncoderDecoder(SemanticIDEncoderDecoder):
    """TIGER backbone + :class:`DigerTokenizer`, trained jointly.

    Parameters
    ----------
    item_content_embeddings
        ``[N, content_dim]`` frozen item content features -- the SAME dense
        flan-T5 embeddings the RQ quantizer was fit on
        (``embeddings/<ds>_merged_predictions_tensor_latest.pt``). Frozen: DIGER
        learns the INDEX, not the content encoder that produced these.
    diger_lambda_vq, diger_lambda_recon
        Weights on the tokenizer's auxiliary terms.
    diger_use_sdud, diger_sdud_lambda
        Standard-deviation uncertainty decay on the generation term.
    diger_commit_every_n_epochs
        0 (default) = commit hard codes only at train end. See the module
        docstring before raising it.
    tokenizer_kwargs
        Forwarded to :class:`DigerTokenizer` (tau, frq_decay_ratio, ema_beta,
        latent_dim, commitment_weight, similarity, encoder_hidden).
    """

    def __init__(
        self,
        *,
        item_content_embeddings: Optional[torch.Tensor] = None,
        diger_lambda_vq: float = 1.0,
        diger_lambda_recon: float = 1.0,
        diger_use_sdud: bool = True,
        diger_sdud_lambda: float = 1.4,
        diger_commit_every_n_epochs: int = 0,
        diger_init_codebooks: bool = True,
        diger_init_seed: int = 0,
        diger_pretrain_steps: int = 2000,
        diger_pretrain_lr: float = 1.0e-3,
        diger_tokenizer_lr: Optional[float] = 1.0e-4,
        diger_tokenizer_weight_decay: Optional[float] = None,
        diger_uncertainty_decay: str = "frqud",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if item_content_embeddings is None:
            raise ValueError(
                "DigerEncoderDecoder requires item_content_embeddings [N, D]; "
                "without them there is nothing to tokenize and the model would "
                "silently degenerate to plain TIGER."
            )
        content = self._as_content_matrix(item_content_embeddings)
        n_items = int(self.codebooks.size(0)) if self.codebooks is not None else None
        if n_items is not None and content.size(0) != n_items:
            raise ValueError(
                f"item_content_embeddings has {content.size(0)} rows but the "
                f"semantic-id map has {n_items} items; they must be aligned on "
                "item id or every tokenized item is the wrong one."
            )
        # Frozen, and NOT persistent: these are an input artifact of the dataset,
        # already on disk, and [N, 2048] would bloat every checkpoint.
        self.register_buffer("item_content", content, persistent=False)

        # n_levels is the SID length BEFORE the dedup digit -- the RKMEANS_HIER
        # convention. The last hierarchy is a tiebreaker with no content
        # semantics (measured: 16 distinct values across beauty's 12,101 items),
        # so there is nothing for a quantizer to learn there.
        self.diger_levels = int(self.num_hierarchies) - 1
        if self.diger_levels < 1:
            raise ValueError(
                f"num_hierarchies={self.num_hierarchies} leaves no quantized "
                "levels once the dedup digit is excluded"
            )
        self.tokenizer = DigerTokenizer(
            content_dim=content.size(1),
            n_levels=self.diger_levels,
            codebook_size=int(self.num_embeddings_per_hierarchy),
            **(tokenizer_kwargs or {}),
        )
        # Fit the fixed input standardization from the frozen catalog NOW, so
        # every later encoder call (including commit under eval()) sees
        # standardized features.
        self.tokenizer.fit_input_norm(content)
        self.diger_lambda_vq = float(diger_lambda_vq)
        self.diger_lambda_recon = float(diger_lambda_recon)
        self.diger_use_sdud = bool(diger_use_sdud)
        self.diger_sdud_lambda = float(diger_sdud_lambda)
        self.diger_commit_every_n_epochs = int(diger_commit_every_n_epochs)
        self.diger_init_codebooks = bool(diger_init_codebooks)
        self.diger_init_seed = int(diger_init_seed)
        self.diger_pretrain_steps = int(diger_pretrain_steps)
        self.diger_pretrain_lr = float(diger_pretrain_lr)
        self.diger_tokenizer_lr = (
            None if diger_tokenizer_lr is None else float(diger_tokenizer_lr)
        )
        self.diger_tokenizer_weight_decay = (
            None
            if diger_tokenizer_weight_decay is None
            else float(diger_tokenizer_weight_decay)
        )
        # FrqUD and SDUD are ALTERNATIVE uncertainty-decay mechanisms in the
        # paper (separate rows of its Table 2), not a stack. On B-Shop -- which
        # is this exact dataset, 22,363 users / 12,101 items -- FrqUD wins:
        # R@10 0.0683 vs 0.0657. Hence the default.
        #   frqud   FrqUD only (noise on hot codes), SDUD off        [default]
        #   sdud    SDUD only; Gumbel noise on ALL codes
        #   both    both mechanisms
        #   none    neither; Gumbel noise on all codes
        #   no_noise  no exploration at all (paper ablation: R@10 0.0283)
        self.diger_uncertainty_decay = str(diger_uncertainty_decay).lower()
        _valid = ("frqud", "sdud", "both", "none", "no_noise")
        if self.diger_uncertainty_decay not in _valid:
            raise ValueError(
                f"diger_uncertainty_decay must be one of {_valid}, got "
                f"{diger_uncertainty_decay!r}"
            )
        mode = self.diger_uncertainty_decay
        self.diger_use_sdud = bool(diger_use_sdud) and mode in ("sdud", "both")
        # SDUD's sigma is the GUMBEL NOISE STANDARD DEVIATION (paper Sec 4.2), not a
        # weight on L_gen. Until 2026-08-25 this was computed for the L_sigma term
        # and then thrown away, so `sdud`/`both` annealed the loss weight and left
        # the noise magnitude fixed -- the exploration schedule the paper describes
        # never actually happened. Wire it to the tokenizer.
        self.tokenizer.sigma_scaled_noise = self.diger_use_sdud
        if mode in ("sdud", "none"):
            # No frequency gate: explore everywhere.
            self.tokenizer.frq_decay_ratio = 0.0
        if mode == "no_noise":
            self.tokenizer.use_gumbel_noise = False
        # Persistent: a resumed run must NOT re-seed a trained codebook.
        self.register_buffer(
            "_codebook_initialized", torch.zeros((), dtype=torch.long)
        )
        self._diger_aux: Dict[str, torch.Tensor] = {}
        self._diger_injected = 0
        # Injection-rate bookkeeping. A single training step with no resolvable
        # item block is NORMAL (a batch can be all padding), so the condition
        # worth warning about is "never fires", not "did not fire once".
        self._diger_steps = 0
        self._diger_steps_injected = 0
        if not hasattr(self, "_adaptive_sorted_keys"):
            self._build_code_to_item_index()

    @staticmethod
    def _as_content_matrix(obj: Any) -> torch.Tensor:
        """Coerce a content-embedding artifact to ``[N, D]`` in ITEM-ID order.

        The repo ships two shapes and Hydra hands whichever it finds straight to
        the constructor, so this has to accept both:

        * a bare ``[N, D]`` tensor (``embeddings/beauty_..._latest.pt``), and
        * a dict ``{"embeddings": [N, D], "item_ids": [N]}``
          (``logs/inference/runs/embeddings/<ds>_*/pickle/...``).

        Passing the dict through ``torch.as_tensor`` raises "Could not infer
        dtype of dict", which is what killed the first four toys runs.

        ROW ORDER IS THE REAL HAZARD. ``item_ids`` is the alignment key, and row
        i of ``embeddings`` is only item i if those ids happen to be ``arange``.
        They are for the toys artifact, but relying on that would silently
        mis-pair every item's content with another item's semantic id the first
        time an artifact is written in a different order. So reorder explicitly.
        """
        if isinstance(obj, dict):
            emb = obj.get("embeddings")
            if emb is None:
                tensors = [v for v in obj.values() if torch.is_tensor(v) and v.dim() == 2]
                if len(tensors) != 1:
                    raise ValueError(
                        "item_content_embeddings dict has no 'embeddings' key and "
                        f"{len(tensors)} 2-D tensors; cannot tell which is the "
                        f"content matrix (keys: {sorted(obj)})"
                    )
                emb = tensors[0]
            emb = torch.as_tensor(emb).float()
            ids = obj.get("item_ids")
            if ids is not None:
                ids = torch.as_tensor(ids).long().flatten()
                if ids.numel() != emb.size(0):
                    raise ValueError(
                        f"item_ids has {ids.numel()} entries but embeddings has "
                        f"{emb.size(0)} rows"
                    )
                n = emb.size(0)
                if int(ids.unique().numel()) != n:
                    raise ValueError("item_ids contains duplicates")
                if int(ids.min()) != 0 or int(ids.max()) != n - 1:
                    raise ValueError(
                        f"item_ids must cover 0..{n - 1} exactly, got range "
                        f"[{int(ids.min())}, {int(ids.max())}]"
                    )
                if not torch.equal(ids, torch.arange(n, dtype=ids.dtype)):
                    log.info(
                        "[diger] content embeddings were not stored in item-id "
                        "order; reordering %d rows by item_ids.", n
                    )
                    emb = emb[torch.argsort(ids)]
            content = emb
        else:
            content = torch.as_tensor(obj).float()
        if content.dim() != 2:
            raise ValueError(
                f"item_content_embeddings must be 2-D [N, D], got {tuple(content.shape)}"
            )
        return content

    # ------------------------------------------------------------- injection
    def _inject_soft_sid_embeddings(
        self, embeds: torch.Tensor, raw_codes: torch.Tensor
    ) -> torch.Tensor:
        """Swap each item's SID token embeddings for DIGER's soft mixture.

        Falls back to the parent (TRACER) behaviour when the DIGER tokenizer is
        not active, so a DIGER checkpoint can still be run through the TRACER
        unlearning path without a second code path.
        """
        if getattr(self, "tokenizer", None) is None:
            return super()._inject_soft_sid_embeddings(embeds, raw_codes)
        num_hierarchies = int(self.num_hierarchies)
        if (
            raw_codes.dim() != 2
            or raw_codes.size(1) == 0
            or raw_codes.size(1) % num_hierarchies != 0
        ):
            # Incremental generation: item identity is not determined yet.
            return embeds
        batch, seq_len = raw_codes.shape
        n_items = seq_len // num_hierarchies
        if embeds.dim() != 3 or embeds.size(0) != batch or embeds.size(1) != seq_len:
            if not getattr(self, "_diger_align_warned", False):
                self._diger_align_warned = True
                log.warning(
                    "[diger] skipping soft-SID injection: embeds %s not aligned to "
                    "raw_codes %s; the tokenizer receives NO gradient here.",
                    tuple(embeds.shape),
                    tuple(raw_codes.shape),
                )
            return embeds

        codes_items = raw_codes.view(batch, n_items, num_hierarchies)
        item_ids = self._codes_to_item_ids(codes_items)               # [b, n]
        valid = item_ids >= 0
        if not bool(valid.any()):
            return embeds

        b_idx, i_idx = valid.nonzero(as_tuple=True)
        sel_items = item_ids[b_idx, i_idx].long()                     # [n_sel]
        # Tokenize each DISTINCT item once: a popular item can appear many times
        # in one batch, and re-running the encoder per occurrence would multiply
        # its weight in L_vq / L_recon on top of wasting the compute.
        uniq, inverse = torch.unique(sel_items, return_inverse=True)
        out = self.tokenizer(self.item_content[uniq])
        table = self.get_embedding_table(table_name="encoder").weight
        K = int(self.num_embeddings_per_hierarchy)
        rows = table[: self.diger_levels * K].view(self.diger_levels, K, -1)
        # e~_v^l = sum_k y_{v,l,k} E_l[k]
        soft = torch.einsum("blk,lkd->bld", out.probs.to(table.dtype), rows)

        emb_items = embeds.view(batch, n_items, num_hierarchies, embeds.size(-1)).clone()
        # Explicit integer indices on dims 0/1 -- a mixed bool-mask/slice index
        # applies the SLICE first and then validates the mask against the wrong
        # shape (this exact bug killed every TRACER step).
        emb_items[b_idx, i_idx, : self.diger_levels, :] = soft[inverse]

        # Stash the auxiliary terms for model_step. Accumulated because the hook
        # fires on BOTH the encoder and decoder side of one forward; model_step
        # averages so the weights mean the same thing either way.
        self._diger_aux["vq"] = self._diger_aux.get("vq", 0.0) + out.vq_loss
        self._diger_aux["recon"] = self._diger_aux.get("recon", 0.0) + out.recon_loss
        self._diger_injected += 1
        return emb_items.view(batch, seq_len, embeds.size(-1))

    # ------------------------------------------------------------- loss ------
    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``L_gen`` (SDUD-weighted) + ``lambda_vq L_vq`` + ``lambda_recon L_recon``."""
        self._diger_aux = {}
        self._diger_injected = 0
        model_output, loss = super().model_step(model_input, label_data)
        if label_data is None:
            return model_output, loss          # inference: no auxiliary terms
        n = max(1, self._diger_injected)
        vq = self._diger_aux.get("vq")
        recon = self._diger_aux.get("recon")
        if self.training:
            self._diger_steps += 1
            if vq is not None:
                self._diger_steps_injected += 1
        if vq is None:
            # No item block was identifiable THIS step, which on its own is
            # unremarkable: a beauty batch is ~93% padding, so an occasional
            # micro-batch resolves nothing. The failure worth catching is the
            # tokenizer NEVER receiving gradient, i.e. silently training plain
            # TIGER under a DIGER config.
            #
            # Warning on the first miss (the previous behaviour) permanently
            # mislabels healthy runs: SMOKE4 emitted it once and still finished
            # with train/diger_vq=3.20 and train/diger_recon=0.52, which are
            # only logged when injection DID fire.
            if (
                self.training
                and not getattr(self, "_diger_noaux_warned", False)
                and self._diger_steps >= _DIGER_NOAUX_GRACE_STEPS
                and self._diger_steps_injected == 0
            ):
                self._diger_noaux_warned = True
                log.warning(
                    "[diger] soft-SID injection has not fired in ANY of the "
                    "first %d training steps; the tokenizer is receiving no "
                    "gradient and this run is plain TIGER with extra loss "
                    "terms. Check that codebooks/item_content are aligned to "
                    "the batch's item ids (src/diagnose_diger.py reports the "
                    "resolve rate on one real batch).",
                    self._diger_steps,
                )
            return model_output, loss
        if self.diger_use_sdud:
            gen = sdud_loss(loss, lam=self.diger_sdud_lambda)
            if self.training:
                # Sets the noise sigma used by the NEXT forward pass; see
                # DigerTokenizer.update_noise_sigma on the one-step lag.
                self.tokenizer.update_noise_sigma(loss, self.diger_sdud_lambda)
        else:
            gen = loss
        total = (
            gen
            + self.diger_lambda_vq * (vq / n)
            + self.diger_lambda_recon * (recon / n)
        )
        if self.training:
            self.log_dict(
                {
                    "train/diger_gen": loss.detach(),
                    "train/diger_vq": (vq / n).detach(),
                    "train/diger_recon": (recon / n).detach(),
                    "train/diger_hot_frac": self.tokenizer.hot_code_fraction().mean(),
                    # The exploration schedule. Should start near sqrt(L_gen) and
                    # decay toward 0; flat at 1.0 means sigma is not wired.
                    "train/diger_noise_sigma": self.tokenizer.noise_sigma.detach(),
                },
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        return model_output, total

    # ------------------------------------------------------------- optimizer
    def configure_optimizers(self) -> Dict[str, Any]:
        """Give the tokenizer its OWN, much lower learning rate.

        Not cosmetic. The paper searches the recommender lr over
        {1e-2 .. 5e-4} but the TOKENIZER lr over {1e-4 .. 1e-6} -- one to three
        orders of magnitude lower. At the recommender's rate the codebook moves
        so fast that the identifier space is rewritten faster than the
        recommender can learn to read it, and the assignment collapses (the
        paper's own plain-STE row lands at R@10 0.0134 against a 0.0610
        two-stage baseline).

        ``diger_tokenizer_lr=None`` disables the split and puts everything on the
        config default -- the ablation for "does the separate rate matter".
        """
        if self.diger_tokenizer_lr is None:
            return super().configure_optimizers()
        tok_params = [p for p in self.tokenizer.parameters() if p.requires_grad]
        if not tok_params:
            return super().configure_optimizers()
        tok_ids = {id(p) for p in tok_params}
        rest = [
            p for p in self.parameters() if p.requires_grad and id(p) not in tok_ids
        ]
        group: Dict[str, Any] = {"params": tok_params, "lr": self.diger_tokenizer_lr}
        if self.diger_tokenizer_weight_decay is not None:
            group["weight_decay"] = self.diger_tokenizer_weight_decay
        optimizer = self.optimizer(params=[{"params": rest}, group])
        log.info(
            "[diger] tokenizer param group: %d tensors at lr=%s (wd=%s); "
            "%d other tensors on config defaults.",
            len(tok_params),
            self.diger_tokenizer_lr,
            self.diger_tokenizer_weight_decay,
            len(rest),
        )
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # -------------------------------------------------------------- schedule
    def on_train_start(self) -> None:
        parent = getattr(super(), "on_train_start", None)
        if callable(parent):
            parent()
        self.maybe_init_codebooks()

    def maybe_init_codebooks(self, force: bool = False) -> bool:
        """k-means-seed the codebooks once. Returns True if it ran.

        Guarded by a PERSISTENT buffer so resuming from a checkpoint does not
        re-seed a codebook that has already been trained -- that would silently
        throw away the learned identifier space at every restart.
        """
        if not self.diger_init_codebooks:
            return False
        if bool(self._codebook_initialized) and not force:
            return False
        # Stage 1 BEFORE seeding: a random deep encoder collapses the catalog
        # onto one point (mean pairwise cosine 0.99995 vs 0.8617 for the raw
        # features), so k-means on top of it would partition noise.
        if self.diger_pretrain_steps > 0:
            before = self.tokenizer.latent_dispersion(self.item_content)
            rec = self.tokenizer.pretrain_autoencoder(
                self.item_content,
                steps=self.diger_pretrain_steps,
                lr=self.diger_pretrain_lr,
                seed=int(getattr(self, "diger_init_seed", 0)),
            )
            after = self.tokenizer.latent_dispersion(self.item_content)
            log.info(
                "[diger] tokenizer autoencoder pretrain: %d steps, recon=%.5f, "
                "mean pairwise latent cosine %.5f -> %.5f",
                self.diger_pretrain_steps, rec, before, after,
            )
            if after > 0.99:
                log.warning(
                    "[diger] latents are still collapsed (mean pairwise cosine "
                    "%.5f > 0.99) after pretraining. The codebook fitted on them "
                    "will partition noise and the ids will not survive training. "
                    "Raise diger_pretrain_steps or shrink the encoder.", after,
                )
        self.tokenizer.init_codebooks_from_content(
            self.item_content, seed=int(getattr(self, "diger_init_seed", 0))
        )
        self._codebook_initialized.fill_(1)
        log.info(
            "[diger] k-means seeded %d codebooks of size %d from %d items",
            self.diger_levels,
            int(self.num_embeddings_per_hierarchy),
            int(self.item_content.size(0)),
        )
        return True

    def on_train_epoch_end(self) -> None:
        parent = getattr(super(), "on_train_epoch_end", None)
        if callable(parent):
            parent()
        self.tokenizer.step_frequency_ema()
        # Cheap health probe, once per epoch. commit_semantic_ids() RAISES if the
        # codebook has collapsed, and it only runs at train end -- so without
        # this a multi-hour run can die at the finish line with no warning. One
        # assign() over the catalog is negligible next to an epoch.
        try:
            codes = self.tokenizer.assign(self.item_content)
            n_lvl = codes.size(1)
            uniq_l0 = int(codes[:, 0].unique().numel())
            prefix = codes[:, : max(1, n_lvl)]
            _, counts = torch.unique(prefix, dim=0, return_counts=True)
            worst = int(counts.max())
            log.info(
                "[diger] epoch %d health: level-0 codes used %d/%d, largest "
                "full-prefix bucket %d (vocab %d), latent dispersion %.4f%s",
                int(self.current_epoch), uniq_l0,
                int(self.num_embeddings_per_hierarchy), worst,
                int(self.num_embeddings_per_hierarchy),
                self.tokenizer.latent_dispersion(self.item_content),
                "  <-- COLLAPSING, commit will fail"
                if worst >= int(self.num_embeddings_per_hierarchy) else "",
            )
        except Exception as exc:  # never let a probe kill training
            log.warning("[diger] health probe failed: %s", exc)
        if self._diger_steps:
            rate = 100.0 * self._diger_steps_injected / self._diger_steps
            log.info(
                "[diger] epoch %d: soft-SID injection fired on %d/%d training "
                "steps (%.1f%%)",
                int(self.current_epoch), self._diger_steps_injected,
                self._diger_steps, rate,
            )
        every = self.diger_commit_every_n_epochs
        if every > 0 and (int(self.current_epoch) + 1) % every == 0:
            changed = self.commit_semantic_ids()
            log.info(
                "[diger] epoch %d: committed semantic ids, %d items reassigned",
                int(self.current_epoch),
                changed,
            )

    def on_train_end(self) -> None:
        parent = getattr(super(), "on_train_end", None)
        if callable(parent):
            parent()
        changed = self.commit_semantic_ids()
        log.info("[diger] train end: committed ids, %d items reassigned", changed)

    # ---------------------------------------------------------------- commit
    @torch.no_grad()
    def commit_semantic_ids(self, batch_size: int = 4096) -> int:
        """Re-derive HARD codes for the whole catalog and rewrite ``self.codebooks``.

        Deterministic (no Gumbel), so committing twice gives the same answer. The
        trailing dedup digit is rebuilt from scratch: it indexes an item within
        its ``c_0..c_{L-1}`` bucket, so it is meaningless once the prefix moves.

        Returns the number of items whose semantic id changed.
        """
        if self.codebooks is None:
            raise RuntimeError("commit_semantic_ids needs the item->SID map")
        device = self.item_content.device
        chunks = [
            self.tokenizer.assign(self.item_content[i : i + batch_size])
            for i in range(0, self.item_content.size(0), batch_size)
        ]
        codes = torch.cat(chunks, dim=0)                       # [N, L]
        dedup = self._dedup_digits(codes)                      # [N]
        new = torch.cat([codes, dedup.unsqueeze(1)], dim=1).to(
            device=self.codebooks.device, dtype=self.codebooks.dtype
        )
        overflow = int((dedup >= int(self.num_embeddings_per_hierarchy)).sum())
        if overflow:
            # Same failure the food dataset hit: a bucket larger than the vocab
            # cannot be separated by one digit. Loud, because the alternative is
            # an out-of-range embedding lookup much later.
            raise RuntimeError(
                f"{overflow} items need a dedup digit >= "
                f"{self.num_embeddings_per_hierarchy}; the codebook has collapsed "
                "a bucket larger than the vocabulary. Lower the learning rate or "
                "add an RQ level."
            )
        changed = int((new != self.codebooks).any(dim=1).sum())
        self.codebooks.copy_(new)
        # The SID-tuple -> item-id map is derived from codebooks; stale entries
        # would silently resolve the WRONG item in the next injection.
        self._build_code_to_item_index()
        return changed

    @staticmethod
    @torch.no_grad()
    def _dedup_digits(codes: torch.Tensor) -> torch.Tensor:
        """Per-item index within its identical-prefix bucket. ``[N, L] -> [N]``.

        Order-stable: items keep ascending item-id order inside a bucket, so the
        digit is a function of the prefix assignment alone and does not churn
        between commits.
        """
        n = codes.size(0)
        # Lexicographic bucket id via a stable sort on the packed prefix.
        keys = [codes[:, j] for j in range(codes.size(1))]
        order = torch.arange(n, device=codes.device)
        for j in reversed(range(len(keys))):
            order = order[torch.argsort(keys[j][order], stable=True)]
        sorted_codes = codes[order]
        same = (sorted_codes[1:] == sorted_codes[:-1]).all(dim=1)
        digits = torch.zeros(n, dtype=torch.long, device=codes.device)
        # Vectorised run-length index over the sorted order.
        starts = torch.cat(
            [torch.ones(1, dtype=torch.bool, device=codes.device), ~same]
        )
        idx = torch.arange(n, device=codes.device)
        run_start = torch.cummax(torch.where(starts, idx, torch.zeros_like(idx)), 0)[0]
        digits_sorted = idx - run_start
        digits[order] = digits_sorted
        return digits

    def write_semantic_id_tensor(self, path: str) -> str:
        """Persist the committed ids in the repo's ``[H, N]`` on-disk layout.

        Evaluation resolves spam/sensitive targets through ``semantic_id_path``;
        writing the model's own committed ids is what keeps SH/ASI/TPM from
        scoring the codes the run STARTED with.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.codebooks.detach().cpu().t().contiguous(), path)
        log.info("[diger] wrote committed semantic ids -> %s", path)
        return path
