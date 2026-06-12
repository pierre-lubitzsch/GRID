import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.metric import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

## Custom Metrics


class CustomMeanReductionMetric(torchmetrics.Metric):
    """
    Custom metric class that uses mean reduction and supports distributed training.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.metric_values = 0
        self.total_values = 0

    def compute(self) -> torch.Tensor:
        # Aggregates the metric accross workers and returns the final value
        metric_values_tensor = torch.tensor(self.metric_values).to(self.device)
        total_values_tensor = torch.tensor(self.total_values).to(self.device)
        # Compute final metric
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.device)
        # Checks if using more than one GPU
        # If so, gather all metric values and total values from all GPUs. Else, return the current
        # worker's metric value
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather all metric values and total values from all GPUs

            metric_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(metric_values_tensor)
            ]
            metric_values_tensor = torch.cat(metric_values_tensor_list).sum()

            total_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(total_values_tensor)
            ]

            total_values_tensor = torch.cat(total_values_tensor_list).sum()

        return metric_values_tensor / total_values_tensor

    def reset(self) -> None:
        self.metric_values = 0
        self.total_values = 0

    def update(self) -> None:
        raise NotImplementedError


class CustomRetrievalMetric(CustomMeanReductionMetric):
    """
    Custom retrieval metric class to calculate ranking metrics.
    """

    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:

        batch_size = int(len(indexes) / (indexes == 0).sum().item())
        preds = preds.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1).int()

        metric = self._metric(preds, target)
        self.metric_values += metric.sum().item()
        self.total_values += batch_size

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NDCG(CustomRetrievalMetric):
    """
    Metric to calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        # Compute DCG
        dcg = torch.sum(
            topk_true
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Compute IDCG
        ideal_indices = torch.topk(target, self.top_k)[1]
        ideal_dcg = torch.sum(
            target.gather(1, ideal_indices)
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Handle cases where IDCG is zero
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)
        return ndcg


class Recall(CustomRetrievalMetric):
    """
    Metric to calculate Recall@K.
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)

        recall = true_positives / total_relevant.minimum(
            torch.tensor(self.top_k, device=self.device)
        ).clamp(
            min=1
        )  # Use clamp to avoid zero
        return recall


## Spam-exposure metrics
#
# These measure how often the spam-boosted target items I_t leak into the top-k
# recommendations, restricted to evaluation examples whose ground-truth label is
# NOT itself a target (y not in I_t). With one eval example per user (leave-one-
# out, |D_u| = 1) these accumulators match the definitions exactly:
#   SH@k  = (# non-target examples with a target in top-k) / (# non-target examples)
#   ASI@k = (sum over non-target examples of |top-k ∩ I_t| / min(|I_t|, k)) / |U|
# where |U| is the total number of users/examples (so target-labelled examples
# contribute 0 to the numerator but still count in ASI's denominator, matching
# the max{1, ...} convention).


class CustomSpamMetric(CustomMeanReductionMetric):
    """Base for spam-exposure metrics that need per-candidate target membership.

    Unlike :class:`CustomRetrievalMetric` (which only sees relevance of the true
    label), spam metrics consume, per example: ``preds`` (B, C) candidate scores,
    ``is_spam_cand`` (B, C) whether each generated candidate is a target item, and
    ``is_spam_label`` (B,) whether the ground-truth label is itself a target.
    """

    def __init__(self, top_k: int, num_targets: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k
        self.num_targets = max(1, int(num_targets))

    def update(
        self,
        preds: torch.Tensor,
        is_spam_cand: torch.Tensor,
        is_spam_label: torch.Tensor,
    ) -> None:
        k = min(self.top_k, preds.size(1))
        topk_idx = torch.topk(preds, k, dim=1)[1]
        spam_in_topk = is_spam_cand.gather(1, topk_idx)  # (B, k) bool
        self._accumulate(spam_in_topk, is_spam_label.bool())

    def _accumulate(
        self, spam_in_topk: torch.Tensor, is_spam_label: torch.Tensor
    ) -> None:
        raise NotImplementedError


class SpamHitRate(CustomSpamMetric):
    """SH@k: fraction of non-target eval examples whose top-k contains a target."""

    def _accumulate(
        self, spam_in_topk: torch.Tensor, is_spam_label: torch.Tensor
    ) -> None:
        nonspam = ~is_spam_label
        hits = spam_in_topk.any(dim=1) & nonspam
        self.metric_values += hits.sum().item()
        self.total_values += nonspam.sum().item()  # denom: # non-target examples


class AvgSpamItems(CustomSpamMetric):
    """ASI@k: mean over users of |top-k ∩ I_t| / min(|I_t|, k) on non-target examples.

    Numerator sums ``|top-k ∩ I_t| / min(|I_t|, k)`` over non-target examples;
    denominator is the total example count |U| (target-labelled examples add 0 to
    the numerator but are counted here, per the max{1, ...} convention).
    """

    def _accumulate(
        self, spam_in_topk: torch.Tensor, is_spam_label: torch.Tensor
    ) -> None:
        nonspam = ~is_spam_label
        cap = max(1, min(self.num_targets, self.top_k))
        per_example = spam_in_topk.sum(dim=1).float() / cap
        per_example = per_example * nonspam.float()  # zero out target-labelled
        self.metric_values += per_example.sum().item()
        self.total_values += is_spam_label.numel()  # denom: |U| (all examples)


## Evaluators

class Evaluator:
    def __init__(self, metrics: Dict[str, Metric], *args, **kwargs):
        self.metrics = metrics

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: torch.device):
        for metric in self.metrics.values():
            metric.to(device=device)


class RetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics.
    It takes model outputs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        should_sample_negatives_from_vocab: bool = True,
        num_negatives: int = 500,
        placeholder_token_buffer: int = 100,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.should_sample_negatives_from_vocab = should_sample_negatives_from_vocab
        self.num_negatives = num_negatives
        self.placeholder_token_buffer = placeholder_token_buffer

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        num_of_samples = query_embeddings.shape[0]
        num_of_candidates = key_embeddings.shape[0]

        if self.should_sample_negatives_from_vocab:
            inbatch_negatives = self.sample_negative_ids_from_vocab(
                num_of_samples=num_of_samples,
                num_of_candidates=num_of_candidates,
                num_negatives=self.num_negatives,
            )
            # we +1 here because we need to include the positive sample
            num_of_candidates = self.num_negatives + 1
            pos_embeddings = key_embeddings[labels]
            key_embeddings = key_embeddings[inbatch_negatives]
            # key_embeddings shape: (bsz, num_negatives+1, emb_dim)
            key_embeddings = torch.cat(
                [pos_embeddings.unsqueeze(1), key_embeddings], dim=1
            )
            # the positive index will always be 0 because the pos embedding will always be the first one.
            labels = torch.zeros(num_of_samples).long()

        # following examples from https://lightning.ai/docs/torchmetrics/stable/retrieval/precision.html
        # indexes refers to the mask of the labels
        indexes = torch.arange(0, query_embeddings.shape[0])
        expanded_indexes = (
            indexes.unsqueeze(-1).expand(num_of_samples, num_of_candidates).reshape(-1)
        )

        if self.should_sample_negatives_from_vocab:
            preds = (
                torch.mul(
                    query_embeddings.unsqueeze(1).expand_as(key_embeddings),
                    key_embeddings,
                )
                .sum(-1)
                .reshape(-1)
            )
        else:
            preds = torch.mm(query_embeddings, key_embeddings.t()).reshape(-1)

        target = torch.zeros(num_of_samples, num_of_candidates).bool()
        target[torch.arange(num_of_samples), labels] = True
        target = target.reshape(-1)

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

    # this method samples random negative samples from the whole vocab
    def sample_negative_ids_from_vocab(
        self,
        num_of_samples: int,
        num_of_candidates: int,
        num_negatives: int,
    ) -> torch.Tensor:
        # num_of_samples: batch size
        # num_of_candidates: number of total vocabs
        # num_negatives: number of negative samples

        # we do randint to accelerate the negative sampling
        # this could have collision with positive pairs but the chance is very low

        # TODO (Clark): in the future we might need to have non-collision negative sampling
        # when K in top-k is very small (e.g., hits@1) and num_negatives is very large
        negative_candidates = torch.randint(
            self.placeholder_token_buffer,
            num_of_candidates,
            (num_of_samples, num_negatives),
        )

        return negative_candidates


class SIDRetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics for semantic IDs.
    It takes model outputs in semantic IDs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        spam_metrics: Optional[Dict[str, CustomSpamMetric]] = None,
        forget_manifest_path: Optional[str] = None,
        semantic_id_path: Optional[str] = None,
        num_hierarchies: Optional[int] = None,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }

        # Optional spam-exposure metrics (SH@k, ASI@k). They require the spam
        # target set I_t as semantic IDs, loaded from the poison forget_manifest
        # and the semantic-ID tensor. If either is missing they are skipped, so
        # clean/unpoisoned runs without a manifest simply don't emit them.
        self.spam_target_sids: Optional[torch.Tensor] = None
        self.num_spam_targets: int = 0
        if spam_metrics:
            self.spam_target_sids, self.num_spam_targets = self._load_spam_target_sids(
                forget_manifest_path, semantic_id_path, num_hierarchies
            )
            if self.spam_target_sids is not None:
                for metric_name, metric_object in spam_metrics.items():
                    for top_k in top_k_list:
                        self.metrics[f"{metric_name}@{top_k}"] = metric_object(
                            top_k=top_k,
                            num_targets=self.num_spam_targets,
                            sync_on_compute=False,
                            compute_with_cache=False,
                        )
            else:
                print(
                    "[SIDRetrievalEvaluator] spam_metrics requested but no spam "
                    f"targets loaded (forget_manifest_path={forget_manifest_path!r}, "
                    f"semantic_id_path={semantic_id_path!r}); SH/ASI skipped."
                )

    @staticmethod
    def _load_spam_target_sids(
        forget_manifest_path: Optional[str],
        semantic_id_path: Optional[str],
        num_hierarchies: Optional[int],
    ):
        """Return (target_sids [T, H] long, num_targets) or (None, 0).

        ``forget_manifest_path`` provides ``target_items`` (I_t); the semantic-ID
        tensor (``[num_hierarchies, num_items]``, raw per-hierarchy codes) maps
        each target item to its code tuple — the same space as ``generated_ids``.
        """
        if not forget_manifest_path or not os.path.isfile(forget_manifest_path):
            return None, 0
        if not semantic_id_path or not os.path.isfile(semantic_id_path):
            return None, 0
        with open(forget_manifest_path, encoding="utf-8") as f:
            targets = json.load(f).get("target_items", [])
        if not targets:
            return None, 0
        sem = torch.load(semantic_id_path, map_location="cpu", weights_only=False)
        if not torch.is_tensor(sem) or sem.ndim != 2:
            return None, 0
        n_items = sem.shape[1]
        idx = torch.tensor([t for t in targets if 0 <= int(t) < n_items], dtype=torch.long)
        if idx.numel() == 0:
            return None, 0
        h = int(num_hierarchies) if num_hierarchies else sem.shape[0]
        h = min(h, sem.shape[0])
        target_sids = sem[:h, idx].t().contiguous().long()  # (T, H)
        return target_sids, int(target_sids.shape[0])

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        labels = labels.reshape(batch_size, 1, num_hierarchies)
        preds = marginal_probs.reshape(-1)

        # check if the generated IDs contain the labels
        # if so, we get the coordinates of the matched IDs
        matched_id_coord = torch.all((generated_ids == labels), dim=2).nonzero()

        # we initialize the ground truth as all false
        target = torch.zeros(batch_size, num_candidates).bool()

        # we set the matched IDs to true if they are in the generated IDs
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)
        expanded_indexes = (
            torch.arange(batch_size)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        for _, metric_object in self.metrics.items():
            if isinstance(metric_object, CustomSpamMetric):
                continue
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

        # Spam-exposure metrics: detect target items among the generated
        # candidates and among the labels (to exclude target-labelled examples).
        if self.spam_target_sids is not None:
            st = self.spam_target_sids.to(generated_ids.device)  # (T, H)
            # candidate is a target iff its full code tuple matches any target sid
            is_spam_cand = (
                (generated_ids.unsqueeze(2) == st.view(1, 1, -1, num_hierarchies))
                .all(dim=3)
                .any(dim=2)
            )  # (B, C) bool
            lbl = labels.reshape(batch_size, num_hierarchies)
            is_spam_label = (
                (lbl.unsqueeze(1) == st.unsqueeze(0)).all(dim=2).any(dim=1)
            )  # (B,) bool
            preds_2d = marginal_probs.reshape(batch_size, num_candidates)
            for _, metric_object in self.metrics.items():
                if isinstance(metric_object, CustomSpamMetric):
                    metric_object.update(
                        preds_2d,
                        is_spam_cand.to(preds.device),
                        is_spam_label.to(preds.device),
                    )
