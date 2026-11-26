"""Validation-based epoch selection (early stopping) for unlearning.

Within one unlearning request, the model is scored on the validation split after
every epoch (one pass over the request's batches). The best epoch so far is kept
as a weight snapshot; once ``patience`` consecutive epochs fail to improve on it,
or ``max_epochs`` is reached, training stops and the best snapshot is restored.

Epochs are ranked by the paper's configuration-selection rule, applied per epoch:

  1. feasible first: tau_P = NDCG@10 / NDCG_ref@10 >= ``tau_floor``;
  2. smaller Delta_SH = |SH@10 - SH_ref@10| (distance to the reference exposure,
     so over-suppression counts as much as residual exposure);
  3. on a Delta_SH tie (within ``sh_tie_tol``), higher tau_P.

The infeasible epochs are ranked by the same (2, 3) key, so when no epoch reaches
the floor the least-exposed one still wins, as in the paper's fallback.

References (``reference``):
  * ``retrained``: SH_ref and NDCG_ref are the retrained model's VALIDATION values,
    read from a metrics.csv (``reference_metrics_csv``, last row, ``test/`` keys as
    written by scripts.eval_ckpt_on_test on a ``_valview``) or given explicitly
    (``ref_sh`` / ``ref_ndcg``).
  * ``original``: NDCG_ref is the starting (poisoned) model's validation NDCG,
    measured once before the first update; SH_ref is ``ref_sh`` (default 0.0).
    Needs no retrained model.
"""

from __future__ import annotations

import csv
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

SH_KEY = "SH@10"
NDCG_KEY = "ndcg@10"


def read_reference_metrics(path: str) -> Tuple[float, float]:
    """(SH@10, NDCG@10) from the last row of a metrics.csv that has them."""
    rows = [r for r in csv.DictReader(open(path))
            if r.get(f"test/{SH_KEY}") not in (None, "")]
    if not rows:
        raise ValueError(f"{path} has no test/{SH_KEY} row")
    r = rows[-1]
    return float(r[f"test/{SH_KEY}"]), float(r[f"test/{NDCG_KEY}"])


def build_val_batches(
    base_train_cfg: Any,
    data_folder: str,
    device: torch.device,
    batch_size: int = 64,
    max_rows: Optional[int] = None,
) -> List[Any]:
    """Drain the validation split into device-resident eval batches.

    Mirrors tiger_train_flat's val_dataloader_config: the training dataset_config
    (same SID map and preprocessing) with the eval collate ``collate_fn_train``,
    which labels each row with its last item.
    """
    from src.data.loading.components.collate_functions import collate_fn_train
    from src.models.modules.semantic_id.tiger_unlearning_module import (
        _build_finite_loader,
        _drain_loader,
    )

    cfg = deepcopy(base_train_cfg)
    cfg.collate_fn = partial(collate_fn_train)
    loader = _build_finite_loader(
        cfg, data_folder, batch_size_per_device_override=int(batch_size)
    )
    batches = _drain_loader(loader, device)
    if max_rows is not None:
        kept, n = [], 0
        for b in batches:
            if n >= int(max_rows):
                break
            kept.append(b)
            n += int(b[0].mask.shape[0])
        batches = kept
    return batches


def _snapshot_modes(model: nn.Module) -> List[Tuple[Any, bool, Optional[bool]]]:
    return [(m, m.training, getattr(m, "is_training", None)) for m in model.modules()]


def _restore_modes(snap: List[Tuple[Any, bool, Optional[bool]]]) -> None:
    for m, training, is_training in snap:
        m.training = training
        if is_training is not None:
            m.is_training = is_training


@torch.no_grad()
def evaluate_batches(model: nn.Module, batches: Sequence[Any]) -> Dict[str, float]:
    """Run the model's own eval_step over ``batches``; return evaluator metrics.

    Equivalent to one Lightning validation epoch (eval mode, deterministic decode),
    without a Trainer. The previous train/eval state of every submodule is
    restored afterwards, so the unlearning loop continues exactly as before.
    """
    from torchmetrics import MeanMetric

    snap = _snapshot_modes(model)
    model.eval()
    if hasattr(model, "_make_deterministic"):
        model._make_deterministic(is_training=False)
    try:
        model.evaluator.reset()
        loss_agg = MeanMetric().to(next(model.parameters()).device)
        for b in batches:
            model.eval_step(b, loss_agg)
        out = {k: float(m.compute()) for k, m in model.evaluator.metrics.items()}
        out["loss"] = float(loss_agg.compute())
        model.evaluator.reset()
    finally:
        _restore_modes(snap)
    return out


@dataclass
class EpochSelector:
    """Per-request early stopping on the validation selection rule."""

    model: nn.Module
    batches: Sequence[Any]
    ref_sh: float
    ref_ndcg: float
    tau_floor: float = 0.95
    sh_tie_tol: float = 0.0
    patience: int = 2
    max_epochs: int = 10
    min_epochs: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_epoch: Optional[int] = None
    _best_key: Optional[Tuple[float, float, float]] = None
    _best_state: Optional[Dict[str, torch.Tensor]] = None
    _since_best: int = 0

    def __post_init__(self) -> None:
        if self.ref_ndcg <= 0:
            raise ValueError(f"ref_ndcg must be > 0, got {self.ref_ndcg}")

    def score(self, metrics: Dict[str, float]) -> Dict[str, float]:
        tau = metrics[NDCG_KEY] / self.ref_ndcg
        return {"tau_P": tau, "delta_sh": abs(metrics[SH_KEY] - self.ref_sh),
                "feasible": float(tau >= self.tau_floor)}

    def _better(self, s: Dict[str, float]) -> bool:
        if self._best_key is None:
            return True
        b_feas, b_dsh, b_tau = self._best_key
        if s["feasible"] != b_feas:
            return s["feasible"] > b_feas
        if s["delta_sh"] < b_dsh - self.sh_tie_tol:
            return True
        if abs(s["delta_sh"] - b_dsh) <= self.sh_tie_tol:
            return s["tau_P"] > b_tau
        return False

    def __call__(self, epoch: int) -> bool:
        """Score ``epoch``; return True to stop the optimization loop."""
        t0 = time.time()
        metrics = evaluate_batches(self.model, self.batches)
        s = self.score(metrics)
        improved = epoch >= self.min_epochs and self._better(s)
        if improved:
            self._best_key = (s["feasible"], s["delta_sh"], s["tau_P"])
            self._best_state = {k: v.detach().clone()
                                for k, v in self.model.state_dict().items()}
            self.best_epoch = epoch
            self._since_best = 0
        elif epoch >= self.min_epochs:
            self._since_best += 1
        self.history.append({
            "epoch": epoch, SH_KEY: metrics[SH_KEY], NDCG_KEY: metrics[NDCG_KEY],
            "val_loss": metrics["loss"], **s, "best": improved,
            "eval_seconds": time.time() - t0,
        })
        log.info(
            "[val_select] epoch=%d SH@10=%.5f ndcg@10=%.5f tau_P=%.4f "
            "dSH=%.5f%s (best=%s, %d since best, %.1fs)",
            epoch, metrics[SH_KEY], metrics[NDCG_KEY], s["tau_P"], s["delta_sh"],
            "" if s["feasible"] else " [below tau floor]",
            self.best_epoch, self._since_best, time.time() - t0,
        )
        return self._since_best >= self.patience or epoch >= self.max_epochs

    def finalize(self) -> Dict[str, Any]:
        """Restore the best epoch's weights; return the selection record."""
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return {
            "best_epoch": self.best_epoch,
            "epochs_run": self.history[-1]["epoch"] if self.history else 0,
            "ref_sh": self.ref_sh, "ref_ndcg": self.ref_ndcg,
            "tau_floor": self.tau_floor, "sh_tie_tol": self.sh_tie_tol,
            "patience": self.patience, "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs, "history": self.history,
        }


# --------------------------------------------------------------------------
# Run-level selection (mode=run): choose ONE n_epochs for the whole sequence.
# --------------------------------------------------------------------------

def run_score(
    metrics: Dict[str, float],
    sh_poison: float,
    ndcg_poison: float,
    lambda_ndcg: float = 1.0,
    tau_floor: float = 0.95,
    sh_floor: float = 0.005,
) -> Dict[str, float]:
    """Trade-off score of a final unlearned model against the poisoned one.

        removal = (SH_P - SH_U) / max(SH_P, sh_floor)    signed: an INCREASE in
                                                         exposure is penalized
        tau     = NDCG_U / NDCG_P
        score   = removal + lambda_ndcg * (tau - 1)

    With lambda_ndcg = 1, removing 1% of the poisoned exposure is worth 1% of
    NDCG. ``sh_floor`` keeps the removal term finite where the attack barely
    landed (0.005 is the "landed" threshold used for PGR); those cells are not
    meaningful for spam removal anyway. ``feasible`` (tau >= tau_floor) ranks
    above the score.
    """
    removal = (sh_poison - metrics[SH_KEY]) / max(sh_poison, sh_floor)
    tau = metrics[NDCG_KEY] / ndcg_poison
    return {"removal": removal, "tau_vs_poison": tau,
            "score": removal + lambda_ndcg * (tau - 1.0),
            "feasible": float(tau >= tau_floor)}


def run_level_selection(
    *,
    model: nn.Module,
    batches: Sequence[Any],
    run_once: Any,
    seed: int,
    lambda_ndcg: float = 1.0,
    tau_floor: float = 0.95,
    sh_floor: float = 0.005,
    patience: int = 2,
    min_epochs: int = 1,
    max_epochs: int = 10,
    score_tol: float = 0.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Pick n_epochs for the whole request sequence on the validation split.

    For E = min_epochs, min_epochs+1, ...: restore the starting weights, reseed,
    run the full sequence with n_epochs=E (``run_once(E)``, which returns
    whatever the caller wants kept for the chosen E), and score the FINAL model
    once. Stops after ``patience`` values of E without improvement, or at
    ``max_epochs``; the best E's final weights are loaded back into ``model``.
    Ranking: feasible first, then the higher run_score. A larger E replaces the
    best only if its score is higher by MORE than ``score_tol``, so near-ties go
    to the fewest epochs. On beauty/mid seed 2 (2026-10-07) the same E=4 run
    from a different RNG start state moved validation NDCG by 2.6%, more than
    the score differences between E >= 3 (one sample, not a noise estimate;
    reruns from the SAME seed agree to ~0.3 tau_P points).
    """
    from lightning.pytorch import seed_everything

    start = {k: v.detach().clone() for k, v in model.state_dict().items()}
    origin = evaluate_batches(model, batches)
    sh_p, ndcg_p = origin[SH_KEY], origin[NDCG_KEY]
    log.info("[val_select:run] starting model: SH@10=%.5f ndcg@10=%.5f", sh_p, ndcg_p)

    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    since = 0
    for E in range(int(min_epochs), int(max_epochs) + 1):
        t0 = time.time()
        model.load_state_dict(start)
        seed_everything(int(seed), workers=False)
        kept = run_once(E)
        m = evaluate_batches(model, batches)
        s = run_score(m, sh_p, ndcg_p, lambda_ndcg, tau_floor, sh_floor)
        key = (s["feasible"], s["score"])
        improved = (best is None or key[0] > best["key"][0]
                    or (key[0] == best["key"][0]
                        and key[1] > best["key"][1] + float(score_tol)))
        if improved:
            best = {"key": key, "E": E, "kept": kept,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            since = 0
        else:
            since += 1
        history.append({"n_epochs": E, SH_KEY: m[SH_KEY], NDCG_KEY: m[NDCG_KEY],
                        "val_loss": m["loss"], **s, "best": improved,
                        "seconds": time.time() - t0})
        log.info("[val_select:run] n_epochs=%d SH@10=%.5f ndcg@10=%.5f removal=%.4f "
                 "tau=%.4f score=%.4f%s (best=%d, %d since best, %.0fs)",
                 E, m[SH_KEY], m[NDCG_KEY], s["removal"], s["tau_vs_poison"],
                 s["score"], "" if s["feasible"] else " [below tau floor]",
                 best["E"], since, time.time() - t0)
        if since >= int(patience):
            break

    model.load_state_dict(best["state"])
    return best["kept"], {
        "mode": "run", "best_n_epochs": best["E"],
        "sh_poison": sh_p, "ndcg_poison": ndcg_p, "lambda_ndcg": lambda_ndcg,
        "tau_floor": tau_floor, "sh_floor": sh_floor, "score_tol": score_tol,
        "patience": patience,
        "min_epochs": min_epochs, "max_epochs": max_epochs, "history": history,
    }
