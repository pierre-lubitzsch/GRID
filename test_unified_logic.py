"""Smoke test for unified_unlearn: no checkpoint, no data, mock model.

Verifies:
  - n_epochs correctly expands to steps = N * min(n_forget, n_retain)
  - gradient accumulation runs without errors for both q_retain > 1 and q_forget > 1
  - parameter actually moves between start and end
  - result dict has the expected keys/values
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Tuple

import torch
from torch import nn

# Monkey-patch the TIGER batch helpers to work with plain tuples (the mock
# batches in this test do not implement the SequentialModelInputData dataclass).
import src.components.unlearning.hvp as _hvp
import src.components.unlearning.unified as _unified

_hvp.batch_to_device = lambda batch, device: batch
_hvp.batch_size = lambda batch: int(batch[0].shape[0])
_unified.batch_to_device = _hvp.batch_to_device
_unified.batch_size = _hvp.batch_size

from src.components.unlearning.unified import unified_unlearn  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test_unified")


class MockTigerModel(nn.Module):
    """Minimum surface that unified_unlearn requires.

    Each batch is a tuple of two tensors (mimicking the (input, label) shape).
    The "loss" is a simple MSE between a learned bias and the batch tensor mean.
    """

    def __init__(self, dim: int = 4):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.bias_ = nn.Parameter(torch.zeros(1))

    def _batch_loss_from_model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        pred = self.lin(x) + self.bias_
        return ((pred - y) ** 2).mean()

    def _sequence_log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Stand-in for a log-prob — return -mean so minimizing this still does something.
        return -((self.lin(x) + self.bias_) ** 2).mean()

    def compute_sep_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *,
        negative_item_ids: Any,
        temperature: float,
    ) -> torch.Tensor:
        x, _ = batch
        return (self.lin(x) ** 2).mean() / max(float(temperature), 1e-3)


def _make_batches(n: int, dim: int = 4, batch_size: int = 8) -> list:
    return [(torch.randn(batch_size, dim), torch.randn(batch_size, dim)) for _ in range(n)]


def _check(label: str, n_forget: int, n_retain: int, n_epochs: int) -> None:
    log.info(
        "=" * 60 + "\n[%s] n_forget=%d n_retain=%d n_epochs=%d",
        label,
        n_forget,
        n_retain,
        n_epochs,
    )
    model = MockTigerModel(dim=4)
    forget_batches = _make_batches(n_forget)
    retain_batches = _make_batches(n_retain)

    p0 = model.bias_.detach().clone()
    info = unified_unlearn(
        model,
        forget_batches=forget_batches,
        retain_batches=retain_batches,
        n_epochs=n_epochs,
        lr=1e-2,
        device=torch.device("cpu"),
    )
    p1 = model.bias_.detach().clone()

    expected_steps = n_epochs * min(n_forget, n_retain)
    assert info["steps"] == expected_steps, (info["steps"], expected_steps)
    assert info["n_epochs"] == n_epochs, info["n_epochs"]
    assert info["optim_steps_per_pass"] == min(n_forget, n_retain), info
    assert info["q_forget"] >= 1 and info["q_retain"] >= 1
    assert (p1 - p0).abs().sum().item() > 0, "param did not change — optimizer never stepped"
    log.info(
        "[%s] OK: steps=%d q_forget=%d q_retain=%d mean_total_loss=%s param_delta=%.4g",
        label,
        info["steps"],
        info["q_forget"],
        info["q_retain"],
        info["mean_total_loss"],
        (p1 - p0).abs().sum().item(),
    )


def main() -> int:
    torch.manual_seed(0)

    # Realistic case from the user's run (n_retain >> n_forget).
    _check("retain_heavy", n_forget=4, n_retain=51, n_epochs=4)
    # Symmetric case (n_forget >> n_retain).
    _check("forget_heavy", n_forget=51, n_retain=4, n_epochs=4)
    # Balanced case.
    _check("balanced", n_forget=8, n_retain=8, n_epochs=4)
    # Steps fallback still works.
    log.info("=" * 60 + "\n[steps_fallback] n_epochs=None, steps=5")
    model = MockTigerModel()
    info = unified_unlearn(
        model,
        forget_batches=_make_batches(4),
        retain_batches=_make_batches(51),
        steps=5,
        n_epochs=None,
        lr=1e-2,
        device=torch.device("cpu"),
    )
    assert info["steps"] == 5
    assert info["n_epochs"] is None
    log.info("[steps_fallback] OK: steps=%d", info["steps"])

    log.info("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
