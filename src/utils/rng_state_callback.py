"""Save / restore RNG states inside Lightning checkpoints.

Lightning checkpoints carry model weights, optimizer states and the fit-loop
state, but NOT the random-number-generator states — so a run resumed via
``trainer.fit(ckpt_path=...)`` re-seeds from scratch and its stochastic
trajectory (dropout masks, samplers, torch randomness) is not the reproducible
continuation the checkpoint implies. This callback closes that gap: it stores
the python/numpy/torch/per-device-CUDA RNG states in every checkpoint and
restores them when a checkpoint is loaded for resume.

Notes
-----
* Restoring happens on whatever process loads the checkpoint; under DDP each
  rank loads the same file, so all ranks are restored to rank-0's saved state.
  Combined with ``DETERMINISTIC=1`` this makes a resumed run reproducible
  (re-running the same resume yields the same trajectory). It does NOT make the
  resumed run identical to a hypothetical uninterrupted run: the streaming
  TFRecord dataloader cannot seek, so the input stream restarts on resume.
* CUDA states are restored only for as many devices as are present at load
  time; a device-count mismatch degrades gracefully (extra saved states are
  ignored, missing ones left untouched).
"""

import logging
import random
from typing import Any, Dict

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

log = logging.getLogger(__name__)

_KEY = "rng_states"


class RngStateCallback(Callback):
    """Persist python/numpy/torch/CUDA RNG states in checkpoints for resume."""

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        states: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state_all()
        checkpoint[_KEY] = states

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        states = checkpoint.get(_KEY)
        if not states:
            log.info(
                "[RngStateCallback] checkpoint has no saved RNG states "
                "(pre-callback checkpoint) — resuming with fresh seeding."
            )
            return
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        cuda_states = states.get("cuda")
        if cuda_states and torch.cuda.is_available():
            n = min(len(cuda_states), torch.cuda.device_count())
            for i in range(n):
                torch.cuda.set_rng_state(cuda_states[i], device=i)
            if len(cuda_states) != torch.cuda.device_count():
                log.warning(
                    "[RngStateCallback] CUDA device count changed "
                    "(saved %d, present %d) — restored the first %d.",
                    len(cuda_states),
                    torch.cuda.device_count(),
                    n,
                )
        log.info("[RngStateCallback] RNG states restored from checkpoint.")
