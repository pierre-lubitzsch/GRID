"""Check that unlearning.sep_checkpoint leaves the generative L_sep and its gradients unchanged.

Loads a trained model, takes one real retain batch, uses check_n random items as separation
negatives, and compares the loss and all parameter gradients with and without checkpointing
(dropout off, so both passes must agree up to float rounding), plus peak memory.

    python -m scripts.check_sep_checkpoint experiment=tiger_unlearn_scif_flat data_dir=<dir> \
        semantic_id_path=<sids> ckpt_path=<ckpt> num_hierarchies=4 +unlearning.check_n=95
"""
import os
from copy import deepcopy

import hydra
import rootutils
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.components.unlearning.hvp import batch_to_device  # noqa: E402
from src.measure_retraining_js import _load_state  # noqa: E402
from src.models.modules.semantic_id.tiger_unlearning_module import _build_finite_loader  # noqa: E402
from src.data.loading.components.collate_functions import collate_fn_train  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    n_neg = int(cfg.unlearning.get("check_n", 95))
    with pipeline_launcher(cfg) as pipeline:
        model = pipeline.model
        _load_state(model, str(cfg.ckpt_path))
        device = torch.device("cuda")
        model = model.to(device).eval()
        load_cfg = deepcopy(pipeline.datamodule.stage_to_config[TrainerFn.FITTING])
        load_cfg.collate_fn = collate_fn_train
        load_cfg.batch_size_per_device = 32
        batch = next(iter(_build_finite_loader(load_cfg, os.path.join(str(cfg.paths.data_dir), "training"))))
        batch = batch_to_device(batch, device)
        g = torch.Generator(device="cpu").manual_seed(0)
        negs = set(torch.randperm(model.codebooks.size(0), generator=g)[:n_neg].tolist())
        results = []
        for ckpt in (False, True):
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            loss = model.compute_sep_loss(batch, negative_item_ids=negs, loss_type="generative",
                                          checkpoint=ckpt)
            loss.backward()
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            results.append((float(loss), grads, torch.cuda.max_memory_allocated() / 2**30))
        (l0, g0, m0), (l1, g1, m1) = results
        diff = max((g0[n] - g1[n]).abs().max().item() for n in g0)
        scale = max(g0[n].abs().max().item() for n in g0)
        print(f"negatives={n_neg}: loss {l0:.6f} vs {l1:.6f}; max |grad diff| {diff:.3e} (max |grad| {scale:.3e}); "
              f"same params with grad: {set(g0) == set(g1)}; peak memory {m0:.1f} GiB vs {m1:.1f} GiB")


if __name__ == "__main__":
    main()
