"""Per-position gradient profile for MANY items, not just the spam target.

The recorded RQ-ID diagnosis reports, for one poisoned model, how forget-set
gradient strength is distributed across the RQ code positions c1..cH, and finds it
concentrated in the coarse codes. With one target per popularity band that is three
numbers, and `scripts/exposure_by_stratum.py` showed the identifier space varies far
more within a band than between bands -- so the ordering could be a property of
those three items rather than of the hierarchy.

Extending the SPAM version to more targets means one poisoned dataset and one
trained model per target (~5 h each), so the whole catalogue is out of reach; the
grid's 15 single-target models are what is affordable there
(`run_rqid_diag_sweep.sh`).

This asks the adjacent question that IS catalogue-scale, on a single model: for an
arbitrary item, where in the code hierarchy does ITS OWN training signal live? Each
item's supervised rows act as its own "forget set", so one backward pass per item
gives a c1..cH profile, and the distribution over thousands of items says whether
"coarse codes carry the signal" is a property of the hierarchy or of the targets.
It is a different quantity from spam influence -- item influence, not attack
influence -- and is reported as such.

    python -m src.item_position_profile experiment=tiger_unlearn_scif_flat \\
        data_dir=src/data/amazon_data/beauty \\
        semantic_id_path=embeddings/beauty/merged_predictions_tensor.pt \\
        ckpt_path=<clean or poisoned ckpt> num_hierarchies=4 \\
        unlearning.profile_max_items=2000

Writes ``${paths.output_dir}/item_position_profile.json``.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import hydra
import rootutils
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402
from src.components.unlearning.position_diagnostics import (  # noqa: E402
    per_position_gradient_report,
)
from src.components.unlearning.target_params import select_target_params  # noqa: E402
from src.models.modules.semantic_id.tiger_unlearning_module import (  # noqa: E402
    TigerUnlearningModule,
)

log = RankedLogger(__name__, rank_zero_only=True)
torch.set_float32_matmul_precision("medium")


def run(cfg: DictConfig) -> Dict[str, Any]:
    with pipeline_launcher(cfg) as pipeline_modules:
        model = pipeline_modules.model
        if not isinstance(model, TigerUnlearningModule):
            raise TypeError(f"expected TigerUnlearningModule, got {type(model).__name__}")
        ck = cfg.get("ckpt_path")
        if not ck:
            raise ValueError("ckpt_path is required")
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(obj["state_dict"], strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        ucfg = OmegaConf.to_container(cfg.unlearning, resolve=True)
        H = int(cfg.get("num_hierarchies", 4))
        train_dl_cfg = pipeline_modules.datamodule.stage_to_config.get(TrainerFn.FITTING)

        # One pass over the training split, bucketing batches by their LABEL item.
        # A batch is attributed to an item when that item is the supervised label,
        # which is the signal the position profile is about.
        ctx = model._prepare_unlearning_context(
            unlearning_cfg=ucfg,
            train_dataloader_config=train_dl_cfg,
            data_dir=cfg.paths.data_dir,
            forget_subdir=cfg.get("forget_subdir", "training_forget"),
            retain_subdir=cfg.get("retain_subdir", "training_retain"),
            retain_subset_dir=os.path.join(cfg.paths.output_dir, "retain_subset"),
            semantic_id_path=cfg.get("semantic_id_path"),
            seed=int(cfg.get("seed", 2)),
            num_hierarchies=H,
            device=device,
        )
        pool = list(ctx["retain_batches"])
        if not pool:
            raise SystemExit("no batches available to profile")
        params = select_target_params(model, ucfg.get("target_params", "tiger"))

        max_items = int(ucfg.get("profile_max_items", 2000))
        rng = random.Random(int(cfg.get("seed", 2)))
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        idx = idx[: min(max_items, len(idx))]
        log.info(f"profiling {len(idx)} batches over {H} code positions")

        rows: List[Dict[str, Any]] = []
        for n, bi in enumerate(idx):
            b = pool[bi]
            rep = per_position_gradient_report(
                model, [b], [b], params, num_hierarchies=H, eval_mode=True
            )
            rows.append({
                "batch_index": bi,
                "norms": [float(p["forget_grad_norm"]) for p in rep["positions"]],
            })
            if (n + 1) % 50 == 0:
                log.info(f"  {n + 1}/{len(idx)}")

        # Distribution of the per-position SHARE, which is what "the signal lives
        # in the coarse codes" actually claims -- absolute norms vary with batch
        # size and item frequency, shares do not.
        shares = defaultdict(list)
        argmax = defaultdict(int)
        for r in rows:
            tot = sum(r["norms"]) or 1.0
            for h, v in enumerate(r["norms"]):
                shares[h].append(v / tot)
            argmax[int(max(range(H), key=lambda h: r["norms"][h]))] += 1

        import statistics as st
        summary = []
        for h in range(H):
            v = shares[h]
            summary.append({
                "code": f"c{h + 1}",
                "mean_share": st.mean(v),
                "sd_share": st.stdev(v) if len(v) > 1 else 0.0,
                "median_share": st.median(v),
                "pct_strongest": 100.0 * argmax[h] / max(1, len(rows)),
            })
        out = {
            "ckpt_path": os.path.abspath(ck),
            "data_dir": str(cfg.paths.data_dir),
            "num_hierarchies": H,
            "n_profiled": len(rows),
            "by_position": summary,
        }
        log.info("=== per-item position profile (share of gradient norm) ===")
        log.info(f"{'code':<6}{'mean share':>12}{'sd':>10}{'% strongest':>14}")
        for s in summary:
            log.info(f"{s['code']:<6}{s['mean_share']:>12.4f}{s['sd_share']:>10.4f}"
                     f"{s['pct_strongest']:>13.1f}%")
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        p = os.path.join(cfg.paths.output_dir, "item_position_profile.json")
        json.dump(out, open(p, "w"), indent=2)
        log.info(f"wrote {p}")
        return out


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
