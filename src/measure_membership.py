"""Loss-based membership inference on the injected fake users.

Members are the fake users of the poisoned training data (`<data_dir>/training_forget`);
non-members are fresh fake users drawn by the same generator with another seed and the
same target (`scripts/make_mia_nonmembers.py`). Each user is scored by the teacher-forced
log-probability of its last item given its history. The membership AUC is the probability
that a random member scores higher than a random non-member: about 0.5 for a model that
never saw the fake users (the retrained model), higher for the poisoned model.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

import hydra
import rootutils
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.measure_neighborhood_performance import _stack  # noqa: E402
from src.measure_retraining_js import _load_state, _row, _to_device  # noqa: E402
from src.models.modules.semantic_id.tiger_unlearning_module import (  # noqa: E402
    TigerUnlearningModule, _build_finite_loader,
)
from src.data.loading.components.collate_functions import collate_fn_train  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402


def _rows(loader: Any) -> list[tuple[Any, torch.Tensor]]:
    rows = []
    for model_input, label_data in loader:
        count = int(model_input.mask.size(0))
        labels = next(iter(label_data.labels.values())).reshape(count, -1)
        rows.extend((_row(model_input, i, count), labels[i].long()) for i in range(count))
    return rows


@torch.inference_mode()
def _logprobs(model: TigerUnlearningModule, rows: list, batch_size: int,
              device: torch.device) -> list[float]:
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs = _to_device(_stack([r[0] for r in batch]), device)
        labels = torch.stack([r[1] for r in batch]).to(device)
        out.extend(model._teacher_forced_log_prob(inputs, labels).float().cpu().tolist())
    return out


def _auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC: P(member score > non-member score), ties count one half."""
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def run(cfg: DictConfig) -> dict[str, Any]:
    params = OmegaConf.to_container(cfg.unlearning, resolve=True)
    keys = tuple(k for k in ("clean", "poison") if f"js_{k}_ckpt" in params)
    keys += tuple(sorted(k[3:-5] for k in params
                         if k.startswith("js_") and k.endswith("_ckpt") and k[3:-5] not in keys))
    checkpoints = {k: str(params[f"js_{k}_ckpt"]) for k in keys}
    member_dir = str(params.get("mia_member_dir") or os.path.join(str(cfg.paths.data_dir), "training_forget"))
    nonmember_dir = str(params["mia_nonmember_dir"])
    batch_size = int(params.get("scope_batch_size", 16))
    with pipeline_launcher(cfg) as pipeline:
        model = pipeline.model
        if not isinstance(model, TigerUnlearningModule):
            raise TypeError(type(model))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        load_cfg = deepcopy(pipeline.datamodule.stage_to_config[TrainerFn.FITTING])
        load_cfg.collate_fn = collate_fn_train
        load_cfg.batch_size_per_device = 32
        members = _rows(_build_finite_loader(load_cfg, member_dir))
        nonmembers = _rows(_build_finite_loader(load_cfg, nonmember_dir))
        # optional control: a second, independent non-member set scored against the first
        # (must give AUC ~0.5 for every model; anything else means the pipeline is biased)
        control_dir = params.get("mia_control_dir")
        control = _rows(_build_finite_loader(load_cfg, str(control_dir))) if control_dir else None
        result = {"dataset": str(params.get("js_dataset", "")), "seed": int(cfg.get("seed", 2)),
                  "stratum": str(params.get("scope_stratum", "mid")),
                  "member_dir": member_dir, "nonmember_dir": nonmember_dir,
                  "n_members": len(members), "n_nonmembers": len(nonmembers),
                  "score": "teacher-forced log p(last item | history)",
                  "checkpoints": checkpoints, "rows": [],
                  # each row's label SID, so AUCs can be split by what the scored item is
                  "member_label_sids": [r[1].tolist() for r in members],
                  "nonmember_label_sids": [r[1].tolist() for r in nonmembers]}
        for name, ckpt in checkpoints.items():
            _load_state(model, ckpt)
            model._make_deterministic(is_training=False)
            pos = _logprobs(model, members, batch_size, device)
            neg = _logprobs(model, nonmembers, batch_size, device)
            row_ctrl = {}
            if control is not None:
                ctl = _logprobs(model, control, batch_size, device)
                row_ctrl = {"control_auc_nonmember_vs_control": _auc(neg, ctl), "control_logprob": ctl}
            result["rows"].append({"method": name, "auc": _auc(pos, neg), **row_ctrl,
                                   "member_mean_logprob": sum(pos) / len(pos),
                                   "nonmember_mean_logprob": sum(neg) / len(neg),
                                   "member_logprob": pos, "nonmember_logprob": neg})
        out = str(params["scope_output"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)
        return result


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
