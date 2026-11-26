"""Catalog-normalized JS distance to a clean retraining reference.

Evaluates the *same* target-labeled forget histories and history-length-matched
retain histories under the clean, poisoned, and unlearned TIGER checkpoints.
Item probabilities are the teacher-forced SID probabilities normalized over
the valid item catalog. This avoids a beam-search truncation bias.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any

import hydra
import rootutils
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.unlearning.deletion_spec import (  # noqa: E402
    load_forget_manifest, load_target_items, resolve_forget_manifest_path,
)
from src.models.modules.semantic_id.tiger_unlearning_module import (  # noqa: E402
    TigerUnlearningModule, _build_finite_loader,
)
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402


def js_divergence(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """JS in nats for two normalized log distributions on one catalog."""
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    p, q = log_p.exp(), log_q.exp()
    value = 0.5 * (torch.where(p > 0, p * (log_p - log_m), 0.0).sum()
                   + torch.where(q > 0, q * (log_q - log_m), 0.0).sum())
    return float(value)


def _row(model_input: Any, index: int, batch_size: int) -> Any:
    def take(value):
        if isinstance(value, torch.Tensor) and value.ndim and value.size(0) == batch_size:
            return value[index:index + 1]
        if isinstance(value, dict):
            return {key: take(item) for key, item in value.items()}
        return value
    return dataclasses.replace(model_input, **{
        field.name: take(getattr(model_input, field.name))
        for field in dataclasses.fields(model_input)
    })


def _queries(loader, target_codes: torch.Tensor | None, limit: int) -> list[Any]:
    rows = []
    for model_input, label_data in loader:
        n = int(model_input.mask.size(0))
        codes = next(iter(label_data.labels.values())).reshape(n, -1)
        if target_codes is None:
            chosen = range(n)
        else:
            chosen = torch.nonzero((codes == target_codes).all(dim=1)).flatten().tolist()
        for index in chosen:
            rows.append(_row(model_input, index, n))
            if len(rows) >= limit:
                return rows
    return rows


def _match_retain(forget: list[Any], retain: list[Any]) -> list[Any]:
    """Greedy matching without replacement on the number of history tokens."""
    pool = list(retain)
    matched = []
    for query in forget:
        length = int(query.mask.sum())
        index = min(range(len(pool)), key=lambda i: abs(int(pool[i].mask.sum()) - length))
        matched.append(pool.pop(index))
    return matched


def _load_state(model: TigerUnlearningModule, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    mismatch = model.load_state_dict(state, strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(
            f"{path}: checkpoint/model mismatch: missing={mismatch.missing_keys[:5]}, "
            f"unexpected={mismatch.unexpected_keys[:5]}")
    model.eval()


@torch.inference_mode()
def _catalog_log_probs(model: TigerUnlearningModule, query: Any,
                       codes: torch.Tensor, chunk: int, device: torch.device) -> torch.Tensor:
    """Reuse one encoder pass, then score the catalog in decoder chunks."""
    query = _to_device(query, device)
    features = {model.feature_to_model_input_map.get(key, key): value
                for key, value in query.transformed_sequences.items()}
    encoded, encoder_mask = model.encoder_forward_pass(
        attention_mask=query.mask, **features)
    if encoded.size(0) != 1:
        raise ValueError("expected one history at a time")
    result = torch.empty(codes.size(0), dtype=torch.float64)
    for start in range(0, codes.size(0), chunk):
        stop = min(start + chunk, codes.size(0))
        future = codes[start:stop].to(device)
        count = future.size(0)
        decoded = model.decoder_forward_pass(
            future_ids=future, encoder_output=encoded.expand(count, -1, -1),
            attention_mask_for_encoder=encoder_mask.expand(count, -1),
            use_cache=False)[:, :-1]
        lp = torch.zeros(count, device=device, dtype=torch.float32)
        for level in range(model.num_hierarchies):
            logits = model.decoder.decoder_mlp[level](decoded[:, level]).float()
            lp += torch.nn.functional.log_softmax(logits, dim=-1).gather(
                1, future[:, level:level + 1]).squeeze(1)
        result[start:stop] = lp.double().cpu()
    return result - torch.logsumexp(result, dim=0)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{
            field.name: _to_device(getattr(value, field.name), device)
            for field in dataclasses.fields(value)})
    return value


def run(cfg: DictConfig) -> dict[str, Any]:
    params = OmegaConf.to_container(cfg.unlearning, resolve=True)
    if not isinstance(params, dict):
        raise ValueError("missing unlearning config")
    checkpoint_keys = ("js_clean_ckpt", "js_poison_ckpt", "js_nau_ckpt",
                       "js_forget_repair_ckpt", "js_forget_only_ckpt", "js_scif_ckpt")
    checkpoints = {key.removeprefix("js_").removesuffix("_ckpt"): str(params[key])
                   for key in checkpoint_keys}
    for path in checkpoints.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    out_path = str(params["js_output"])
    limit = int(params.get("js_n_queries", 16))
    chunk = int(params.get("js_item_chunk", 128))
    if limit <= 0 or chunk <= 0:
        raise ValueError("query limit and item chunk must be positive")
    torch.manual_seed(int(cfg.get("seed", 2)))

    with pipeline_launcher(cfg) as pipeline:
        model = pipeline.model
        if not isinstance(model, TigerUnlearningModule):
            raise TypeError(type(model))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        codes = model.codebooks.detach().cpu().long()
        manifest = load_forget_manifest(resolve_forget_manifest_path(str(cfg.paths.data_dir)))
        targets = sorted(load_target_items(manifest))
        if len(targets) != 1:
            raise ValueError(f"expected one spam target, got {targets}")
        target = targets[0]
        target_codes = codes[target]
        train_cfg = pipeline.datamodule.stage_to_config.get(TrainerFn.FITTING)
        data_dir = str(cfg.paths.data_dir)
        forget = _queries(_build_finite_loader(
            train_cfg, os.path.join(data_dir, "training_forget")),
            target_codes, limit)
        if not forget:
            raise RuntimeError("no target-labeled forget histories")
        retain_pool = _queries(_build_finite_loader(
            train_cfg, os.path.join(data_dir, "training_retain")),
            None, max(128, 8 * len(forget)))
        if len(retain_pool) < len(forget):
            raise RuntimeError("too few retain histories to match")
        retain = _match_retain(forget, retain_pool)
        queries = {"forget": forget, "retain": retain}
        # A fixed SID prefix group includes all items sharing the target's first
        # semantic code except the target itself. This does not depend on method.
        neighborhood = (codes[:, 0] == target_codes[0])
        neighborhood[target] = False
        if not bool(neighborhood.any()):
            raise RuntimeError("target has no depth-one SID neighbors")

        distributions: dict[str, dict[str, list[torch.Tensor]]] = {}
        for name, checkpoint in checkpoints.items():
            _load_state(model, checkpoint)
            distributions[name] = {}
            for group, group_queries in queries.items():
                distributions[name][group] = [
                    _catalog_log_probs(model, query, codes, chunk, device)
                    for query in group_queries]
        reference = distributions["clean"]
        rows = []
        for name in checkpoints:
            for group in ("forget", "retain"):
                values = [js_divergence(log_p, log_r)
                          for log_p, log_r in zip(distributions[name][group],
                                                  reference[group])]
                nbr_diff = [abs(float(log_p[neighborhood].exp().sum())
                                - float(log_r[neighborhood].exp().sum()))
                            for log_p, log_r in zip(distributions[name][group],
                                                    reference[group])]
                rows.append({"method": name, "group": group, "js_mean": sum(values) / len(values),
                             "js_per_query": values, "neighborhood_mass_abs_mean":
                             sum(nbr_diff) / len(nbr_diff), "n_queries": len(values)})
        output = {"dataset": str(params["js_dataset"]), "seed": int(cfg.get("seed", 2)),
                  "stratum": "mid", "target_item": target,
                  "catalog_items": int(codes.size(0)), "sid_neighbor_items": int(neighborhood.sum()),
                  "normalization": "teacher-forced item probabilities conditioned on valid catalog",
                  "forget_history_lengths": [int(q.mask.sum()) for q in forget],
                  "retain_history_lengths": [int(q.mask.sum()) for q in retain],
                  "checkpoints": checkpoints, "rows": rows}
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as file:
            json.dump(output, file, indent=2)
        return output


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
