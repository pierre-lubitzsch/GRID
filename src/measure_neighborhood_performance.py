"""Held-out SH@10 and NDCG@10 by relation to a spam target's SID.

The same sampled test histories are evaluated under every checkpoint with the
model's usual beam generation. The neighbor group has a true next item sharing
the target's first semantic token; the control group shares no first token.
The target itself is excluded from both groups, as in the paper's SH@10.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
from copy import deepcopy
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
from src.measure_retraining_js import _load_state, _row, _to_device  # noqa: E402
from src.models.modules.semantic_id.tiger_unlearning_module import (  # noqa: E402
    TigerUnlearningModule, _build_finite_loader,
)
from src.data.loading.components.collate_functions import collate_fn_train  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402


def _reservoir_add(pool: list[Any], row: Any, seen: int, cap: int,
                   rng: random.Random) -> None:
    if len(pool) < cap:
        pool.append(row)
    else:
        position = rng.randrange(seen)
        if position < cap:
            pool[position] = row


def _test_groups(loader: Any, target_code: torch.Tensor, max_neighbors: int,
                 seed: int, all_unrelated: bool = False, is_neighbor: Any = None
                 ) -> tuple[dict[str, list[tuple[Any, torch.Tensor]]], dict[str, int]]:
    """Held-out rows split by the true next item's relation to the target.

    Default: a uniform reservoir of unrelated rows, length-matched one-to-one to
    the neighbor rows. That controls history length but leaves the control group
    the same size as the neighbor group -- a few hundred rows, which is noisy,
    and the matching selects easier-than-average queries (the clean model scores
    5.1 on the matched sample against 3.8 on the whole test set).

    all_unrelated=True instead keeps EVERY prefix-disjoint row: no cap, no
    matching. The control group becomes the real population rather than a sample,
    at the cost of no longer controlling for history length -- so the neighbor
    and unrelated numbers are then each comparable ACROSS models, but the gap
    between the two groups within a model also reflects their length profiles.
    """
    rng = random.Random(seed)
    # "target" keeps EVERY held-out row whose true next item IS the promoted item:
    # the legitimate demand for it, which retraining still serves and unlearning
    # should not destroy. Few rows (0 to 12 per seed), so no cap and no sampling.
    pools: dict[str, list[tuple[Any, torch.Tensor]]] = {"neighbor": [], "unrelated": [], "target": []}
    seen = {"neighbor": 0, "unrelated": 0, "target": 0}
    for model_input, label_data in loader:
        count = int(model_input.mask.size(0))
        labels = next(iter(label_data.labels.values())).reshape(count, -1)
        for index in range(count):
            label = labels[index].long()
            if torch.equal(label, target_code):
                seen["target"] += 1
                pools["target"].append((_row(model_input, index, count), label))
                continue
            if is_neighbor is None:
                group = "neighbor" if label[0] == target_code[0] else "unrelated"
            else:
                group = "neighbor" if is_neighbor(label) else "unrelated"
            seen[group] += 1
            if all_unrelated and group == "unrelated":
                pools[group].append((_row(model_input, index, count), label))
                continue
            cap = max_neighbors if group == "neighbor" else max(2048, 8 * max_neighbors)
            _reservoir_add(pools[group], (_row(model_input, index, count), label),
                           seen[group], cap, rng)
    if not pools["neighbor"] and not getattr(is_neighbor, "allow_empty", False):
        raise RuntimeError("no held-out examples with a neighbor as the true next item")
    if all_unrelated:
        return pools, seen
    available = pools["unrelated"]
    if len(available) < len(pools["neighbor"]):
        raise RuntimeError("too few unrelated controls")
    matched = []
    for query, _label in pools["neighbor"]:
        length = int(query.mask.sum())
        index = min(range(len(available)),
                    key=lambda i: abs(int(available[i][0].mask.sum()) - length))
        matched.append(available.pop(index))
    pools["unrelated"] = matched
    return pools, seen


def _stack(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {key: _stack([value[key] for value in values]) for key in first}
    if dataclasses.is_dataclass(first):
        return dataclasses.replace(first, **{
            field.name: _stack([getattr(value, field.name) for value in values])
            for field in dataclasses.fields(first)})
    if first is None:
        return None
    if isinstance(first, list):
        return sum(values, [])
    raise TypeError(f"cannot stack {type(first)}")


@torch.inference_mode()
def _metrics(model: TigerUnlearningModule, rows: list[tuple[Any, torch.Tensor]],
             target_code: torch.Tensor, batch_size: int,
             device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "ndcg_at_10": None, "sh_at_10": None, "recall_at_10": None,
                "ndcg_per_query": [], "sh_per_query": [], "rank_per_query": [],
                "label_logprob_per_query": []}
    ndcgs, hits, ranks, logps = [], [], [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs = _to_device(_stack([row[0] for row in batch]), device)
        labels = torch.stack([row[1] for row in batch]).to(device)
        features = {model.feature_to_model_input_map.get(key, key): value
                    for key, value in inputs.transformed_sequences.items()}
        generated, scores = model.generate(attention_mask=inputs.mask, **features)
        k = min(10, generated.size(1))
        order = scores.topk(k, dim=1).indices
        top = generated.gather(1, order.unsqueeze(-1).expand(-1, -1, generated.size(-1)))
        relevant = (top == labels[:, None, :]).all(dim=-1)
        spam = (top == target_code.to(device)[None, None, :]).all(dim=-1)
        discount = (1 / torch.log2(torch.arange(k, device=device, dtype=torch.float32) + 2))
        ndcgs.extend((relevant.float() * discount).sum(dim=1).cpu().tolist())
        hits.extend(spam.any(dim=1).int().cpu().tolist())
        # 1-based position of the true next item in the top-k, 0 if absent
        pos = torch.where(relevant.any(dim=1), relevant.float().argmax(dim=1) + 1,
                          torch.zeros_like(relevant[:, 0], dtype=torch.long))
        ranks.extend(pos.cpu().tolist())
        # teacher-forced log p(true next item | history): continuous, so it still
        # resolves differences when the item is outside every model's top 10
        logps.extend(model._teacher_forced_log_prob(inputs, labels).float().cpu().tolist())
    return {"n": len(rows), "ndcg_at_10": sum(ndcgs) / len(ndcgs),
            "sh_at_10": sum(hits) / len(hits),
            "recall_at_10": sum(r > 0 for r in ranks) / len(ranks),
            "ndcg_per_query": ndcgs, "sh_per_query": hits,
            "rank_per_query": ranks, "label_logprob_per_query": logps}


def _neighbor_predicate(neighbor_def: str, codes: torch.Tensor, target: int,
                        params: dict[str, Any]) -> tuple[Any, str]:
    """Neighbor test for a label SID, and a description of the definition.

    prefix1 (default): the next item shares the target's first SID token.
    prefix2: the next item shares the target's first two SID tokens.
    knn<K>: the next item is among the K items closest to the target by cosine
    similarity of the content embeddings (``scope_embedding_path``), independent of the SIDs.
    """
    if neighbor_def == "none":
        # Score EVERY held-out row (use with scope_all_unrelated=true): all non-target rows
        # land in "unrelated", and each group records the true next item's id per query
        # (items_per_query), so any neighborhood can be defined offline on the same
        # per-query arrays (random-SID control, paper/plotting/random_sid_table.py).
        def never(label):
            return False
        never.allow_empty = True
        return never, ("no split: every non-target row is in 'unrelated' with its next item "
                        "id (items_per_query); target: next item IS the target")
    if neighbor_def == "prefix1":
        return None, ("neighbor: test next item shares first SID token with target; "
                      "unrelated: shares no first token; target: next item IS the target")
    if neighbor_def == "prefix2":
        t = codes[target]
        return (lambda label: bool(torch.equal(label[:2], t[:2]))), (
            "neighbor: test next item shares the first two SID tokens with target; "
            "unrelated: all other non-target rows; target: next item IS the target")
    if neighbor_def.startswith("knn"):
        k = int(neighbor_def[3:])
        emb = torch.load(str(params["scope_embedding_path"]), map_location="cpu")
        if isinstance(emb, dict):
            row = {int(i): r for r, i in enumerate(emb["item_ids"].long().tolist())}
            emb = emb["embeddings"][torch.tensor([row[i] for i in range(codes.size(0))])]
        emb = torch.nn.functional.normalize(emb.float(), dim=1)
        sims = emb @ emb[target]
        sims[target] = float("-inf")
        knn = set(torch.topk(sims, k).indices.tolist())
        code_to_item = {tuple(codes[i].tolist()): i for i in range(codes.size(0))}
        return (lambda label: code_to_item.get(tuple(label.tolist())) in knn), (
            f"neighbor: test next item is among the {k} nearest items to the target by content "
            "embedding cosine; unrelated: all other non-target rows; target: next item IS the target")
    raise ValueError(f"unknown scope_neighbor_def {neighbor_def!r}")


def run(cfg: DictConfig) -> dict[str, Any]:
    params = OmegaConf.to_container(cfg.unlearning, resolve=True)
    # DISCOVER the methods from the overrides instead of hardcoding them. The
    # fixed tuple silently dropped any arm the launcher passed but this list did
    # not name -- `retain_only` was handed over as
    # +unlearning.js_retain_only_ckpt and scored nowhere, with exit 0 and no
    # warning, which is indistinguishable from the arm having no checkpoint.
    # Seed the order with the Figure-1 arms, but only those actually passed:
    # the off-mid strata carry the ablation arms alone, and a fixed tuple made
    # every one of those evals die with KeyError: 'js_forget_repair_ckpt'.
    keys = tuple(k for k in ("clean", "poison", "nau", "forget_repair",
                             "forget_only", "scif")
                 if f"js_{k}_ckpt" in params)
    keys += tuple(sorted(k[len("js_"):-len("_ckpt")] for k in params
                         if k.startswith("js_") and k.endswith("_ckpt")
                         and k[len("js_"):-len("_ckpt")] not in keys))
    checkpoints = {key: str(params[f"js_{key}_ckpt"]) for key in keys}
    for path in checkpoints.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    cap = int(params.get("scope_max_neighbors", 256))
    batch_size = int(params.get("scope_batch_size", 16))
    if cap <= 0 or batch_size <= 0:
        raise ValueError("scope_max_neighbors and scope_batch_size must be positive")
    torch.manual_seed(int(cfg.get("seed", 2)))
    with pipeline_launcher(cfg) as pipeline:
        model = pipeline.model
        if not isinstance(model, TigerUnlearningModule):
            raise TypeError(type(model))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        codes = model.codebooks.detach().cpu().long()
        targets = sorted(load_target_items(load_forget_manifest(
            resolve_forget_manifest_path(str(cfg.paths.data_dir)))))
        if len(targets) != 1:
            raise ValueError(f"expected one spam target, got {targets}")
        target = targets[0]
        target_code = codes[target]
        test_cfg = deepcopy(pipeline.datamodule.stage_to_config[TrainerFn.FITTING])
        test_cfg.collate_fn = collate_fn_train
        test_cfg.batch_size_per_device = 32
        test_dir = os.path.join(str(cfg.paths.data_dir), "testing")
        all_unrel = bool(params.get("scope_all_unrelated", False))
        neighbor_def = str(params.get("scope_neighbor_def", "prefix1"))
        is_neighbor, definition = _neighbor_predicate(neighbor_def, codes, target, params)
        groups, population = _test_groups(
            _build_finite_loader(test_cfg, test_dir), target_code, cap,
            int(cfg.get("seed", 2)), all_unrelated=all_unrel, is_neighbor=is_neighbor)
        result = {"dataset": str(params["js_dataset"]), "seed": int(cfg.get("seed", 2)),
                  "stratum": str(params.get("scope_stratum", "mid")), "target_item": target,
                  "neighbor_def": neighbor_def,
                  "definition": definition,
                  "unrelated_selection": ("all prefix-disjoint rows" if all_unrel
                                          else "length-matched sample, one per neighbor"),
                  "ranking": "standard TIGER beam generation, top 10 by marginal score",
                  "population_counts": population,
                  "selected_counts": {key: len(value) for key, value in groups.items()},
                  "checkpoints": checkpoints, "rows": []}
        if neighbor_def == "none":
            if not all_unrel:
                raise ValueError("scope_neighbor_def=none needs scope_all_unrelated=true")
            code_to_item = {tuple(codes[i].tolist()): i for i in range(codes.size(0))}
            result["semantic_id_path"] = str(cfg.get("semantic_id_path"))
            result["items_per_query"] = {
                group: [code_to_item[tuple(label.tolist())] for _row_input, label in rows]
                for group, rows in groups.items()}
        for name, checkpoint in checkpoints.items():
            _load_state(model, checkpoint)
            model._make_deterministic(is_training=False)
            for group, rows in groups.items():
                scores = _metrics(model, rows, target_code, batch_size, device)
                result["rows"].append({"method": name, "group": group, **scores})
        out = str(params["scope_output"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as file:
            json.dump(result, file, indent=2)
        return result


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
