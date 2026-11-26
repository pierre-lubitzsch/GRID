"""Collateral drift: do retained items that SHARE codes with a target lose more?

The third RQ-ID diagnosis, and the one `src/diagnose_rq_ids.py` does not answer.
That module reports where the forget signal sits (position-wise gradients) and how
many retained items structurally share a target's prefix (code sharing). Both are
STATIC: they describe the identifier space, not what unlearning did to it.

This measures the DRIFT. Over a fixed set of evaluation histories it scores, under
BOTH the pre-unlearning and post-unlearning model, the teacher-forced
log-probability of each probe item's semantic id, and reports the change bucketed
by how deeply that item shares a code prefix with a target:

    bucket d = the item shares exactly its first d codes with some target
               (d = 0 shares nothing, d = H means it IS a target)

Hypothesis under test -- over-shared codes cause collateral forgetting. If true,
drift falls monotonically with d and items at d >= 1 lose materially more than the
d = 0 control. If drift is flat in d, the coarse codes are NOT a shared damage
channel and unlearning is item-local.

WHY TEACHER FORCING, NOT GENERATION. Beam search only reports items that survive
the beam, so an item pushed out entirely reads as "absent" rather than as a
measured loss -- and the items most likely to move are exactly the borderline ones.
Teacher forcing scores every probe item on every query, so the two models are
exactly commensurable and no item can silently drop out of the measurement.

    python -m src.measure_collateral_drift experiment=tiger_unlearn_scif_flat \\
        data_dir=src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1 \\
        semantic_id_path=embeddings/beauty/merged_predictions_tensor.pt \\
        ckpt_path=<POISONED base ckpt> \\
        num_hierarchies=4 \\
        unlearning.drift_unlearned_ckpt=<UNLEARNED ckpt> \\
        unlearning.drift_n_queries=256 \\
        unlearning.drift_control_items=600

Writes ``${paths.output_dir}/collateral_drift.json``.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402
from src.models.modules.semantic_id.tiger_unlearning_module import (  # noqa: E402
    TigerUnlearningModule,
)

log = RankedLogger(__name__, rank_zero_only=True)
torch.set_float32_matmul_precision("medium")


def prefix_share_depth(
    codes: torch.Tensor, targets: List[int], n_levels: int
) -> torch.Tensor:
    """Longest prefix length each item shares with ANY target. ``[N]`` in 0..n_levels.

    ``codes`` is ``[N, H]`` of raw per-hierarchy codes. `cumprod` over the equality
    mask gives 1 up to the first mismatch, so its row sum is the shared prefix
    length -- and taking the max over targets means an item is bucketed by its
    CLOSEST target, which is the exposure that matters.
    """
    depth = torch.zeros(codes.size(0), dtype=torch.long)
    for t in targets:
        eq = codes[:, :n_levels] == codes[t, :n_levels].unsqueeze(0)
        run = torch.cumprod(eq.to(torch.long), dim=1)
        depth = torch.maximum(depth, run.sum(dim=1))
    return depth


def _load_state(model: torch.nn.Module, ckpt_path: str) -> None:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in obj:
        raise KeyError(f"{ckpt_path} has no 'state_dict' (keys: {sorted(obj)[:6]})")
    res = model.load_state_dict(obj["state_dict"], strict=False)
    if res.missing_keys:
        log.warning(f"{len(res.missing_keys)} missing keys, e.g. {res.missing_keys[:3]}")


@torch.no_grad()
def _score_probe(
    model: TigerUnlearningModule,
    batches: List[Any],
    probe_codes: torch.Tensor,
    device: torch.device,
    item_chunk: int,
    group_pos: Optional[Dict[str, List[int]]] = None,
) -> Tuple[torch.Tensor, int, Dict[str, float], Dict[str, torch.Tensor]]:
    """MEAN teacher-forced log-prob per query per probe item, plus group MASS.

    Returns ``([P], n_rows, {group: mean per-query log-mass},
    {group: [n_rows] per-query log-mass})``.

    THE FOURTH RETURN VALUE IS THE PER-QUERY VECTOR, kept so the caller can put
    an interval on the delta. The scalar mass is a mean over query rows, and
    collapsing it here used to throw away the only spread this probe has: the
    `target` group holds ONE item, so there is no item-level spread to resample,
    and the campaign has no replicate runs. Rows are the unit of resampling.

    THE MASS IS THE POINT OF THE THIRD RETURN VALUE. Averaging per-item log-probs
    and then differencing lets a group that half rises and half falls look
    unchanged, and it carries a per-item estimation floor (~0.75 nats at 6 query
    batches) that can exceed the effect. The mass

        log sum_{i in G} p(i | H)

    sums inside the log, so per-item noise averages out before the log is taken,
    and it is exactly the quantity L_neighborhood optimises -- the figure then
    measures what the objective acts on rather than a proxy for it.

    Computed PER QUERY and then averaged, never logsumexp of the averaged
    log-probs: those differ whenever queries vary, and only the former is "the
    share of probability this group holds, on average".

    For each query batch the probe items are scored in chunks; a probe item's
    codes are broadcast across the batch so one forward pass covers
    ``batch_size`` queries for one item.

    The per-query MEAN, not the sum. A sum scales with
    ``drift_n_queries * batch_size``, so two runs configured with different query
    budgets would produce drifts differing by an order of magnitude for reasons
    that have nothing to do with unlearning, and could not be put on one axis.
    The divisor is returned alongside so the sum is still recoverable.
    """
    total = torch.zeros(probe_codes.size(0), dtype=torch.float64)
    mass_tot: Dict[str, float] = {g: 0.0 for g in (group_pos or {})}
    mass_rows: Dict[str, List[torch.Tensor]] = {g: [] for g in (group_pos or {})}
    n_rows = 0
    from src.components.unlearning.hvp import batch_size as _bsz

    for bi, (model_input, _label) in enumerate(batches):
        bsz = _bsz((model_input, _label))
        if bi == 0:
            log.info(f"  batch size {bsz} x {probe_codes.size(0)} probe items "
                     f"= {bsz * probe_codes.size(0)} teacher-forced rows per batch")
        # [P, B] for this batch: needed because the mass is a logsumexp ACROSS
        # items within one query, which cannot be recovered from per-item means.
        lp_mat = torch.empty(probe_codes.size(0), bsz, dtype=torch.float64)
        for lo in range(0, probe_codes.size(0), item_chunk):
            hi = min(lo + item_chunk, probe_codes.size(0))
            for j in range(lo, hi):
                # future_ids must be [B, H] aligned to the query batch, so one
                # probe item's codes are broadcast across every query in it.
                fut = probe_codes[j].unsqueeze(0).to(device).expand(bsz, -1)
                lp = model._teacher_forced_log_prob(model_input, fut)
                row = lp.double().detach().cpu().reshape(-1)
                lp_mat[j] = row
                total[j] += float(row.sum())
            if (hi % 64) == 0:
                log.info(f"    batch {bi + 1}: {hi}/{probe_codes.size(0)} items")
        for g, pos in (group_pos or {}).items():
            if pos:
                # [B]: this group's log-mass for each query in the batch.
                per_q = torch.logsumexp(
                    lp_mat[torch.tensor(pos, dtype=torch.long)], dim=0)
                mass_tot[g] += float(per_q.sum())
                mass_rows[g].append(per_q.clone())
        n_rows += bsz
        log.info(f"  scored query batch {bi + 1}/{len(batches)}")
    return (total / max(1, n_rows), n_rows,
            {g: v / max(1, n_rows) for g, v in mass_tot.items()},
            {g: (torch.cat(v) if v else torch.empty(0, dtype=torch.float64))
             for g, v in mass_rows.items()})


def _bootstrap_mass_delta(
    base_rows: torch.Tensor,
    unl_rows: torch.Tensor,
    n_boot: int = 2000,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """Paired percentile bootstrap over QUERY ROWS for a group's mass delta.

    Both checkpoints are scored on the same `batches` list, in the same order,
    so row i of each vector is the same query and the difference is paired.
    The mean of the paired differences equals the reported `delta` exactly
    (both are sums over rows divided by the same n_rows), so the interval is an
    interval on the published number, not on a near relative of it.

    WHAT THIS DOES NOT COVER: one unlearning run per cell. The interval is the
    query-sampling spread of a fixed pair of checkpoints, so it says whether the
    probe resolves the gap, NOT whether re-running the unlearning would land in
    the same place. Seed-to-seed variation needs replicate runs.
    """
    if base_rows.numel() == 0 or base_rows.numel() != unl_rows.numel():
        return None
    d = (unl_rows - base_rows).double()
    n = d.numel()
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=gen)
    means = d[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"delta_se": float(means.std(unbiased=True)),
            "delta_ci95": [float(lo), float(hi)],
            "n_boot": int(n_boot)}


def run(cfg: DictConfig) -> Dict[str, Any]:
    with pipeline_launcher(cfg) as pipeline_modules:
        model = pipeline_modules.model
        if not isinstance(model, TigerUnlearningModule):
            raise TypeError(f"expected TigerUnlearningModule, got {type(model).__name__}")

        ucfg = OmegaConf.to_container(cfg.unlearning, resolve=True)
        base_ckpt = cfg.get("ckpt_path")
        unl_ckpt = ucfg.get("drift_unlearned_ckpt")
        if not base_ckpt or not unl_ckpt:
            raise ValueError(
                "need ckpt_path (pre-unlearning) and unlearning.drift_unlearned_ckpt"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_h = int(cfg.get("num_hierarchies", 4))
        n_levels = n_h - 1  # last digit is the dedup counter, not semantic

        # ---- probe set: every prefix-sharing item, plus a control sample ------
        codes = model.codebooks.detach().cpu().to(torch.long)
        if codes.size(1) != n_h:
            codes = codes.t()
        manifest = cfg.get("spam_forget_manifest") or os.path.join(
            cfg.paths.data_dir, "forget_manifest.json"
        )
        with open(manifest) as fh:
            targets = sorted(int(i) for i in json.load(fh)["target_items"])
        # NEIGHBOURHOOD DEFINITION.
        #
        # `prefix` (default) buckets by shared code-prefix length. That is the
        # channel the update travels down, but the bucket SIZE is a property of
        # the codebook, not of the experiment: on beauty it is 64 / 70 / 36 items
        # for the unpopular / mid / popular targets, so the neighbour bar means a
        # different sample size in each stratum and the three are not directly
        # comparable.
        #
        # `embedding` takes the k nearest catalog items by cosine in the
        # pre-quantization embedding space, so every stratum gets exactly k
        # neighbours. That is also the space L_n itself samples from
        # (coherence_neighbor_method=embedding), which makes the measurement
        # agree with what the term operates on.
        #
        # Implementation note: the group is encoded as a SYNTHETIC depth
        # (target -> n_levels, neighbour -> 1, control -> 0) so the bucket
        # schema, the JSON and the plot's grouping all keep working unchanged.
        nb_mode = str(ucfg.get("drift_neighbor_mode", "prefix")).lower()
        if nb_mode not in ("prefix", "embedding"):
            raise ValueError(f"drift_neighbor_mode must be prefix|embedding, "
                             f"got {nb_mode!r}")
        if nb_mode == "prefix":
            depth = prefix_share_depth(codes, targets, n_levels)
        else:
            from src.components.unlearning.neighborhood_sampler import (
                load_dense_embeddings, topk_embedding_neighbors,
            )
            emb_path = ucfg.get("embedding_path") or cfg.get("embedding_path")
            if not emb_path or not os.path.isfile(str(emb_path)):
                raise ValueError(
                    "drift_neighbor_mode=embedding needs unlearning.embedding_path "
                    f"to point at the pre-quantization embeddings (got {emb_path!r})")
            emb = load_dense_embeddings(str(emb_path))
            k_nb = int(ucfg.get("drift_neighbor_k", 32))
            nbrs: List[int] = []
            for t in targets:
                nbrs.extend(topk_embedding_neighbors(
                    t, emb, k_nb,
                    metric=str(ucfg.get("coherence_embedding_metric", "cosine")),
                    exclude_ids=set(targets)))
            nbrs = [i for i in dict.fromkeys(nbrs) if i not in set(targets)]
            depth = torch.zeros(codes.size(0), dtype=torch.long)
            for i in nbrs:
                if 0 <= i < depth.numel():
                    depth[i] = 1
            for t in targets:
                depth[t] = n_levels
            log.info(f"embedding neighbourhood: k={k_nb} per target -> "
                     f"{len(nbrs)} unique neighbours "
                     f"(metric={ucfg.get('coherence_embedding_metric', 'cosine')})")

        rng = random.Random(int(cfg.get("seed", 2)))
        sharing = [i for i in range(codes.size(0)) if depth[i] > 0 and i not in targets]

        # CONTROL POOL: items that share NO code prefix with any target.
        #
        # In prefix mode `depth` already is the shared-prefix depth, so depth==0
        # says exactly that. In EMBEDDING mode `depth` was overwritten above to
        # mean "in the top-k by cosine", and depth==0 then only says "outside the
        # neighbourhood" -- such an item can still share a coarse code with the
        # target, which makes it a poor control for a claim about shared codes.
        # So in embedding mode the pool is intersected with the TRUE prefix
        # depth, and controls are drawn from items that are both outside the
        # embedding neighbourhood and prefix-disjoint from the target.
        true_depth = (depth if nb_mode == "prefix"
                      else prefix_share_depth(codes, targets, n_levels))
        control_pool = [i for i in range(codes.size(0))
                        if depth[i] == 0 and true_depth[i] == 0 and i not in targets]
        n_ctrl = min(int(ucfg.get("drift_control_items", 600)), len(control_pool))
        control = rng.sample(control_pool, n_ctrl)
        probe = targets + sharing + control
        log.info(
            f"probe set: {len(targets)} target(s), {len(sharing)} "
            f"{'prefix-sharing' if nb_mode == 'prefix' else 'embedding-neighbour'}, "
            f"{len(control)} control drawn from {len(control_pool)} items that "
            f"share no code prefix with the target"
            + ("" if nb_mode == "prefix" else " and lie outside the top-k")
        )

        # ---- query histories: reuse the retain batches the module already builds
        train_dl_cfg = pipeline_modules.datamodule.stage_to_config.get(TrainerFn.FITTING)
        ctx = model._prepare_unlearning_context(
            unlearning_cfg=ucfg,
            train_dataloader_config=train_dl_cfg,
            data_dir=cfg.paths.data_dir,
            forget_subdir=cfg.get("forget_subdir", "training_forget"),
            retain_subdir=cfg.get("retain_subdir", "training_retain"),
            retain_subset_dir=os.path.join(cfg.paths.output_dir, "retain_subset"),
            semantic_id_path=cfg.get("semantic_id_path"),
            seed=int(cfg.get("seed", 2)),
            num_hierarchies=n_h,
            device=device,
        )
        n_q = int(ucfg.get("drift_n_queries", 8))

        # WHICH QUERIES THE PROBE SCORES OVER.
        #   retain (default) the retain TRAINING batches the unlearning context
        #                    already built. Backward compatible: every recorded
        #                    drift run used these.
        #   test             the held-out test split -- the same population
        #                    NDCG@10 and SH@10 are computed on, so drift is
        #                    measured where the reported metrics are measured,
        #                    and is not a statement about training rows.
        q_split = str(ucfg.get("drift_query_split", "retain")).lower()
        if q_split not in ("retain", "test"):
            raise ValueError(f"drift_query_split must be retain|test, got {q_split!r}")
        if q_split == "test":
            # NOT via datamodule.test_dataloader(): that calls setup(), which
            # needs a Lightning trainer attached (it reads trainer.world_size to
            # shard files) and this script never runs a Trainer. Build the loader
            # the same way the unlearning context builds its own -- from the
            # training dataloader config with the folder swapped -- so the test
            # split is read with identical collation and batch size.
            from src.models.modules.semantic_id.tiger_unlearning_module import (
                _build_finite_loader,
            )
            test_dir = os.path.join(str(cfg.paths.data_dir),
                                    str(cfg.get("test_subdir", "testing")))
            if not os.path.isdir(test_dir):
                raise SystemExit(f"drift_query_split=test but {test_dir} is missing")
            # Batches must be MOVED TO THE DEVICE. The retain path gets this for
            # free because _prepare_unlearning_context drains its loaders through
            # _drain_loader -> batch_to_device; a raw DataLoader yields CPU
            # tensors, which works on CPU and dies on GPU with "Expected all
            # tensors to be on the same device". That is exactly how the first
            # GPU run of this path failed while the CPU runs passed.
            from src.components.unlearning.hvp import batch_to_device
            loader = _build_finite_loader(
                base_train_cfg=train_dl_cfg, data_folder=test_dir)
            batches = []
            for b in loader:
                batches.append(batch_to_device(b, device))
                if len(batches) >= n_q:
                    break
            if not batches:
                raise SystemExit(f"no batches read from {test_dir}")
        else:
            batches = list(ctx["retain_batches"])[:n_q]
        log.info(f"scoring over {len(batches)} {q_split} query batches")

        probe_codes = codes[torch.tensor(probe)]
        # Group -> POSITIONS IN `probe` (not item ids): _score_probe indexes its
        # [P, B] matrix by position, and probe ids are catalog ids.
        group_pos: Dict[str, List[int]] = {"target": [], "nbr": [], "ctrl": []}
        for pos, it in enumerate(probe):
            d = int(depth[it])
            group_pos["target" if d >= n_levels
                      else ("ctrl" if d == 0 else "nbr")].append(pos)
        log.info("mass groups: " + ", ".join(f"{g}={len(v)}"
                                             for g, v in group_pos.items()))
        model = model.to(device).eval()

        _load_state(model, base_ckpt)
        model = model.to(device).eval()
        base, n_query_rows, mass_base, mass_rows_base = _score_probe(
            model, batches, probe_codes, device,
            int(ucfg.get("drift_item_chunk", 64)), group_pos)

        _load_state(model, unl_ckpt)
        model = model.to(device).eval()
        unl, _, mass_unl, mass_rows_unl = _score_probe(
            model, batches, probe_codes, device,
            int(ucfg.get("drift_item_chunk", 64)), group_pos)

        delta = (unl - base).tolist()
        # The ABSOLUTE levels, not only the difference. A delta of +0.09 nats
        # means something different on a log p of -4 than on -12, and the
        # clean-vs-poisoned comparison is unreadable without the level. Kept
        # additively so every existing consumer of this JSON is unaffected.
        base_lp, unl_lp = base.tolist(), unl.tolist()
        buckets: Dict[int, List[float]] = {}
        buckets_base: Dict[int, List[float]] = {}
        buckets_unl: Dict[int, List[float]] = {}
        for pos, item in enumerate(probe):
            d = int(depth[item])
            buckets.setdefault(d, []).append(delta[pos])
            buckets_base.setdefault(d, []).append(base_lp[pos])
            buckets_unl.setdefault(d, []).append(unl_lp[pos])

        rows = []
        for d in sorted(buckets):
            v = sorted(buckets[d])
            n = len(v)
            mean = sum(v) / n
            # Dispersion, not just the mean: the buckets are small at large d
            # (one target, tens of sharers) and a mean alone cannot say whether a
            # gap between depths is real.
            var = sum((x - mean) ** 2 for x in v) / (n - 1) if n > 1 else 0.0
            rows.append(
                {
                    "prefix_share_depth": d,
                    "n_items": n,
                    "mean_delta_logprob": mean,
                    "sd_delta_logprob": var ** 0.5,
                    "sem_delta_logprob": (var / n) ** 0.5 if n > 1 else 0.0,
                    "median": v[n // 2],
                    "p10": v[int(0.10 * (n - 1))],
                    "p90": v[int(0.90 * (n - 1))],
                    "min": v[0],
                    "max": v[-1],
                    "frac_negative": sum(1 for x in v if x < 0) / n,
                    "mean_logp_base": sum(buckets_base[d]) / n,
                    "mean_logp_unlearned": sum(buckets_unl[d]) / n,
                    "median_logp_base": sorted(buckets_base[d])[n // 2],
                    "median_logp_unlearned": sorted(buckets_unl[d])[n // 2],
                    "is_target_bucket": d >= n_levels,
                }
            )
        out = {
            "base_ckpt": os.path.abspath(base_ckpt),
            "unlearned_ckpt": os.path.abspath(unl_ckpt),
            "data_dir": str(cfg.paths.data_dir),
            "semantic_id_path": str(cfg.get("semantic_id_path")),
            "num_hierarchies": n_h,
            "semantic_levels": n_levels,
            "neighbor_mode": nb_mode,
            "neighbor_k": int(ucfg.get("drift_neighbor_k", 32)) if nb_mode == "embedding" else None,
            "target_items": targets,
            "n_query_batches": len(batches),
            "n_query_rows": n_query_rows,
            "units": "mean change in teacher-forced log p(sid | history) per query",
            "query_split": q_split,
            # Exact group probability mass: mean over queries of
            # log sum_{i in group} p(i | H), under each model, and the change.
            # Immune to the sign cancellation that afflicts a mean of per-item
            # deltas, and free of the per-item estimation floor.
            "group_log_mass": {
                g: {"base": mass_base.get(g), "unlearned": mass_unl.get(g),
                    "delta": (None if mass_base.get(g) is None
                              else mass_unl[g] - mass_base[g]),
                    "n_items": len(group_pos[g]),
                    **(_bootstrap_mass_delta(
                        mass_rows_base.get(g, torch.empty(0)),
                        mass_rows_unl.get(g, torch.empty(0)),
                        n_boot=int(ucfg.get("drift_n_boot", 2000)),
                        seed=int(cfg.get("seed", 2) or 2)) or {})}
                for g in group_pos
            },
            "by_prefix_share_depth": rows,
            # Per-item deltas, so a figure can show distributions rather than
            # three means, and so the buckets can be re-pooled across runs.
            "per_item": [
                {"item": int(it), "depth": int(depth[it]),
                 "delta_logprob": delta[pos],
                 "logp_base": base_lp[pos], "logp_unlearned": unl_lp[pos]}
                for pos, it in enumerate(probe)
            ],
        }
        log.info("=== collateral drift (mean change in teacher-forced log-prob) ===")
        log.info(f"{'depth':<7}{'n':>7}{'mean dlogp':>16}")
        for r in rows:
            log.info(f"{r['prefix_share_depth']:<7}{r['n_items']:>7}"
                     f"{r['mean_delta_logprob']:>16.5f}")

        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        p = os.path.join(cfg.paths.output_dir, "collateral_drift.json")
        with open(p, "w") as fh:
            json.dump(out, fh, indent=2)
        log.info(f"wrote {p}")
        return out


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
