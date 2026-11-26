"""Predictive log-prob on REAL test labels, split by code-neighbourhood, for
three models: clean, poisoned and unlearned.

WHY THIS AND NOT measure_collateral_drift. That script scores a PROBE item set
counterfactually -- each probe item is force-fed as `future_ids` against
*training-retain* histories -- and reports only the DIFFERENCE between two
checkpoints. Both choices limit what it can say:

  * counterfactual mass answers "how much probability would this model put on
    item X for a typical user", not "how well does it predict what users
    actually did next";
  * a difference cannot show that the unlearned model LANDED near the clean one,
    only that it moved.

This measures, for every held-out test sample, the teacher-forced log-probability
of the item the user ACTUALLY consumed next, under each of the three models, and
groups the samples by how much code prefix their true label shares with the
forget target:

    neighbour  label shares 1..L-1 code levels with a forget target
    unrelated  label shares none

The forget-target group is deliberately ABSENT. Only 1 of 22,363 beauty test
samples has the forget target as its true next item (0.004%), so that bar would
be a single observation; counterfactual probe scoring is the right instrument
for the target itself, and measure_collateral_drift already provides it.

The result is a utility statement on real predictions: if unlearning damaged the
neighbourhood, the unlearned model loses log-probability on genuine next items
that merely share codes with the target, while the unrelated group -- 99.5% of
the test set -- stays put.

    STRATUM=mid ARM=<arm> python -m src.measure_label_logprob \\
      experiment=tiger_unlearn_scif_flat ...
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import hydra
import rootutils
import random as _rnd
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.measure_collateral_drift import _load_state, prefix_share_depth  # noqa: E402
from src.models.modules.semantic_id.tiger_generation_model import (  # noqa: E402
    SemanticIDEncoderDecoder,
)
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402

log = logging.getLogger(__name__)


def _subbatch(mi, ld, idx: List[int]):
    """A new (model_input, labels) pair holding only rows ``idx``.

    Needed because the probe scan is targeted: the neighbour SAMPLE group is
    0.5% of the test set, so finding it requires a full pass, but scoring 49
    probe items x 3 models on all 22k samples costs 3.3M teacher-forced rows.
    Selecting the rows first cuts that by ~30x.
    """
    import copy
    sel = torch.as_tensor(idx, dtype=torch.long)
    mi2 = copy.copy(mi)
    mi2.mask = mi.mask[sel]
    mi2.transformed_sequences = {k: v[sel] for k, v in mi.transformed_sequences.items()}
    ld2 = copy.copy(ld)
    ld2.labels = {k: v.reshape(mi.mask.size(0), -1)[sel] for k, v in ld.labels.items()}
    return mi2, ld2


def _probe_logprobs(model, batches, probe_codes, device, chunk: int):
    """Per-sample log p(probe_item | history) for every probe item.

    Returns a list of ``[B, P]`` tensors, one per batch. Unlike
    ``measure_collateral_drift._score_probe`` this keeps the PER-SAMPLE axis
    instead of averaging over queries, because the whole point here is to split
    the queries by what their own true label was.
    """
    out = []
    P = probe_codes.size(0)
    with torch.no_grad():
        for mi, _ld in batches:
            bsz = mi.mask.size(0)
            got = torch.empty(bsz, P, dtype=torch.float32)
            for lo in range(0, P, chunk):
                hi = min(lo + chunk, P)
                for j in range(lo, hi):
                    fut = probe_codes[j].unsqueeze(0).to(device).expand(bsz, -1)
                    got[:, j] = model._teacher_forced_log_prob(
                        mi, fut).detach().float().cpu()
            out.append(got)
    return out


def _label_logprobs(model, batches, device) -> List[torch.Tensor]:
    """Per-sample log p(true label | history) for each batch. [B] tensors."""
    out = []
    with torch.no_grad():
        for mi, ld in batches:
            fut = None
            for k in ld.labels:
                fut = ld.labels[k].reshape(mi.mask.size(0), -1)
            out.append(model._teacher_forced_log_prob(mi, fut).detach().float().cpu())
    return out


def _label_items(model, batches) -> List[torch.Tensor]:
    """Item id of each sample's true label. -1 when the code tuple is unknown."""
    out = []
    with torch.no_grad():
        for mi, ld in batches:
            fut = None
            for k in ld.labels:
                fut = ld.labels[k].reshape(mi.mask.size(0), -1)
            out.append(model._codes_to_item_ids(fut).detach().cpu())
    return out


def run(cfg: DictConfig) -> Dict[str, Any]:
    with pipeline_launcher(cfg) as pm:
        model = pm.model
        # Runs under experiment=tiger_train_flat, not the unlearn config: that
        # one defines only a TRAIN dataloader (val/test/predict are all None), so
        # get_dataloader(TESTING) raises KeyError there. tiger_train_flat has a
        # test loader over `testing/` with collate_fn_train, i.e. one example per
        # record with the true last item as the label -- which is exactly the
        # leave-one-out sample this measurement needs.
        # SemanticIDEncoderDecoder is what tiger_train_flat instantiates, and it
        # owns both helpers this needs (_teacher_forced_log_prob at line ~1340,
        # _codes_to_item_ids at ~1260). TigerUnlearningModule subclasses it, so
        # this check accepts either.
        if not isinstance(model, SemanticIDEncoderDecoder):
            raise TypeError(
                f"expected SemanticIDEncoderDecoder, got {type(model).__name__}")

        ckpts = {
            "clean": cfg.get("clean_ckpt"),
            "poisoned": cfg.get("ckpt_path"),
            "unlearned": cfg.get("unlearned_ckpt"),
        }
        missing = [k for k, v in ckpts.items() if not v]
        if missing:
            raise ValueError(f"need checkpoints for: {missing} "
                             "(clean_ckpt, ckpt_path, unlearned_ckpt)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_h = int(cfg.get("num_hierarchies", 4))
        n_levels = n_h - 1          # last digit is the dedup counter

        codes = model.codebooks.detach().cpu().to(torch.long)
        if codes.size(1) != n_h:
            codes = codes.t()
        manifest = cfg.get("spam_forget_manifest") or os.path.join(
            cfg.paths.data_dir, "forget_manifest.json")
        with open(manifest) as fh:
            targets = sorted(int(i) for i in json.load(fh)["target_items"])
        depth = prefix_share_depth(codes, targets, n_levels)

        # held-out TEST samples, not the retain histories the drift script uses
        # setup() reads self.trainer.world_size to shard files across workers,
        # and Lightning only attaches the trainer when a fit/test loop runs --
        # which train=False / test=False deliberately skips. The launcher already
        # built a trainer, so wire it up directly rather than starting a loop we
        # do not want.
        if getattr(pm.datamodule, "trainer", None) is None:
            pm.datamodule.trainer = pm.trainer
        pm.datamodule.setup()
        # get_dataloader returns a 1-TUPLE containing the loader
        # (`return (DataloaderWithIterationRetry(...),)`), even though both its
        # annotation and its docstring say DataLoader. Unwrap it, tolerating a
        # bare loader in case that is ever fixed.
        dl = pm.datamodule.get_dataloader(TrainerFn.TESTING)
        if isinstance(dl, (tuple, list)):
            if len(dl) != 1:
                raise ValueError(f"expected 1 test dataloader, got {len(dl)}")
            dl = dl[0]
        # Default to the WHOLE test set: the neighbour sample group is ~0.5% of
        # it, so a 16-batch sample yielded n=1 and made that group unusable.
        n_b = int(cfg.get("label_lp_n_batches", 10**9))
        batches = []
        for i, b in enumerate(dl):
            if i >= n_b:
                break
            # Index rather than unpack: the test dataloader yields more than two
            # elements, and the model's own eval_step reads batch[0]/batch[1]
            # for exactly this reason.
            mi, ld = b[0], b[1]
            mi.mask = mi.mask.to(device)
            mi.transformed_sequences = {k: v.to(device)
                                        for k, v in mi.transformed_sequences.items()}
            ld.labels = {k: v.to(device) for k, v in ld.labels.items()}
            batches.append((mi, ld))
        log.info(f"[label_lp] {len(batches)} test batches")

        model = model.to(device).eval()
        # `_codes_to_item_ids` reads the buffers that `_build_code_to_item_index`
        # registers, and that only runs inside `enable_adaptive_item_offset`.
        # Every experiment here uses adaptive_codes=false, so the index does not
        # exist and the lookup raises. Build it explicitly -- it depends only on
        # `codebooks`, so it is safe to call standalone and is idempotent in
        # effect (it just re-registers the same non-persistent buffers).
        if not hasattr(model, "_adaptive_code_powers"):
            model._build_code_to_item_index()
        items = _label_items(model, batches)
        tset = set(targets)
        groups = []
        for t in items:
            g = []
            for it in t.reshape(-1).tolist():
                if it < 0 or it >= depth.numel():
                    g.append("skip")
                elif it in tset:
                    g.append("target")          # recorded but not plotted (n~1)
                elif int(depth[it]) > 0:
                    g.append("neighbour")
                else:
                    g.append("unrelated")
            groups.append(g)

        # Probe items to score against EVERY history: the target(s), a sample of
        # code neighbours, and a sample of unrelated controls. Subsampled because
        # cost is batch_size x n_probe teacher-forced rows per batch per model.
        rng = _rnd.Random(int(cfg.get("seed", 2)))
        nbr_pool = [i for i in range(codes.size(0))
                    if int(depth[i]) > 0 and i not in tset]
        ctl_pool = [i for i in range(codes.size(0)) if int(depth[i]) == 0]
        k = int(cfg.get("label_lp_probe_items", 24))
        probe_nbr = rng.sample(nbr_pool, min(k, len(nbr_pool)))
        probe_ctl = rng.sample(ctl_pool, min(k, len(ctl_pool)))
        probe = list(targets) + probe_nbr + probe_ctl
        probe_kind = (["target"] * len(targets) + ["neighbour_items"] * len(probe_nbr)
                      + ["unrelated_items"] * len(probe_ctl))
        probe_codes = codes[torch.tensor(probe)]
        log.info("[label_lp] probe: %d target, %d neighbour, %d unrelated items",
                 len(targets), len(probe_nbr), len(probe_ctl))

        SGROUPS = ("neighbour", "unrelated", "target")
        BASE_IK = ("label", "target", "neighbour_items", "unrelated_items")
        # per-item-group statistics live alongside the plain (mean) key
        IKINDS = tuple(BASE_IK) + tuple(
            f"{k}__{st}" for k in ("target", "neighbour_items", "unrelated_items")
            for st in ("max", "lse"))

        # Targeted probe set: EVERY sample in a rare group, plus a capped random
        # sample of the abundant `unrelated` group. Probe cost is then bounded by
        # this selection instead of by the size of the test set.
        cap = int(cfg.get("label_lp_probe_rows_per_group", 400))
        rng2 = _rnd.Random(int(cfg.get("seed", 2)) + 1)
        picked: Dict[str, List] = {sg: [] for sg in SGROUPS}
        for bi, g in enumerate(groups):
            for row, sg in enumerate(g):
                if sg != "skip":
                    picked[sg].append((bi, row))
        probe_rows: Dict[str, List] = {}
        for sg in SGROUPS:
            v = picked[sg]
            probe_rows[sg] = v if len(v) <= cap else rng2.sample(v, cap)
        by_batch: Dict[int, List] = {}
        for sg in SGROUPS:
            for bi, row in probe_rows[sg]:
                by_batch.setdefault(bi, []).append((row, sg))
        log.info("[label_lp] probe rows per sample group: %s (cap %d)",
                 {sg: len(probe_rows[sg]) for sg in SGROUPS}, cap)

        per_model: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        kind_idx = {kind: [i for i, kk in enumerate(probe_kind) if kk == kind]
                    for kind in ("target", "neighbour_items", "unrelated_items")}
        for name, path in ckpts.items():
            _load_state(model, path)
            model = model.to(device).eval()
            acc = {sg: {ik: [] for ik in IKINDS} for sg in SGROUPS}

            # pass 1: label log-prob on every sample (one forward per batch)
            lps = _label_logprobs(model, batches, device)
            for lp, g in zip(lps, groups):
                for lv, sg in zip(lp.tolist(), g):
                    if sg != "skip":
                        acc[sg]["label"].append(lv)

            # pass 2: probe scoring, only on the selected rows
            chunk = int(cfg.get("label_lp_item_chunk", 24))
            for bi, rows in by_batch.items():
                mi, ld = batches[bi]
                idx = [r for r, _ in rows]
                sub = _subbatch(mi, ld, idx)
                pr = _probe_logprobs(model, [sub], probe_codes, device, chunk)[0]
                for pos, (_r, sg) in enumerate(rows):
                    for kind, sel in kind_idx.items():
                        if not sel:
                            continue
                        v = pr[pos, sel]
                        # Store THREE statistics, because the choice is not
                        # cosmetic and cannot be changed after the fact:
                        #   mean  = mean of log-probs = log of the GEOMETRIC
                        #           mean probability. "how well the model
                        #           supports a typical member of the group" --
                        #           not an exposure measure.
                        #   max   = the most exposed item in the group. Size
                        #           robust, and since the target group holds a
                        #           single item all three groups become the same
                        #           kind of quantity.
                        #   lse   = logsumexp = log of the TOTAL probability
                        #           mass on the group (what TPM@k measures).
                        #           Correct as exposure but confounded by group
                        #           size: 1 target item vs 24 neighbours vs 24
                        #           unrelated.
                        acc[sg][kind].append(float(v.mean()))
                        acc[sg][kind + "__max"].append(float(v.max()))
                        acc[sg][kind + "__lse"].append(
                            float(torch.logsumexp(v.double(), dim=0)))
            per_model[name] = acc
            log.info("[label_lp] %-9s %s", name,
                     {sg: {ik: (round(sum(v)/len(v), 3) if v else None)
                           for ik, v in d.items()} for sg, d in acc.items()})

        def stats(v: List[float]) -> Dict[str, Any]:
            n = len(v)
            if n == 0:
                return {"n": 0, "mean": None, "sem": None}
            m = sum(v) / n
            var = sum((x - m) ** 2 for x in v) / (n - 1) if n > 1 else 0.0
            return {"n": n, "mean": m, "sem": (var / n) ** 0.5 if n > 1 else 0.0}

        out = {
            "dataset": os.path.basename(str(cfg.paths.data_dir)),
            "target_items": targets,
            "semantic_levels": n_levels,
            "checkpoints": ckpts,
            "units": "teacher-forced log p(item | history), nats",
            "statistics": {
                "<kind>": "mean of log-probs over items in the group "
                          "(log geometric-mean probability)",
                "<kind>__max": "max log-prob = most exposed item in the group",
                "<kind>__lse": "logsumexp = log total probability mass on the group",
            },
            "probe_counts": {"target": len(targets), "neighbour_items": len(probe_nbr),
                             "unrelated_items": len(probe_ctl)},
            # by_sample_group[sample_group][model][scored_item_kind] -> stats
            "by_sample_group": {
                sg: {name: {ik: stats(per_model[name][sg][ik]) for ik in IKINDS}
                     for name in per_model}
                for sg in SGROUPS},
            # kept for the simpler 2-group view: label log-prob only
            "by_group": {name: {sg: stats(acc[sg]["label"]) for sg in SGROUPS}
                         for name, acc in per_model.items()},
        }
        dest = os.path.join(cfg.paths.output_dir, "label_logprob.json")
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=2)
        log.info(f"[label_lp] wrote {dest}")
        return out


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
