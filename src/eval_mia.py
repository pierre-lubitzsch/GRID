"""Membership-inference evaluation in MUNKEY's protocol (MUNKEY/experiments/base_module.py,
_logreg_mia_auc_shadow), adapted to a generative recommender.

Per example (one user history, labelled with its last item by the eval collate):
  loss    = -log p(SID of the label item | history)   (teacher-forced, summed over levels)
  entropy, confidence (max prob), margin (top-1 minus top-2 logit) of the FIRST SID
          level's distribution, the analogue of MUNKEY's class distribution.
A logistic-regression attack is fitted on RETAIN (members, 1) vs TEST (non-members, 0)
and evaluated on FORGET (1) vs TEST (0): AUC near 0.5 means forgotten users look like
strangers; retain-vs-test AUC is reported as the attack's strength.

Usage (from GRID/; the model overrides must match the checkpoint):
  python -m src.eval_mia experiment=tiger_train_flat data_dir=<training data dir> \
      semantic_id_path=<sid file> num_hierarchies=4 ckpt_path=<ckpt> \
      +mia.forget_dir=<dir> +mia.retain_dir=<dir> +mia.test_dir=<dir> \
      +mia.max_rows=3000 +mia.out=<json> [model overrides]
  +mia.munkey_delete_from=<forget_manifest.json>: MUNKEY zero-shot deletion applied
  in-process (the manifest's spam_user_ids keys deleted; no weight changes), so no
  unlearned checkpoint is needed.
"""

import json
import os
from typing import Any, Dict, List

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402, F401, F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)
FEATURES = ("loss", "entropy", "conf", "margin")


@torch.no_grad()
def row_features(model: Any, batches: List[Any]) -> torch.Tensor:
    """(N, 4) features [loss, entropy, conf, margin] for every row of ``batches``."""
    model.eval()
    H = int(model.num_hierarchies)
    out = []
    for model_input, label_data in batches:
        fut = None
        for k in label_data.labels:
            fut = label_data.labels[k].reshape(model_input.mask.size(0), -1)
        mo = model.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut,
            **{model.feature_to_model_input_map.get(k, k): v
               for k, v in model_input.transformed_sequences.items()},
        )[:, :-1]
        lp_sum = 0.0
        for h in range(H):
            logits = model.decoder.decoder_mlp[h](mo[:, h]).float()
            logp = torch.log_softmax(logits, dim=-1)
            lp_sum = lp_sum + logp.gather(1, fut[:, h].long().unsqueeze(1)).squeeze(1)
            if h == 0:
                p = logp.exp()
                entropy = -(p * logp).sum(1)
                conf = p.max(1).values
                top2 = logits.topk(2, dim=-1).values
                margin = top2[:, 0] - top2[:, 1]
        out.append(torch.stack([-lp_sum, entropy, conf, margin], dim=1).cpu())
    return torch.cat(out)


def fit_logreg(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-3, iters: int = 200):
    """Standardised logistic regression (L-BFGS). Returns a scoring function."""
    mu, sd = x.mean(0), x.std(0).clamp(min=1e-8)
    z = ((x - mu) / sd).double()
    w = torch.zeros(z.size(1), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    yy = y.double()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, yy) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    w_, b_ = w.detach(), b.detach()
    return lambda xn: (((xn - mu) / sd).double() @ w_ + b_)


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC AUC as the Mann-Whitney statistic (ties count half)."""
    s, y = scores.double(), labels.bool()
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    order = allv.argsort()
    ranks = torch.empty_like(allv)
    ranks[order] = torch.arange(1, len(allv) + 1, dtype=torch.float64)
    # average ranks over ties
    uniq, inv = torch.unique(allv, return_inverse=True)
    sums = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, ranks)
    cnts = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, torch.ones_like(ranks))
    ranks = (sums / cnts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mia_aucs(f_retain: torch.Tensor, f_test: torch.Tensor, f_forget: torch.Tensor) -> Dict[str, float]:
    x = torch.cat([f_retain, f_test])
    y = torch.cat([torch.ones(len(f_retain)), torch.zeros(len(f_test))])
    score = fit_logreg(x, y)
    return {
        "auc_forget_vs_test": auc(torch.cat([score(f_forget), score(f_test)]),
                                  torch.cat([torch.ones(len(f_forget)), torch.zeros(len(f_test))])),
        "auc_retain_vs_test": auc(score(x), y),
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    from src.components.unlearning.val_selection import build_val_batches
    from src.models.components.network_blocks.product_key_memory import resolve_auto_query_norm
    from src.unlearn_sequential import _resolve_train_dl_cfg

    m = cfg.mia
    with pipeline_launcher(cfg) as pm:
        model = pm.model
        sd = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        resolve_auto_query_norm(model, sd, adopt=True)
        r = model.load_state_dict(sd, strict=False)
        if r.missing_keys or r.unexpected_keys:
            log.warning(f"[mia] load: missing {r.missing_keys[:5]} unexpected {r.unexpected_keys[:5]}")
        if m.get("munkey_delete_from"):
            mm = getattr(model, "munkey_memory", None)
            if mm is None:
                raise ValueError("mia.munkey_delete_from needs model.munkey_memory")
            uids = torch.tensor(json.load(open(str(m.munkey_delete_from)))["spam_user_ids"])
            lut = mm.uid_to_row
            rows = lut[uids[(uids >= 0) & (uids < lut.numel())]]
            log.info(f"[mia] MUNKEY key deletion: {mm.delete(rows[rows >= 0])} of {len(uids)} users")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dl = _resolve_train_dl_cfg(pm, cfg)
        feats = {}
        for name in ("retain", "test", "forget"):
            folder = str(m[f"{name}_dir"])
            b = build_val_batches(dl, folder, device, batch_size=int(m.get("batch_size", 64)),
                                  max_rows=m.get("max_rows"))
            feats[name] = row_features(model, b)
            log.info(f"[mia] {name}: {len(feats[name])} rows from {folder}")
        res = mia_aucs(feats["retain"], feats["test"], feats["forget"])
        res.update({f"mean_{f}_{n}": float(feats[n][:, i].mean()) for n in feats for i, f in enumerate(FEATURES)})
        res.update({"ckpt_path": str(cfg.ckpt_path), "n_rows": {k: len(v) for k, v in feats.items()}})
        log.info(f"[mia] {json.dumps({k: v for k, v in res.items() if k.startswith('auc')})}")
        if m.get("out"):
            os.makedirs(os.path.dirname(os.path.abspath(str(m.out))), exist_ok=True)
            json.dump(res, open(str(m.out), "w"), indent=2)


if __name__ == "__main__":
    main()
