"""Does the spam live in the forget-specific PKM slots? Slot-deletion diagnostic.

For a trained PKM model: count per-slot reads on the forget data (the fake users) and on
the FULL retain set, then delete slot sets at evaluation (masked slots are never selected,
HashingMemory.slot_mask) and score the test split:
  none          no deletion
  exclusive     slots read by the forget data and by NO retain row (strict)
  afihf_<t>     top-t slots per memory by AF-IHF (forget reads, discounted by retain reads)
  dom<r>        forget-DOMINATED slots: read by the forget data at >= r times the per-row
                rate of the retain data (slots no retain row reads count as dominated)
For each condition: test N@10 / SH@10, slots deleted per memory, and the share of the
forget data's slot reads that the deleted set covers.
A large SH drop at small utility cost means the spam is stored in forget-specific slots
(so a routing / slot-deletion unlearning term can exploit separability); no drop means it
lives in the backbone or in shared slots.

Usage (from GRID/, CPU is fine):
  python -m scripts.pkm_slot_mask_diag --ckpt <last.ckpt> \
      --overrides "model.pkm_layers={encoder:[1],decoder:null} model.pkm_mode=replace" \
      --data src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1 --out <json>
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")
import src.utils  # noqa: F401,E402  (import order as in src.train)
from src.utils.custom_hydra_resolvers import *  # noqa: F401,F403,E402  (config resolvers)
import hydra  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--sid", default="embeddings/beauty_rqvae/merged_predictions_tensor.pt")
    ap.add_argument("--tops", default="100,1000,10000")
    ap.add_argument("--dominance", default="1,10", help="rate ratios r for the dom<r> conditions")
    ap.add_argument("--conds", default=None,
                    help="comma list of conditions to run (default: all); 'none' is always run")
    ap.add_argument("--retain_rows", type=int, default=None, help="default: the full retain set")
    ap.add_argument("--test_rows", type=int, default=6000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from src.components.unlearning.slot_selection import _access_sweep, _memories, select_top_t_slots
    from src.components.unlearning.val_selection import build_val_batches, evaluate_batches
    from src.models.components.network_blocks.product_key_memory import resolve_auto_query_norm

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            "experiment=tiger_train_flat", f"data_dir={a.data}", f"semantic_id_path={a.sid}",
            "num_hierarchies=4", *a.overrides.split()])
    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    resolve_auto_query_norm(model, sd, adopt=True)
    r = model.load_state_dict(sd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        print("WARNING load: missing", r.missing_keys[:5], "unexpected", r.unexpected_keys[:5])
    mems = _memories(model)
    if not mems:
        raise SystemExit("no PKM memory in this model")
    dl = hydra.utils.instantiate(cfg.data_loading.train_dataloader_config.dataloader)
    dev = torch.device("cpu")
    t0 = time.time()
    forget = build_val_batches(dl, os.path.join(a.data, "training_forget"), dev)
    retain = build_val_batches(dl, os.path.join(a.data, "training_retain"), dev, max_rows=a.retain_rows)
    test = build_val_batches(dl, os.path.join(a.data, "testing"), dev, max_rows=a.test_rows)
    nrow = lambda bs: sum(int(b[0].mask.shape[0]) for b in bs)
    print(f"[diag] forget {nrow(forget)} / retain {nrow(retain)} / test {nrow(test)} rows "
          f"({time.time() - t0:.0f}s)", flush=True)

    fa = _access_sweep(model, mems, forget)
    ra = _access_sweep(model, mems, retain)
    conds = {"none": {n: torch.zeros_like(fa[n], dtype=torch.bool) for n, _ in mems},
             "exclusive": {n: (fa[n] > 0) & (ra[n] == 0) for n, _ in mems}}
    nf, nr = max(nrow(forget), 1), max(nrow(retain), 1)
    for rr in [float(x) for x in a.dominance.split(",") if x]:
        conds[f"dom{rr:g}"] = {n: (fa[n] > 0) & (fa[n] / nf >= rr * ra[n] / nr) for n, _ in mems}
    want = None if a.conds is None else set(a.conds.split(",")) | {"none"}
    for t in [int(x) for x in a.tops.split(",") if x]:
        if want is not None and f"afihf_{t}" not in want:
            continue
        masks, _ = select_top_t_slots(model, forget, retain, criterion="af_ihf", top_t=t)
        conds[f"afihf_{t}"] = {n: masks[n].flatten().bool().cpu() for n, _ in mems}
    if want is not None:
        conds = {k: v for k, v in conds.items() if k in want}

    res = {"ckpt": a.ckpt, "rows": {"forget": nrow(forget), "retain": nrow(retain), "test": nrow(test)},
           "forget_slots_read": {n: int((fa[n] > 0).sum()) for n, _ in mems},
           "retain_slots_read": {n: int((ra[n] > 0).sum()) for n, _ in mems},
           "conditions": {}}
    for name, cm in conds.items():
        for n, m in mems:
            m.slot_mask = cm[n] if bool(cm[n].any()) else None
        mtr = evaluate_batches(model, test)
        cover = {n: float(fa[n][cm[n]].sum() / fa[n].sum().clamp(min=1)) for n, _ in mems}
        res["conditions"][name] = {
            "ndcg@10": mtr.get("ndcg@10"), "SH@10": mtr.get("SH@10"),
            "slots_deleted": {n: int(cm[n].sum()) for n, _ in mems},
            "forget_read_share_deleted": cover,
            "retain_read_share_deleted": {n: float(ra[n][cm[n]].sum() / ra[n].sum().clamp(min=1))
                                          for n, _ in mems},
        }
        c = res["conditions"][name]
        print(f"[diag] {name:12s} N {c['ndcg@10']:.4f} SH {c['SH@10']:.4f}  deleted "
              f"{sum(c['slots_deleted'].values())} slots, forget-read share "
              f"{sum(cover.values()) / len(cover):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    for _, m in mems:
        m.slot_mask = None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
