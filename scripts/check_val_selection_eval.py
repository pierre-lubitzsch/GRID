"""Check the in-process validation eval used by unlearning.val_selection.

Builds the unlearn config's model and training dataloader config, loads a
checkpoint, drains the validation split exactly as the EpochSelector does, and
scores it with evaluate_batches. With ``--expect`` (a metrics.csv from
scripts.eval_ckpt_on_test on the same checkpoint and a ``_valview``), it asserts
that SH@10 and NDCG@10 agree, i.e. that the selector sees the same numbers as the
reported validation eval. Use ``--max-rows`` for a quick wiring check only (the
comparison is skipped then).

    python -m scripts.check_val_selection_eval \\
        --data-dir src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1 \\
        --ckpt logs/train/runs/2026-07-01/16-00-19_job9872117_beauty_poison_pct1_n1_tgtmid/checkpoints/checkpoint_epoch=000_step=004400.ckpt \\
        --expect logs/diagnose/valsel_refs/rkmeans_beauty_mid_s2_poison/csv/version_0/metrics.csv
"""

from __future__ import annotations

import argparse
import sys
import time

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

import src.utils.custom_hydra_resolvers  # noqa: E402,F401
from src.components.unlearning.val_selection import (  # noqa: E402
    build_val_batches,
    evaluate_batches,
    read_reference_metrics,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--semantic-id-path", default="embeddings/beauty/merged_predictions_tensor.pt")
    ap.add_argument("--experiment", default="tiger_unlearn_unified_sequential")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--expect", default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    import os
    with initialize_config_dir(version_base="1.3", config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="unlearn.yaml", overrides=[
            f"experiment={a.experiment}", f"data_dir={a.data_dir}",
            f"semantic_id_path={a.semantic_id_path}", "num_hierarchies=4",
            "hydra.run.dir=."])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    res = model.load_state_dict(sd, strict=False)
    print(f"loaded {a.ckpt} (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")
    model = model.to(device)
    dl_cfg = hydra.utils.instantiate(cfg.data_loading.train_dataloader_config.dataloader)

    t0 = time.time()
    batches = build_val_batches(dl_cfg, f"{a.data_dir}/evaluation", device,
                                batch_size=a.batch_size, max_rows=a.max_rows)
    n = sum(int(b[0].mask.shape[0]) for b in batches)
    print(f"{n} validation rows in {len(batches)} batches ({time.time() - t0:.1f}s)")
    t0 = time.time()
    m = evaluate_batches(model, batches)
    print(f"eval {time.time() - t0:.1f}s on {device}: "
          f"SH@10={m.get('SH@10')} ndcg@10={m.get('ndcg@10')} loss={m['loss']:.4f}")
    if "SH@10" not in m:
        print("FAIL: no SH@10 (spam manifest not loaded?)")
        return 1
    if a.expect and a.max_rows is None:
        sh, ndcg = read_reference_metrics(a.expect)
        ok = abs(m["SH@10"] - sh) <= a.tol and abs(m["ndcg@10"] - ndcg) <= a.tol
        print(f"expected SH@10={sh} ndcg@10={ndcg} -> {'MATCH' if ok else 'MISMATCH'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
