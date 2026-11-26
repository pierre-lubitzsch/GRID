"""Non-member fake users for the membership-inference check.

Runs the bandwagon generator exactly as for the poisoned dataset (same attack, ratio,
placement, and target item) but with a different random seed, so the fake users are
drawn from the same distribution as the injected ones while no model was trained on
them. The target is pinned to the poisoned dataset's target (from its manifest), and
only the new fake-user shards are written (no copy of the clean data).

    python -m scripts.make_mia_nonmembers --poison_dir src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1 \
        --clean_dir src/data/amazon_data/beauty --out_dir logs/diagnose/mia/nonmembers/beauty_mid_s2
"""
import argparse
import json
import os

from src.data.poisoning import bandwagon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poison_dir", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed_offset", type=int, default=1000)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.poison_dir, "forget_manifest.json")))
    targets = [int(t) for t in man["target_items"]]
    assert man["method"] == "bandwagon" and man["attack_type"] == "bandwagon", man["method"]

    bandwagon._select_target_items = lambda *args, **kwargs: list(targets)
    bandwagon._copy_clean_shards = lambda *args, **kwargs: None
    bandwagon._copy_sibling_subdirs = lambda *args, **kwargs: None

    bandwagon.main(
        data_dir=a.clean_dir, out_dir=a.out_dir, attack="bandwagon",
        target_strategy=man["target_strategy"], poisoning_ratio=float(man["poisoning_ratio"]),
        n_target_items=len(targets), placement=man["placement"], seed=int(man["seed"]) + a.seed_offset,
        rows_per_shard=5000, overwrite=True, p_two_targets=float(man["p_two_targets"]),
        method="bandwagon",
    )
    out = json.load(open(os.path.join(a.out_dir, "forget_manifest.json")))
    assert [int(t) for t in out["target_items"]] == targets, (out["target_items"], targets)
    print(f"non-members: {out['n_spam_users']} fake users promoting {targets} -> {a.out_dir}/training")


if __name__ == "__main__":
    main()
