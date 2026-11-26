"""Spam data dirs where only part of the fake users is known to the deleter.

For each Beauty spam cell (mid and unpopular targets, 5 data seeds) and each known fraction f,
builds <spam dir>_known<100f>/: training/, items/, evaluation/, testing/ symlinked to the cell's
spam dir, a forget_manifest.json whose spam_user_ids is a fixed random subset (fraction f) of the
fake users, and a forget/retain split of that manifest. The undetected fake users stay in
training_retain/ like ordinary users. The rows are parsed (no --segregated-shards): copying the
spam shards wholesale would put every fake user into the forget set.

The reference stays the fully clean (retrained) model: the goal is still to remove the attack.

    python scripts/make_partial_spam_dirs.py            # build all (skips finished dirs)
    python scripts/make_partial_spam_dirs.py --dry-run
"""
import argparse
import json
import os
import random
import subprocess

ROOT = "src/data/amazon_data"
FRACTIONS = (0.25, 0.5, 0.75)
STRATA = ("unpopular", "mid")
SEEDS = (2, 3, 5, 7, 11)


def spam_dir(st, sd):
    tok = "" if st == "unpopular" else f"_tgt{st}"
    return f"{ROOT}/beauty_spam{tok}_seed{sd}_pct1_n1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for st in STRATA:
        for sd in SEEDS:
            src = spam_dir(st, sd)
            man = json.load(open(f"{src}/forget_manifest.json"))
            users = sorted(int(u) for u in man["spam_user_ids"])
            for f in FRACTIONS:
                dst = f"{src}_known{int(round(100 * f))}"
                if os.path.isdir(f"{dst}/training_retain"):
                    print("skip", dst); continue
                rng = random.Random(1000 * sd + int(round(100 * f)) + (0 if st == "unpopular" else 7))
                known = sorted(rng.sample(users, max(1, int(round(f * len(users))))))
                print(f"{dst}: {len(known)} of {len(users)} fake users known")
                if a.dry_run:
                    continue
                os.makedirs(dst, exist_ok=True)
                for sub in ("training", "items", "evaluation", "testing"):
                    if not os.path.exists(f"{dst}/{sub}"):
                        os.symlink(os.path.realpath(f"{src}/{sub}"), f"{dst}/{sub}")
                m = dict(man)
                m.update(spam_user_ids=known, n_spam_users=len(known), known_fraction=f,
                         all_spam_user_ids=users, n_all_spam_users=len(users), source_spam_dir=os.path.abspath(src))
                json.dump(m, open(f"{dst}/forget_manifest.json", "w"), indent=1)
                subprocess.run(["python", "-m", "src.data.unlearning.split_forget_retain", "--data_dir", dst,
                                "--forget_manifest", f"{dst}/forget_manifest.json"], check=True,
                               env={**os.environ, "PYTHONPATH": "."})


if __name__ == "__main__":
    main()
