"""Build a forget_manifest.json for SENSITIVE-CATEGORY unlearning.

The second unlearning scenario besides spam. Spam is *maliciously injected*;
sensitive-category deletion removes legitimate items a user must stop being
recommended -- ERASE's framing is "meat for users adopting a vegetarian diet or
alcohol for users with addiction issues" (arXiv:2603.08341).

Why it matters for NAU: deleting one meat item while its near-duplicates stay
recommendable is not partial success, it is failure. The neighbourhood IS the
concept, which is where neighbourhood-aware unlearning stops being a marginal
utility gain.

WHAT THIS EMITS. The same schema ``bandwagon.py`` writes, so every downstream
consumer works unmodified:

    target_items    the sensitive items I_s          (read by deletion_spec.py,
                    eval_metrics._load_spam_target_sids -> SH/ASI/TPM)
    spam_user_ids   users whose sessions touch I_s   (read by
                    split_forget_retain._load_forget_user_ids)
    scenario        "sensitive"                      (how extractors tell the
                    two scenarios apart)

The ``spam_user_ids`` name is deliberate reuse, not sloppiness:
``split_forget_retain`` routes rows by user id and hardcodes that key, so reusing
it means the whole split / eval path needs no changes at all.

WHERE THE CATEGORY COMES FROM. No metadata download is needed for the Amazon P5
datasets: each ``items/`` record carries a ``text`` feature shaped

    Title: <t>; Brand: <b>; Categories: ['Toys & Games', 'Games', ...]; Price: <p>;

so the category path is recoverable exactly. Matching a level of that path is
cleaner than ERASE's keyword regex (which was written for a raw metadata dump
that does not exist here), but ``--keywords`` reproduces the ERASE behaviour when
comparability matters.

Examples
--------
    # toys, the "Baby & Toddler Toys" level-2 category (399 items, 3.3%)
    python -m scripts.build_sensitive_manifest \
        --data_dir src/data/amazon_data/toys \
        --category "Baby & Toddler Toys" --category_level 1 \
        --out src/data/amazon_data/toys_sensitive_baby/forget_manifest.json

    # ERASE-style keyword matching
    python -m scripts.build_sensitive_manifest --data_dir <dir> \
        --keywords meat beef pork chicken lamb turkey bacon ham sausage salami
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import tensorflow as tf

BRAND_RE = re.compile(r"Brand:\s*([^;]+);")
CATEGORIES_RE = re.compile(r"Categories:\s*(\[.*?\])\s*;")


def _shards(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".tfrecord.gz")
    ]


def parse_categories(text: str) -> List[str]:
    """Category path out of an item's ``text`` feature; [] when absent."""
    m = CATEGORIES_RE.search(text)
    if not m:
        return []
    try:
        val = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return []
    return [str(x) for x in val] if isinstance(val, (list, tuple)) else []


def scan_items(
    items_dir: str,
    category: Optional[str],
    category_level: Optional[int],
    keywords: Optional[Sequence[str]],
    brand: Optional[str] = None,
) -> Tuple[Set[int], int, Dict[str, int]]:
    """Return (matching item ids, total items, level-`category_level` histogram).

    Keyword matching is whole-word and case-insensitive over the whole ``text``
    field, matching ERASE's ``identify_sensitive_items.py``. ``brand`` matches the
    item's ``Brand:`` field exactly (a takedown of one brand's whole catalog) and
    takes precedence over keywords and category.
    """
    pat = None
    if keywords:
        pat = re.compile(
            r"\b(" + "|".join(re.escape(k.lower()) for k in keywords) + r")\b",
            re.IGNORECASE,
        )
    hits: Set[int] = set()
    total = 0
    hist: Dict[str, int] = {}
    for shard in _shards(items_dir):
        for rec in tf.data.TFRecordDataset(shard, compression_type="GZIP"):
            ex = tf.train.Example.FromString(rec.numpy())
            feat = ex.features.feature
            item_id = int(feat["id"].int64_list.value[0])
            text = feat["text"].bytes_list.value[0].decode("utf-8", "replace")
            total += 1
            cats = parse_categories(text)
            if category_level is not None and len(cats) > category_level:
                hist[cats[category_level]] = hist.get(cats[category_level], 0) + 1
            if brand is not None:
                m = BRAND_RE.search(text)
                if m and m.group(1).strip() == brand:
                    hits.add(item_id)
            elif pat is not None:
                if pat.search(text):
                    hits.add(item_id)
            elif category is not None:
                if category_level is None:
                    if any(c == category for c in cats):
                        hits.add(item_id)
                elif len(cats) > category_level and cats[category_level] == category:
                    hits.add(item_id)
    return hits, total, hist


def scan_users(
    training_dir: str, targets: Set[int]
) -> Tuple[List[Tuple[int, int]], int, int, int]:
    """Scan the training split once.

    Returns ``(touching, n_users, n_touch_interactions, n_interactions_total)``
    where ``touching`` is ``[(user_id, n_sensitive_interactions), ...]``.
    """
    touching: List[Tuple[int, int]] = []
    n_users = 0
    n_touch = 0
    n_total = 0
    for shard in _shards(training_dir):
        for rec in tf.data.TFRecordDataset(shard, compression_type="GZIP"):
            ex = tf.train.Example.FromString(rec.numpy())
            feat = ex.features.feature
            uid = int(feat["user_id"].int64_list.value[0])
            seq = list(feat["sequence_data"].int64_list.value)
            n_users += 1
            n_total += len(seq)
            k = sum(1 for i in seq if i in targets)
            if k:
                touching.append((uid, k))
                n_touch += k
    return touching, n_users, n_touch, n_total


def sample_forget_users(
    touching: Sequence[Tuple[int, int]],
    n_interactions_total: int,
    forget_ratio: float,
    seed: int,
) -> Tuple[List[int], int]:
    """ERASE-style forget-set sizing: sample users until their sensitive
    interactions reach ``forget_ratio`` of ALL interactions.

    ERASE generates sensitive forget sets at 0.01% / 0.001% / 0.0001% of the
    interactions (``generate_forget_sets.py --forget-ratio``), NOT "every user
    who ever touched the category". That distinction is load-bearing here:
    on toys, "Baby & Toddler Toys" is touched by 21.4% of all sessions, and
    since ``split_forget_retain`` routes rows BY USER, taking all of them would
    move a fifth of the real users -- and all their unrelated interactions --
    out of the retain set. Utility would then fall for reasons that have nothing
    to do with unlearning.

    Users are shuffled with ``seed`` and accumulated whole (a request is one
    user's sensitive interactions, per ERASE's sequential protocol), so the
    realized count lands at or just above the budget.
    """
    budget = forget_ratio * float(n_interactions_total)
    rng = random.Random(seed)
    order = list(touching)
    rng.shuffle(order)
    chosen: List[int] = []
    acc = 0
    for uid, k in order:
        if acc >= budget:
            break
        chosen.append(uid)
        acc += k
    return sorted(chosen), acc


def write_forget_pairs(
    training_dir: str,
    targets: Set[int],
    forget_users: Set[int],
    out_path: str,
) -> int:
    """Persist the forget set as ERASE does: one ROW PER INTERACTION, TSV.

    ERASE's ``generate_forget_sets.py`` writes

        {dataset}_unlearn_pairs_sensitive_category_{cat}_seed_{s}_unlearning_fraction_{r}.inter

    with the RecBole header ``user_id:token item_id:token rating:float
    timestamp:float``, one line per forgotten interaction, taking ALL of a
    selected user's sensitive interactions. Same here, so the two benchmarks'
    forget sets are directly diffable.

    DEVIATION: GRID's TFRecords carry no ratings and no timestamps (features are
    user_id / sequence_data / text / embedding). ``rating`` is written as 0.0 and
    ``timestamp`` as the item's POSITION in the session, which preserves order but
    is not a real time. Recorded in the manifest as ``rating_threshold: null``.
    """
    n = 0
    with open(out_path, "w") as fh:
        fh.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for shard in _shards(training_dir):
            for rec in tf.data.TFRecordDataset(shard, compression_type="GZIP"):
                ex = tf.train.Example.FromString(rec.numpy())
                feat = ex.features.feature
                uid = int(feat["user_id"].int64_list.value[0])
                if uid not in forget_users:
                    continue
                for pos, item in enumerate(feat["sequence_data"].int64_list.value):
                    if int(item) in targets:
                        fh.write(f"{uid}\t{int(item)}\t0.0\t{float(pos)}\n")
                        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default=None, help="default <data_dir>/forget_manifest.json")
    p.add_argument("--category", default=None, help='e.g. "Baby & Toddler Toys"')
    p.add_argument(
        "--category_level",
        type=int,
        default=None,
        help="index into the category path; omit to match at any level",
    )
    p.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="ERASE-style whole-word match over the item text (overrides --category)",
    )
    p.add_argument(
        "--brand",
        default=None,
        help='exact match on the item\'s "Brand:" field, e.g. "e.l.f. Cosmetics" '
             "(an item takedown; overrides --keywords and --category)",
    )
    p.add_argument("--deletion_spec", default="item_pairs",
                   choices=["session", "item", "item_pairs"])
    p.add_argument("--seed", type=int, default=2,
                   help="ERASE uses seeds 2/3/5/7/11")
    p.add_argument(
        "--forget_ratio",
        type=float,
        default=1e-4,
        help="fraction of ALL interactions to place in the forget set "
             "(ERASE: 1e-4 / 1e-5 / 1e-6). Use 0 to take every touching user.",
    )
    p.add_argument(
        "--list_categories",
        type=int,
        default=None,
        metavar="LEVEL",
        help="print the category histogram at LEVEL and exit (pick a category)",
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    items_dir = os.path.join(args.data_dir, "items")
    training_dir = os.path.join(args.data_dir, "training")
    for d in (items_dir, training_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"missing directory: {d}")

    if args.list_categories is not None:
        _, total, hist = scan_items(items_dir, None, args.list_categories, None)
        print(f"{total} items; level-{args.list_categories} categories:")
        for name, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {n:6d}  ({100.0 * n / max(total,1):5.2f}%)  {name}")
        return 0

    if not args.category and not args.keywords and not args.brand:
        raise SystemExit("give --category, --keywords or --brand (or --list_categories LEVEL)")

    targets, total_items, _ = scan_items(
        items_dir, args.category, args.category_level, args.keywords, brand=args.brand
    )
    if not targets:
        raise SystemExit(
            "no items matched. Run --list_categories <level> to see the available "
            "category names at that depth."
        )
    share = 100.0 * len(targets) / max(total_items, 1)
    print(f"[items] {len(targets)} / {total_items} matched ({share:.2f}% of catalog)")
    if share > 15.0:
        print(
            f"[warn] {share:.1f}% of the catalog is sensitive. This does NOT "
            "inflate the forget set (--forget_ratio controls that independently), "
            "but it does weaken the METRIC: Sensitive@k / SH@k asks whether any "
            "I_s item is in the top-k, and with a category this large a random "
            "model already scores high, so there is little headroom to show "
            "removal. Consider a narrower sub-category for the reported numbers."
        )

    touching, n_users, n_touch, n_total = scan_users(training_dir, targets)
    print(
        f"[users] {len(touching)} / {n_users} sessions touch a sensitive item "
        f"({100.0 * len(touching) / max(n_users,1):.2f}%), "
        f"{n_touch} of {n_total} interactions "
        f"({100.0 * n_touch / max(n_total,1):.3f}%)"
    )

    if args.forget_ratio and args.forget_ratio > 0:
        users, n_forget_inter = sample_forget_users(
            touching, n_total, args.forget_ratio, args.seed
        )
        print(
            f"[forget] ERASE-style ratio {args.forget_ratio:g}: sampled "
            f"{len(users)} users / {n_forget_inter} sensitive interactions "
            f"(realized {100.0 * n_forget_inter / max(n_total,1):.4f}% of all "
            f"interactions, budget {args.forget_ratio * n_total:.1f})"
        )
        if not users:
            print(
                "[warn] the ratio rounds to zero users. Raise --forget_ratio or "
                "pass --forget_ratio 0 to take every touching user."
            )
    else:
        users = sorted(u for u, _ in touching)
        n_forget_inter = n_touch
        print(
            f"[forget] --forget_ratio 0: taking ALL {len(users)} touching users "
            f"({100.0 * len(users) / max(n_users,1):.1f}% of the catalog's users). "
            "split_forget_retain routes BY USER, so their unrelated interactions "
            "leave the retain set too."
        )

    manifest = {
        "scenario": "sensitive",
        "deletion_spec": args.deletion_spec,
        "target_items": sorted(int(i) for i in targets),
        # Deliberately named spam_user_ids: split_forget_retain hardcodes that key,
        # so reusing it keeps the whole split/eval path unmodified.
        "spam_user_ids": [int(u) for u in users],
        "n_spam_users": len(users),
        "n_target_items": len(targets),
        "category": args.category,
        "category_level": args.category_level,
        "keywords": list(args.keywords) if args.keywords else None,
        "brand": args.brand,
        "seed": args.seed,
        "source_dataset": os.path.abspath(args.data_dir),
        "catalog_size": total_items,
        "target_share_pct": round(share, 4),
        # ERASE-style forget-set sizing (generate_forget_sets.py --forget-ratio).
        "forget_ratio": args.forget_ratio,
        "n_forget_interactions": int(n_forget_inter),
        "n_interactions_total": int(n_total),
        "realized_forget_ratio": round(n_forget_inter / max(n_total, 1), 8),
        "n_users_touching_category": len(touching),
        # ERASE has a --rating-threshold 3.5; GRID's TFRecords carry no ratings
        # (features are user_id / sequence_data / text / embedding), so every
        # interaction counts. Recorded so the deviation is not silent.
        "rating_threshold": None,
    }
    out = args.out or os.path.join(args.data_dir, "forget_manifest.json")
    # ERASE's forget-set filename, so the two benchmarks' sets are comparable.
    cat_tok = (
        "brand_" + re.sub(r"[^a-z0-9]+", "_", args.brand.lower()).strip("_")
        if args.brand
        else args.category.lower().replace(" & ", "_").replace(" ", "_")
        if args.category
        else ("_".join(args.keywords[:2]).lower() if args.keywords else "sensitive")
    )
    pairs_name = (
        f"{os.path.basename(os.path.normpath(args.data_dir))}"
        f"_unlearn_pairs_sensitive_category_{cat_tok}"
        f"_seed_{args.seed}_unlearning_fraction_{args.forget_ratio:g}.inter"
    )
    pairs_path = os.path.join(os.path.dirname(os.path.abspath(out)), pairs_name)

    if args.dry_run:
        print(f"[dry-run] would write {out}")
        print(f"[dry-run] would write {pairs_path}")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[write] {out}")
    n_pairs = write_forget_pairs(training_dir, targets, set(users), pairs_path)
    print(f"[write] {pairs_path}  ({n_pairs} interactions)")
    manifest["forget_pairs_file"] = pairs_name
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
