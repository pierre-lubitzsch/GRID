"""Scrubbed val/test views for the unwanted-item ("sensitive") scenario.

WHY. `build_retrain_dataset.py` removes each requesting user's unwanted items
from that user's TRAINING row only and copies `evaluation/` and `testing/`
verbatim. So at val/test time the requesting users' input histories still hold
the deleted interactions, for every model including the retrained reference. A
deployed system would delete them from the stored history too. This builds a
view in which they are gone, so the paper's metrics can be re-measured on it.

WHAT IS REMOVED. Exactly what build_retrain_dataset.py removes, read from the
same manifest (the `manifest` field of the retrain dir's retrain_meta.json, which
is byte-identical to the sens dir's forget_manifest.json): for each user u in
`spam_user_ids`, D_u = target_items  intersect  items(u's training row). Every
occurrence of an item in D_u is dropped from u's val and test HISTORIES. Labels
are never changed; a label in D_u is kept and counted.

HOW THE WINDOW IS REBUILT. The splits are leave-one-out with a 20-item history:
    training   = s[:-2]                (full)
    evaluation = (s[:-2] + [s[-2]])[-W:]
    testing    = (s[:-1] + [s[-1]])[-W:]      W = 21 (20 history + 1 label)
The script first CHECKS that rule reproduces every user's original eval/test row
(sequence, text and embedding), then applies it to the scrubbed sequence, so a
removed item lets an older interaction re-enter the 20-item window, as it would
in a deployed system. (`--no_refill` instead only drops items inside the
existing window.) The per-position `text` and `embedding` features are kept
aligned with `sequence_data`.

LAYOUT. Per (dataset, seed), next to the existing `<ds>_sens_<cat>_1e-4_seed<sd>`:

    <sens>_scrubtestview/   testing/, evaluation/: real dirs. Partition files with
                            no requesting user are symlinks to the originals;
                            files holding one are rewritten, with every other
                            record copied as its original serialized bytes.
                            training, items, manifests, training_* -> symlinks.
                            scrub_meta.json: counts and per-user detail.
    <sens>_scrubvalview/    like `_valview`: testing -> scrubbed evaluation, so
                            `trainer.test` scores the scrubbed validation split.

    python -m scripts.build_scrubbed_eval_view --datasets beauty sports toys
    python -m scripts.build_scrubbed_eval_view --verify_only   # re-run checks
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Set

import tensorflow as tf

ROOT = "src/data/amazon_data"
CATEGORY = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
# forget-ratio token of the sens/retrain dirs; --ratio overrides (default 1e-4 = the paper setting)
RATIO = "1e-4"
SEQ, UID, TXT, EMB = "sequence_data", "user_id", "text", "embedding"
SPLITS = ("evaluation", "testing")


def shards(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, "*.tfrecord.gz")))


def records(path: str):
    for r in tf.data.TFRecordDataset(path, compression_type="GZIP"):
        yield r.numpy()


def parse(raw: bytes):
    return tf.train.Example.FromString(raw)


def uid_of(ex) -> int:
    return int(ex.features.feature[UID].int64_list.value[0])


def positions(ex):
    """Per-position (item, text, embedding-slice) tuples of one row."""
    f = ex.features.feature
    seq = list(f[SEQ].int64_list.value)
    txt = list(f[TXT].bytes_list.value)
    emb = list(f[EMB].float_list.value)
    assert len(txt) == len(seq), "text not per-position"
    assert len(emb) % max(len(seq), 1) == 0, "embedding not per-position"
    d = len(emb) // max(len(seq), 1)
    return [(int(seq[i]), txt[i], tuple(emb[i * d:(i + 1) * d])) for i in range(len(seq))]


def make_row(template, pos, uid):
    ex = tf.train.Example()
    ex.CopyFrom(template)
    f = ex.features.feature
    del f[SEQ].int64_list.value[:]
    f[SEQ].int64_list.value.extend(p[0] for p in pos)
    del f[TXT].bytes_list.value[:]
    f[TXT].bytes_list.value.extend(p[1] for p in pos)
    del f[EMB].float_list.value[:]
    for p in pos:
        f[EMB].float_list.value.extend(p[2])
    assert uid_of(ex) == uid
    return ex.SerializeToString()


def sens_name(ds: str, sd: int) -> str:
    return f"{ds}_sens_{CATEGORY[ds]}_{RATIO}_seed{sd}"


def load_dataset(ds: str, keep_users: Set[int]):
    """One pass over the clean dataset: sequences of every user (for the
    reconstruction check), full positions of `keep_users`, and where each
    user's eval/test row lives."""
    base = os.path.join(ROOT, ds)
    train_seq: Dict[int, List[int]] = {}
    train_pos: Dict[int, list] = {}
    for sh in shards(os.path.join(base, "training")):
        for raw in records(sh):
            ex = parse(raw)
            u = uid_of(ex)
            assert u not in train_seq, f"{ds}: user {u} has two training rows"
            train_seq[u] = [int(x) for x in ex.features.feature[SEQ].int64_list.value]
            if u in keep_users:
                train_pos[u] = positions(ex)
    split_rows = {}
    for sp in SPLITS:
        rows, where = {}, {}
        for sh in shards(os.path.join(base, sp)):
            for raw in records(sh):
                ex = parse(raw)
                u = uid_of(ex)
                assert u not in rows, f"{ds}/{sp}: user {u} has two rows"
                rows[u] = [int(x) for x in ex.features.feature[SEQ].int64_list.value]
                where[u] = os.path.basename(sh)
        split_rows[sp] = (rows, where)
    return train_seq, train_pos, split_rows


def check_window_rule(ds, train_seq, split_rows) -> int:
    """W such that eval = (train+[val])[-W:] and test = (train+[val]+[test])[-W:]
    for EVERY user; raises if the rule does not reproduce the data."""
    ev, _ = split_rows["evaluation"]
    te, _ = split_rows["testing"]
    W = max(len(r) for r in te.values())
    assert set(ev) == set(te) == set(train_seq), f"{ds}: user sets differ across splits"
    bad = 0
    for u, tr in train_seq.items():
        val, tst = ev[u][-1], te[u][-1]
        if ev[u] != (tr + [val])[-W:] or te[u] != (tr + [val, tst])[-W:]:
            bad += 1
    if bad:
        raise SystemExit(f"{ds}: window rule fails for {bad} users; use --no_refill")
    print(f"[{ds}] window rule (W={W}) reproduces eval/test rows of all {len(train_seq)} users")
    return W


def build_seed(ds, sd, train_seq, train_pos, split_rows, W, refill, dry):
    sens = os.path.join(ROOT, sens_name(ds, sd))
    retrain = os.path.join(ROOT, f"{ds}_retrain_{CATEGORY[ds]}_{RATIO}_seed{sd}")
    rmeta = json.load(open(os.path.join(retrain, "retrain_meta.json")))
    man_path = rmeta["manifest"]
    man = json.load(open(man_path))
    with open(os.path.join(sens, "forget_manifest.json"), "rb") as a, open(man_path, "rb") as b:
        assert a.read() == b.read(), "sens manifest != retrain manifest"
    targets = {int(i) for i in man["target_items"]}
    users = sorted(int(u) for u in man["spam_user_ids"])
    D = {u: targets & set(train_seq[u]) for u in users}
    n_train_removed = sum(sum(1 for x in train_seq[u] if x in targets) for u in users)
    assert n_train_removed == rmeta["interactions_removed"], (n_train_removed, rmeta)

    out = os.path.join(ROOT, sens_name(ds, sd) + "_scrubtestview")
    valv = os.path.join(ROOT, sens_name(ds, sd) + "_scrubvalview")
    meta = {"dataset": ds, "seed": sd, "category": man.get("category"), "manifest": man_path,
            "retrain_dir": retrain, "refill": refill, "window_items": W,
            "n_requesting_users": len(users),
            "n_users_with_deleted_items": sum(1 for u in users if D[u]),
            "train_interactions_removed_by_retrain_builder": n_train_removed,
            "splits": {}, "users": {}}
    new_rows = {sp: {} for sp in SPLITS}
    ev, _ = split_rows["evaluation"]
    te, _ = split_rows["testing"]
    for sp in SPLITS:
        rows, where = split_rows[sp]
        st = dict(rows_changed=0, history_occurrences_removed=0, label_in_deleted=0,
                  label_is_unwanted_item=0, unwanted_items_left_in_history=0,
                  rows_refilled=0, empty_history=0, files_rewritten=0)
        for u in users:
            orig = rows[u]
            hist_orig, label = orig[:-1], orig[-1]
            if sp == "evaluation":
                full_hist = list(train_pos[u])
            else:
                # val label position: from the user's own eval row (last position)
                full_hist = list(train_pos[u]) + [split_rows["_evpos"][u]]
            row_pos = split_rows["_rowpos"][sp][u]
            lab_pos = row_pos[-1]
            assert lab_pos[0] == label
            # the window rule must reproduce the ORIGINAL row incl. text/embedding
            assert (full_hist + [lab_pos])[-W:] == row_pos, (ds, sp, u)
            if refill:
                kept = [p for p in full_hist if p[0] not in D[u]]
                new = (kept + [lab_pos])[-W:]
                # occurrences removed = those inside the ORIGINAL window
                removed = sum(1 for x in hist_orig if x in D[u])
            else:
                win = (full_hist + [lab_pos])[-W:]
                new = [p for p in win[:-1] if p[0] not in D[u]] + [lab_pos]
                removed = sum(1 for x in hist_orig if x in D[u])
            new_seq = [p[0] for p in new]
            st["history_occurrences_removed"] += removed
            st["label_in_deleted"] += int(label in D[u])
            st["label_is_unwanted_item"] += int(label in targets)
            st["unwanted_items_left_in_history"] += sum(1 for x in new_seq[:-1] if x in targets)
            st["empty_history"] += int(len(new_seq) == 1)
            if new_seq != orig:
                st["rows_changed"] += 1
                if refill and removed and len(new_seq) - 1 > len(hist_orig) - removed:
                    st["rows_refilled"] += 1
                new_rows[sp][u] = new
            meta["users"].setdefault(str(u), {"deleted_items": sorted(D[u])})[sp] = {
                "orig": orig, "scrubbed": new_seq, "removed_in_window": removed,
                "label_in_deleted": label in D[u]}
        files = sorted({where[u] for u in new_rows[sp]})
        st["files_rewritten"] = len(files)
        meta["splits"][sp] = st
        print(f"[{ds} s{sd}] {sp}: {st}")
        if dry:
            continue
        d = os.path.join(out, sp)
        os.makedirs(d, exist_ok=True)
        src_dir = os.path.realpath(os.path.join(ROOT, ds, sp))
        opts = tf.io.TFRecordOptions(compression_type="GZIP")
        for sh in shards(src_dir):
            name = os.path.basename(sh)
            dst = os.path.join(d, name)
            if os.path.lexists(dst):
                raise SystemExit(f"{dst} exists; refusing to overwrite (views are write-once)")
            if name not in files:
                os.symlink(os.path.relpath(sh, d), dst)
                continue
            with tf.io.TFRecordWriter(dst, options=opts) as w:
                for raw in records(sh):
                    u = uid_of(parse(raw))
                    if u in new_rows[sp]:
                        tmpl = parse(raw)
                        w.write(make_row(tmpl, new_rows[sp][u], u))
                    else:
                        w.write(raw)
    if dry:
        return meta
    for name in ("training", "items"):
        os.symlink(os.path.relpath(os.path.realpath(os.path.join(ROOT, ds, name)), out),
                   os.path.join(out, name))
    for name in ("forget_manifest.json", "forget_retain_split.json", "training_forget",
                 "training_retain", "training_forget_item_pairs"):
        if os.path.lexists(os.path.join(sens, name)):
            os.symlink(os.path.join("..", sens_name(ds, sd), name), os.path.join(out, name))
    with open(os.path.join(out, "scrub_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    os.makedirs(valv, exist_ok=False)
    tv = os.path.basename(out)
    for name in os.listdir(out):
        if name in ("testing", "evaluation"):
            continue
        os.symlink(os.path.join("..", tv, name), os.path.join(valv, name))
    os.symlink(os.path.join("..", tv, "evaluation"), os.path.join(valv, "evaluation"))
    os.symlink(os.path.join("..", tv, "evaluation"), os.path.join(valv, "testing"))
    return meta


def verify(ds, sd):
    """Record-level check of a built view against the original split."""
    out = os.path.join(ROOT, sens_name(ds, sd) + "_scrubtestview")
    meta = json.load(open(os.path.join(out, "scrub_meta.json")))
    req = {int(u) for u in meta["users"]}
    res = {}
    for sp in SPLITS:
        src = os.path.join(ROOT, ds, sp)
        a = shards(src)
        b = shards(os.path.join(out, sp))
        assert [os.path.basename(x) for x in a] == [os.path.basename(x) for x in b]
        n_same = n_req = n_req_changed = n_links = 0
        for x, y in zip(a, b):
            if os.path.islink(y) and os.path.realpath(y) == os.path.realpath(x):
                n_links += 1
                for raw in records(x):
                    u = uid_of(parse(raw))
                    if u in req:   # allowed only if scrubbing left the row unchanged
                        r = meta["users"][str(u)][sp]
                        assert r["orig"] == r["scrubbed"], (u, sp)
                        n_req += 1
                    else:
                        n_same += 1
                continue
            ra, rb = list(records(x)), list(records(y))
            assert len(ra) == len(rb), (x, y)
            for p, q in zip(ra, rb):
                ua, ub = uid_of(parse(p)), uid_of(parse(q))
                assert ua == ub, "row order changed"
                if ua in req:
                    n_req += 1
                    pe, qe = parse(p), parse(q)
                    want = meta["users"][str(ua)][sp]["scrubbed"]
                    got = [int(v) for v in qe.features.feature[SEQ].int64_list.value]
                    assert got == want, (ua, got, want)
                    pq = positions(qe)
                    assert len(pq) == len(got)
                    n_req_changed += int(p != q)
                else:
                    assert p == q, f"non-requesting user {ua} not byte-identical"
                    n_same += 1
        res[sp] = dict(files_symlinked=n_links, files_rewritten=len(a) - n_links,
                       non_requesting_rows_identical=n_same, requesting_rows_in_rewritten=n_req,
                       requesting_rows_changed=n_req_changed)
    print(f"[verify {ds} s{sd}] {res}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(CATEGORY))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--no_refill", action="store_true")
    ap.add_argument("--dry", action="store_true", help="counts only, write nothing")
    ap.add_argument("--verify_only", action="store_true")
    ap.add_argument("--ratio", default="1e-4",
                    help="forget-ratio token of the sens/retrain dirs (e.g. 5e-4 for the larger-request variant)")
    ap.add_argument("--category", default=None,
                    help="category token of the sens/retrain dirs for the given datasets, overriding the "
                         "default (e.g. takedown_elf with --ratio 0 for beauty_sens_takedown_elf_0_seed<N>)")
    a = ap.parse_args()
    global RATIO
    RATIO = a.ratio
    if a.category:
        for ds in a.datasets:
            CATEGORY[ds] = a.category
    summary = {}
    for ds in a.datasets:
        if a.verify_only:
            for sd in a.seeds:
                summary[f"{ds}_s{sd}"] = verify(ds, sd)
            continue
        req = set()
        for sd in a.seeds:
            m = json.load(open(os.path.join(ROOT, sens_name(ds, sd), "forget_manifest.json")))
            req |= {int(u) for u in m["spam_user_ids"]}
        train_seq, train_pos, split_rows = load_dataset(ds, req)
        W = check_window_rule(ds, train_seq, split_rows)
        # label positions (item, text, emb) of requesting users, from the original rows
        split_rows["_rowpos"] = {sp: {} for sp in SPLITS}
        split_rows["_evpos"] = {}
        for sp in SPLITS:
            _, where = split_rows[sp]
            for sh in shards(os.path.join(ROOT, ds, sp)):
                if os.path.basename(sh) not in {where[u] for u in req}:
                    continue
                for raw in records(sh):
                    ex = parse(raw)
                    u = uid_of(ex)
                    if u in req:
                        rp = positions(ex)
                        split_rows["_rowpos"][sp][u] = rp
                        if sp == "evaluation":
                            split_rows["_evpos"][u] = rp[-1]
        for sd in a.seeds:
            m = build_seed(ds, sd, train_seq, train_pos, split_rows, W, not a.no_refill, a.dry)
            summary[f"{ds}_s{sd}"] = {k: m[k] for k in ("n_requesting_users", "n_users_with_deleted_items",
                                                         "train_interactions_removed_by_retrain_builder")}
            summary[f"{ds}_s{sd}"].update({sp: m["splits"][sp] for sp in SPLITS})
            if not a.dry:
                summary[f"{ds}_s{sd}"]["verify"] = verify(ds, sd)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
