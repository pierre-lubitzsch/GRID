"""Calibration check (critique follow-up, addendum 2): is there an NAU setting whose first-token
neighborhood gain over Forget+Repair is a repair, i.e. the trained neighbors (A) stay near the
retrained level instead of being over-boosted, while the rest of the neighborhood (B) is not worse?

Beauty, 15 cells (3 strata x 5 seeds), test and validation. Inputs: logs/diagnose/calib_scope_none{,_val}/
(run_calib_scope_batch.sh, plan scripts/plan_calibration_scope.py): every held-out row with its next
item, scored by clean (retrained), start, canonical NAU (lambda_s=lambda_n=0.01, k=8), canonical
Forget+Repair, the NAU weight grid with checkpoints, the k sweep, and (if present) step-2 arms.

First-token neighborhood = held-out cases whose next item shares the target's first SID token
(target excluded). A8 = those whose next item is among the target's 8 nearest content-embedding items
(the canonical NAU's training neighbors), B8 = the rest. For the k sweep also A_own / B_own with the
run's own k nearest. Levels: pooled NDCG@10 as % of the retrained model on the same cases (95%
paired bootstrap over cases for A). Gains: arm minus Forget+Repair in points of the retrained
NDCG on that subset, paired bootstrap 95% CI. Whole-set Delta^retr_SH and tau_P from each run's own
eval / eval_val (as paper/plotting/nau_k_table.py; mean over seeds of the mean over strata).

    PYTHONPATH=. python paper/plotting/calibration_table.py [--out paper/revision/calibration_check.md]
"""
import argparse, glob, importlib.util, json, os, re, statistics as S, sys
from collections import defaultdict
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "critique_followup"))
from knn_excluding_trained_all import load_codes, load_emb  # noqa: E402
spec = importlib.util.spec_from_file_location("vt", os.path.join(HERE, "valsel_table.py"))
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)

SEEDS = (2, 3, 5, 7, 11); STRATA = ("unpopular", "mid", "popular")
# arm -> (label, k, group)
ARMS = [("forget_repair", "Forget+Repair (ls 0, ln 0)", 8, "grid"),
        ("g_s0_n0p01", "ls 0, ln 0.01", 8, "grid"),
        ("g_s0_n0p1", "ls 0, ln 0.1", 8, "grid"),
        ("g_s0p01_n0", "ls 0.01, ln 0", 8, "grid"),
        ("nau", "**ls 0.01, ln 0.01 (NAU, k 8)**", 8, "grid"),
        ("g_s0p1_n0", "ls 0.1, ln 0", 8, "grid"),
        ("g_s0p1_n0p01", "ls 0.1, ln 0.01", 8, "grid"),
        ("g_s0p1_n0p1", "ls 0.1, ln 0.1", 8, "grid"),
        ("nau_k2", "k 2", 2, "k"), ("nau_k4", "k 4", 4, "k"), ("nau", "**k 8 (NAU)**", 8, "k"),
        ("nau_k16", "k 16", 16, "k"), ("nau_k32", "k 32", 32, "k"), ("nau_k64", "k 64", 64, "k"),
        ("ln0p001", "ls 0.01, ln 0.001", 8, "ln"), ("ln0p003", "ls 0.01, ln 0.003", 8, "ln"),
        ("nau", "**ls 0.01, ln 0.01 (NAU)**", 8, "ln")]
STEM = os.environ.get("CALIB_STEM", "calib_scope_none")
N_BOOT = 3000


def jpath(st, sd, split):
    v = "" if split == "test" else "_val"
    return f"logs/diagnose/{STEM}{v}/" + (f"beauty_seed{sd}.json" if st == "mid" else f"beauty_{st}_seed{sd}.json")


def run_dir(ckpt):
    p = os.path.dirname(ckpt)
    if os.path.basename(p) == "checkpoints":
        p = os.path.dirname(p)
    return "logs/unlearn/runs/" + os.path.basename(p)


def pool(split, codes, emb):
    """(subset, arm) -> list of per-case NDCG; also the per-arm run dirs per cell."""
    P, dirs, cells = defaultdict(list), defaultdict(dict), 0
    for st in STRATA:
        for sd in SEEDS:
            f = jpath(st, sd, split)
            if not os.path.isfile(f):
                continue
            d = json.load(open(f)); t = d["target_item"]
            items = np.array(d["items_per_query"]["unrelated"])
            r = {x["method"]: np.array(x["ndcg_per_query"]) for x in d["rows"] if x["group"] == "unrelated"}
            f2 = f.replace(f"/{STEM}", f"/{STEM}_ln", 1)   # step-2 arms (lambda_n sweep), same rows
            if os.path.isfile(f2):
                d2 = json.load(open(f2))
                assert d2["target_item"] == t and d2["items_per_query"]["unrelated"] == d["items_per_query"]["unrelated"], f2
                r2 = {x["method"]: np.array(x["ndcg_per_query"]) for x in d2["rows"] if x["group"] == "unrelated"}
                for m in ("clean", "forget_repair", "nau"):   # deterministic re-scoring check
                    if not np.allclose(r2[m], r[m]):
                        print(f"WARNING: {m} re-scored differently in {f2}", file=sys.stderr)
                for m in ("ln0p001", "ln0p003"):
                    if m in r2:
                        r[m] = r2[m]; d["checkpoints"][m] = d2["checkpoints"][m]
            sims = emb @ emb[t]; sims[t] = -1e9
            order = torch.topk(sims, 64).indices.numpy()
            first = (codes[items, 0] == codes[t, 0]).numpy() & (items != t)
            masks = {"first": first}
            for k in (2, 4, 8, 16, 32, 64):
                ink = np.isin(items, order[:k])
                masks[f"A{k}"] = first & ink; masks[f"B{k}"] = first & ~ink
            for name, m in masks.items():
                for meth, v in r.items():
                    P[(name, meth)].extend(v[m].tolist())
            for meth, ck in d["checkpoints"].items():
                dirs[meth][(st, str(sd))] = run_dir(ck)
            cells += 1
    return {k: np.asarray(v) for k, v in P.items()}, dirs, cells


def ci_diff(x, y, ref):
    s = 100 / ref.mean(); d = x - y
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return s * d.mean(), s * lo, s * hi


def whole(dirs):
    """arm -> (val dSH, val tau, test dSH, test tau, n cells) as in nau_k_table.collect()."""
    R = vt.refs()
    test_ref = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("paper/data/spam_references/*.json")}
    out = {}
    for arm, cd in dirs.items():
        if arm in ("clean", "poison"):
            continue
        per = defaultdict(list)
        for (st, sd), d in cd.items():
            te, va = vt.metrics(d + "/eval"), vt.metrics(d + "/eval_val")
            cv = R.get(("rkmeans", "beauty", st, sd, "clean"))
            rt = test_ref.get("beauty" + ("" if sd == "2" else f"_seed{sd}"))
            if te and va and cv and rt:
                per[sd].append((abs(va[0] - cv[0]), va[1] / cv[1], abs(te[0] - rt["clean_SH@10"][st]), te[1] / rt["clean_ndcg@10"]))
        n = sum(len(v) for v in per.values())
        agg = lambda j: 100 * S.mean(S.mean(x[j] for x in v) for v in per.values()) if per else float("nan")
        out[arm] = (agg(0), agg(1), agg(2), agg(3), n)
    return out


def fmt(t, sign=True):
    return (f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]" if sign else f"{t[0]:.0f} [{t[1]:.0f}, {t[2]:.0f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/revision/calibration_check.md")
    ap.add_argument("--reading", default="paper/revision/calibration_reading.md",
                    help="hand-written reading appended to the generated tables (if the file exists)")
    a = ap.parse_args()
    codes = load_codes("beauty"); emb = load_emb("beauty", codes.shape[0])
    L = ["# Calibration check: does any NAU setting repair the neighborhood?", "",
         "Generated by `paper/plotting/calibration_table.py` (do not edit the tables by hand). "
         "Beauty, TIGER + RQ-KMeans, 3 strata x 5 seeds, lambda_f = 0.1 throughout. Inputs "
         f"`logs/diagnose/{STEM}{{,_val}}/` (jobs listed in PROGRESS_rebuttal.md). "
         "First-token neighborhood = held-out cases whose next item shares the target's first SID token. "
         "A8 = those whose next item is one of the target's 8 nearest content-embedding items (the canonical NAU's "
         "training neighbors); B8 = the rest. Levels are pooled NDCG@10 in % of the retrained model on the same cases "
         "(\"A8 minus retr.\": the arm's excess over the retrained model on A8 in points, 95% paired bootstrap). Gains are arm minus Forget+Repair in points of the retrained NDCG on "
         "that subset, 95% paired bootstrap over cases. Whole-set Delta^retr_SH (%, lower is better) and tau_P (%) "
         "come from each run's own evaluation of the same split. A setting *repairs* if A8 is near 100%, B8 is not "
         "below Forget+Repair, and the first-token gain is positive.", ""]
    summary = {}
    for split in ("test", "val"):
        P, dirs, cells = pool(split, codes, emb)
        if not cells:
            L += [f"## {split}: no inputs yet", ""]; continue
        W = whole(dirs)
        have = {m for (_, m) in P}
        ref = {s: P[(s, "clean")] for s in ("first", "A8", "B8")}
        fr = {s: P[(s, "forget_repair")] for s in ref}
        L += [f"## {'Test' if split == 'test' else 'Validation'} split ({cells}/15 cells; first-token n = {len(ref['first'])}, "
              f"A8 n = {len(ref['A8'])} ({100*len(ref['A8'])/len(ref['first']):.1f}%))", "",
              "| Setting | A8 % of retr. | A8 minus retr. | B8 % of retr. | Gain over F+R: A8 | B8 | first-token | whole dSH | whole tau_P |",
              "|---|---|---|---|---|---|---|---|---|"]
        for nm, lab in (("clean", "Retrained"), ("poison", "Start (poisoned)")):
            ex = "" if nm == "clean" else fmt(ci_diff(P[("A8", nm)], ref["A8"], ref["A8"]))
            L.append(f"| {lab} | {100*P[('A8', nm)].mean()/ref['A8'].mean():.0f} | {ex} | {100*P[('B8', nm)].mean()/ref['B8'].mean():.1f} | | | | | |")
        groups_done = set()
        for arm, lab, k, grp in ARMS:
            if arm not in have or (grp == "ln" and not any(x in have for x in ("ln0p001", "ln0p003"))):
                continue
            if grp not in groups_done:
                groups_done.add(grp)
                head = {"grid": "*weight grid, k 8*", "k": "*k sweep, ls 0.01, ln 0.01*", "ln": "*lambda_n below 0.01 (step 2), k 8*"}[grp]
                L.append(f"| {head} | | | | | | | | |")
            lvA = 100 * P[("A8", arm)].mean() / ref["A8"].mean(); exA = ci_diff(P[("A8", arm)], ref["A8"], ref["A8"]); lvB = 100 * P[("B8", arm)].mean() / ref["B8"].mean()
            if arm == "forget_repair":
                gA = gB = gF = None
            else:
                gA, gB, gF = (ci_diff(P[(s, arm)], fr[s], ref[s]) for s in ("A8", "B8", "first"))
            w = W.get(arm, (float("nan"),) * 5)
            dsh, tau = (w[2], w[3]) if split == "test" else (w[0], w[1])
            g = lambda t: "" if t is None else fmt(t)
            L.append(f"| {lab} | {lvA:.0f} | {fmt(exA)} | {lvB:.1f} | {g(gA)} | {g(gB)} | {g(gF)} | {dsh:.3f} | {tau:.1f} |")
            summary[(split, grp, arm)] = (lvA, exA, lvB, gA, gB, gF, dsh, tau)
        # own-k split for the k sweep
        L += ["", f"k sweep with each run's own trained set (A_k = next item among the target's k nearest), {split}:", "",
              "| k | A_k n | A_k % of retr.: F+R / NAU_k | B_k % of retr.: F+R / NAU_k | Gain over F+R: A_k | B_k |", "|---|---|---|---|---|---|"]
        for arm, lab, k, grp in ARMS:
            if grp != "k" or arm not in have:
                continue
            ra, rb = P[(f"A{k}", "clean")], P[(f"B{k}", "clean")]
            fa, fb = P[(f"A{k}", "forget_repair")], P[(f"B{k}", "forget_repair")]
            xa, xb = P[(f"A{k}", arm)], P[(f"B{k}", arm)]
            L.append(f"| {k} | {len(ra)} | {100*fa.mean()/ra.mean():.0f} / {100*xa.mean()/ra.mean():.0f} | "
                     f"{100*fb.mean()/rb.mean():.1f} / {100*xb.mean()/rb.mean():.1f} | {fmt(ci_diff(xa, fa, ra))} | {fmt(ci_diff(xb, fb, rb))} |")
        L.append("")
    if os.path.isfile(a.reading):
        L += [open(a.reading).read().rstrip(), ""]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote", a.out)


if __name__ == "__main__":
    main()
