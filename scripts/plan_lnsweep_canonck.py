"""Step 2 of the calibration check: NAU at lambda_n below 0.01 (lambda_f=0.1, lambda_s=0.01, k=8) on the
15 Beauty cells, replaying the canonical NAU runs (`*_nau_s0p01_n0p01_keepck`) exactly except lambda_n,
checkpoints kept, test + validation eval (run_valsel_batch.sh rows). The rows are the k sweep rows of
logs/nau_k_keepck_plan.tsv (verified to differ from the canonical config only in neighborhood_count and
inert keys) with neighborhood_count=8 and lambda_n replaced. Tags `bw_tgt<st>_nau_s0p01_n0p00X_lnsweep_canonck[_seed<N>]`
(no `keepck`, so no canonical reader picks them up). Submit with VALSEL_KEEP_CKPT=1, or
run_valsel_one.sh deletes the checkpoint after the evals (that happened to the first submission, tags
`_lnsweep_canonck`: whole-set metrics only; the resubmission uses `_lnsweep2_canonck`). Writes logs/valsel_batches/lnsweep2_canonck_<i>.tsv.
    python scripts/plan_lnsweep_canonck.py [lambda_n ...]   (default 0.001 0.003)"""
import sys
lns = sys.argv[1:] or ["0.001", "0.003"]
rows = []
for line in open("logs/nau_k_keepck_plan.tsv"):
    f = line.rstrip("\n").split("\t")
    if "_k4_keepck" not in f[4]:
        continue
    for ln in lns:
        g = list(f)
        g[4] = f[4].replace("_n0p01_k4_keepck", f"_n{ln.replace('.', 'p')}_lnsweep2_canonck")
        ex = g[13].split()
        ex = [("unlearning.lambda_n=" + ln) if e.startswith("unlearning.lambda_n=") else
              "unlearning.neighborhood_count=8" if e.startswith("unlearning.neighborhood_count=") else e for e in ex]
        assert "unlearning.lambda_n=" + ln in ex and "unlearning.neighborhood_count=8" in ex
        g[13] = " ".join(ex); rows.append("\t".join(g))
assert len(rows) == 15 * len(lns), len(rows)
n = 3
for i in range(n):
    open(f"logs/valsel_batches/lnsweep2_canonck_{i}.tsv", "w").write("\n".join(rows[i::n]) + "\n")
print(len(rows), "rows ->", [f"logs/valsel_batches/lnsweep2_canonck_{i}.tsv" for i in range(n)])
