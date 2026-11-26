#!/usr/bin/env bash
# R2 Q2: apply NAU's neighborhood tie-break to the tuned Forget+Repair grid (Beauty validation).
#
# Tie set on the existing grid's validation metrics (paper/plotting/fr_tiebreak_table.py --tie-only):
# among F+R settings with whole-set val tau_P >= 95%, those whose per-cell val Delta_SH is not
# resolvably different from the lowest (paired seed-level 95% t-interval includes 0):
#   lambda_f=0.3 ep2 (lowest), lambda_f=0.1 ep4 (= canonical F+R, ledger forget_repair_lf0.1,
#   checkpoints kept), lambda_f=0.03 ep8.
# This re-runs the two tied settings WITHOUT a checkpointed execution on all 15 Beauty cells with
# VALSEL_KEEP_CKPT=1 (test + val + checkpoint), each replaying the cell's RECORDED `_valsel` grid
# run exactly (same poisoned ckpt file, SID, unlearning seed, data dir and unlearning.* overrides
# from its .hydra/overrides.yaml). Tag suffix _canonck (not _valsel*, so grid readers ignore them).
# Then scores every tied setting + canonical NAU on the first-token neighborhood (Figure 2
# protocol) on the TEST split and on the VALIDATION split (_valview), afterok dependency:
#   logs/diagnose/fig1_scope_frtie/ and logs/diagnose/fig1_scope_frtie_val/ (run_scope_frtie.sh).
#
#   bash run_fr_tiebreak_launch.sh             # DRY RUN (default)
#   DRYRUN=0 bash run_fr_tiebreak_launch.sh    # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
mkdir -p logs/fr_tiebreak
PLAN="logs/fr_tiebreak/canonck_plan.tsv"

python3 - <<'EOF' > "${PLAN}"
import glob, os, re, yaml
SEEDS = (2, 3, 5, 7, 11)
STRATA = ("unpopular", "mid", "popular")
ARMS = ("unified_abl_lf0.3_ls0.0_ln0.0_ep2", "unified_abl_lf0.03_ls0.0_ln0.0_ep8")
POS = ("unlearning.algorithm=", "unlearning.neighborhood_aware=",
       "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=")
jid = lambda p: int(re.match(r"job(\d+)_", os.path.basename(p)).group(1))
for arm in ARMS:
    for st in STRATA:
        for sd in SEEDS:
            sfx = "" if sd == 2 else f"_seed{sd}"
            # the recorded grid run of this cell: exact tag, finished test AND val eval, latest job
            c = [p for p in glob.glob(f"logs/unlearn/runs/job*_bs1_bw_tgt{st}_{arm}_valsel{sfx}")
                 if os.path.isfile(p + "/eval/csv/version_0/metrics.csv")
                 and os.path.isfile(p + "/eval_val/csv/version_0/metrics.csv")]
            assert c, (arm, st, sd)
            tmpl = max(c, key=jid)
            ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
            kv = dict(o.split("=", 1) for o in ov)
            useed = str(int(kv["seed"]))
            ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
            assert os.path.isfile(ckpt), ckpt; assert os.path.isfile(sid), sid
            # wrong-seed guard: the poisoned checkpoint must be this cell's
            base = ckpt.split("/checkpoints/")[0]
            toks = {t for t in STRATA if f"tgt{t}" in base}
            assert (toks == {st}) or (not toks and st == "unpopular"), ckpt
            assert (sd == 2 and "_seed" not in os.path.basename(base)) or f"_seed{sd}_" in base + "_", ckpt
            want = f"beauty_spam_seed{sd}_pct1_n1" if st == "unpopular" else f"beauty_spam_tgt{st}_seed{sd}_pct1_n1"
            assert kv["data_dir"].rstrip("/").endswith(want), kv["data_dir"]
            algo = kv["unlearning.algorithm"]
            na = kv.get("unlearning.neighborhood_aware", "false"); rbs = kv.get("unlearning.request_batch_size", "1")
            rate = kv.get("unlearning.neighborhood_aware_sample_rate", "0.0")
            extra = [o for o in ov if o.startswith("unlearning.") and not o.startswith(POS)]
            tag = f"bw_tgt{st}_{arm}_canonck{sfx}"
            print("\t".join(["beauty", st, str(sd), useed, tag, ckpt, algo, sid, na, rbs, rate,
                             kv["data_dir"], tmpl, " ".join(extra)]))
EOF
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^(vsb_frtie|scope_frtie)' <<<"${QUEUED}"; then
  echo "Refusing: an frtie job is already queued/running:"; grep -E '^(vsb_frtie|scope_frtie)' <<<"${QUEUED}"; exit 1
fi
todo=0
while IFS=$'\t' read -r ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  exists: ${tag}"; continue; fi
  resolved="$(POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1 POISON_SEED="${sd}" TARGET_STRATEGY="${st}" \
    bash -c 'source scripts/resolve_unlearn_dataset.sh >/dev/null 2>&1; resolve_unlearn_dataset "$0" >/dev/null 2>&1; echo "${UNLEARN_DATA_DIR}"' "${ds}")"
  [ "${resolved}" = "${data_dir}" ] || { echo "MISMATCH ${tag}: ${resolved} vs ${data_dir}" >&2; exit 1; }
  echo "  todo:   ${tag}  <- $(basename "${tmpl}")  [${algo} ${extra}]"
  todo=$((todo+1))
done < "${PLAN}"

dep=()
if [ "${todo}" -gt 0 ]; then
  ucmd=(sbatch --parsable --job-name=vsb_frtie_canonck --time=08:00:00 run_valsel_batch.sh "${PLAN}")
  if [ "${DRYRUN}" = "1" ]; then printf 'VALSEL_KEEP_CKPT=1'; printf ' %q' "${ucmd[@]}"; printf '\n'; UJID="<unlearn_jid>"
  else UJID="$(VALSEL_KEEP_CKPT=1 "${ucmd[@]}")"; echo "submitted unlearning batch ${UJID}"; fi
  dep=(--dependency=afterok:"${UJID}")
fi
for split in test val; do
  scmd=(sbatch --parsable --job-name=scope_frtie_${split} "${dep[@]}" --export=ALL,SPLIT=${split} run_scope_frtie.sh)
  if [ "${DRYRUN}" = "1" ]; then printf ' %q' "${scmd[@]}"; printf '\n'
  else SJID="$("${scmd[@]}")"; echo "submitted scope eval (${split}) ${SJID} (${dep[*]:-no dependency})"; fi
done
[ "${DRYRUN}" = "1" ] && echo "DRY RUN (DRYRUN=0 to submit)"
exit 0
