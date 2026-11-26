#!/usr/bin/env bash
# Figure 1 fix, option (a) of paper/revision/canonical_runs_plan.md 3.2.
#
# Re-runs Table 1's Forget only (unified_abl_lf1.0_ls0.0_ln0.0_lr0.0: lambda_f=1, lambda_r=0) and
# Fanchuan t0.07 on Beauty mid x seeds 2,3,5,7,11 (10 runs) through run_valsel_batch.sh ->
# run_valsel_one.sh (unlearn + test eval + validation eval) with VALSEL_KEEP_CKPT=1, each
# replaying the RECORDED Table 1 run of its cell exactly (same poisoned ckpt file, SID, unlearning
# seed, data dir and unlearning.* overrides from its .hydra/overrides.yaml, as run_valsel_launch.sh
# does). Tags get the suffix _canonck (NOT _valsel*, so valsel_table.py's *_valsel* glob ignores
# them). Then scores them on the Figure 1 neighborhood evaluation (run_fig1_fix_eval.sh,
# afterok dependency) -> logs/diagnose/fig1_scope_allunrel_fix/beauty_seed<N>.json.
#
#   bash run_fig1_fix_launch.sh             # DRY RUN (default)
#   DRYRUN=0 bash run_fig1_fix_launch.sh    # submit both jobs
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
mkdir -p logs/fig1_fix
PLAN="logs/fig1_fix/canonck_plan.tsv"

python3 - <<'EOF' > "${PLAN}"
import glob, os, re, sys, yaml
SEEDS = (2, 3, 5, 7, 11)
ARMS = ("unified_abl_lf1.0_ls0.0_ln0.0_lr0.0", "fanchuan_t0.07")
POS = ("unlearning.algorithm=", "unlearning.neighborhood_aware=",
       "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=")
jid = lambda p: int(re.match(r"job(\d+)_", os.path.basename(p)).group(1))
for arm in ARMS:
    for sd in SEEDS:
        sfx = "" if sd == 2 else f"_seed{sd}"
        # the recorded Table 1 run of this cell: exact tag, finished test eval, latest job
        c = [p for p in glob.glob(f"logs/unlearn/runs/job*_bs1_bw_tgtmid_{arm}{sfx}")
             if os.path.isfile(p + "/eval/csv/version_0/metrics.csv")]
        assert c, (arm, sd)
        tmpl = max(c, key=jid)
        ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
        kv = dict(o.split("=", 1) for o in ov)
        useed = str(int(kv["seed"]))
        ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
        assert os.path.isfile(ckpt), ckpt; assert os.path.isfile(sid), sid
        # wrong-seed guard: the poisoned checkpoint must be this cell's
        base = ckpt.split("/checkpoints/")[0]
        assert "tgtmid" in base and ((sd == 2 and "_seed" not in os.path.basename(base)) or f"_seed{sd}_" in base + "_"), ckpt
        assert kv["data_dir"].rstrip("/").endswith(f"beauty_spam_tgtmid_seed{sd}_pct1_n1"), kv["data_dir"]
        algo = kv["unlearning.algorithm"]
        na = kv.get("unlearning.neighborhood_aware", "false"); rbs = kv.get("unlearning.request_batch_size", "1")
        rate = kv.get("unlearning.neighborhood_aware_sample_rate", "0.0")
        extra = [o for o in ov if o.startswith("unlearning.") and not o.startswith(POS)]
        tag = f"bw_tgtmid_{arm}_canonck{sfx}"
        print("\t".join(["beauty", "mid", str(sd), useed, tag, ckpt, algo, sid, na, rbs, rate,
                         kv["data_dir"], tmpl, " ".join(extra)]))
EOF
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^(vsb_fig1fix|fig1fix_scope)' <<<"${QUEUED}"; then
  echo "Refusing: a fig1fix job is already queued/running:"; grep -E '^(vsb_fig1fix|fig1fix_scope)' <<<"${QUEUED}"; exit 1
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
  ucmd=(sbatch --parsable --job-name=vsb_fig1fix_canonck --time=03:00:00 run_valsel_batch.sh "${PLAN}")
  if [ "${DRYRUN}" = "1" ]; then printf 'VALSEL_KEEP_CKPT=1'; printf ' %q' "${ucmd[@]}"; printf '\n'; UJID="<unlearn_jid>"
  else UJID="$(VALSEL_KEEP_CKPT=1 "${ucmd[@]}")"; echo "submitted unlearning batch ${UJID}"; fi
  dep=(--dependency=afterok:"${UJID}")
fi
scmd=(sbatch --parsable --job-name=fig1fix_scope "${dep[@]}" run_fig1_fix_eval.sh)
if [ "${DRYRUN}" = "1" ]; then printf ' %q' "${scmd[@]}"; printf '\nDRY RUN (DRYRUN=0 to submit)\n'
else SJID="$("${scmd[@]}")"; echo "submitted scope eval ${SJID} (${dep[*]:-no dependency})"; fi
