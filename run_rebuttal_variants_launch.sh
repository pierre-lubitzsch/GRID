#!/usr/bin/env bash
# Launch the rebuttal-variant runs (plans from scripts/plan_rebuttal_variants.py):
#   1 smoke job  (2 rows: Beauty popular seed 2 NAU capref, Sports seed 2 NAU anchor 1.0)
#   2 spam jobs  (capref, the other 24 rows)        afterany:smoke, guarded on the smoke rows
#   3 sens jobs  (anchor, the other 29 rows)        afterany:smoke, guarded on the smoke rows
# All pgpu with 2 GPUs (2 lanes per job), VALSEL_KEEP_CKPT=1. Idempotent: rows whose run
# dir exists are skipped by the packers; refuses if an rbv_ job is already queued.
#
#   bash run_rebuttal_variants_launch.sh            # DRY RUN (default)
#   TEST_JOB=<id of run_test_ref_cap_anchor.sh> makes the smoke job afterok on the unit test
#   DRYRUN=0 bash run_rebuttal_variants_launch.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
D=logs/rebuttal_variants
python3 scripts/plan_rebuttal_variants.py
QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^rbv_' <<<"${QUEUED}"; then echo "Refusing: rbv_ jobs queued:"; grep -E '^rbv_' <<<"${QUEUED}"; exit 1; fi

SMOKE_SPAM_TAG=bw_tgtpopular_nau_s0p01_n0p01_capref_canonck
SMOKE_SENS_TAG=sens_sports_guns_unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0_anchor1p0_canonck
awk -F'\t' -v t="${SMOKE_SPAM_TAG}" '$5==t' ${D}/plan_capref.tsv > ${D}/smoke_spam.tsv
awk -F'\t' -v t="${SMOKE_SENS_TAG}" '$5==t' ${D}/plan_anchor.tsv > ${D}/smoke_sens.tsv
[ "$(wc -l < ${D}/smoke_spam.tsv)" = 1 ] && [ "$(wc -l < ${D}/smoke_sens.tsv)" = 1 ] || { echo "smoke rows not found"; exit 1; }
awk -F'\t' -v t="${SMOKE_SPAM_TAG}" '$5!=t' ${D}/plan_capref.tsv > ${D}/rest_capref.tsv
awk -F'\t' -v t="${SMOKE_SENS_TAG}" '$5!=t' ${D}/plan_anchor.tsv > ${D}/rest_anchor.tsv
# interleave strata/datasets across jobs (round robin by line) so each job mixes cell types
awk -v n=2 -v p="${D}/capref_" '{print > (p (NR-1)%n ".tsv")}' ${D}/rest_capref.tsv
awk -v n=3 -v p="${D}/anchor_" '{print > (p (NR-1)%n ".tsv")}' ${D}/rest_anchor.tsv
wc -l ${D}/smoke_*.tsv ${D}/capref_?.tsv ${D}/anchor_?.tsv

sub() {  # name time dep args...
  local name="$1" time="$2" dep="$3"; shift 3
  local cmd=(sbatch --parsable --job-name="${name}" --time="${time}")
  [ -n "${dep}" ] && cmd+=(--dependency="afterany:${dep}")
  [ -n "${SUBDEP_OK:-}" ] && cmd+=(--dependency="afterok:${SUBDEP_OK}")   # smoke waits for the unit test
  cmd+=(run_rebuttal_variants_batch.sh "$@")
  if [ "${DRYRUN}" = 1 ]; then printf '%q ' "${cmd[@]}" >&2; echo >&2; echo "DRY"; else "${cmd[@]}"; fi
}
SMOKE="$(SUBDEP_OK="${TEST_JOB:-}" sub rbv_smoke 02:00:00 "" smoke ${D}/smoke_spam.tsv ${D}/smoke_sens.tsv)"
echo "smoke: ${SMOKE}"
for i in 0 1; do
  echo "rbv_capref_${i}: $(sub rbv_capref_${i} 04:00:00 "${SMOKE}" spam ${D}/capref_${i}.tsv "${SMOKE_SPAM_TAG}")"
done
for i in 0 1 2; do
  echo "rbv_anchor_${i}: $(sub rbv_anchor_${i} 05:00:00 "${SMOKE}" sens ${D}/anchor_${i}.tsv "${SMOKE_SENS_TAG}")"
done
[ "${DRYRUN}" = 1 ] && echo "DRY RUN (DRYRUN=0 to submit)"
exit 0
