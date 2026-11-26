#!/usr/bin/env bash
# Canonical re-runs WITH checkpoint of +L_sep at lambda_s=0.01 (config nau_ls0.01_ln0.0:
# lambda_f=0.1, lambda_s=0.01, lambda_n=0) for the 25 spam cells whose canonical execution is a
# checkpoint-less `_valsel` run: Sports all strata (15) and Toys unpopular/popular (10). Needed for
# the SASRec vs TIGER contrast (paper/revision/sasrec_vs_tiger.md), which scope-evaluates them.
#
# Same mechanics as run_canonck_missing_launch.sh: each row replays the recorded `_valsel` run of
# its cell exactly (poisoned ckpt, SID, unlearning seed, data dir and unlearning.* overrides from
# .hydra/overrides.yaml; unlearning.checkpoint_fractions is re-added by the runner), through
# run_valsel_batch.sh -> run_valsel_one.sh (unlearn + test eval + validation eval) with
# VALSEL_KEEP_CKPT=1. Tag: <ds>_bw_tgt<st>_nau_s0p01_n0p0_canonck[_seedN] (no `_valsel`, so the
# valsel globs ignore it); scripts/build_canonical_runs.py designates it for these cells.
# Then the scope eval (run_scope_sep001.sh, afterok) -> logs/diagnose/fig1_scope_sep001/.
#
#   bash run_canonck_sep001_launch.sh             # DRY RUN (default)
#   DRYRUN=0 bash run_canonck_sep001_launch.sh    # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
D=logs/canonck_sep001
mkdir -p "${D}"
PLAN="${D}/plan.tsv"

if [ ! -s "${PLAN}" ]; then
python3 - <<'EOF' > "${PLAN}.tmp"
import csv, os, re, yaml
POS = ("unlearning.algorithm=", "unlearning.neighborhood_aware=",
       "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=",
       "unlearning.checkpoint_fractions=")
rows = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
        if r["setting"] == "spam" and r["track"] == "tiger+rkmeans" and r["config"] == "nau_ls0.01_ln0.0"
        and (r["dataset"] == "sports" or (r["dataset"] == "toys" and r["stratum"] != "mid"))]
assert len(rows) == 25, len(rows)
for r in sorted(rows, key=lambda r: (r["dataset"], r["stratum"], int(r["seed"]))):
    assert r["source"] == "valsel_replay" and r["has_ckpt"] == "0" and r["status"] == "ok", r
    ds, st, sd, tmpl, rtag = r["dataset"], r["stratum"], int(r["seed"]), r["run_dir"], r["tag"]
    ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
    kv = dict(o.split("=", 1) for o in ov)
    assert kv["unlearning_run_tag"] == rtag, (kv["unlearning_run_tag"], rtag)
    useed = str(int(kv["seed"])); assert int(useed) == sd, (tmpl, useed)
    assert (kv["unlearning.lambda_f"], kv["unlearning.lambda_s"], kv["unlearning.lambda_n"]) == ("0.1", "0.01", "0.0"), kv
    ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
    assert os.path.isfile(ckpt), ckpt; assert os.path.isfile(sid), sid
    assert sid == f"embeddings/{ds}/merged_predictions_tensor.pt", sid
    assert kv.get("num_hierarchies", "4") == "4"
    base = os.path.basename(ckpt.split("/checkpoints/")[0])          # wrong-seed guard
    assert f"_{ds}_poison_pct1_n1" in base and base.endswith(f"_tgt{st}"), ckpt
    assert (sd == 2 and "_seed" not in base) or f"_seed{sd}_" in base, ckpt
    algo = kv["unlearning.algorithm"]; assert algo == "unified"
    assert kv["experiment"] == "tiger_unlearn_unified_sequential", kv["experiment"]
    na = kv.get("unlearning.neighborhood_aware", "false"); rbs = kv.get("unlearning.request_batch_size", "1")
    rate = kv.get("unlearning.neighborhood_aware_sample_rate", "0.0")
    extra = [o for o in ov if o.startswith("unlearning.") and not o.startswith(POS)]
    tag = f"{ds}_bw_tgt{st}_nau_s0p01_n0p0_canonck" + ("" if sd == 2 else f"_seed{sd}")
    print("\t".join([ds, st, str(sd), useed, tag, ckpt, algo, sid, na, rbs, rate,
                     kv["data_dir"], tmpl, " ".join(extra)]))
EOF
mv "${PLAN}.tmp" "${PLAN}"
fi
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^(vsb_canonck_sep001|scope_sep001)' <<<"${QUEUED}"; then
  echo "Refusing: already queued/running:"; grep -E '^(vsb_canonck_sep001|scope_sep001)' <<<"${QUEUED}"; exit 1
fi
: > "${D}/todo.tsv"
while IFS=$'\t' read -r ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  exists: ${tag}"; continue; fi
  resolved="$(POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1 POISON_SEED="${sd}" TARGET_STRATEGY="${st}" \
    bash -c 'source scripts/resolve_unlearn_dataset.sh >/dev/null 2>&1; resolve_unlearn_dataset "$0" >/dev/null 2>&1; echo "${UNLEARN_DATA_DIR}"' "${ds}")"
  [ "${resolved}" = "${data_dir}" ] || { echo "MISMATCH ${tag}: ${resolved} vs ${data_dir}" >&2; exit 1; }
  echo "  todo:   ${tag}  <- $(basename "${tmpl}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" \
    "${ckpt}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${data_dir}" "${tmpl}" "${extra}" >> "${D}/todo.tsv"
done < "${PLAN}"
rm -f "${D}"/todo_[0-9]*.tsv
split -n l/2 -d --additional-suffix=.tsv "${D}/todo.tsv" "${D}/todo_"
deps=()
for f in "${D}"/todo_0*.tsv; do
  n=$(wc -l < "${f}"); [ "${n}" -gt 0 ] || continue
  name="vsb_canonck_sep001_$(basename "${f}" .tsv | sed 's/todo_//')"
  cmd=(sbatch --parsable --job-name="${name}" --time=06:00:00 run_valsel_batch.sh "${f}")
  if [ "${DRYRUN}" = "1" ]; then printf 'VALSEL_KEEP_CKPT=1'; printf ' %q' "${cmd[@]}"; printf '   # %s rows\n' "${n}"
  else j=$(VALSEL_KEEP_CKPT=1 "${cmd[@]}"); deps+=("${j}"); echo "submitted ${name} (${n} rows): ${j}"; fi
done
dep=""; [ "${#deps[@]}" -gt 0 ] && dep="--dependency=afterok:$(IFS=:; echo "${deps[*]}")"
scmd=(sbatch --parsable --job-name=scope_sep001 ${dep} run_scope_sep001.sh)
if [ "${DRYRUN}" = "1" ]; then printf '%q ' "${scmd[@]}"; printf '\nDRY RUN (DRYRUN=0 to submit)\n'
else echo "submitted scope_sep001: $("${scmd[@]}")"; fi
