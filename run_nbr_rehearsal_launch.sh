#!/usr/bin/env bash
# R3: "simpler neighborhood regularizer at comparable neighborhood size and budget".
# Forget+Repair (canonical: unified lf0.1 ls0 ln0, ledger forget_repair_lf0.1) whose retain batches
# are enriched with retain rows whose NEXT item is one of NAU's 8 neighbors (top-8 cosine in the
# pre-quantization embedding space, forget items excluded; the same call NAU's L_n makes), via the
# opt-in unlearning.nbr_rehearsal_rate (TigerUnlearningModule._apply_nbr_rehearsal). Each retain
# batch keeps its row count; a share r of its rows is replaced, so batches, rows/step, epochs and
# optimizer steps equal F+R's. Two rates: r = 0.25 and 0.5.
# Each row replays the cell's CANONICAL F+R run exactly (same poisoned ckpt file, SID, unlearning
# seed, data dir and unlearning.* overrides from .hydra/overrides.yaml) plus the new keys, through
# run_valsel_batch.sh with VALSEL_KEEP_CKPT=1 (test + val + checkpoint).
# Tags bw_tgt<st>_fr_nbrrehearse<25|50>_canonck[_seed<N>] (arm "fr_..." -> algo "other" in
# scripts/run_tags.py, so no table reader treats them as an existing config).
# Gate: CPU smoke test (scripts/smoke_nbr_rehearsal.py) -> unlearning batches (afterok) ->
# neighborhood scoring test + val (run_scope_nbrreh.sh, afterok).
#
#   bash run_nbr_rehearsal_launch.sh            # DRY RUN
#   DRYRUN=0 bash run_nbr_rehearsal_launch.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
mkdir -p logs/nbr_rehearsal
PLAN="logs/nbr_rehearsal/plan.tsv"
python3 - <<'EOF' > "${PLAN}"
import csv, os, yaml
POS = ("unlearning.algorithm=", "unlearning.neighborhood_aware=",
       "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=")
NEW = "unlearning.nbr_rehearsal_count=8 unlearning.nbr_rehearsal_neighbor_method=embedding " \
      "unlearning.coherence_embedding_metric=cosine " \
      "unlearning.embedding_path=embeddings/beauty_merged_predictions_tensor_latest.pt"
rows = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
        if r["setting"] == "spam" and r["dataset"] == "beauty" and r["config"] == "forget_repair_lf0.1"]
assert len(rows) == 15, len(rows)
for rate in ("0.25", "0.5"):
    rt = str(int(round(float(rate) * 100)))
    for r in sorted(rows, key=lambda r: (["unpopular", "mid", "popular"].index(r["stratum"]), int(r["seed"]))):
        st, sd, tmpl = r["stratum"], int(r["seed"]), r["run_dir"]
        ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
        kv = dict(o.split("=", 1) for o in ov)
        assert kv["unlearning.lambda_f"] == "0.1" and kv["unlearning.lambda_s"] == "0.0" and kv["unlearning.lambda_n"] == "0.0"
        assert not any(k.startswith("unlearning.nbr_rehearsal") for k in kv)
        useed = str(int(kv["seed"])); assert int(useed) == sd
        ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
        assert os.path.isfile(ckpt), ckpt; assert os.path.isfile(sid), sid
        base = os.path.basename(ckpt.split("/checkpoints/")[0])
        assert "_beauty_poison_pct1_n1" in base and base.endswith(f"_tgt{st}"), ckpt
        assert (sd == 2 and "_seed" not in base) or f"_seed{sd}_" in base, ckpt
        extra = [o for o in ov if o.startswith("unlearning.") and not o.startswith(POS)]
        extra.append(f"unlearning.nbr_rehearsal_rate={rate}")
        sfx = "" if sd == 2 else f"_seed{sd}"
        tag = f"bw_tgt{st}_fr_nbrrehearse{rt}_canonck{sfx}"
        print("\t".join(["beauty", st, str(sd), useed, tag, ckpt, kv["unlearning.algorithm"], sid,
                         kv.get("unlearning.neighborhood_aware", "false"),
                         kv.get("unlearning.request_batch_size", "1"),
                         kv.get("unlearning.neighborhood_aware_sample_rate", "0.0"),
                         kv["data_dir"], tmpl, " ".join(extra) + " " + NEW]))
EOF
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"
QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^(nbrreh_)' <<<"${QUEUED}"; then echo "Refusing: nbrreh_ jobs queued"; grep -E '^nbrreh_' <<<"${QUEUED}"; exit 1; fi
todo=0
while IFS=$'\t' read -r ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  exists: ${tag}"; continue; fi
  resolved="$(POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1 POISON_SEED="${sd}" TARGET_STRATEGY="${st}" \
    bash -c 'source scripts/resolve_unlearn_dataset.sh >/dev/null 2>&1; resolve_unlearn_dataset "$0" >/dev/null 2>&1; echo "${UNLEARN_DATA_DIR}"' "${ds}")"
  [ "${resolved}" = "${data_dir}" ] || { echo "MISMATCH ${tag}: ${resolved} vs ${data_dir}" >&2; exit 1; }
  todo=$((todo+1))
done < "${PLAN}"
echo "todo: ${todo}"
# two packed batch jobs (15 rows each, 2 lanes)
head -15 "${PLAN}" > logs/nbr_rehearsal/plan_r25.tsv; tail -n +16 "${PLAN}" > logs/nbr_rehearsal/plan_r50.tsv
run() { if [ "${DRYRUN}" = "1" ]; then printf '%q ' "$@" >&2; echo >&2; echo "<jid>"; else "$@"; fi; }
SM=$(run sbatch --parsable --job-name=nbrreh_smoke --partition=compute --cpus-per-task=4 --mem=48G --time=01:00:00 \
  --output=logs/nbr_rehearsal/smoke_%j.out \
  --wrap "set -e; export CUDA_VISIBLE_DEVICES=; python -u scripts/smoke_nbr_rehearsal.py logs/nbr_rehearsal/smoke_work")
echo "smoke ${SM}"
U=()
for r in 25 50; do
  j=$(VALSEL_KEEP_CKPT=1 run sbatch --parsable --job-name=nbrreh_r${r} --dependency=afterok:${SM} --time=06:00:00 \
      run_valsel_batch.sh logs/nbr_rehearsal/plan_r${r}.tsv)
  echo "unlearn r${r} ${j}"; U+=("${j}")
done
dep="afterok:${U[0]}:${U[1]}"
for split in test val; do
  j=$(run sbatch --parsable --job-name=nbrreh_scope_${split} --dependency=${dep} --export=ALL,SPLIT=${split} run_scope_nbrreh.sh)
  echo "scope ${split} ${j}"
done
