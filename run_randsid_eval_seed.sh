#!/usr/bin/env bash
#SBATCH --job-name=randsid_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=03:00:00
#
# Random-SID control, evaluation of ONE data seed (Beauty mid).
#   KIND=randsid (default): TIGER trained on embeddings/beauty_randsid.
#     1. whole-set references: retrained (clean, best epoch ckpt, as the paper's references)
#        and poisoned (last.ckpt = unlearning start) on test and validation, with the spam
#        manifest -> logs/diagnose/randsid_refs/s<seed>_{clean,poison}_{test,val}/
#     2. scope eval of every held-out row (--neighbor-def none --all-unrelated: per-query
#        NDCG / log p / item id) for clean, poison and the four unlearned arms, on test and
#        validation -> logs/diagnose/randsid_scope{,_val}/beauty_seed<seed>.json
#   KIND=orig: the same scope eval for the paper's RQ-KMeans TIGER and the canonical runs of
#     the same four arms (clean = recorded retrained model, poison = the runs' start ckpt)
#     -> logs/diagnose/randsid_scope_orig{,_val}/beauty_seed<seed>.json
# Neighborhoods are defined offline from items_per_query (paper/plotting/random_sid_table.py).
#
#   SEED=3 sbatch run_randsid_eval_seed.sh            # after the seed's unlearn job
#   KIND=orig SEED=3 sbatch run_randsid_eval_seed.sh
#   DATASET=sports SEED=3 [KIND=orig] sbatch run_randsid_eval_seed.sh   # also toys; same outputs
#     named <ds>_seed<seed>.json, refs in logs/diagnose/randsid_refs/<ds>_s<seed>_* (beauty: s<seed>_*)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SD="${SEED:?}"; KIND="${KIND:-randsid}"; DS="${DATASET:-beauty}"
RSID="embeddings/${DS}_randsid/merged_predictions_tensor.pt"
SPAM="src/data/amazon_data/${DS}_spam_tgtmid_seed${SD}_pct1_n1"
if [ "${DS}" = beauty ]; then REFPFX="s${SD}"; else REFPFX="${DS}_s${SD}"; fi

mapfile -t R < <(python3 - "${KIND}" "${SD}" "${DS}" <<'EOF'
import csv, glob, os, sys
sys.path.insert(0, "scripts")
from plan_reference_evals import ckpt_of
kind, sd, ds = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tds = "" if ds == "beauty" else f"{ds}_"
sfx = "" if sd == 2 else f"_seed{sd}"
arms = {"finetune": ("finetune", "finetune"), "forget_repair": ("forget_repair_lf0.1", "fr_lf0p1"),
        "sep_ls0p1": ("nau_ls0.1_ln0.0", "sep_ls0p1"), "nau": ("nau_ls0.01_ln0.01", "nau_s0p01_n0p01")}
def one(pattern):
    hits = [d for d in glob.glob(pattern) if glob.glob(os.path.join(d, "checkpoints", "*.ckpt"))]
    if len(hits) != 1:
        sys.exit(f"expected one run for {pattern}, got {hits}")
    return hits[0]
out = []
if kind == "randsid":
    clean = ckpt_of(one(f"logs/train/runs/*/*_job*_{ds}_clean{sfx}_randsid"))
    poison = one(f"logs/train/runs/*/*_job*_{ds}_poison_pct1_n1{sfx}_tgtmid_randsid") + "/checkpoints/last.ckpt"
    for arm, (_, tag) in arms.items():
        d = one(f"logs/unlearn/runs/job*_bs1_randsid_{tds}bw_tgtmid_{tag}{sfx}")
        out.append(f"{arm}={os.path.realpath(d + '/checkpoints/unlearned.ckpt')}")
else:
    led = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
           if r["setting"] == "spam" and r["dataset"] == ds and r["track"] == "tiger+rkmeans"
           and r["stratum"] == "mid" and int(r["seed"]) == sd]
    clean, poison = "", None
    for arm, (cfg, _) in arms.items():
        r = [x for x in led if x["config"] == cfg]
        assert len(r) == 1 and r[0]["status"] == "ok", (cfg, r)
        out.append(f"{arm}={os.path.realpath(r[0]['ckpt_resolved'])}")
        ov = dict(o[2:].split("=", 1) for o in open(os.path.join(r[0]["run_dir"], ".hydra/overrides.yaml")).read().splitlines() if o.startswith("- "))
        p = ov["ckpt_path"].strip("'\"")
        assert poison in (None, p), (poison, p)   # all four arms start from the same checkpoint
        poison = p
for x in [clean, poison] + [a.split("=", 1)[1] for a in out]:
    if x and not os.path.isfile(x):
        sys.exit(f"missing {x}")
print(clean); print(poison); print(*out, sep="\n")
EOF
)
CLEAN="${R[0]}"; POISON="${R[1]}"; ARMS=("${R[@]:2}")
[ "${#ARMS[@]}" -eq 4 ] || { echo "arm resolution failed: ${R[*]:-}" >&2; exit 1; }
echo "[randsid-eval] ${DS} kind=${KIND} seed=${SD} clean=${CLEAN:-<recorded>} poison=${POISON}"
CK=(); for a in "${ARMS[@]}"; do CK+=(--ckpt "${a}"); echo "  arm ${a}"; done

if [ "${KIND}" = "randsid" ]; then
  for which in clean poison; do
    if [ "${which}" = clean ]; then C="${CLEAN}"; else C="${POISON}"; fi
    for split in test val; do
      if [ "${split}" = test ]; then DD="src/data/amazon_data/${DS}"; else DD="src/data/amazon_data/${DS}_valview"; fi
      OUTD="logs/diagnose/randsid_refs/${REFPFX}_${which}_${split}"
      if [ -s "${OUTD}/csv/version_0/metrics.csv" ]; then echo "exists ${OUTD}"; continue; fi
      python -u -m scripts.eval_ckpt_on_test experiment=tiger_train_flat data_dir="${DD}" \
        "semantic_id_path='${RSID}'" "ckpt_path='${C}'" "spam_forget_manifest='${SPAM}/forget_manifest.json'" \
        num_hierarchies=4 seed="${SD}" train=False test=True trainer.devices=1 trainer.strategy=auto \
        trainer.sync_batchnorm=false trainer.num_nodes=1 trainer.deterministic=true \
        callbacks.model_checkpoint=null callbacks.early_stopping=null hydra.run.dir="${OUTD}"
    done
  done
  STEM=randsid_scope; EXTRA=(--sid-path "${RSID}" --clean-ckpt "${CLEAN}" --poison-ckpt "${POISON}")
else
  STEM=randsid_scope_orig; EXTRA=(--poison-ckpt "${POISON}")
fi
for split in test val; do
  if [ "${split}" = test ]; then S="${STEM}"; V=(); else S="${STEM}_val"; V=(--data-view _valview); fi
  if [ -s "logs/diagnose/${S}/${DS}_seed${SD}.json" ]; then echo "exists ${S}"; continue; fi
  python -u scripts/run_fig1_js_eval.py --dataset "${DS}" --seed "${SD}" --stratum mid --metric scope \
    --all-unrelated --neighbor-def none --out-stem "${S}" "${V[@]}" "${EXTRA[@]}" "${CK[@]}"
  [ -s "logs/diagnose/${S}/${DS}_seed${SD}.json" ] || { echo "scope eval ${S} failed" >&2; exit 1; }
done
echo "[randsid-eval] done ${DS} kind=${KIND} seed=${SD}"
