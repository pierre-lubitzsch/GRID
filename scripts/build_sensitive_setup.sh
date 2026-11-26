#!/usr/bin/env bash
# Build the full sensitive-category setup for one (dataset, seed): manifest,
# forget/retain split, and the retrain reference dataset.
#
# The recipe existed only as ad-hoc commands, which is how the seed-2 dirs were
# made; this pins it so all 5 seeds are built the same way.
#
# Three things worth knowing about the layout it produces:
#   * The <ds>_sens_<cat>_<ratio>_seed<N>/ dir SYMLINKS training/, items/,
#     evaluation/ and testing/ back to the clean dataset and materializes only
#     training_forget/ and training_retain/. Evaluation must stay byte-identical
#     across the retrained, unlearned and base models or the metrics are not
#     comparable.
#   * The retrain dataset is a real copy (~1.4 G for beauty), because it removes
#     interactions from inside real users' sessions. For SPAM the clean model is
#     already the retrain reference; for a sensitive deletion it is not.
#   * The manifest writes its user list under `spam_user_ids`. That key name is
#     reused deliberately: every downstream consumer (split_forget_retain,
#     deletion_spec, the eval metrics) then works unmodified.
#
#   ./scripts/build_sensitive_setup.sh beauty 3
#   DRYRUN=1 ./scripts/build_sensitive_setup.sh sports 2
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

KEYWORDS=""
DS="${1:?usage: build_sensitive_setup.sh <dataset> <seed>}"
SEED="${2:?usage: build_sensitive_setup.sh <dataset> <seed>}"
RATIO="${FORGET_RATIO:-1e-4}"
DRYRUN="${DRYRUN:-0}"
ROOT="src/data/amazon_data"

# dataset -> category token, category string, level.
# Verified against the item metadata on 2026-08-19 (counts are of the catalog).
# ITEM TAKEDOWN (opt-in): BRAND="<exact Brand: field>" TOK=<token> FORGET_RATIO=0 removes
# every interaction with that brand's items for every user (all touching users), e.g.
#   BRAND="e.l.f. Cosmetics" TOK=takedown_elf FORGET_RATIO=0 ./scripts/build_sensitive_setup.sh beauty 2
if [ -n "${BRAND:-}" ]; then
  TOK="${TOK:?set TOK=<token> together with BRAND}"; CAT=""; LEVEL=""
else
case "${DS}" in
  beauty)  TOK=hairloss; CAT="Hair Loss Products";  LEVEL=2 ;;   # 106 items, 0.88%
  # toys uses an ERASE-style KEYWORD concept, not a taxonomy node: levels 1-3 of
  # the toys tree hold exactly one plausible deletion request (Grown-Up Toys) and
  # the previous choice, "Baby & Toddler Toys", was unusable -- it occupies ~22%
  # of top-10 slots, so its RETRAIN reference came out ABOVE its base on 3 of 5
  # seeds. KEYWORDS is set instead of CAT/LEVEL and the manifest step branches on it.
  toys)    TOK=toyweapons; KEYWORDS="gun guns rifle rifles pistol pistols weapon weapons sword swords knife knives dagger"; CAT=""; LEVEL="" ;;  # 68 items, 0.37% of interactions
  sports)  TOK=guns;     CAT="Guns & Rifles";       LEVEL=3 ;;   # 202 items, 1.10%
  food_k5) TOK=meat;     CAT="Meat & Seafood";      LEVEL=1 ;;   # 1904 items, 1.14%
  *) echo "unknown dataset '${DS}'" >&2; exit 1 ;;
esac
fi

CLEAN="${ROOT}/${DS}"
MAN="${CLEAN}/forget_manifest_${TOK}_${RATIO}_seed${SEED}.json"
SENS="${ROOT}/${DS}_sens_${TOK}_${RATIO}_seed${SEED}"
RETRAIN="${ROOT}/${DS}_retrain_${TOK}_${RATIO}_seed${SEED}"

echo "== ${DS} seed ${SEED}: ${BRAND:+brand '${BRAND}' }${KEYWORDS:+keywords '${KEYWORDS}'}${CAT:+category '${CAT}' (level ${LEVEL})}, ratio ${RATIO}"
echo "   manifest: ${MAN}"
echo "   sens dir: ${SENS}"
echo "   retrain : ${RETRAIN}"
[ "${DRYRUN}" = "1" ] && exit 0

# 1. manifest
if [ -f "${MAN}" ]; then
  echo "[skip] manifest exists"
else
  if [ -n "${BRAND:-}" ]; then
    PYTHONPATH=. python -m scripts.build_sensitive_manifest \
      --data_dir "${CLEAN}" \
      --brand "${BRAND}" \
      --seed "${SEED}" --forget_ratio "${RATIO}" \
      --deletion_spec item_pairs \
      --out "${MAN}"
  elif [ -n "${KEYWORDS}" ]; then
    # shellcheck disable=SC2086  # KEYWORDS is a deliberate word list
    PYTHONPATH=. python -m scripts.build_sensitive_manifest \
      --data_dir "${CLEAN}" \
      --keywords ${KEYWORDS} \
      --seed "${SEED}" --forget_ratio "${RATIO}" \
      --deletion_spec item_pairs \
      --out "${MAN}"
  else
    PYTHONPATH=. python -m scripts.build_sensitive_manifest \
      --data_dir "${CLEAN}" \
      --category "${CAT}" --category_level "${LEVEL}" \
      --seed "${SEED}" --forget_ratio "${RATIO}" \
      --deletion_spec item_pairs \
      --out "${MAN}"
  fi
fi

# 2. sensitive dataset dir: symlink the shared splits, materialize the partition
if [ -d "${SENS}/training_retain" ]; then
  echo "[skip] ${SENS} already split"
else
  mkdir -p "${SENS}"
  for sub in training items evaluation testing; do
    [ -e "${SENS}/${sub}" ] || ln -s "$(readlink -f "${CLEAN}/${sub}")" "${SENS}/${sub}"
  done
  cp "${MAN}" "${SENS}/forget_manifest.json"
  PYTHONPATH=. python -m src.data.unlearning.split_forget_retain \
    --data_dir "${SENS}" \
    --forget_manifest "${SENS}/forget_manifest.json"
fi

# 3. retrain reference
if [ -d "${RETRAIN}/training" ]; then
  echo "[skip] ${RETRAIN} exists"
else
  PYTHONPATH=. python -m scripts.build_retrain_dataset \
    --data_dir "${CLEAN}" --manifest "${MAN}" --out_dir "${RETRAIN}"
fi

echo "== done: ${DS} seed ${SEED}"
